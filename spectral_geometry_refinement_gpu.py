"""
Spectral Geometry Refinement (SGR)

Post-processing refinement for a global clustering partition obtained from
Dynamic (MF)^2A. SGR analyzes complete spectra using local spectral landmarks,
regional spectral trends, GMM+BIC, view agreement, co-association, and
consensus validation.

This 2-GPU vectorized version avoids forcing artificial landmarks when no significant local
extrema are present and batches the main CUDA GMM linear algebra across mixture components. Candidate landmarks are retained only when they have
sufficient prominence/width in the population mean and are supported by a
fraction of the individual spectra. Regional/global trend views remain active,
so slope/curvature structure can still be detected without requiring a band.
"""

import os
import multiprocessing as mp
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score

try:
    import cupy as cp
    _CUPY_AVAILABLE=True
except Exception:
    cp=None
    _CUPY_AVAILABLE=False

class _CuPyFullGMM:
    """Small full-covariance Gaussian mixture model executed with CuPy.

    It is used only for SGR view fitting. The statistical criterion remains
    GMM + BIC; CPU scikit-learn remains the fallback backend.
    """
    def __init__(self,n_components,n_init=1,max_iter=1500,reg_covar=1e-5,tol=1e-3,random_state=42):
        self.n_components=int(n_components); self.n_init=int(n_init); self.max_iter=int(max_iter)
        self.reg_covar=float(reg_covar); self.tol=float(tol); self.random_state=int(random_state)
        self.weights_=None; self.means_=None; self.covariances_=None; self.lower_bound_=-cp.inf if _CUPY_AVAILABLE else None

    @staticmethod
    def _logsumexp(A,axis=1):
        m=cp.max(A,axis=axis,keepdims=True); return cp.squeeze(m,axis=axis)+cp.log(cp.sum(cp.exp(A-m),axis=axis))

    def _estimate_log_prob(self,X,weights,means,covs):
        # Batched CUDA likelihood across all mixture components (K <= refine_kmax).
        n,d=X.shape; eye=cp.eye(d,dtype=X.dtype)[None,:,:]; covs=0.5*(covs+cp.swapaxes(covs,1,2))
        L=None
        for attempt in range(7):
            try:
                jitter=(10.0**attempt)*self.reg_covar if attempt else self.reg_covar
                L=cp.linalg.cholesky(covs+jitter*eye); break
            except Exception: pass
        if L is None: L=cp.linalg.cholesky(covs+1e-2*eye)
        diff=cp.transpose(X[None,:,:]-means[:,None,:],(0,2,1))
        sol=cp.linalg.solve(L,diff); maha=cp.sum(sol*sol,axis=1).T
        logdet=2.0*cp.sum(cp.log(cp.diagonal(L,axis1=1,axis2=2)),axis=1)
        log2pi=cp.log(cp.asarray(2.0*cp.pi,dtype=X.dtype))
        return -0.5*(d*log2pi+logdet[None,:]+maha)+cp.log(cp.maximum(weights,1e-15))[None,:]

    def _init_params(self,X,seed):
        n,d=X.shape; rng=cp.random.RandomState(seed); idx=rng.choice(n,size=self.n_components,replace=False); means=X[idx].copy()
        # Short GPU k-means warm start, analogous in purpose to sklearn GMM's default k-means initialization.
        labels=cp.zeros(n,dtype=cp.int32)
        for _ in range(25):
            dist=cp.sum((X[:,None,:]-means[None,:,:])**2,axis=2); new_labels=cp.argmin(dist,axis=1)
            if bool(cp.all(new_labels==labels).item()) and _>0: break
            labels=new_labels
            for k in range(self.n_components):
                mask=labels==k
                if int(cp.sum(mask).item())>0: means[k]=cp.mean(X[mask],axis=0)
                else: means[k]=X[int(rng.randint(0,n))]
        counts=cp.bincount(labels,minlength=self.n_components).astype(X.dtype); weights=cp.maximum(counts,1.0); weights/=cp.sum(weights)
        covs=cp.empty((self.n_components,d,d),dtype=X.dtype); eye=cp.eye(d,dtype=X.dtype)
        global_centered=X-cp.mean(X,axis=0); global_cov=(global_centered.T@global_centered)/max(n,1)+self.reg_covar*eye
        for k in range(self.n_components):
            mask=labels==k; nk=int(cp.sum(mask).item())
            if nk>1:
                diff=X[mask]-means[k]; covs[k]=(diff.T@diff)/nk+self.reg_covar*eye
            else: covs[k]=global_cov
        return weights,means,covs

    def fit(self,X):
        X=cp.asarray(X); n,d=X.shape; best=None; best_lb=-cp.inf
        for init in range(self.n_init):
            weights,means,covs=self._init_params(X,self.random_state+104729*self.n_components+1009*init); prev=None
            for _ in range(self.max_iter):
                logp=self._estimate_log_prob(X,weights,means,covs); lognorm=self._logsumexp(logp,axis=1); lb=cp.mean(lognorm); resp=cp.exp(logp-lognorm[:,None])
                nk=cp.sum(resp,axis=0)+10.0*cp.finfo(X.dtype).eps; weights=nk/n; means=(resp.T@X)/nk[:,None]; eye=cp.eye(d,dtype=X.dtype)
                diff=X[None,:,:]-means[:,None,:]; covs=cp.einsum("kn,kni,knj->kij",resp.T,diff,diff)/nk[:,None,None]+self.reg_covar*eye[None,:,:]
                if prev is not None and abs(float((lb-prev).item()))<self.tol: break
                prev=lb
            if float(lb.item())>float(best_lb.item() if hasattr(best_lb,'item') else best_lb):
                best_lb=lb; best=(weights.copy(),means.copy(),covs.copy())
        if best is None: raise RuntimeError('GPU GMM failed to converge.')
        self.weights_,self.means_,self.covariances_=best; self.lower_bound_=best_lb; self.n_features_in_=d; return self

    def score_samples(self,X):
        X=cp.asarray(X); logp=self._estimate_log_prob(X,self.weights_,self.means_,self.covariances_); return self._logsumexp(logp,axis=1)

    def predict(self,X):
        X=cp.asarray(X); logp=self._estimate_log_prob(X,self.weights_,self.means_,self.covariances_); return cp.asnumpy(cp.argmax(logp,axis=1)).astype(int)

    def bic(self,X):
        X=cp.asarray(X); n,d=X.shape; ll=cp.sum(self.score_samples(X)); K=self.n_components
        n_params=(K-1)+K*d+K*d*(d+1)/2.0
        return float((-2.0*ll+n_params*cp.log(n)).item())

@dataclass
class SGRConfig:
    seed: int = 42
    max_refine_depth: int = 2
    refine_kmax: int = 4
    min_parent_n: int = 80
    min_child_n: int = 3
    min_child_frac: float = 0.0
    prom_frac: float = 0.035
    peak_distance: int = 3
    landmark_min_width_channels: float = 3.0
    landmark_support_frac: float = 0.50
    landmark_support_tolerance_channels: int = 8
    landmark_support_prom_scale: float = 0.75
    delta_bic_min: float = 10.0
    ari_view_min: float = 0.30
    min_coherent_views: int = 2
    cons_sil_min: float = 0.15
    cons_ari_min: float = 0.30
    gmm_n_init: int = 15
    gmm_max_iter: int = 1500
    gmm_tol: float = 1e-3
    gpu_backend: str = "gpu"  # forced CUDA GPU backend; use "cpu" only if explicitly requested
    gpu_dtype: str = "float32"
    gpu_devices: tuple = (0,1)
    parallel_root_clusters: bool = True
    gap_factor: float = 2.5
    gap_min: float = 0.05
    use_regional_trends: bool = True
    trend_window_channels: tuple = ()
    trend_step_fraction: float = 1.0
    trend_min_channels: int = 5
    trend_include_full_region: bool = False
    trend_regions: tuple = ((0.45,0.55),(0.55,2.50),(0.45,2.50))

@dataclass
class SGRResult:
    global_labels: np.ndarray
    refined_labels: np.ndarray
    history: pd.DataFrame
    split_records: list
    wavelengths: np.ndarray
    regions: list
    X: np.ndarray = field(repr=False)

    @property
    def n_global_clusters(self): return len(np.unique(self.global_labels))
    @property
    def n_refined_clusters(self): return len(np.unique(self.refined_labels))
    @property
    def n_splits(self): return len(self.split_records)

def _sgr_gpu_worker(device,cfg,X,wave,global_labels,root_labels,q):
    try:
        if not _CUPY_AVAILABLE: raise RuntimeError("CuPy/CUDA is not available in SGR worker.")
        cp.cuda.Device(int(device)).use()
        ref=SpectralGeometryRefiner(config=cfg); ref.X=np.asarray(X,float); ref.wave=np.asarray(wave,float); ref.regions=ref._build_regions(); ref.history=[]; ref.split_records=[]
        roots=[]
        unique_all=list(np.unique(global_labels)); display={k:i+1 for i,k in enumerate(unique_all)}
        for k in root_labels:
            idx=np.where(global_labels==k)[0]; leaves=ref._refine_node(idx,0,f"G{display[k]}",k); roots.append((k,leaves))
        q.put({"ok":True,"device":int(device),"roots":roots,"history":ref.history,"split_records":ref.split_records})
    except Exception as e:
        import traceback; q.put({"ok":False,"device":int(device),"error":repr(e),"traceback":traceback.format_exc()})

class SpectralGeometryRefiner:
    def __init__(self, config=None):
        self.cfg=config if config is not None else SGRConfig()
        if self.cfg.gpu_backend not in ("auto","gpu","cpu"): raise ValueError("gpu_backend must be 'auto', 'gpu', or 'cpu'.")
        if self.cfg.gpu_dtype not in ("float32","float64"): raise ValueError("gpu_dtype must be 'float32' or 'float64'.")
        if self.cfg.parallel_root_clusters and self.cfg.gpu_backend=="gpu" and len(tuple(self.cfg.gpu_devices))<1: raise ValueError("gpu_devices must contain at least one CUDA device.")
        self.use_gpu=self.cfg.gpu_backend=="gpu" or (self.cfg.gpu_backend=="auto" and _CUPY_AVAILABLE)
        if self.cfg.gpu_backend=="gpu" and not _CUPY_AVAILABLE: raise RuntimeError("gpu_backend='gpu' requested, but CuPy/CUDA is not available.")
        if self.cfg.min_child_n<1: raise ValueError("min_child_n must be >= 1.")
        if not 0.0<=self.cfg.min_child_frac<=1.0: raise ValueError("min_child_frac must be in [0,1].")
        if self.cfg.min_parent_n<2*self.cfg.min_child_n: raise ValueError("min_parent_n must be at least 2*min_child_n.")
        if self.cfg.landmark_min_width_channels<1: raise ValueError("landmark_min_width_channels must be >= 1.")
        if not 0.0<=self.cfg.landmark_support_frac<=1.0: raise ValueError("landmark_support_frac must be in [0,1].")
        if self.cfg.landmark_support_tolerance_channels<0: raise ValueError("landmark_support_tolerance_channels must be >= 0.")
        if self.cfg.landmark_support_prom_scale<=0: raise ValueError("landmark_support_prom_scale must be > 0.")
        if self.cfg.trend_min_channels<3: raise ValueError("trend_min_channels must be >= 3.")
        if not 0.0<self.cfg.trend_step_fraction<=1.0: raise ValueError("trend_step_fraction must be in (0,1].")
        for n in self.cfg.trend_window_channels:
            if int(n)<self.cfg.trend_min_channels: raise ValueError("Each trend_window_channels value must be >= trend_min_channels.")
        for bounds in self.cfg.trend_regions:
            if len(bounds)!=2 or not np.isfinite(bounds[0]) or not np.isfinite(bounds[1]) or bounds[0]>=bounds[1]:
                raise ValueError("Each trend_regions entry must be a finite (lambda_start, lambda_end) pair with start < end.")
        self.X=None; self.wave=None; self.regions=None; self.history=[]; self.split_records=[]

    def _validate_inputs(self,X,wavelengths,global_labels):
        X=np.asarray(X,float); wave=np.asarray(wavelengths,float).ravel(); labels=np.asarray(global_labels).ravel()
        if X.ndim!=2: raise ValueError("X must be a 2D matrix with shape (N, d).")
        if len(wave)!=X.shape[1]: raise ValueError("wavelengths must have length d, matching X.shape[1].")
        if len(labels)!=X.shape[0]: raise ValueError("global_labels must have length N, matching X.shape[0].")
        if X.shape[1]<3: raise ValueError("At least three spectral features are required.")
        if not np.all(np.isfinite(wave)): raise ValueError("wavelengths contains non-finite values.")
        order=np.argsort(wave); wave=wave[order]; X=X[:,order]
        if np.any(np.diff(wave)<=0): raise ValueError("wavelengths must contain unique values.")
        X=pd.DataFrame(X).replace([np.inf,-np.inf],np.nan)
        if X.isna().all(axis=0).any(): raise ValueError("At least one spectral column contains no finite values.")
        X=X.fillna(X.median()).to_numpy(float)
        if not np.all(np.isfinite(X)): raise ValueError("X still contains non-finite values after median filling.")
        return X,wave,labels

    def _build_regions(self):
        dw=np.diff(self.wave)
        if len(dw)==0: return [np.arange(len(self.wave))]
        positive=dw[dw>0]; med=float(np.median(positive)) if len(positive) else 0.0
        gap_thr=max(self.cfg.gap_min,self.cfg.gap_factor*med); cuts=np.where(dw>gap_thr)[0]+1
        regions=[r for r in np.split(np.arange(len(self.wave)),cuts) if len(r)>=3]
        return regions if regions else [np.arange(len(self.wave))]

    def _derivatives(self,X):
        if self.use_gpu:
            dtype=cp.float32 if self.cfg.gpu_dtype=="float32" else cp.float64; Xg=cp.asarray(X,dtype=dtype); D1g=cp.full_like(Xg,cp.nan); D2g=cp.full_like(Xg,cp.nan)
            for r in self.regions:
                rg=cp.asarray(r); wg=cp.asarray(self.wave[r],dtype=dtype); d1=cp.gradient(Xg[:,rg],wg,axis=1,edge_order=2); D1g[:,rg]=d1; D2g[:,rg]=cp.gradient(d1,wg,axis=1,edge_order=2)
            return cp.asnumpy(D1g),cp.asnumpy(D2g)
        D1=np.full_like(X,np.nan,float); D2=np.full_like(X,np.nan,float)
        for r in self.regions:
            D1[:,r]=np.gradient(X[:,r],self.wave[r],axis=1,edge_order=2); D2[:,r]=np.gradient(D1[:,r],self.wave[r],axis=1,edge_order=2)
        return D1,D2

    @staticmethod
    def _quad_vertex(w,y,idx,lo,hi,kind="min",half=2):
        s=max(lo,idx-half); e=min(hi,idx+half)
        if e-s+1<3: return float(w[idx]),float(y[idx]),np.nan
        try:
            a,b,c=np.polyfit(w[s:e+1],y[s:e+1],2); xv=-b/(2*a) if abs(a)>1e-14 else w[idx]
            good=w[s]<=xv<=w[e] and ((kind=="min" and a>0) or (kind=="max" and a<0))
            if not good: return float(w[idx]),float(y[idx]),np.nan
            return float(xv),float(a*xv*xv+b*xv+c),float(2*a)
        except Exception: return float(w[idx]),float(y[idx]),np.nan

    @staticmethod
    def _slope_fit(w,y,a,b):
        if b-a+1<2: return np.nan
        try: return float(np.polyfit(w[a:b+1],y[a:b+1],1)[0])
        except Exception: return np.nan

    @staticmethod
    def _zero_cross(w,d,a,b,center_idx,side):
        values=[]
        for j in range(a,b):
            y1,y2=d[j],d[j+1]
            if not (np.isfinite(y1) and np.isfinite(y2)): continue
            if y1==0: x=w[j]
            elif y1*y2<0: x=w[j]-y1*(w[j+1]-w[j])/(y2-y1)
            else: continue
            center=.5*(w[j]+w[j+1])
            if side=="left" and center<w[center_idx]: values.append(x)
            if side=="right" and center>w[center_idx]: values.append(x)
        if not values: return np.nan
        return float(max(values) if side=="left" else min(values))

    @staticmethod
    def _band_shoulders(w,y,imin,a,b):
        if imin<=a or imin>=b: return np.nan,None,None
        yl=y[a:imin]; yr=y[imin+1:b+1]
        if len(yl)==0 or len(yr)==0 or not np.any(np.isfinite(yl)) or not np.any(np.isfinite(yr)): return np.nan,None,None
        ileft=a+int(np.nanargmax(yl)); iright=imin+1+int(np.nanargmax(yr))
        if not (a<=ileft<imin<iright<=b): return np.nan,None,None
        return float(w[iright]-w[ileft]),ileft,iright

    def _supported_landmarks(self,YR,mean_flux,kind):
        base_signal=-mean_flux if kind=="min" else mean_flux
        mean_prom=max(self.cfg.prom_frac*np.ptp(mean_flux),1e-6)
        candidates,_=find_peaks(base_signal,prominence=mean_prom,distance=self.cfg.peak_distance,width=self.cfg.landmark_min_width_channels)
        if len(candidates)==0 or self.cfg.landmark_support_frac<=0: return candidates.astype(int)
        tol=int(self.cfg.landmark_support_tolerance_channels); support=np.zeros(len(candidates),dtype=int)
        for y in YR:
            signal=-y if kind=="min" else y
            individual_prom=max(self.cfg.landmark_support_prom_scale*mean_prom,1e-6)
            peaks,_=find_peaks(signal,prominence=individual_prom,distance=self.cfg.peak_distance,width=self.cfg.landmark_min_width_channels)
            if len(peaks)==0: continue
            for j,c in enumerate(candidates):
                if np.any(np.abs(peaks-c)<=tol): support[j]+=1
        frac=support/max(1,len(YR))
        return candidates[frac>=self.cfg.landmark_support_frac].astype(int)

    def _trend_windows(self):
        if not self.cfg.use_regional_trends: return []
        windows=[]; seen=set()
        def add_window(region_number,idx,label):
            idx=np.asarray(idx,dtype=int)
            if len(idx)<self.cfg.trend_min_channels: return
            key=tuple(idx.tolist())
            if key in seen: return
            seen.add(key); windows.append((region_number,idx,label))
        if self.cfg.trend_regions:
            for manual_number,(lo,hi) in enumerate(self.cfg.trend_regions,1):
                for region_number,ridx in enumerate(self.regions,1):
                    idx=ridx[(self.wave[ridx]>=lo)&(self.wave[ridx]<=hi)]; add_window(region_number,idx,f"M{manual_number}")
            return windows
        for region_number,ridx in enumerate(self.regions,1):
            if self.cfg.trend_include_full_region: add_window(region_number,ridx,"ALL")
            for n_ch in sorted(set(int(n) for n in self.cfg.trend_window_channels)):
                if len(ridx)<n_ch: continue
                step=max(1,int(round(n_ch*self.cfg.trend_step_fraction))); starts=list(range(0,len(ridx)-n_ch+1,step)); last=len(ridx)-n_ch
                if starts and starts[-1]!=last: starts.append(last)
                for window_number,start in enumerate(starts,1): add_window(region_number,ridx[start:start+n_ch],f"C{n_ch}_{window_number}")
        return windows

    def _regional_trend_view(self,Xsub,idx):
        # Vectorized quadratic regression for all spectra at once. When GPU is
        # enabled, slope/curvature/delta are computed entirely with CuPy.
        W=self.wave[idx]; center=float(np.mean(W)); x=W-center
        if self.use_gpu:
            dtype=cp.float32 if self.cfg.gpu_dtype=="float32" else cp.float64
            xg=cp.asarray(x,dtype=dtype); Yg=cp.asarray(Xsub[:,idx],dtype=dtype)
            A=cp.stack((xg*xg,xg,cp.ones_like(xg)),axis=1)
            pinv=cp.linalg.pinv(A)
            C=Yg@pinv.T
            slope=C[:,1]; curvature=2.0*C[:,0]; delta=Yg[:,-1]-Yg[:,0]
            return cp.asnumpy(cp.stack((slope,curvature,delta),axis=1))
        A=np.column_stack((x*x,x,np.ones_like(x))); pinv=np.linalg.pinv(A); Y=np.asarray(Xsub[:,idx],float)
        C=Y@pinv.T; slope=C[:,1]; curvature=2.0*C[:,0]; delta=Y[:,-1]-Y[:,0]
        return np.column_stack((slope,curvature,delta))

    def _extract_views(self,Xsub):
        D1,D2=self._derivatives(Xsub); views=[]; meta=[]
        for region_number,ridx in enumerate(self.regions,1):
            W=self.wave[ridx]; YR=Xsub[:,ridx]; G1=D1[:,ridx]; G2=D2[:,ridx]
            mean_flux=np.nanmean(YR,axis=0)
            mins=self._supported_landmarks(YR,mean_flux,"min"); maxs=self._supported_landmarks(YR,mean_flux,"max")
            for bi,m in enumerate(mins,1):
                L=maxs[maxs<m]; R=maxs[maxs>m]; a=int(L[-1]) if len(L) else 0; b=int(R[0]) if len(R) else len(W)-1
                if b-a<4: continue
                rows=[]
                for i in range(len(Xsub)):
                    y,d1,d2=YR[i],G1[i],G2[i]; imin=a+int(np.argmin(y[a:b+1]))
                    lambda_min,Rmin,curvature=self._quad_vertex(W,y,imin,a,b,"min")
                    left=max(a,imin-3); right=min(b,imin+3)
                    slope_left=self._slope_fit(W,y,left,imin); slope_right=self._slope_fit(W,y,imin,right)
                    idown=a+int(np.nanargmin(d1[a:imin+1])) if imin>a else imin; iup=imin+int(np.nanargmax(d1[imin:b+1])) if imin<b else imin
                    inf_left=self._zero_cross(W,d2,a,b,imin,"left"); inf_right=self._zero_cross(W,d2,a,b,imin,"right")
                    width,ileft,iright=self._band_shoulders(W,y,imin,a,b)
                    if ileft is not None and iright is not None:
                        lambda_left,lambda_right=W[ileft],W[iright]; Rleft,Rright=y[ileft],y[iright]
                        Rcont=Rleft+(Rright-Rleft)*(lambda_min-lambda_left)/(lambda_right-lambda_left); dlocal=Rcont-Rmin
                        depth=float(dlocal) if np.isfinite(dlocal) and dlocal>0 and lambda_left<=lambda_min<=lambda_right else np.nan
                    else: depth=np.nan
                    rows.append([lambda_min,W[idown],W[iup],inf_left,inf_right,depth,width,slope_left,slope_right,curvature])
                A=np.asarray(rows,float); base=f"R{region_number}_B{bi}"
                views+=[{"name":base+"_positions","Z":A[:,0:5]},{"name":base+"_shape","Z":A[:,[0,5,6,7,8,9]]},{"name":base+"_core","Z":A[:,[0,7,8,9]]}]
                meta.append({"name":base,"type":"minimum","left":float(W[a]),"right":float(W[b]),"center":float(W[m])})
            for pi,m in enumerate(maxs,1):
                L=mins[mins<m]; R=mins[mins>m]; a=int(L[-1]) if len(L) else 0; b=int(R[0]) if len(R) else len(W)-1
                if b-a<4: continue
                rows=[]
                for i in range(len(Xsub)):
                    y=YR[i]; imax=a+int(np.argmax(y[a:b+1])); lambda_max,Rmax,curvature=self._quad_vertex(W,y,imax,a,b,"max"); rows.append([lambda_max,Rmax,curvature])
                base=f"R{region_number}_P{pi}"; views.append({"name":base+"_maximum","Z":np.asarray(rows,float)})
                meta.append({"name":base,"type":"maximum","left":float(W[a]),"right":float(W[b]),"center":float(W[m])})
        for region_number,idx,label in self._trend_windows():
            Z=self._regional_trend_view(Xsub,idx); base=f"R{region_number}_T{label}"; views.append({"name":base+"_regional_trend","Z":Z})
            meta.append({"name":base,"type":"regional_trend","left":float(self.wave[idx[0]]),"right":float(self.wave[idx[-1]]),"center":float(np.mean(self.wave[idx])),"n_channels":int(len(idx)),"features":["slope","curvature","delta_signal"]})
        return views,meta

    def _fit_view(self,view):
        Z=pd.DataFrame(np.asarray(view["Z"],float)).replace([np.inf,-np.inf],np.nan); Z=Z.loc[:,Z.notna().any(axis=0)]
        if Z.shape[1]==0 or len(Z)<2*self.cfg.min_child_n: return None
        Z=Z.fillna(Z.median()).to_numpy(float)
        if not np.all(np.isfinite(Z)): return None
        vr=np.var(Z,axis=0); Z=Z[:,np.isfinite(vr)&(vr>1e-12)]
        if Z.shape[1]==0: return None
        kmax=min(self.cfg.refine_kmax,max(1,len(Z)//self.cfg.min_child_n)); fits=[]
        if self.use_gpu:
            dtype=cp.float32 if self.cfg.gpu_dtype=="float32" else cp.float64
            Zg=cp.asarray(Z,dtype=dtype); mu=cp.mean(Zg,axis=0); sd=cp.std(Zg,axis=0); sd=cp.where(sd>0,sd,1); Zg=(Zg-mu)/sd
            for K in range(1,kmax+1):
                try:
                    g=_CuPyFullGMM(n_components=K,n_init=self.cfg.gmm_n_init,max_iter=self.cfg.gmm_max_iter,reg_covar=1e-5,tol=self.cfg.gmm_tol,random_state=self.cfg.seed).fit(Zg)
                    fits.append((K,g.bic(Zg),g))
                except Exception: continue
            Zpred=Zg
        else:
            Z=StandardScaler().fit_transform(Z)
            for K in range(1,kmax+1):
                try:
                    g=GaussianMixture(n_components=K,covariance_type="full",n_init=self.cfg.gmm_n_init,max_iter=self.cfg.gmm_max_iter,tol=self.cfg.gmm_tol,reg_covar=1e-5,random_state=self.cfg.seed).fit(Z)
                    fits.append((K,g.bic(Z),g))
                except Exception: continue
            Zpred=Z
        if not fits: return None
        Kbest,bestbic,g=min(fits,key=lambda z:z[1]); one=[q for q in fits if q[0]==1]
        if not one: return None
        labels=np.asarray(g.predict(Zpred),int); sizes=np.bincount(labels,minlength=Kbest); delta=one[0][1]-bestbic
        valid=Kbest>1 and delta>=self.cfg.delta_bic_min and sizes.min()>=self.cfg.min_child_n and sizes.min()/len(labels)>=self.cfg.min_child_frac
        return {"name":view["name"],"K":Kbest,"delta":float(delta),"labels":labels,"sizes":sizes,"valid":valid}

    def _consensus_from_views(self,fitted):
        valid=[r for r in fitted if r is not None and r["valid"]]
        if len(valid)<self.cfg.min_coherent_views: return None
        m=len(valid); M=np.eye(m)
        for i in range(m):
            for j in range(i+1,m): M[i,j]=M[j,i]=adjusted_rand_score(valid[i]["labels"],valid[j]["labels"])
        adjacency=M>=self.cfg.ari_view_min; nc,groups=connected_components(csr_matrix(adjacency.astype(int)),directed=False)
        components=[np.where(groups==q)[0] for q in range(nc) if np.sum(groups==q)>=self.cfg.min_coherent_views]
        if not components: return None
        def component_score(comp):
            aris=[M[comp[ii],comp[jj]] for ii in range(len(comp)) for jj in range(ii+1,len(comp))]
            return len(comp),np.median(aris) if aris else 1.0,np.median([valid[i]["delta"] for i in comp])
        comp=max(components,key=component_score); family=[valid[i] for i in comp]; n=len(family[0]["labels"]); A=np.zeros((n,n),float)
        for r in family:
            z=r["labels"]; A+=(z[:,None]==z[None,:])
        A/=len(family); np.fill_diagonal(A,1); D=1-A; np.fill_diagonal(D,0); L=linkage(squareform(D,checks=False),method="average")
        kmax=min(self.cfg.refine_kmax,max(r["K"] for r in family)); candidates=[]
        for Ktarget in range(2,kmax+1):
            z=fcluster(L,Ktarget,criterion="maxclust")-1; u=np.unique(z); mp={v:i for i,v in enumerate(u)}; z=np.array([mp[v] for v in z],int)
            Kactual=len(u); sizes=np.bincount(z,minlength=Kactual)
            if Kactual<2 or sizes.min()<self.cfg.min_child_n or sizes.min()/n<self.cfg.min_child_frac: continue
            sil=silhouette_score(D,z,metric="precomputed"); agreement=np.median([adjusted_rand_score(z,r["labels"]) for r in family])
            if sil>=self.cfg.cons_sil_min and agreement>=self.cfg.cons_ari_min: candidates.append((sil,agreement,Kactual,z,sizes))
        if not candidates: return None
        sil,agreement,K,z,sizes=max(candidates,key=lambda q:(q[0],q[1]))
        return {"labels":z,"K":K,"sizes":sizes,"silhouette":float(sil),"agreement":float(agreement),"family":[r["name"] for r in family],"A":A}

    def _refine_node(self,indices,depth,node,root_label):
        if depth>=self.cfg.max_refine_depth or len(indices)<self.cfg.min_parent_n:
            self.history.append([node,root_label,depth,len(indices),"STOP",0,1,[],np.nan,np.nan]); return [indices]
        views,meta=self._extract_views(self.X[indices]); fitted=[r for r in (self._fit_view(v) for v in views) if r is not None]
        nvalid=sum(int(r["valid"]) for r in fitted); split=self._consensus_from_views(fitted)
        if split is None:
            self.history.append([node,root_label,depth,len(indices),"STOP",nvalid,1,[],np.nan,np.nan]); return [indices]
        self.history.append([node,root_label,depth,len(indices),"SPLIT",nvalid,split["K"],split["family"],split["silhouette"],split["agreement"]])
        self.split_records.append({"node":node,"root_label":root_label,"indices":indices.copy(),"labels":split["labels"].copy(),"family":split["family"],"K":split["K"],"sizes":split["sizes"].copy(),"silhouette":split["silhouette"],"agreement":split["agreement"],"landmarks":meta})
        leaves=[]
        for c in range(split["K"]):
            child=indices[split["labels"]==c]; leaves+=self._refine_node(child,depth+1,f"{node}.{c+1}",root_label)
        return leaves

    def fit(self,X,wavelengths,global_labels):
        self.X,self.wave,global_labels=self._validate_inputs(X,wavelengths,global_labels); self.regions=self._build_regions(); self.history=[]; self.split_records=[]; roots=list(np.unique(global_labels)); root_leaves={}
        devices=tuple(int(d) for d in self.cfg.gpu_devices) if self.use_gpu else ()
        do_parallel=self.use_gpu and self.cfg.parallel_root_clusters and len(devices)>1 and len(roots)>1
        if do_parallel:
            assignments=[roots[i::len(devices)] for i in range(len(devices))]; assignments=[a for a in assignments if a]
            ctx=mp.get_context("spawn"); q=ctx.Queue(); ps=[]
            for device,assigned in zip(devices,assignments):
                p=ctx.Process(target=_sgr_gpu_worker,args=(device,self.cfg,self.X,self.wave,global_labels,assigned,q)); p.start(); ps.append(p)
            rows=[q.get() for _ in ps]
            for p in ps: p.join()
            bad=[r for r in rows if not r.get("ok",False)]
            if bad: raise RuntimeError("SGR 2-GPU worker failed: "+" | ".join(f"GPU {r.get('device')}: {r.get('error')}" for r in bad))
            for r in rows:
                self.history.extend(r["history"]); self.split_records.extend(r["split_records"])
                for k,leaves in r["roots"]: root_leaves[k]=leaves
            self.history.sort(key=lambda x:str(x[0])); self.split_records.sort(key=lambda x:str(x["node"]))
        else:
            if self.use_gpu and devices: cp.cuda.Device(devices[0]).use()
            for display_idx,k in enumerate(roots,1):
                idx=np.where(global_labels==k)[0]; root_leaves[k]=self._refine_node(idx,0,f"G{display_idx}",k)
        all_leaves=[]
        for k in roots: all_leaves.extend(root_leaves[k])
        refined=np.full(len(global_labels),-1,dtype=int)
        for label,idx in enumerate(all_leaves): refined[idx]=label
        if np.any(refined<0): raise RuntimeError("Some observations were not assigned to a refined cluster.")
        hist=pd.DataFrame(self.history,columns=["node","global_cluster","depth","N","decision","valid_views","local_K","coherent_landmarks","consensus_silhouette","consensus_ARI"])
        return SGRResult(global_labels=np.asarray(global_labels),refined_labels=refined,history=hist,split_records=self.split_records.copy(),wavelengths=self.wave.copy(),regions=[r.copy() for r in self.regions],X=self.X.copy())

def print_sgr_summary(result):
    print("\n===== SPECTRAL GEOMETRY REFINEMENT (SGR) =====")
    print("Global clusters :",result.n_global_clusters); print("Accepted splits :",result.n_splits); print("Refined clusters:",result.n_refined_clusters)
    if result.n_splits:
        for rec in result.split_records: print(f"{rec['node']}: K={rec['K']} | N={len(rec['indices'])} | silhouette={rec['silhouette']:.4f} | agreement={rec['agreement']:.4f}")

def save_sgr_results(result,output_dir,prefix="sgr"):
    os.makedirs(output_dir,exist_ok=True); history_path=os.path.join(output_dir,f"{prefix}_history.csv"); labels_path=os.path.join(output_dir,f"{prefix}_labels.csv")
    result.history.to_csv(history_path,index=False); pd.DataFrame({"row_index":np.arange(len(result.global_labels)),"global_cluster":result.global_labels,"refined_cluster":result.refined_labels+1}).to_csv(labels_path,index=False)
    return {"history":history_path,"labels":labels_path}

def _derivatives_for_result(result,X):
    D1=np.full_like(X,np.nan,float); D2=np.full_like(X,np.nan,float)
    for r in result.regions:
        D1[:,r]=np.gradient(X[:,r],result.wavelengths[r],axis=1,edge_order=2); D2[:,r]=np.gradient(D1[:,r],result.wavelengths[r],axis=1,edge_order=2)
    return D1,D2

def plot_accepted_splits(result,max_plots=6,output_dir=None,show=True):
    if output_dir is not None: os.makedirs(output_dir,exist_ok=True)
    records=result.split_records if max_plots is None else result.split_records[:max_plots]; saved=[]
    for rec in records:
        idx=rec["indices"]; labels=rec["labels"]; Xparent=result.X[idx]; D1,D2=_derivatives_for_result(result,Xparent)
        for A,ylabel,suffix in [(Xparent,"Spectral signal","signal"),(D1,r"$dF/d\lambda$","first_derivative"),(D2,r"$d^2F/d\lambda^2$","second_derivative")]:
            fig,ax=plt.subplots(figsize=(9,5))
            for c in range(rec["K"]):
                m=labels==c
                for rr,r in enumerate(result.regions): ax.plot(result.wavelengths[r],np.nanmean(A[m][:,r],axis=0),lw=2,label=f"Subcluster {c+1}" if rr==0 else None)
            ax.set_xlabel(r"Wavelength ($\mu$m)"); ax.set_ylabel(ylabel); ax.set_title(f"{rec['node']} -> K={rec['K']} | silhouette={rec['silhouette']:.3f} | agreement={rec['agreement']:.3f}"); ax.legend(frameon=False); fig.tight_layout()
            if output_dir is not None:
                path=os.path.join(output_dir,f"{rec['node'].replace('.','_')}_{suffix}.png"); fig.savefig(path,dpi=300,bbox_inches="tight"); saved.append(path)
            if show: plt.show()
            else: plt.close(fig)
    return saved

def plot_refined_mean_spectra(result,output_dir=None,show=True):
    if output_dir is not None: os.makedirs(output_dir,exist_ok=True)
    fig,ax=plt.subplots(figsize=(10,6))
    for c in np.unique(result.refined_labels):
        m=result.refined_labels==c
        for rr,r in enumerate(result.regions): ax.plot(result.wavelengths[r],np.nanmean(result.X[m][:,r],axis=0),lw=1.8,label=f"R{c+1}" if rr==0 else None)
    ax.set_xlabel(r"Wavelength ($\mu$m)"); ax.set_ylabel("Spectral signal"); ax.set_title("Refined partition - mean spectra"); ax.legend(frameon=False,ncol=2,bbox_to_anchor=(1.02,1),loc="upper left"); fig.tight_layout(); path=None
    if output_dir is not None:
        path=os.path.join(output_dir,"sgr_mean_spectra.png"); fig.savefig(path,dpi=300,bbox_inches="tight")
    if show: plt.show()
    else: plt.close(fig)
    return path

def run_sgr(X,wavelengths,global_labels,config=None,output_dir=None,make_plots=True,max_split_plots=6,show_plots=True):
    refiner=SpectralGeometryRefiner(config=config); result=refiner.fit(X,wavelengths,global_labels); print_sgr_summary(result); paths={}
    if output_dir is not None: paths.update(save_sgr_results(result,output_dir))
    if make_plots:
        split_dir=os.path.join(output_dir,"split_plots") if output_dir is not None else None
        paths["split_plots"]=plot_accepted_splits(result,max_plots=max_split_plots,output_dir=split_dir,show=show_plots); paths["mean_spectra"]=plot_refined_mean_spectra(result,output_dir=output_dir,show=show_plots)
    result.output_paths=paths; return result

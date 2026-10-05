"""INFERRED native-center geometry. Forward frame->reference, inverse reference->frame.
No dynamic-object deghosting. OpenCV is needed only by the estimated aligner.
"""
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F

@dataclass
class Geometry:
    forward: torch.Tensor
    inverse: torch.Tensor
    reports: list

def correct_lca(rgb, offsets=None, scale=2):
    """Observed channel(x)=ideal(x+offset); correction samples x-offset.
    Offsets [3,2] are dx,dy native pixels. Bilinear, border replication.
    """
    if offsets is None:
        return rgb
    offsets = torch.as_tensor(offsets, device=rgb.device, dtype=rgb.dtype)
    if offsets.shape != (3,2) or not torch.isfinite(offsets).all():
        raise ValueError('LCA requires finite [3,2] native offsets')
    h,w=rgb.shape[-2:]
    y,x=torch.meshgrid(torch.arange(h,device=rgb.device,dtype=rgb.dtype),torch.arange(w,device=rgb.device,dtype=rgb.dtype),indexing='ij')
    out=[]
    for c in range(3):
        grid=torch.stack((2*(x-offsets[c,0]*scale+.5)/w-1,2*(y-offsets[c,1]*scale+.5)/h-1),-1)
        out.append(F.grid_sample(rgb[:,c:c+1],grid[None].expand(len(rgb),-1,-1,-1),align_corners=False,padding_mode='border'))
    return torch.cat(out,1)

def basis(points, degree, width, height):
    x=2*points[:,0]/max(width-1,1)-1; y=2*points[:,1]/max(height-1,1)-1
    return np.stack([x**i*y**j for n in range(degree+1) for i in range(n+1) for j in [n-i]],1)

def fit_residual(points, residual, width, height, max_degree=4):
    """Conditioned least squares, descending degree; at least twice terms."""
    for degree in range(max_degree,-1,-1):
        a=basis(points,degree,width,height)
        if len(a)<2*a.shape[1] or np.linalg.cond(a)>1e5:
            continue
        coef=np.linalg.lstsq(a,residual,rcond=None)[0]
        if np.isfinite(coef).all():
            return degree,coef
    return -1,None

@torch.no_grad()
def estimate_geometry(raw, max_degree=4):
    """GFTT+LK local controls, bidirectional and actual frame-to-frame cycle checks.
    Homography RANSAC plus conditioned polynomial residual in native pixels.
    Low texture falls back to identity with reason; no oracle shifts are read.
    """
    import cv2
    if raw.ndim!=5 or raw.shape[2]!=1 or min(raw.shape[-2:])<8 or not torch.isfinite(raw).all() or any(d%2 for d in raw.shape[-2:]):
        raise ValueError('finite even B,K,1,H,W required')
    if not isinstance(max_degree,int) or not 0<=max_degree<=4:
        raise ValueError('polynomial degree must be integer 0..4')
    cv2.setRNGSeed(0)
    b,k,_,h,w=raw.shape
    y,x=np.mgrid[:h,:w]; grid=np.stack((x,y),-1).astype(np.float64); flat=grid.reshape(-1,2)
    images=[]
    for batch in range(b):
        ims=[]
        for frame in range(k):
            r=raw[batch,frame,0].cpu().numpy()
            g=.5*(r[0::2,1::2]+r[1::2,0::2])
            g=cv2.resize(g,(w,h))
            lo,hi=np.percentile(g,[1,99]); ims.append(np.uint8(np.clip((g-lo)/max(hi-lo,1e-8),0,1)*255))
        images.append(ims)
    forwards=[]; inverses=[]; reports=[]
    for ims in images:
        fs=[grid.copy()]; invs=[grid.copy()]; rr=[{'status':'reference','degree':-1}]
        controls=cv2.goodFeaturesToTrack(ims[0],300,.02,4)
        tracked=[]
        for im in ims:
            tracked.append(None if controls is None else cv2.calcOpticalFlowPyrLK(ims[0],im,controls,None,winSize=(15,15),maxLevel=3))
        for frame in range(1,k):
            report={'status':'fallback','reason':'insufficient reliable texture','degree':-1,'controls':0,'confidence':0.}
            forward=grid.copy(); inverse=grid.copy()
            if controls is not None:
                q,ok,_=tracked[frame]; back,okback,_=cv2.calcOpticalFlowPyrLK(ims[frame],ims[0],q,None,winSize=(15,15),maxLevel=3)
                good=(ok[:,0]>0)&(okback[:,0]>0)&(np.linalg.norm(back[:,0]-controls[:,0],axis=1)<.75)
                cycle_errors=[]
                for other in range(k):
                    if other in (0,frame): continue
                    via,status,_=cv2.calcOpticalFlowPyrLK(ims[frame],ims[other],q,None,winSize=(15,15),maxLevel=3)
                    target,target_ok,_=tracked[other]
                    error=np.linalg.norm(via[:,0]-target[:,0],axis=1)
                    good &= (status[:,0]>0)&(target_ok[:,0]>0)&(error<1.)
                    cycle_errors.extend(error[good].tolist())
                src=q[:,0][good]; dst=controls[:,0][good]
                report['controls']=len(src); report['confidence']=float(len(src)/len(controls)); report['cross_frame_mean_px']=float(np.mean(cycle_errors)) if cycle_errors else None
                if len(src)>=8:
                    hom,mask=cv2.findHomography(src,dst,cv2.RANSAC,1.)
                    if hom is not None and mask is not None and mask.sum()>=8 and mask.mean()>=.5 and np.isfinite(hom).all() and abs(np.linalg.det(hom))>1e-8 and np.linalg.cond(hom)<1e6:
                        keep=mask[:,0]>0; src=src[keep]; dst=dst[keep]
                        projected=cv2.perspectiveTransform(src[None],hom)[0]
                        degree,coef=fit_residual(src,dst-projected,w,h,max_degree)
                        def mapping(p):
                            z=cv2.perspectiveTransform(p[None].astype(np.float64),hom)[0]
                            return z if coef is None else z+basis(p,degree,w,h)@coef
                        forward=mapping(flat).reshape(h,w,2)
                        # Invert the whole fitted mapping by finite-difference Newton, never use forward as sampling grid.
                        p=cv2.perspectiveTransform(flat[None],np.linalg.inv(hom))[0]
                        for _ in range(12):
                            err=mapping(p)-flat
                            jx=(mapping(p+np.array([.01,0]))-mapping(p))/.01
                            jy=(mapping(p+np.array([0,.01]))-mapping(p))/.01
                            det=jx[:,0]*jy[:,1]-jx[:,1]*jy[:,0]
                            safe=np.abs(det)>1e-6
                            delta=np.zeros_like(p)
                            delta[safe,0]=(err[safe,0]*jy[safe,1]-err[safe,1]*jy[safe,0])/det[safe]
                            delta[safe,1]=(jx[safe,0]*err[safe,1]-jx[safe,1]*err[safe,0])/det[safe]
                            p-=delta
                        inverse=p.reshape(h,w,2)
                        error=np.linalg.norm(mapping(p)-flat,axis=1)
                        if not np.isfinite(forward).all() or not np.isfinite(inverse).all() or error.max()>.05:
                            forward=grid.copy(); inverse=grid.copy(); report['reason']='inverse convergence failed'
                        else:
                            report.update(status='estimated',reason='polynomial' if degree>=0 else 'homography only: insufficient conditioned controls',degree=degree,requested_degree=max_degree,degree_fallback=(degree<max_degree),homography=hom.tolist(),residual_rms_px=float(np.sqrt(np.mean((mapping(src)-dst)**2))),inverse_max_error_px=float(error.max()))
                    else:
                        report['reason']='RANSAC failed or too few inliers, singular or ill-conditioned homography'
            fs.append(forward); invs.append(inverse); rr.append(report)
        forwards.append(fs); inverses.append(invs); reports.append(rr)
    return Geometry(torch.as_tensor(np.array(forwards),device=raw.device,dtype=raw.dtype),torch.as_tensor(np.array(inverses),device=raw.device,dtype=raw.dtype),reports)


def validate_geometry(geometry, raw):
    b,k,_,h,w=raw.shape
    expected=(b,k,h,w,2)
    for field in (geometry.forward,geometry.inverse):
        if field.shape!=expected or field.device!=raw.device or not torch.isfinite(field).all():
            raise ValueError('geometry must contain finite native [B,K,H,W,2] maps on RAW device')
        yy,xx=torch.meshgrid(torch.arange(h,device=raw.device),torch.arange(w,device=raw.device),indexing='ij')
        identity=torch.stack((xx,yy),-1).to(field)
        if not torch.allclose(field[:,0],identity.expand(b,-1,-1,-1),atol=1e-5,rtol=0):
            raise ValueError('reference geometry must be identity')

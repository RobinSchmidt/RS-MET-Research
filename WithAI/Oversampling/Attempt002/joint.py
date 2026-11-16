import numpy as np, sys, time
from scipy.optimize import minimize
sys.path.insert(0,'/home/claude')
import oversampling_pair as op
import oversampling_matched as mt

def mirror(N, drop_ends):
    mid=(N-1)//2; m=mid if drop_ends else mid+1
    S=np.zeros((N,m))
    for k in range(m): S[mid-k,k]=1; S[mid+k,k]=1
    return S

class Joint:
    """u (gain L, symmetric, free ends) and d (gain 1, symmetric, end taps fixed to 0), exact Nyquist, balanced stopband objective."""
    def __init__(s, L, Nu, Nd, p):
        s.L,s.Nu,s.Nd,s.p=L,Nu,Nd,p
        s.n0=(Nu+Nd-2)//2; assert s.n0%L==0 and (Nu+Nd-2)%2==0
        s.K=s.n0//L
        s.Su,s.Sd=mirror(Nu,False),mirror(Nd,True)
        ws=(2-p)*np.pi/L
        s.Qu=s.Su.T@mt.stopband_matrix(Nu,ws)@s.Su/L**2
        s.Qd=s.Sd.T@mt.stopband_matrix(Nd,ws)@s.Sd
        s.mu=s.Su.shape[1]; s.md=s.Sd.shape[1]
        s.idx=s.n0+L*np.arange(s.K)            # k=K is implied by d's zero end taps
        s.tgt=(np.arange(s.K)==0).astype(float)
        s.gu=s.Su.sum(axis=0)                  # gain row for u
    def split(s,x): return x[:s.mu],x[s.mu:]
    def filt(s,x): a,b=s.split(x); return s.Su@a, s.Sd@b
    def r(s,x):
        u,d=s.filt(x); h=np.convolve(u,d)
        return np.concatenate([h[s.idx]-s.tgt,[u.sum()-s.L]])
    def J(s,x):
        u,d=s.filt(x); K=s.K; Ju=np.zeros((K,s.Nu)); Jd=np.zeros((K,s.Nd))
        ju=np.arange(s.Nu); jd=np.arange(s.Nd)
        for k,pp in enumerate(s.idx):
            i=pp-ju; ok=(i>=0)&(i<s.Nd); Ju[k,ju[ok]]=d[i[ok]]
            i=pp-jd; ok=(i>=0)&(i<s.Nu); Jd[k,jd[ok]]=u[i[ok]]
        top=np.hstack([Ju@s.Su,Jd@s.Sd]); g=np.concatenate([s.gu,np.zeros(s.md)])[None,:]
        return np.vstack([top,g])
    def f(s,x): a,b=s.split(x); return 0.5*(a@s.Qu@a+b@s.Qd@b)
    def grad(s,x): a,b=s.split(x); return np.concatenate([s.Qu@a,s.Qd@b])
    def polish(s,x,iters=40):
        for _ in range(iters):
            rr=s.r(x)
            if np.max(abs(rr))<1e-16: break
            x=x-np.linalg.lstsq(s.J(x),rr,rcond=None)[0]
        return x
    def optimise(s,x,restarts=4,maxiter=120,verbose=False):
        rng=np.random.default_rng(0); stalls=0
        x=s.polish(x)
        for _ in range(restarts):
            f0=s.f(x); sc=1.0/f0
            start=x if stalls==0 else s.polish(x+1e-6*np.sqrt(f0)*rng.standard_normal(len(x)))
            res=minimize(lambda t:sc*s.f(t),start,jac=lambda t:sc*s.grad(t),method='SLSQP',
                         constraints=[dict(type='eq',fun=s.r,jac=s.J)],options=dict(maxiter=maxiter,ftol=1e-14))
            xn=s.polish(res.x)
            if verbose: print('     slsqp',res.status,res.nit,f'f {f0:.2e}->{s.f(xn):.2e}',flush=True)
            if s.f(xn)<0.999*f0: x,stalls=xn,0
            else:
                stalls+=1
                if stalls>=2: break
        return x
    def metrics(s,x):
        u,d=s.filt(x); w=np.linspace(0,np.pi,4000); sb=w>=(2-s.p)*np.pi/s.L; pb=w<=s.p*np.pi/s.L
        F=lambda h:np.abs(np.exp(-1j*np.outer(w,np.arange(len(h))))@h)
        U,D=F(u)/s.L,F(d); h=np.convolve(u,d); idx=s.n0+s.L*np.arange(-s.K,s.K+1)
        return dict(u_db=20*np.log10(U[sb].max()),d_db=20*np.log10(D[sb].max()),
                    rip=max(20*np.log10(U[pb].max()/U[pb].min()),20*np.log10(D[pb].max()/D[pb].min())),
                    nyq=float(np.max(abs(h[idx]-(idx==s.n0)))))

if __name__=='__main__':
    L,A,p=4,100,0.9
    u0,d0,info=op.design_oversampling_pair(L,A,p,"linear"); print('start pair',len(u0),len(d0),'K',info['K'])
    print('u ends',u0[0],u0[1],' d ends',d0[0],d0[1],d0[L])
    Nu,Nd=len(u0),len(d0)
    jt=Joint(L,Nu,Nd,p)
    dz=d0.copy(); dz[0]=dz[-1]=0
    x=np.concatenate([[u0[(Nu-1)//2-k] for k in range(jt.mu)],[dz[(Nd-1)//2-k] for k in range(jt.md)]])
    print('residual of start (d ends zeroed):',np.max(abs(jt.r(x))))
    t=time.time(); x=jt.optimise(x,verbose=True); print('joint optimise at start split:',jt.metrics(x),f'({time.time()-t:.0f}s)')

import numpy as np, sys, time, importlib.util
sys.path.insert(0,'/tmp/claude-0/-home-claude/5ca760b1-3a7d-5e92-b3ae-83292b04bb97/scratchpad'); sys.path.insert(0,'/home/claude')
from joint import *
spec=importlib.util.spec_from_file_location("old","/home/claude/robinschmidt/rs-met-research/WithAI/Oversampling/Attempt001/oversampling_pr_design.py")
old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
L,Nu,Nd=int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]); p=float(sys.argv[4]); beta=float(sys.argv[5]) if len(sys.argv)>5 else 8.0
T0=time.time(); n0=(Nu+Nd-2)//2; assert n0%L==0; Neq=(Nu+Nd)//2; K=(Neq-1)//L; assert L*K+1==Neq and (L*K)%2==0
print(f"=== L={L}, passband {p}: start Nu={Nu}, Nd={Nd}; equal-length target N={Neq} (K={K}); free params of exact matched g: {(L-2)*K//2}")
u0=old.design_u(L,Nu,beta); d0=old.design_d(u0,L,Nd,1-p)
def to_x(jt,u,d): return np.concatenate([[u[(jt.Nu-1)//2-k] for k in range(jt.mu)],[d[(jt.Nd-1)//2-k] for k in range(jt.md)]])
d0s=0.5*(d0+d0[::-1]); d0s[0]=d0s[-1]=0
jt=Joint(L,Nu,Nd,p); m=jt.metrics(jt.polish(to_x(jt,u0,d0s)))
print(f"old-script start pair: u {m['u_db']:.1f} dB, d {m['d_db']:.1f} dB")
u,d=u0.copy(),d0s; rng=np.random.default_rng(0); cu,cd=Nu,Nd
print(f"{'Nu':>4} {'Nd':>4} {'u_stop':>8} {'d_stop':>8} {'worst':>8}")
while True:
    jt=Joint(L,cu,cd,p); x=to_x(jt,u,d); x=jt.optimise(x,restarts=4,maxiter=150)
    m=jt.metrics(x); u,d=jt.filt(x)
    print(f"{cu:4d} {cd:4d} {m['u_db']:8.1f} {m['d_db']:8.1f} {max(m['u_db'],m['d_db']):8.1f}",flush=True)
    if cd-2<cu+2-2: break
    u=np.pad(u,1); u=u+1e-7*rng.standard_normal(len(u))*(np.abs(u)<1e-12); u=0.5*(u+u[::-1]); cu+=2
    d=d[1:-1].copy(); d[0]=d[-1]=0; cd-=2
assert cu==cd==Neq
class Tie(Joint):
    lam=0.0
    def diff(s,x): a,b=s.split(x); return (s.Su@a)/np.sqrt(s.L)-(s.Sd@b)*np.sqrt(s.L)
    def f(s,x): dd=s.diff(x); return Joint.f(s,x)+0.5*s.lam*dd@dd
    def grad(s,x):
        dd=s.diff(x); g=Joint.grad(s,x)
        g[:s.mu]+=s.lam*(s.Su.T@dd)/np.sqrt(s.L); g[s.mu:]-=s.lam*(s.Sd.T@dd)*np.sqrt(s.L); return g
tj=Tie(L,Neq,Neq,p); x=to_x(tj,u,d); f0=Joint.f(tj,x); base=np.linalg.norm(tj.diff(x))**2
print("tie step (penalty on |u/sqrt(L)-d*sqrt(L)|^2, Nyquist exact):")
print(f"{'rel':>8} {'rel.diff':>9} {'u_stop':>8} {'d_stop':>8}")
for rel in (0,1e1,1e2,1e3,1e4,1e5,1e6,1e8,1e10):
    tj.lam=rel*f0/base; x=tj.optimise(x,restarts=4,maxiter=200)
    m=tj.metrics(x); uu,dd_=tj.filt(x); rd=np.linalg.norm(tj.diff(x))/np.linalg.norm(uu/np.sqrt(L))
    print(f"{rel:8.0e} {rd:9.4f} {m['u_db']:8.1f} {m['d_db']:8.1f}",flush=True)
gavg=0.5*(uu/np.sqrt(L)+dd_*np.sqrt(L)); gavg[0]=gavg[-1]=0
ma=mt.measure(gavg,L,p); print(f"averaged g: stop {ma['stop_db']:.1f} dB, Nyquist err of g*g {ma['nyquist_error']:.1e}")
g=mt.design_matched(L,K,p,g_init=gavg); mg=mt.measure(g,L,p)
print(f"refined g (SLSQP at exact Nyquist, started from average): stop {mg['stop_db']:.1f} dB, ripple {mg['pass_ripple_db']:.4f} dB, Nyquist err {mg['nyquist_error']:.1e}, moved {np.linalg.norm(g-gavg)/np.linalg.norm(gavg):.1e}")
xr=np.random.default_rng(3).standard_normal(5000); yr=mt.roundtrip(xr,g,L,K); print(f"   roundtrip error u=d=g: {np.max(abs(yr-xr)):.1e}")
gs=mt.design_matched(L,K,p,kaiser_beta=4.0); ms=mt.measure(gs,L,p)
print(f"from-scratch matched solver (Kaiser start, same N={Neq}): stop {ms['stop_db']:.1f} dB, Nyquist err {ms['nyquist_error']:.1e}")
np.save(f'/tmp/claude-0/-home-claude/5ca760b1-3a7d-5e92-b3ae-83292b04bb97/scratchpad/g_L{L}_refined.npy',g)
print(f"(total {time.time()-T0:.0f}s)")

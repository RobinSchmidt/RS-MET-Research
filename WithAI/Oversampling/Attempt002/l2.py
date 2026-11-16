import numpy as np, sys, time, importlib.util, pickle
sys.path.insert(0,'/tmp/claude-0/-home-claude/5ca760b1-3a7d-5e92-b3ae-83292b04bb97/scratchpad'); sys.path.insert(0,'/home/claude')
from joint import *
spec=importlib.util.spec_from_file_location("old","/home/claude/robinschmidt/rs-met-research/WithAI/Oversampling/Attempt001/oversampling_pr_design.py")
old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)

L,p=2,0.8; Nu,Nd=63,127
u0=old.design_u(L,Nu,8.0); d0=old.design_d(u0,L,Nd,1-p)          # exactly the old script's KKT design
print('old-script start: u sym err %.1e, d sym err %.1e, d ends %.1e %.1e, d gain %.4f'%(abs(u0-u0[::-1]).max(),abs(d0-d0[::-1]).max(),d0[0],d0[-1],d0.sum()))
def to_x(jt,u,d): return np.concatenate([[u[(jt.Nu-1)//2-k] for k in range(jt.mu)],[d[(jt.Nd-1)//2-k] for k in range(jt.md)]])
d0s=0.5*(d0+d0[::-1]); d0s[0]=d0s[-1]=0
jt=Joint(L,Nu,Nd,p); x=to_x(jt,u0,d0s)
print('unknowns: u %d, d %d, constraints %d (+1 gain)'%(jt.mu,jt.md,jt.K))
m=jt.metrics(jt.polish(x)); print('start pair      : u %.1f dB, d %.1f dB, nyq %.0e'%(m['u_db'],m['d_db'],m['nyq']),flush=True)

u,d=u0.copy(),d0s; STEP=2; res=[]; rng=np.random.default_rng(0)
print(f"{'Nu':>4} {'Nd':>4} {'u_stop':>8} {'d_stop':>8} {'worst':>8} {'rip':>8} {'|u/sqL-d*sqL|/|u/sqL|':>22}")
while True:
    jt=Joint(L,Nu,Nd,p); x=to_x(jt,u,d); x=jt.optimise(x,restarts=4,maxiter=150)
    m=jt.metrics(x); u,d=jt.filt(x); worst=max(m['u_db'],m['d_db'])
    if Nu==Nd: rel=np.linalg.norm(u/np.sqrt(L)-d*np.sqrt(L))/np.linalg.norm(u/np.sqrt(L))
    else: rel=float('nan')
    res.append((Nu,Nd,m['u_db'],m['d_db'],worst,u.copy(),d.copy()))
    print(f"{Nu:4d} {Nd:4d} {m['u_db']:8.1f} {m['d_db']:8.1f} {worst:8.1f} {m['rip']:8.4f} {rel:22.3f}",flush=True)
    if Nd-STEP<Nu+STEP-2: break
    u=np.pad(u,1); u=u+1e-7*rng.standard_normal(len(u))*(np.abs(u)<1e-12); u=0.5*(u+u[::-1]); Nu+=2
    d=d[1:-1].copy(); d[0]=d[-1]=0; Nd-=2
pickle.dump(res,open('/tmp/claude-0/-home-claude/5ca760b1-3a7d-5e92-b3ae-83292b04bb97/scratchpad/l2_path.pkl','wb'))
i=int(np.argmin([r[4] for r in res])); print('tracked minimum of the worse stopband: Nu=%d Nd=%d at %.1f dB'%(res[i][0],res[i][1],res[i][4]))

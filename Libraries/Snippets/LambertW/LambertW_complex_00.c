/* Keith Briggs 2017-10-11
 Complex Lambert W function.
 Based on C++ code by István Mező from
 https://sites.google.com/site/istvanmezo81/others,
 but completely rewritten in C (c99 standard).

 See also https://github.com/scipy/scipy/blob/v0.19.1/scipy/special/lambertw.pxd

 R. M. Corless et al., On the Lambert W Function, Advances in Computational Mathematics, v.5 (1996) 329-359. (W-adv-cm.pdf)

 Run tests:
   gcc -D__TEST_LAMBERTW__ -Wall -O3 -std=c99 LambertW_complex_00.c -o LambertW_complex -lm && ./LambertW_complex | p -x -10 10
*/

#include <math.h>
#include <stdio.h>
#include <complex.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_E
#define M_E  2.7182818284590452354
#endif

complex initial_point(complex z, int k) {
  // Determine the initial point for the root finding
  complex twopi_k_I,p;
  if (cabs(z-(-1.0/M_E))<=1.0) { // we are close to the branch cut
    p=csqrt(2.0*(M_E*z+1.0));
    if (k==0) return -1.0+p-p*p/3.0+11.0*p*p*p/72.0;
    if ((k==1 && cimag(z)<0.0) || (k==-1 && cimag(z)>0.0))
      return -1.0-p-p*p/3.0-11.0*p*p*p/72.0;
  }
  if (cabs(z-0.5)<=0.5) {
    if (k== 0) // (1,1) Pade approximant for W(0,a)
      return (0.35173371*(0.1237166+7.061302897*z))/(2.0+0.827184*(1.0+2.0*z));
    if (k==-1) // (1,1) Pade approximant for W(-1,a)
     return -(((2.2591588985+4.22096*I)*((-14.073271-33.767687754*I)*z-(12.7127-19.071643*I)*(1.0+2.0*z)))/(2.0-(17.23103-10.629721*I)*(1.0+2.0*z)));
  }
  twopi_k_I=2.0*M_PI*k*I;
  // initial point from the general asymptotic approximation
  return clog(z)+twopi_k_I-clog(clog(z)+twopi_k_I);
}

complex LambertW(complex z, int k) {
  // are the next three useful?
  if (z==0.0) return (k==0)?0.0:-INFINITY;
  if (z==-1.0/M_E && (k==0 || k==-1)) return -1.0;
  if (z== 1.0/M_E &&  k==0) return 1.0;
  complex w=initial_point(z,k),wprev,cexpw,wexpw,wexpw_d,wexpw_dd,d;
  const unsigned int maxiter=15;
  unsigned int iter=0;
  double eps=1.0e-12;
  do { // Halley iteration
    wprev=w;
    cexpw=cexp(w);
    wexpw=w*cexpw;
    wexpw_d=cexpw+wexpw;
    wexpw_dd=2.0*cexpw+wexpw;
    d=wexpw-z;
    w-=d/(wexpw_d-0.5*d*wexpw_dd/wexpw_d);
    if (++iter>maxiter) {
      fprintf(stderr,"# Convergence failure in LambertW for z=%g+%gi, k=%d\n",creal(z),cimag(z),k);
      return NAN;
    }
  } while (cabs(w-wprev)>eps*(1.0+cabs(w)));
  return w;
}

#ifdef __TEST_LAMBERTW__

#include <stdlib.h>

int test_00() { // Stratton example
  int k;
  complex z=-7.25e-7-7.25e-7*I,w,y;
  printf("z=%g+%gi\n",creal(z),cimag(z));
  for (k=-5; k<=5; k++) {
    w=LambertW(z,k);
    printf("w=W(z,%2d)=%g+%gi   ",k,creal(w),cimag(w));
    y=cexp(w);
    printf("exp(w)=%g+%gi\n",creal(y),cimag(y));
  }
  return 0;
}

int test_01(int k) { // random - check branches are correct
  // Corless et al., On the Lambert W Function p.344
  // ./LambertW_complex | p -x -10 10
  int i;
  complex z,w;
  double q=1.0+RAND_MAX;
  printf("#m=%d,S=1\n",k+6);
  for (i=0; i<10000; i++) {
    z=1.0e6*(rand()/q-0.5)+1.0e6*(rand()/q-0.5)*I;
    w=LambertW(z,k);
    printf("%g\t%g\n",creal(w),cimag(w));
  }
  return 0;
}

int main() {
  int i,kk;
  double x,t;
  test_01(-2);
  test_01(-1);
  test_01( 0);
  test_01( 1);
  test_01( 2);
  // draw bounding curves...
  for (kk=0; kk<3; kk++) {
    printf("\n#m=-3,S=1\n");
    for (i=1; i<1000; i++) {
      t=M_PI*(2.0*kk+i/1000.0);
      x=-t/tan(t);
      printf("%g\t%g\n",x, t);
      printf("%g\t%g\n",x,-t);
    }
  }
  return 0;
}

#endif

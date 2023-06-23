/* quintic.c
   Classify a general quintic with real coefficients.   
   KMB 97 June 19 12:00 

   Compile: gcc -Wall quintic.c -o quintic 
   Use:     for a5*x^5 + a4*x^4 + ... + a0 do `quintic a5 a4 a3 a2 a1 a0'

   Test script: tryquintic
   Passes all 12 tests with e=1e-6

   To do:
     Hornerize p,q,r,s expressions.
     macros for d4,d5 etc.
     accurate summation in d5 (Higham)
     bisection macro
     root bounds for bisection
     computation of actual roots by bisection in each case
     obfuscation
*/

#include <math.h>   /* for double fabs(double); */
#include <stdio.h>  /* for int printf(char*,...); */
#include <stdlib.h> /* for double atof(char*); */
#define pows(x) x##2=x*x; x##3=x##2*x; x##4=x##2*x##2; x##5=x##3*x##2; 
#define D4 12*p4*r-4*p3*q2+117*p*r*q2-88*r2*p2-40*q*p2*s+125*p*s2-27*q4-300*q*r*s+160*r3
typedef double real;

int main(int argc, char **argv) {
  int debug=1;
  int i; real e=1e-6;  /* epsilon for zero test */
  real a[6],p,q,r,s,p2,q2,r2,s2,p3,q3,r3,s3,p4,q4,r4,s4,p5,q5,r5,s5,d2,d3,d4,d5,e2,f2;
  for (i=5; i>=0; i--) { a[i]=atof(argv[6-i]); a[i]/=a[5]; }
  printf("  Input quintic: x^5+%fx^4+%fx^3+%fx^2+%fx+%f\n",a[4],a[3],a[2],a[1],a[0]);
  /* get p,q,r,s (x->x-a4/5).  See maple/tschirn.map 
  this translates roots by -a[4]/5, so this must be put back at the end.
  p=   a3-2/5*a4^2
  q=   a2+4/25*a4^3-3/5*a3*a4
  r=   -2/5*a2*a4-3/125*a4^4+3/25*a3*a4^2+a1
  s=   -1/5*a1*a4+a0+1/25*a2*a4^2+4/3125*a4^5-1/125*a3*a4^3 */
  s=a[4]*a[4]/5; /* temp */
  p=a[3]-2*s; d2=-p;
  q=a[2]+a[4]*(0.8*s-0.6*a[3]);
  r=a[1]-a[4]*(0.4*a[2]+0.12*a[4]*(s-a[3]));
  s=a[0]+0.2*a[2]*s+a[4]*(0.032*s*s-0.04*a[3]*s-0.2*a[1]);
  printf("Reduced quintic: x^5+%fx^3+%fx^2+%fx+%f\n",p,q,r,s);
  pows(p) pows(q) pows(r) pows(s)
  d5=-1600*q*s*r3-3750*p*s3*q+2000*p*s2*r2-4*p3*q2*r2+16*p3*q3*s-900*r*s2*p3+
      825*q2*p2*s2+144*p*q2*r3+2250*q2*r*s2+16*p4*r3+108*p5*s2-128*r4*p2-
      27*q4*r2+108*q5*s+256*r5+3125*s4-72*p4*r*s*q+560*r2*p2*s*q-630*p*r*q3*s;
  if (debug) printf("d5=%e\n",d5);
  if (d5<-e) {
    printf("case (3), 3 distinct real roots, 1 imaginary pair\n");
  } else if (d5>e) { /* cases (1) and (2) */
    if (d2<e) {
      printf("case (2), 1 real root, 2 imaginary pairs\n");
    } else { /* d2>0 */
      d3=p*(40*r-12*p2)-45*q2;
      if (d3<e) printf("case (2), 1 real root, 2 imaginary pairs\n");
      else { /* d3>0 */
        if (D4<e) printf("case (2), 1 real root, 2 imaginary pairs\n");
        else { /* d4>0 */
          printf("case (1), 5 distinct real roots\n");
        }
      }
    }
  } else { /* |d5|<e, so assume d5==0 */
    d4=D4; 
    if (debug) printf("d4=%e\n",d4);
    if (d4>e) {
      printf("case (4), 4 real roots, 1 double\n");
    } else if (d4<-e) {
      printf("case (5), 2 real roots, 1 double, 1 imaginary pair\n");
    } else { /* d5==0 && d4==0 */
      d3=40*r*p-12*p3-45*q2;
      if (fabs(d3)>e) { /* cases (6) to (9) */
        e2=r*(r*(160*p3+900*q2)+p*(60*q2*p+1500*s*q-48*p4))+16*q2*p4
           +s*(625*s*p2-3375*q3-1100*q*p3);
	if (debug) printf("e2=%e\n",e2);
        if (d3>e) {
          if (fabs(e2)>e) printf("case (6), 3 real roots, 2 double\n");
                     else printf("case (7), 3 real roots, 1 triple\n");
        } else { /* d3<0 */
          if (fabs(e2)>e) printf("case (8), 1 real root, 2 imaginary pairs\n");
                     else printf("case (9), 1 triple real root, 1 imaginary pair\n");
        }
      } else { /* d3==0 */
        f2=3*q2-8*r*p;
	if (debug) printf("f2=%e\n",f2);
        if (fabs(d2)>e) {
          if (fabs(f2)>e) printf("case (10), 2 real roots, 1 double, 1 triple\n");
                     else printf("case (11), 2 real roots, 1 quadruple\n");
        } else { /* d2==0 */
          printf("case (12), 1 quintuple real root\n");
        }
      }
    }
  }
  return 0;
}

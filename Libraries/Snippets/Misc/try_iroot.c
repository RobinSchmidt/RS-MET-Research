// KMB 2006 Jul 07; Dec 12
// Try all-integer root-finding methods
// gcc -Wall -O3 try_iroot.c -lgmp && ./a.out
// mcopy -o -v try_iroot.c a:

#include <stdio.h>
#include <gmp.h>

void isqrt(mpz_t x, mpz_t n) {
  /* x=floor(sqrt(n))
     Cohen "A course in computational algebraic number theory" 
     Algorithm 1.7.1 page 38 - convergence proven
     Iteration is x <- (x+n//x)//2 where // is truncating divide.
     Not really Newton's method because the divisions all truncate */
  mpz_t y;
  mpz_init(y);
  mpz_set(x,n); // should use better initial condition!
  do {
    mpz_div(y,n,x);
    mpz_add(y,y,x);
    mpz_tdiv_q_2exp(y,y,1);
    if (mpz_cmp(y,x)>=0)  { mpz_clear(y); return; }
    mpz_set(x,y);
  } while (1);
}

void iroot(mpz_t x, mpz_t n, unsigned int k) {
/* x=floor(n^(1/k))
  KMB's algorithm 
  Iteration is x <- ((k-1)*x+n//x^(k-1))//k where // is truncating divide.
  Reduces to isqrt for k=2 
  Convergence proof by Paul Zimmermann from
  http://sympa.loria.fr/wwsympa/arc/mpfr/2005-04/msg00030.html:
  Root(n,k) = {
    local (x, y);
    x = n+1;
    y = n;
    while (y<x,x=y;y=divrem((k-1)*x+divrem(n,x^(k-1))[1],k)[1]);
    return(x);
  }
  - as long as y < x, the sequence of x's is decreasing, so we only have to
  consider what happens when y >= x
  - since y = floor([(k-1)x + n/x^(k-1)] / k), then y >= x implies
  n/x^(k-1) >= x, thus x^k <= n
  - consider the function f(t) = ((k-1)t+n/t^(k-1))/k. Its derivative is
  [(k-1) - (k-1)*n/t^k]/k, which is negative for t < n^(1/k), and
  positive for t > n^(1/k), thus we have f(t) >= f(n^(1/k)) = n^(1/k).
  This proves that x, y >= q := floor(n^(1/k)) along the algorithm.
  - now assume x is too large: then n - x^k < 0, hence
  y - x = floor([(k-1)x + n/x^(k-1)] / k) - x
  = floor([n/x^(k-1) - x] / k)
  = floor([n - x^k] / x^(k-1) / k) < 0, which contradicts y >= x.
*/
  mpz_t t;
  mpz_t y;
  mpz_init(t);
  mpz_init(y);
  mpz_set(x,n); // should use better initial condition!
  do {
    mpz_pow_ui(t,x,k-1); // bad for large k!
    mpz_div(y,n,t);
    mpz_mul_ui(t,x,k-1);
    mpz_add(y,y,t);
    mpz_div_ui(y,y,k);
    if (mpz_cmp(y,x)>=0) { mpz_clear(y); mpz_clear(t); return; }
    mpz_set(x,y);
  } while (1);
}

int main() {
  unsigned int k;
  mpz_t x,y,n;
  mpz_init(x);
  mpz_init(y);
  mpz_init_set_str(n,"123456789987654321",10);
  printf("n=")+mpz_out_str(stdout,10,n)+printf("\n");
  isqrt(x,n);
  printf("x=floor(sqrt(n))=")+mpz_out_str(stdout,10,x)+printf("\n");
  mpz_mul(y,x,x);
  mpz_sub(y,n,y);
  printf("n-x^2=")+mpz_out_str(stdout,10,y)+printf("\n");
  for (k=2; k<60; k++) {
    iroot(x,n,k);
    printf("floor(n^(1/%u))=",k)+mpz_out_str(stdout,10,x)+printf("\n");
  }
  return 0;
}

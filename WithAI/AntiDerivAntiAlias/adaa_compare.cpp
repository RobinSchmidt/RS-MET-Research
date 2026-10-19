// Spectral comparison: naive vs 1st-order vs 2nd-order ADAA hard clipper.
// Build: g++ -O2 -std=c++17 adaa_compare.cpp -o adaa_compare
//
// Math (f = hard clip, F1' = f, F2' = F1):
//   ADAA1: y = (F1(x0) - F1(x1)) / (x0 - x1)
//   ADAA2: y = 2 * F2[x0, x1, x2]   (second divided difference of F2)
//            = 2 * (mu(x0,x1) - mu(x1,x2)) / (x0 - x2)
//   where mu(a,b) = (F2(a) - F2(b)) / (a - b)  = F2[a,b]
//
// Degenerate cases (limits of the divided differences):
//   a ~ b         : F2[a,b]   -> F1((a+b)/2)  (midpoint approximation)
//   x0 ~ x2 (m = (x0+x2)/2), x1 different:
//                   F2[m,m,x1] = (F1(m) - mu(m,x1)) / (m - x1)
//   all three ~ equal: F2[a,a,a] = F1'(a) = f(a)

#include <cmath>
#include <cstdio>
#include <vector>
#include <complex>
#include <algorithm>

namespace hc
{
  inline double f(double x) { return std::clamp(x, -1.0, 1.0); }

  inline double F1(double x)
  {
    double a = std::fabs(x);
    return (a <= 1.0) ? 0.5 * x * x : a - 0.5;
  }

  inline double F2(double x)
  {
    double a = std::fabs(x);
    if (a <= 1.0)
      return x * x * x / 6.0;
    double r = 1.0 / 6.0 + 0.5 * a * a - 0.5 * a;
    return (x > 0.0) ? r : -r;
  }
}

struct Naive
{
  double process(double x) { return hc::f(x); }
};

struct ADAA1
{
  double process(double x)
  {
    double y;
    double dx = x - x1;
    if (std::fabs(dx) < eps) y = hc::f(0.5 * (x + x1));
    else                     y = (hc::F1(x) - hc::F1(x1)) / dx;
    x1 = x;
    return y;
  }
  double x1 = 0.0, eps = 1e-5;
};

struct ADAA2
{
  // First divided difference of F2 (symmetric in a, b).
  double mu(double a, double b) const
  {
    double d = a - b;
    if (std::fabs(d) < eps) return hc::F1(0.5 * (a + b));
    return (hc::F2(a) - hc::F2(b)) / d;
  }

  double process(double x0)
  {
    double y;
    double mu0 = mu(x0, x1);     // F2[x0, x1]
    double d02 = x0 - x2;

    if (std::fabs(d02) >= eps)
    {
      y = 2.0 * (mu0 - mu1) / d02;       // regular case, mu1 = F2[x1, x2]
    }
    else
    {
      // x0 ~ x2: use the limit F2[m, m, x1]
      double m  = 0.5 * (x0 + x2);
      double dm = m - x1;
      if (std::fabs(dm) < eps) y = hc::f(m);   // all three ~ equal
      else y = 2.0 * (hc::F1(m) - mu(m, x1)) / dm;
    }

    x2  = x1;
    x1  = x0;
    mu1 = mu0;
    return y;
  }

  double x1 = 0.0, x2 = 0.0, mu1 = 0.0, eps = 1e-5;
};

// Returns energy outside the fundamental bin relative to the fundamental, in dB.
// The nonlinearity is odd, so only odd harmonics are generated. With k = 557
// and N = 4096 the 3rd harmonic is already above Nyquist, so all non-
// fundamental energy is aliasing or a folded harmonic. It also includes the
// in-band attenuation of the ADAA lowpass, so it is not a pure alias measure.
template <class Proc>
double measure(Proc p, double gain, int N, int k)
{
  const double pi = 3.14159265358979323846;
  auto in = [&](int n) { return gain * std::sin(2.0 * pi * k * n / N); };

  for (int n = 0; n < N; ++n) p.process(in(n));   // one period to flush state

  std::vector<double> y(N);
  for (int n = 0; n < N; ++n) y[n] = p.process(in(n));

  double fund = 0.0, other = 0.0;
  for (int b = 1; b < N / 2; ++b)
  {
    std::complex<double> s = 0.0;
    for (int n = 0; n < N; ++n)
      s += y[n] * std::polar(1.0, -2.0 * pi * b * n / N);
    (b == k ? fund : other) += std::norm(s);
  }
  return 10.0 * std::log10(other / fund);
}

int main()
{
  const int N = 4096, k = 557;
  std::printf("Non-fundamental energy relative to fundamental (dB, lower = better)\n");
  std::printf("%6s %10s %10s %10s\n", "gain", "naive", "ADAA1", "ADAA2");
  for (double g : {1.5, 2.0, 4.0, 8.0})
    std::printf("%6.1f %10.2f %10.2f %10.2f\n", g,
                measure(Naive{}, g, N, k),
                measure(ADAA1{}, g, N, k),
                measure(ADAA2{}, g, N, k));

  // Sanity: below the clip threshold, ADAA2 should return approximately
  // the input delayed by one sample for slowly varying signals.
  ADAA2 p;
  const double pi = 3.14159265358979323846;
  double prev = 0.0, maxErr = 0.0;
  for (int n = 0; n < 200; ++n)
  {
    double x = 0.5 * std::sin(2.0 * pi * 0.01 * n);
    double y = p.process(x);
    if (n > 3) maxErr = std::max(maxErr, std::fabs(y - prev));
    prev = x;
  }
  std::printf("max |y[n] - x[n-1]| for slow unclipped sine: %.2e\n", maxErr);
  return 0;
}
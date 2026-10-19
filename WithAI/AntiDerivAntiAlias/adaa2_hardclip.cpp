// Second-order ADAA hard clipper (f(x) = clamp(x, -1, 1)).
// Build: g++ -O2 -std=c++17 adaa2_hardclip.cpp -o adaa2
//
// Antiderivatives of f:
//   F1(x) = x^2/2                  for |x| <= 1,   |x| - 1/2 otherwise  (even)
//   F2(x) = x^3/6                  for |x| <= 1,
//           sign(x)*(1/6 + x^2/2 - |x|/2)   otherwise                   (odd)
//
// Second-order ADAA (Bilbao et al. style):
//   mu(a, b) = (F2(a) - F2(b)) / (a - b)       if |a - b| >= eps
//            = F1((a + b) / 2)                 otherwise
//   y[n]     = 2 / (x[n] - x[n-2]) * (mu(x[n], x[n-1]) - mu(x[n-1], x[n-2]))
// If |x[n] - x[n-2]| < eps, fall back to f of the triangular-weighted
// mean (x[n] + 2 x[n-1] + x[n-2]) / 4.
//
// The effective filter is two cascaded one-sample box filters, so the
// output is delayed by 1 sample relative to the ideal nonlinearity.

#include <cmath>
#include <cstdio>
#include <algorithm>

struct HardClipADAA2
{
  static double f(double x)  { return std::clamp(x, -1.0, 1.0); }

  static double F1(double x)
  {
    double a = std::fabs(x);
    return (a <= 1.0) ? 0.5 * x * x : a - 0.5;
  }

  static double F2(double x)
  {
    double a = std::fabs(x);
    if (a <= 1.0)
      return x * x * x / 6.0;
    double r = 1.0 / 6.0 + 0.5 * a * a - 0.5 * a;
    return (x > 0.0) ? r : -r;
  }

  // Divided difference of F2 (approximates F1 at the midpoint when a ~ b)
  double mu(double a, double b) const
  {
    double d = a - b;
    if (std::fabs(d) < eps)
      return F1(0.5 * (a + b));
    return (F2(a) - F2(b)) / d;
  }

  double process(double x)
  {
    const double mu0 = mu(x, x1);      // uses x[n], x[n-1]
    // mu1 = mu(x[n-1], x[n-2]) was computed on the previous call; reuse it
    // instead of calling F2 again. (Valid because mu is symmetric.)
    const double dx = x - x2;
    double y;
    if (std::fabs(dx) < eps)
      y = f(0.25 * (x + 2.0 * x1 + x2));
    else
      y = 2.0 * (mu0 - mu1) / dx;

    x2  = x1;
    x1  = x;
    mu1 = mu0;
    return y;
  }

  double x1 = 0.0, x2 = 0.0;
  double mu1 = 0.0;      // mu(x[n-1], x[n-2]); zero input gives 0 initially
  double eps = 1e-5;     // use larger (~1e-3) for float
};

int main()
{
  // Quick sanity check: for a slowly varying input well inside [-1, 1],
  // the output should equal the input delayed by one sample.
  HardClipADAA2 p;
  const double pi = 3.14159265358979323846;
  for (int n = 0; n < 8; ++n)
  {
    double x = 0.5 * std::sin(2.0 * pi * 0.01 * n);
    std::printf("n=%d x=%9.6f y=%9.6f\n", n, x, p.process(x));
  }
  // For heavy overdrive (gain 4), y should stay within [-1, 1] and show
  // smoothed corners instead of hard ones.
  return 0;
}
// First-order antiderivative antialiasing (ADAA) demo.
// Build: g++ -O2 -std=c++17 adaa_demo.cpp -o adaa_demo
//
// Core idea: treat the input as piecewise linear between samples, apply the
// nonlinearity f to that continuous signal, average over one sample interval
// (box lowpass), then sample. The average has a closed form using the
// antiderivative F (F' = f):
//
//   y[n] = (F(x[n]) - F(x[n-1])) / (x[n] - x[n-1])
//
// If x[n] ~= x[n-1] this is 0/0, so we fall back to f((x[n] + x[n-1]) / 2).

#include <cmath>
#include <cstdio>
#include <vector>
#include <complex>
#include <algorithm>

// ---------------------------------------------------------------------------
// Nonlinearities f(x) and their first antiderivatives F(x)
// ---------------------------------------------------------------------------

// Hard clipper: f(x) = clamp(x, -1, 1)
struct HardClip
{
  static double f(double x) { return std::clamp(x, -1.0, 1.0); }

  // F(x) = x^2/2 for |x| <= 1, |x| - 1/2 otherwise.
  // The two pieces match in value and slope at |x| = 1, so F is C1.
  static double F(double x)
  {
    double a = std::fabs(x);
    return (a <= 1.0) ? 0.5 * x * x : a - 0.5;
  }
};

// tanh: f(x) = tanh(x), F(x) = log(cosh(x))
struct Tanh
{
  static double f(double x) { return std::tanh(x); }

  // log(cosh(x)) overflows for large |x|, so use the identity
  // log(cosh(x)) = |x| + log1p(exp(-2|x|)) - log(2)
  static double F(double x)
  {
    double a = std::fabs(x);
    return a + std::log1p(std::exp(-2.0 * a)) - 0.6931471805599453;
  }
};

// ---------------------------------------------------------------------------
// Memoryless nonlinearity processors
// ---------------------------------------------------------------------------

// Naive version: apply f directly. This aliases.
template <class NL>
struct Naive
{
  double process(double x) { return NL::f(x); }
};

// First-order ADAA. Needs one sample of state: x[n-1] and F(x[n-1]).
template <class NL>
struct ADAA1
{
  double process(double x)
  {
    const double Fx = NL::F(x);
    double y;
    const double dx = x - x1;

    if (std::fabs(dx) < eps)
      y = NL::f(0.5 * (x + x1));   // ill-conditioned: use the midpoint value
    else
      y = (Fx - Fx1) / dx;         // the actual ADAA formula

    x1  = x;                       // update state
    Fx1 = Fx;
    return y;
  }

  // Caching F(x[n-1]) avoids evaluating F twice per sample.
  double x1 = 0.0, Fx1 = NL::F(0.0);
  double eps = 1e-5;
};

// ---------------------------------------------------------------------------
// Test: drive a sine hard into the nonlinearity and measure aliasing.
// ---------------------------------------------------------------------------

// Run the processor on a sine and return the energy (dB) outside the
// fundamental bin, relative to the fundamental. Since the fundamental sits
// exactly on a DFT bin (see below), everything else is harmonics or aliases.
// With k=557, N=4096, the 3rd harmonic is already above Nyquist, so ALL the
// non-fundamental energy is either aliasing or folded harmonics.
template <class Proc>
double measure(Proc p, double gain, int N, int k)
{
  const double pi = 3.14159265358979323846;
  std::vector<double> y(N);

  // Warm up one sample so the ADAA state is valid.
  p.process(0.0);
  for (int n = 0; n < N; ++n)
    y[n] = p.process(gain * std::sin(2.0 * pi * k * n / N));

  double fund = 0.0, other = 0.0;
  for (int b = 1; b < N / 2; ++b)         // naive DFT, fine for N = 4096
  {
    std::complex<double> s = 0.0;
    for (int n = 0; n < N; ++n)
      s += y[n] * std::polar(1.0, -2.0 * pi * b * n / N);
    double e = std::norm(s);
    (b == k ? fund : other) += e;
  }
  return 10.0 * std::log10(other / fund);
}

int main()
{
  const int    N    = 4096;
  const int    k    = 557;     // ~6 kHz at 44.1 kHz; bin-exact, so no leakage
  const double gain = 4.0;     // overdrive so the nonlinearity is exercised

  std::printf("Aliasing + harmonic energy relative to fundamental (dB, lower is better)\n");
  std::printf("HardClip naive: %7.2f\n", measure(Naive<HardClip>{}, gain, N, k));
  std::printf("HardClip ADAA1: %7.2f\n", measure(ADAA1<HardClip>{}, gain, N, k));
  std::printf("Tanh     naive: %7.2f\n", measure(Naive<Tanh>{},     gain, N, k));
  std::printf("Tanh     ADAA1: %7.2f\n", measure(ADAA1<Tanh>{},     gain, N, k));
  return 0;
}
#pragma once

#include <cmath>

/**
  State variable filter (SVF), designed by Andrew Simper of Cytomic.

  http://cytomic.com/files/dsp/SvfLinearTrapOptimised2.pdf

  The frequency response of this filter is the same as of BZT filters.
  This is a second-order filter. It has a cutoff slope of 12 dB/octave.
  Q = 0.707 means no resonant peaking.

  This filter will self-oscillate when Q is very high (can be forced by
  setting the `k` coefficient to zero).

  This filter is stable when modulated at high rates.

  This implementation uses the generic formulation that can morph between the
  different responses by altering the mix coefficients m0, m1, m2.
 */
class StateVariableFilter
{
public:
    StateVariableFilter() : m0(0.0f), m1(0.0f), m2(0.0f) { }

    void setCoefficients(double sampleRate, double freq, double Q) noexcept
    {
        g = std::tan(M_PI * freq / sampleRate);
        k = 1.0 / Q;
        a1 = 1.0 / (1.0 + g * (g + k));
        a2 = g * a1;
        a3 = g * a2;
    }

    void lowpass(double sampleRate, double freq, double Q) noexcept
    {
        setCoefficients(sampleRate, freq, Q);
        m0 = 0.0f;
        m1 = 0.0f;
        m2 = 1.0f;
    }

    void highpass(double sampleRate, double freq, double Q) noexcept
    {
        setCoefficients(sampleRate, freq, Q);
        m0 = 1.0f;
        m1 = -k;
        m2 = -1.0f;
    }

    void bandpass(double sampleRate, double freq, double Q) noexcept
    {
        setCoefficients(sampleRate, freq, Q);
        m0 = 0.0f;
        m1 = k;     // paper says 1, but that is not same as RBJ bandpass
        m2 = 0.0f;
    }

    void notch(double sampleRate, double freq, double Q) noexcept
    {
        setCoefficients(sampleRate, freq, Q);
        m0 = 1.0f;
        m1 = -k;
        m2 = 0.0f;
    }

    void allpass(double sampleRate, double freq, double Q) noexcept
    {
        setCoefficients(sampleRate, freq, Q);
        m0 = 1.0f;
        m1 = -2.0f * k;
        m2 = 0.0f;
    }

    // Note: This is not the same as the RBJ peaking filter, since no dbGain.
    void peaking(double sampleRate, double freq, double Q) noexcept
    {
        setCoefficients(sampleRate, freq, Q);
        m0 = 1.0f;
        m1 = -k;
        m2 = -2.0f;
    }

    // Note: This is the same as the RBJ peaking EQ.
    void bell(double sampleRate, double freq, double Q, double dbGain) noexcept
    {
        const double A = std::pow(10.0, dbGain / 40.0);
        g = std::tan(M_PI * freq / sampleRate);
        k = 1.0 / (Q * A);
        a1 = 1.0 / (1.0 + g * (g + k));
        a2 = g * a1;
        a3 = g * a2;
        m0 = 1.0f;
        m1 = k * (A*A - 1.0);
        m2 = 0.0f;
    }

    void lowShelf(double sampleRate, double freq, double Q, double dbGain) noexcept
    {
        const double A = std::pow(10.0, dbGain / 40.0);
        g = std::tan(M_PI * freq / sampleRate) / std::sqrt(A);
        k = 1.0 / Q;
        a1 = 1.0 / (1.0 + g * (g + k));
        a2 = g * a1;
        a3 = g * a2;
        m0 = 1.0f;
        m1 = k * (A - 1.0);
        m2 = (A*A - 1.0);
    }

    void highShelf(double sampleRate, double freq, double Q, double dbGain) noexcept
    {
        const double A = std::pow(10.0, dbGain / 40.0);
        g = std::tan(M_PI * freq / sampleRate) * std::sqrt(A);
        k = 1.0 / Q;
        a1 = 1.0 / (1.0 + g * (g + k));
        a2 = g * a1;
        a3 = g * a2;
        m0 = A * A;
        m1 = k * (1.0 - A) * A;
        m2 = (1.0 - A*A);
    }

    void reset() noexcept
    {
        ic1eq = 0.0f;
        ic2eq = 0.0f;
    }

    float processSample(float v0) noexcept
    {
        float v3 = v0 - ic2eq;
        float v1 = a1 * ic1eq + a2 * v3;
        float v2 = ic2eq + a2 * ic1eq + a3 * v3;
        ic1eq = 2.0f * v1 - ic1eq;
        ic2eq = 2.0f * v2 - ic2eq;
        return m0 * v0 + m1 * v1 + m2 * v2;
    }

private:
    float g, k, a1, a2, a3;  // filter coefficients
    float m0, m1, m2;        // mix coefficients
    float ic1eq, ic2eq;      // internal state
};

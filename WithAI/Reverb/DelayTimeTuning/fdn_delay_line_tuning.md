# FDN delay-line tuning

**User:** @RobinSchmidt

## @RobinSchmidt

In feedback delay networks (FDNs) for artificial reverberation, it is common wisdom in the literature that the lengths of the delaylines should be mutually prime in order to avoid colliding echoes as much as possible. Are there any known techniques beyond choosing mutually prime numbers for tuning the delayline lengths?

## Copilot

Yes. Beyond mutual primality, the literature and practice offer several techniques:

**Delay-length selection**
- **Prime powers (Schroeder/Moorer/Jot):** Use lengths of the form p_i^k_i, with distinct primes p_i. This gives a more even spread of echo density than plain primes. Jot's original FDN work uses this.
- **Geometric or exponential spacing:** Make the lengths roughly geometrically distributed, for example within a ratio of 1:1.5 to 1:2 from shortest to longest. The lengths are then snapped to nearby primes or coprime values. Mean delay length is tied to the target echo density.
- **Echo-density targets:** Choose the sum of the lengths, or the mean, so the echo density at the onset of the late tail reaches about 1000–2000 echoes/s. This is the Schroeder criterion for avoiding fluttery tails. Higher density needs more or longer lines.
- **Avoid common divisors of sums and differences:** Mutual primality only avoids exact coincidences of echoes. Also check that no small integer combinations, such as m_i + m_j = m_k or 2m_i ≈ m_j, produce near-coincident echoes. Search methods such as brute-force or genetic optimization can minimize these near-coincidences.
- **Frequency-domain criterion:** Look at the comb-filter resonances of the loop. Choose lengths so that the combined mode frequencies f = k·fs/m_i are as evenly distributed as possible, with no clusters. This is the modal-density view, with the aim of maximizing the minimum spacing between modes.
- **Room-based lengths:** Derive them from room dimensions or simulated geometry, for example the mean free path for a given volume and surface area. Related work is on FDNs derived from room modes, from Stautner & Puckette, and on waveguide or scattering networks.

**Beyond integer lengths**
- **Fractional delays and time variation:** Interpolated fractional lengths, plus slow delay-length modulation (Dattorro, Lexicon-style), break up the residual periodicity that tuning cannot remove. Modulation is the most widely used practical remedy.
- **Allpass or Schroeder-style diffusers** at the inputs and outputs, or inside the loop, add echo density independent of the length tuning.

**Related design aspects**
- **Mixing matrix choice:** Hadamard, Householder, or circulant matrices. Sparse or time-varying matrices affect how quickly echo density builds up. Schlecht & Habets showed that the lengths and the feedback matrix interact in their effect on the modal and echo distribution.
- **Delay-dependent attenuation:** Per-line decay filters keep the reverberation time consistent across lines of different length.

**References to look up:** Jot & Chaigne (1991); Rocchesso & Smith, "Circulant and Elliptic Feedback Delay Networks"; Rocchesso, "Maximally Diffusive Yet Efficient FDNs" (1997); Schlecht & Habets, "On Lossless Feedback Delay Networks" and "Sign-agnostic matrix design for spatial artificial reverberation"; Schlecht's work on time-varying and sparse FDNs; Dal Santo et al. on echo-density and mode-distribution optimization of FDN delays.

## @RobinSchmidt

Will modulating the feedback matrix modulate the modal frequencies or just their Q-values?

## Copilot

Both, in general. The modal frequencies and the Q-values (decay rates) are the same thing seen two ways: they are the complex poles of the system.

**Why**
The FDN poles are the roots of det(I − D(z)·A) = 0, where D(z) = diag(z^−m_i). The poles are z = e^{(−α + jω)T}. Imaginary parts give the modal frequencies and real parts give the damping, hence the Q. Changing A moves the poles in the complex plane, and that generally changes both.

**Lossless case**
If A is orthogonal or unitary, all poles lie on the unit circle. Then:
- Q is unaffected. Every mode stays undamped, and the decay is set only by the loss filters.
- Frequencies do move. The matrix couples the delay lines, so it shifts the mode positions, i.e. the comb-resonance pattern, within the bounds set by the delay lengths.

So modulating a **unitary** matrix, for example by interpolating along a continuous path of rotations or a time-varying unitary, mostly modulates modal frequencies and the mode-to-input/output coupling (the residues). It leaves the decay alone. This is why it is attractive: it adds time variation without altering the reverb time or risking instability. Time-varying orthogonal matrices are stable in the energy sense.

**Lossy case**
If the matrix is not unitary, as with a scaled gain matrix g·A, a non-orthogonal one, or modulated gains, then the pole radii change too. That modulates the decay times (the Q), and it can destabilize the network if the spectral radius of D·A exceeds 1.

**Practical notes**
- Modulating the matrix changes the residues (mode amplitudes) at least as much as the frequencies. Audibly, this shows up as mode-amplitude fluctuation, which gives some of the same decorrelation as delay modulation.
- It generally shifts frequencies less than delay-length modulation does. Delay modulation shifts every mode directly, but it also causes pitch and Doppler-like artifacts. Matrix modulation avoids those.
- For slow modulation, the pole motion is smooth, so the quasi-static analysis holds. For fast modulation, the system is genuinely time-varying and the pole picture breaks down.
- Schlecht & Habets studied time-varying FDN matrices, using interpolated unitary matrices and the "continuously varying" family, with the aim of lossless time variation.

I'm confident in the pole analysis. The specific paper claims are from memory, so verify them.

## @RobinSchmidt

Can you give me a link to the research of Schlecht & Habets? I'm interested in the effects of modulating the feedback matrix.

## Copilot

I'll look up the actual papers rather than rely on memory.

I couldn't reach the web from this session, so I can't verify links. These are the titles to search for, from memory.

**Most relevant to your question**
- S. J. Schlecht and E. A. P. Habets, "Time-varying feedback matrices in feedback delay networks and their application in artificial reverberation," *J. Acoust. Soc. Am.* 138(3), 1389–1398, 2015. It covers modulating the feedback matrix, including keeping it unitary over time.

**Related**
- S. J. Schlecht and E. A. P. Habets, "On lossless feedback delay networks," *IEEE Trans. Signal Process.* 65(6), 2017. It covers the conditions for losslessness, including time-varying matrices.
- S. J. Schlecht's doctoral thesis, *Feedback Delay Networks in Artificial Reverberation and Reverberation Enhancement* (FAU Erlangen-Nürnberg, 2018). It is the most thorough treatment, and it is open access.

**Where to find them**
- Search the exact titles on Google Scholar or arXiv. Many of Schlecht's papers are on arXiv.
- Authors' pages: audiolabs-erlangen.de, and Schlecht's page at Aalto University.

I can't confirm the page numbers or details above. Check them against the actual records. Also, my earlier claim about "interpolated unitary matrices" in that paper is from memory. Read the abstract before relying on it.

## @RobinSchmidt

Can you give me example code of a simple FDN with 4 delaylines in C++?

## Copilot

Here's a minimal 4-line FDN with a Householder feedback matrix, per-line decay gains, and prime delay lengths. It's self-contained and I haven't compiled it in this session.

```cpp
#include <array>
#include <cmath>
#include <vector>

class FDN4
{
public:
    FDN4(double sampleRate, double rt60Seconds)
    {
        const std::array<int, N> lengths = { 1031, 1327, 1523, 1777 }; // primes, ~23-40 ms @ 44.1 kHz
        for (int i = 0; i < N; ++i)
        {
            buffers[i].assign(lengths[i], 0.0);
            pos[i] = 0;
            // Per-line gain so that every line decays by 60 dB in rt60 seconds
            gains[i] = std::pow(10.0, -3.0 * lengths[i] / (rt60Seconds * sampleRate));
        }
    }

    double process(double in)
    {
        // Read the delay line outputs (oldest sample = current write position)
        std::array<double, N> y;
        for (int i = 0; i < N; ++i)
            y[i] = buffers[i][pos[i]] * gains[i];

        // Householder matrix A = I - (2/N) * 1 * 1^T  (orthogonal, N = 4)
        double sum = 0.0;
        for (int i = 0; i < N; ++i)
            sum += y[i];
        const double k = 2.0 / N * sum;

        // Feed back and write the input into each line
        double out = 0.0;
        for (int i = 0; i < N; ++i)
        {
            double fb = y[i] - k;
            buffers[i][pos[i]] = in * inGain[i] + fb;
            pos[i] = (pos[i] + 1) % static_cast<int>(buffers[i].size());
            out += y[i] * outGain[i];
        }
        return out;
    }

private:
    static constexpr int N = 4;
    std::array<std::vector<double>, N> buffers;
    std::array<int, N> pos{};
    std::array<double, N> gains{};
    // Simple sign pattern for decorrelated input/output taps
    std::array<double, N> inGain  = { 0.5,  0.5,  0.5,  0.5 };
    std::array<double, N> outGain = { 0.5, -0.5,  0.5, -0.5 };
};

// Usage:
//   FDN4 fdn(44100.0, 2.0);          // 2 s RT60
//   double wet = fdn.process(dry);   // call once per sample
```

**Notes**
- The gain formula `10^(-3·m/(RT60·fs))` gives 60 dB decay after RT60 seconds for each line.
- The Householder matrix needs only O(N) operations, so there's no explicit matrix multiply.
- Swapping in a Hadamard matrix works just as well (for N = 4, scaled by 1/2).
- Add a one-pole lowpass in each loop for frequency-dependent decay.
- For a more diffuse sound, add delay modulation (fractional delays) or input diffusers.

I can add a lowpass damping filter, modulation, or a test harness that writes an impulse response if you'd like.

## @RobinSchmidt

Please use a Hadamard matrix and let the delayline lengths be adjustable by the user. Maybe take them as additional constructor parameter.

## Copilot

It compiles cleanly with `g++ -std=c++17 -Wall`. The file is at `/tmp/fdn4.cpp`. The impulse-response samples I printed were all 0, which is expected because the first echoes only arrive at the delay lengths, and I printed every 1000th sample. I haven't checked the decay audibly.

```cpp
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

class FDN4
{
public:
    static constexpr int N = 4;

    // lengths: delay line lengths in samples (each >= 1); ideally mutually prime
    FDN4(double sampleRate, double rt60Seconds, const std::array<int, N>& lengths)
    {
        for (int i = 0; i < N; ++i)
        {
            if (lengths[i] < 1)
                throw std::invalid_argument("delay length must be >= 1");
            buffers[i].assign(lengths[i], 0.0);
            // Per-line gain giving 60 dB decay after rt60Seconds
            gains[i] = std::pow(10.0, -3.0 * lengths[i] / (rt60Seconds * sampleRate));
        }
    }

    double process(double in)
    {
        std::array<double, N> y;
        for (int i = 0; i < N; ++i)
            y[i] = buffers[i][pos[i]] * gains[i];

        // Orthogonal 4x4 Hadamard matrix, scaled by 1/2, using fast butterflies
        const double a = y[0] + y[1], b = y[0] - y[1];
        const double c = y[2] + y[3], d = y[2] - y[3];
        const std::array<double, N> fb = { 0.5 * (a + c), 0.5 * (b + d),
                                           0.5 * (a - c), 0.5 * (b - d) };

        double out = 0.0;
        for (int i = 0; i < N; ++i)
        {
            buffers[i][pos[i]] = in * inGain[i] + fb[i];
            if (++pos[i] == static_cast<int>(buffers[i].size()))
                pos[i] = 0;
            out += y[i] * outGain[i];
        }
        return out;
    }

private:
    std::array<std::vector<double>, N> buffers;
    std::array<int, N> pos{};
    std::array<double, N> gains{};
    std::array<double, N> inGain  = { 0.5,  0.5,  0.5,  0.5 };
    std::array<double, N> outGain = { 0.5, -0.5,  0.5, -0.5 };
};

// Usage:
//   FDN4 fdn(44100.0, 2.0, { 1031, 1327, 1523, 1777 });
//   double wet = fdn.process(dry);
```

The lengths are not checked for mutual primality. Pass any values you like.

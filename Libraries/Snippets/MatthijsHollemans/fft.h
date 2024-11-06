// Code is from here:
// https://gist.github.com/hollance/6dec14d8532f67af8642d6171b7fa192

#pragma once

#include <vector>

/**
  Very basic Cooley-Tukey FFT, not optimized for speed at all.

  This is a complex-valued FFT, so you need to provide both real and imaginary
  input values (set imaginary to zero for audio input). The complex values are
  interleaved.

  No normalization is performed. If you need normalization, manually divide by
  the FFT length after the inverse transform.
 */
class FFT
{
public:
    FFT(int order)
    {
        numBits = size_t(order);
        length = 1 << numBits;
        createRearrangeTable();
    }

    void perform(float* data, bool inverse)
    {
        rearrange(data);

        float twoPi = 6.283185307179586231995926937088370323f;
        if (!inverse) { twoPi *= -1.0f; }

        for (size_t i = 0; i < numBits; ++i) {
            size_t m = 1 << i;
            size_t n = m << 1;

            for (size_t k = 0; k < m; ++k) {
                float a = float(k) / float(n) * twoPi;
                float c = std::cos(a);
                float s = std::sin(a);

                for (size_t j = k; j < length; j += n) {
                    size_t vr = (j + m) << 1;
                    size_t vi = vr + 1;
                    size_t wr = j << 1;
                    size_t wi = wr + 1;

                    float tr = c * data[vr] - s * data[vi];
                    float ti = s * data[vr] + c * data[vi];

                    data[vr] = data[wr] - tr;
                    data[vi] = data[wi] - ti;

                    data[wr] += tr;
                    data[wi] += ti;
                }
            }
        }
    }

private:
    void createRearrangeTable()
    {
        indices.resize(length);
        indices[0] = 0;

        for (size_t limit = 1, bit = length / 2; limit < length; limit <<= 1, bit >>= 1) {
            for (size_t i = 0; i < limit; ++i) {
                indices[i + limit] = indices[i] + bit;
            }
        }

        for (size_t i = 0; i < length; ++i) {
            if (indices[i] == i) {
                indices[i] = 0;           // don't swap
            } else {
                indices[indices[i]] = 0;  // swap each side only once
            }
        }
    }

    void rearrange(float* data)
    {
        for (size_t i = 0; i < length; ++i) {
            if (indices[i] != 0) {
                size_t j = indices[i];
                std::swap(data[i*2    ], data[j*2    ]);  // real
                std::swap(data[i*2 + 1], data[j*2 + 1]);  // imag
            }
        }
    }

    size_t numBits;
    size_t length;
    std::vector<size_t> indices;
};

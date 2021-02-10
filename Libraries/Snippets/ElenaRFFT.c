// https://www.kvraudio.com/forum/viewtopic.php?f=33&t=559722

// ----------------------------------------------------------------------------------------------------
// ELENA REAL FFT - (C) ELENA DESIGN 2021 - VERSION 1.0
// by Elena Novaretti
// www.elenadomain.it/listing/SE - www.kvraudio.com/developer/elena-design - elena@elenadomain.it
// 
// This source code can be freely used, distributed, adapted or even modified until the present
// accompanying header is not removed
// ----------------------------------------------------------------------------------------------------
//
// DESCRIPTION:
// A couple of compact, optimized routines performing direct/inverse 1D Fast Fourier Transform working
// *directly* with Real numbers and half-complex spectra, about 30% faster than the "standard"
// approach of feeding a Complext FFT with two Real sequences and post-processing the result.
//
// MOTIVATION:
// The Fast Fourier Transform is an inherently Complex procedure, converting a Complex sequence
// to a Complex spectrum and vice-versa. However in the real world and in common audio applications
// we are pretty always dealing with Real sequences and we need to perform spectral processing and
// analysis on a half-complex spectrum (i.e containing only the positive frequencies from 0 to Nyquist).
// To achieve that efficiently, it is established practice to feed both the Real and the Imaginary inputs
// of a FFT with an appositely processed and splitted Real sequence twice as long, computing the Complex
// transform and post-processing the result to obtain a half-complex spectrum.
// I always wondered whether a more direct and compact approach working natively on real sequences
// existed, thus avoiding many redundant passages and operations; unfortunately despite a long search
// I could not find anything. Therefore I decided to write my own dedicated routines.
// I tried to keep things as compact as possible, and I adopted C (quite optimized = hardly readable)
// for performance reasons; however nothing prevents the incorporation of the present code inside a CPP
// class if needed.
//
// USAGE:
// Given the pointer to a Real sequence of z floats, where z must be an integer power of 2, rfft()
// replaces the data with their Discrete Fourier Transform in form of a z-floats long half-complex
// spectrum [F0][F1.re][F1.im]...[Fz/2-1.re][Fz/2-1.im][Fz/2].
// Since DC (F0) and Nyquist (Fz/2) always have zero imaginary parts they are returned as singlets
// rather than regular 'bins', to preserve the relationship "z floats in, z floats out".
// However they can be converted to regular bins externally by simply copying the data block to
// another block larger by two units and setting their imaginaries to zero, whenever spectral
// processing required all bins to be identical.
// rifft() does the inverse job, i.e converts a z-sized half-complex spectrum to the corresponding
// z-sized real sequence.
// The two tables used (roots of unity and bit-reversed indexes) shall be pre-computed once for all
// at initialization stage by the apposite functions computeRoots() and computeIndexes().
// The FFT_MAX_SIZE #definition shall always be set to the maximum planned size (must always be an
// integer power of two aswell)
// Warning: to avoid too many conditionals, no check is done against a value of z which is not a power
// of two or which exceeds FFT_MAX_SIZE!
//
// PRINCIPLE:
// To facilitate their progressive merging, data are first scrambled in bit-reversed order.
// With w increasing from 1 to z/2, starting from the initial trivial case of w=1 (the half-complex
// spectrum of one real sample corresponds to the sample itself), partial w spectra are joined in
// interleaved fashion two by two into 2w spectra up to the last stage (w=z/2), where the spectrum
// corresponding to the even samples and the one corresponding to the odd samples will be merged to
// produce the final spectrum.
// Given a half-complex spectrum A corresponding to a time-domain (TD) sequence of even samples:
// [A0][A1.re][A1.im]...[An/2-1.re][An/2-1.im][An/2]
// and a spectrum B corresponding to a TD sequence of odd samples:
// [B0][B1.re][B1.im]...[Bn/2-1.re][Bn/2-1.im][Bn/2]
// the first spectrum (A) is first doubled by concatenating the flipped copy of itself, conjugated,
// expanding the Nyquist singlets to regular bins (null imaginaries), overlapping (summed together):
// [A0][A1.re][A1.im]...[An/2-1.re][An/2-1.im] [2An/2][0] [An/2-1.re][-An/2-1.im]...[A1.re][-A1.im][A0]
// this operation produces a double spectrum A2 corresponding to the original TD sequence at even
// positions interleaved by zeros at the odd positions.
// The same operation is performed on the spectrum B to produce B2:
// [B0][B1.re][B1.im]...[Bn/2-1.re][Bn/2-1.im] [2Bn/2][0] [Bn/2-1.re][-Bn/2-1.im]...[B1.re][-B1.im][B0]
// Its corresponding TD sequence is then circularly shifted by one position to the right by multiplying
// every bin by exp(-i*2*PI*n/(2*w)), and the two double spectra A2 and B2 then summed together.
// Since every doubling operation actually produces a spectrum corresponding to a TD sequence of double
// magnitude, the overall magnitude shall be scaled by 1/z.
// To inverse-transform, the same exact operations are performed in reverse order. A separate function
// has to be used, since the scheme and the optimizations adopted prevent the simple usage of a common
// code suitable for both forward and inverse transforms by just changing a sign.
// To maximize performance, both bit-reversed indexes and roots of unity are stored in pre-computed
// tables; to minimize access to the table, roots are computed only up to PI/4 and all the inherent
// symmetries exploited; no more than one read and one write operation on every data cell is performed
// per pass.
// ----------------------------------------------------------------------------------------------------



#define FFT_MAX_SIZE 16384 // Always set the max planned FFT size, must be a power of 2

#include <math.h>

typedef struct { float re, im; } complex_t;

// The pre-computed tables

complex_t fftRoots[FFT_MAX_SIZE >> 3];

unsigned int fftIndexes[FFT_MAX_SIZE >> 1];


// The functions to compute the tables

void computeRoots(void)
{
	unsigned int c;
	double p, k = 6.283185307179586476925286766559 / FFT_MAX_SIZE;

	for (c = 0, p = 0.0; c < FFT_MAX_SIZE >> 3; c++, p -= k)
		fftRoots[c] = { (float)cos(p),(float)sin(p) };

}

void computeIndexes()
{
	unsigned int c, j;
	const unsigned int m = FFT_MAX_SIZE >> 1;

	for (c = j = 0; c < m; c++)
	{
		fftIndexes[c] = j;
		j ^= m - m / ((c ^ (c + 1)) + 1);
	}
}



// ---------------------------      The FFT functions     ----------------------------------------------

void rfft(float* data, const int z)
{
	unsigned int w, w2, i, p, inc;
	float x, y, x1, y1, x2, y2, rx, ry, tmp;
	const float scal = 1.f / (float)z;
	float* data2, * dataover = data + z, * ptr1, * ptr2;
	complex_t* rotptr, * rotators = fftRoots;
	unsigned int* bitrev = fftIndexes;

	// Scramble data in bit-reverse order and already generate two-points spectra to save one passage

	for (i = 0, ptr1 = data, inc = FFT_MAX_SIZE / z, w = z >> 1; i < z; i += 2, bitrev += inc)
	{
		p = *bitrev;
		x = *(ptr1++);
		if (p > i) { tmp = *(ptr2 = data + p); *ptr2 = x; x = tmp; }
		p += w;
		y = *ptr1;
		if (p > i + 1) { tmp = *(ptr2 = data + p); *ptr2 = y; y = tmp; }
		*(ptr1--) = (x - y) * scal;
		*ptr1 = (x + y) * scal;
		ptr1 += 2;
	}

	// Progressively merge consecutive interleaved w spectra into w2 spectra

	for (w = 2, w2 = 4, inc = FFT_MAX_SIZE >> 2; w < z; w <<= 1, w2 <<= 1, inc >>= 1)
	{
		for (data2 = data; data2 < dataover; data2 += w2) // Process every w2 group of two partial spectra
		{
			// Process N,DC singlets and central bin

			x = data2[0];
			y = data2[w];
			data2[0] = x + y;
			x -= y;
			y = data2[w2 - 1];
			data2[w2 - 1] = x;
			data2[w - 1] *= 2.f;
			data2[w] = -(y + y);

			if (w < 4) continue;

			// Process the two PI/4 bins 

			ptr1 = data2 + (w >> 1) - 1;
			ptr2 = ptr1 + w;
			x = *(ptr1++);	y = *ptr1;
			x1 = *(ptr2++);	y1 = *ptr2;
			tmp = (x1 + y1) * 0.7071067812f;
			y1 = (y1 - x1) * 0.7071067812f;
			x1 = tmp;
			*(ptr1--) = y + y1;
			*ptr1 = x + x1;
			*(ptr2--) = y1 - y;
			*ptr2 = x - x1;

			if (w < 8) continue;

			// Process all other bins

			for (ptr1 = data2 + 1, ptr2 = data2 + w - 3, rotptr = rotators + inc; ptr1 < ptr2; rotptr += inc)
			{
				rx = rotptr->re; ry = rotptr->im;
				x = *ptr1; y = *(ptr1 + 1);
				x1 = *(ptr1 + w); y1 = *(ptr1 + w + 1);
				x2 = *(ptr2 + w); y2 = *(ptr2 + w + 1);
				tmp = rx * x1 - ry * y1; y1 = rx * y1 + ry * x1; x1 = tmp;
				*ptr1 = x + x1;	*(ptr1 + 1) = y + y1;
				*(ptr2 + w) = x - x1; *(ptr2 + w + 1) = y1 - y;
				x = *ptr2; y = *(ptr2 + 1);
				tmp = rx * y2 - ry * x2; y2 = -rx * x2 - ry * y2; x2 = tmp;
				*ptr2 = x + x2;	*(ptr2 + 1) = y + y2;
				*(ptr1 + w) = x - x2; *(ptr1 + w + 1) = y2 - y;
				ptr1 += 2; ptr2 -= 2;
			}
		}
	}
}





void rifft(float* data, const int z)
{
	unsigned int w, w2, i, p, inc;
	float x, y, x1, y1, x2, y2, rx, ry;
	float* data2, * dataover = data + z, * ptr1, * ptr2;
	complex_t* rotptr, * rotators = fftRoots;
	unsigned int* bitrev = fftIndexes;

	data[0] += data[0];
	data[z - 1] += data[z - 1];

	// Progressively split w2 spectra into consecutive interleaved w spectra

	for (w = z >> 1, w2 = z, inc = FFT_MAX_SIZE / z; w > 1; w >>= 1, w2 >>= 1, inc <<= 1)
	{
		for (data2 = data; data2 < dataover; data2 += w2) // Split every w2 group in two w interleaved spectra
		{
			// Process N,DC singlets and central bin

			x = data2[0];
			y = data2[w2 - 1];
			data2[0] = x + y;
			data2[w - 1] += data2[w - 1];
			x -= y;
			y = data2[w];
			data2[w] = x;
			data2[w2 - 1] = -(y + y);


			if (w < 4) continue;

			// Process the two PI/4 bins 

			ptr1 = data2 + (w >> 1);
			ptr2 = ptr1 + w;
			y = *(ptr1--);
			x = *ptr1;
			y1 = *(ptr2--);
			x1 = *ptr2;
			*(ptr1++) = x + x1;
			*ptr1 = y - y1;
			*(ptr2++) = 0.7071067812f * (x - y - x1 - y1);
			*ptr2 = 0.7071067812f * (x + y + y1 - x1);

			if (w < 8) continue;

			// Process all other bins

			for (ptr1 = data2 + 1, ptr2 = data2 + w - 3, rotptr = rotators + inc; ptr1 < ptr2; rotptr += inc)
			{
				rx = rotptr->re; ry = rotptr->im;
				x = *ptr1; y = *(ptr1 + 1);
				x1 = *(ptr2 + w); y1 = *(ptr2 + w + 1);
				*ptr1 = x + x1; *(ptr1 + 1) = y - y1;
				x -= x1; y += y1;
				x1 = *(ptr1 + w); y1 = *(ptr1 + w + 1);
				*(ptr1 + w) = rx * x + ry * y;
				*(ptr1 + w + 1) = rx * y - ry * x;
				x = *ptr2; y = *(ptr2 + 1);
				*ptr2 = x + x1; *(ptr2 + 1) = y - y1;
				y += y1; x -= x1;
				*(ptr2 + w) = -rx * y - ry * x;
				*(ptr2 + w + 1) = rx * x - ry * y;
				ptr1 += 2; ptr2 -= 2;
			}
		}

	}

	// Convert two-points spectra to samples and scramble back

	for (i = 0, ptr1 = data + 1, inc = FFT_MAX_SIZE / z, w = z >> 1; i < z; i += 2, bitrev += inc)
	{
		p = *bitrev;
		y = *(ptr1--) * .5f;
		x = *ptr1 * .5f;
		x1 = *ptr1 = x + y;
		if (p < i) { *ptr1 = *(ptr2 = data + p); *ptr2 = x1; }
		p += w;
		ptr1++;
		x1 = *ptr1 = x - y;
		if (p < i + 1) { *ptr1 = *(ptr2 = data + p); *ptr2 = x1; }
		ptr1 += 2;
	}

}
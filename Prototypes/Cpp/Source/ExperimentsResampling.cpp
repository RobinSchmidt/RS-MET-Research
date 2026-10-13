// Maybe rename file to ScriptsResampling.cpp

// Contains some scripts with experiments that try to implement the ideas explained in the file:
// RS-MET-Research/Notes/DSP/TransparentOversampling.txt

//=================================================================================================
// Helper functions:

/** Zero stuffs the signal x by the factor M, i.e. produces a new signal y in which between any 
pair of values in x, M-1 zeros will be interspersed. This operation can be viewed as the first step
in upsampling a signal by the factor M. The second step would then be to filter the result of the 
zero-stuffing with a filter (typically a lowpass) that is called the anti-imaging filter in this
context. @see rsUpSample(). */
template<class T>
std::vector<T> rsZeroStuff(const std::vector<T>& x, int M)
{
  size_t Nx = x.size();
  size_t Ny = (M-1)*(Nx-1) + Nx;       // Length of the result
  std::vector<T> y(Ny);                // Will init y with all zeros
  for(size_t n = 0; n < Nx; n++)
    y[n*M] = x[n];
  return y;

  // Notes:
  // 
  // - The length Ny of the result y is (M-1)*(Nx-1) + Nx because between any two elements in x, we
  //   insert M-1 zeros and there are Nx-1 "gaps" between the elements of x where we need to insert
  //   these zeros. The + Nx is for the original elements in x. If we would use just M*Nx instead, 
  //   we would produce a vector y whose last M-1 values would be all zeros, so the somewhat 
  //   complicated formula can be seen as a way to scrap away these extra zeros at the end of the 
  //   resulting signal. Maybe in some contexts it would be more convenient to keep them? I'm not 
  //   sure. Maybe if that turns out to be the case, we could add an optional boolean parameter 
  //   "trimTrailingZeros" or something similar (defaulting to true).
  //
  // - We rely on zero intitialization of std::vector here. When implementing this for some other
  //   array class that perhaps doesn't init the allocated memory, we would have to add some
  //   fillWithZeros() step after creating y.
}

/** Upsamples the signal x by the factor M using zero stuffing and then convolves the result of 
that with kernel h which can be interpreted as anti-imaging filter. This operation can also be 
interpreted as follows: If we assume that x is itself a filter kernel that is used for upsampling
by some factor L (the value of which is irrelevant here) and then the result of that upsampling by
L is again upsampled by a factor of M using the kernel h, we will at the end of the day have 
upsampled our signal by the factor L*M with some resulting kernel. This function computes that 
resulting kernel. */
template<class T>
std::vector<T> rsUpSample(const std::vector<T>& x, int M, const std::vector<T>& h)
{
  std::vector<T> xz = rsZeroStuff(x, M);
  return rsConvolve(xz, h);

  // ToDo:
  //
  // - Maybe rename to rsUpSample1D. Or maybe not. Maybe 1D should be the default assumption.
}

/** Downsamples the signal x by the factor M. We first convolve x with the anti-aliasing filter
kernel h and then decimate the result. You can also set up an offset into filtered signal from 
which we start the resampling. This makes sense because usually, the filter kernel h may need some 
warm-up time and introduce some delay. For a symmetric kernel h with an odd length L, it may make 
sense to shift by (L+1)/2 such that the middle sample of the filter kernel is interpreted as origin
of our (sampled) time axis. This is the typical case - at least for what I currently do. */
template<class T>
std::vector<T> rsDownSample(const std::vector<T>& x, int M, const std::vector<T>& h, 
  size_t shift = 0)
{
  std::vector<T> y = rsConvolve(x, h);
  rsShiftLeft(y, shift);
  return rsDecimate(y, M);
}
// Needs more tests! What about "atypical" cases?

/** Experimental. Needs more tests and may still be buggy.

Returns true, iff the upsampling/downsampling roundtrip by factor "M" for the given signal "x",
using "hu" as the anti-imaging filter kernel for upsampling and "hd" as the anti-aliasing filter 
kernel for downsampling, is an identity operation up to the given numerical tolerance "tol". */
template<class T>
bool testUpDownSampleRoundTrip(const std::vector<T>& x, int M, 
  const std::vector<T>& hu, const std::vector<T>& hd, T tol = T(0))
{
  using Vec = std::vector<T>;

  //size_t shift = (hd.size()+1) / 2;          // Compute warm-up length in downsampling kernel
  size_t shift = hu.size()/2 + hd.size()/2;

  Vec xu = rsUpSample(  x,  M, hu);          // Upsample x by M with hu as anti-imaging filter
  Vec xd = rsDownSample(xu, M, hd, shift);   // Downsample with hd as anti-aliasing filter
  rsRemoveRange(xd, x.size(), xd.size()-1);  // Remove trailing zeros
  //rsPlotVectors(x, xd);                    // Plot original and downsampled - for debugging
  return rsIsCloseTo(xd, x, tol);            // Compare original and downsampled signals


  // Notes:
  //
  // - For our test case with M = 2, it seems to almost work. It's just that the decimated signal 
  //   xd has two additional zero samples at the end compared to the original x.
  //   ...is this still true? I think, I may have fixed that already. -> Verify and maybe delete 
  //   this comment
  //
  //
  // ToDo:
  //
  // - Verify and document formula for shift amount! I sort of guessed it. The rationale is that we
  //   we usually will have a symmetric kernel hd with an odd length and we want to skip forward to
  //   its middle sample. Does that make sense in general? It seems to work in our particular test 
  //   case. What if the length of h is even? Does the formula still make sense?
  //
  // - Maybe we need to shorten xd? It seems like in our test case, the decimated result xd is 2 
  //   samples longer than the original x. It has two additional samples of value zero, so it's the
  //   original with a bit of zero-padding at the end. Maybe in general, the amount of additional 
  //   samples is M? Or maybe the length of hd also matters? ...done
  //
  // - More test cases are needed with different values for M and different lengths for hu and hd.
  //   Maybe try also asymmetric and/or even length kernels.
}
// Needs more tests!



/** UNDER CONSTRUCTION. Does not yet work!

Generates a filter kernel that can be used for sinc interpolation ...TBC...

  L: Length of the kernel
  M: Upsampling factor (determines positions of zero crossings)

*/
template<class T>
std::vector<T> rsSincUpSampleKernel(int L, int M)
{
  rsAssert(rsIsOdd(L), "Only odd lengths are supported in rsSincOverSampleKernel");

  std::vector<T> h(L);    // Filter kernel (aka impulse response)
  int c = L/2;            // Center sample

  h[c] = 1;
  for(int i = 1; i <= c; i++)
  {
    T v = RAPT::rsNormalizedSinc(T(i) / T(M));  // VERIFY!

    //T v = RAPT::rsNormalizedSinc(T(2*i) / T(M));  // VERIFY!

    h[c+i] = v;
    h[c-i] = v;
  }

  /*
  // Normalize to unit DC gain:
  T sum(0);
  for(int i = 1; i <= c; i++)
    sum += 2*h[c+i];
  T scaler = T(M-1) / sum;      // Explain this!
  for(int i = 1; i <= c; i++)
  {
    h[c+i] *= scaler;
    h[c-i] *= scaler;
  }
  */
  // ToDo: Make this optional. Actually, it may make sense to factor it out into a function. Maybe
  // it should be called rsNormalizeUpsampleKernel()
  // The reason is that we may want to do this kind of normalization after applying a window
  // But the plot looks wrong! Maybe simple uniform scaling is not the right approach. Maybe we 
  // need to shift instead - or apply a window, i.e. scale non-uniformly. But how? That seems to be
  // a not so simple task. Maybe make an ansatz of using a linear combination of 1 and 1-x^2 where
  // x = (c+i)/(c) and try figure out the correct mixing coeff. Maybe that leads to an equation 
  // that can be solved? Or maybe use the simpler ansatz of a linear combination of 1 and 1-x.
  // I think, to figure out the right linear combination, we could do the following: let a[n] and 
  // b[n] be two different kernels. They can be anything but here we will take a[n] to be our raw 
  // sinc kernel and b[n] a windowed sinc kernel. Let sa be the sum of all values in a and sb the 
  // sum of all values in b. Assume sa > 1 and sb < 1. I think, what we need to to is to transform
  // the interval sb..sa to the interval 0..1 via an affine transform and we get our linear 
  // interpolation coeff by transforming 1 using the exact same affine trafo. Then just form the 
  // weighted sum of a and b using the found coeff as weight. I think this should work because of
  // linearity of summation. If we don't have sa > 1 and sb < 1, it may still work but the coeff 
  // may not be in 0..1 as it usually is in interpolation settings.

  /*
  // For test/debug:
  sum = 0;
  for(int i = 0; i < L; i++)
    sum += h[i];
  */
  // The sum should now be equal to M up to roundoff error. Maybe wrap this test into a function
  // rsIsInterpolatingUpsampler. This function should check that the middle sample is 1, that 
  // zero-crossings occur at the right places and that the total sum is equal to M

  //rsStemPlot(h);          // For debug

  return h;

  // ToDo:
  //
  // - Maybe optionally normalize the result such that the sum of all values except the middle one
  //   gives M-1. I think, this is the condition to achieve unit DC gain in the oversampled domain.
  //
  // - Try to give it some parameter of type T such that the compile can infer the type from it and
  //   we don't have to call it like rsSincUpSampleKernel<Real>(L, M). But what could that 
  //   parameter be such that it makes sense and is useful? Maybe the number M could be of type T
  //   rather than an integer?
  //
  // - I think, we may need to compress it along the time axis by a factor of 0.5. Currently, when
  //   calling it with rsSincUpSampleKernel<Real>(201, 10); we get a sample of 1 at 100 and zero 
  //   crossings at 90,110, 80,120, 70,130, ... But that does not correspond to an upsampling 
  //   factor of 10 but rather 20, I think. Hmm...using v = RAPT::rsNormalizedSinc(T(2*i) / T(M));
  //   look good - but that formula looks really bad for M=2. We get zeros everywhere except in the
  //   middle - this is clearly wrong! Or is it? Maybe that is what we should expect? It may be 
  //   plausible. Maybe it means that sinc interpolation makes sense only for higher upsampling 
  //   factors? And maybe those factors need to be even?
}

/** A function that can be used to tweak a given kernel h such that its sum is equal to 1. 

...TBC...

*/
template<class T>
std::vector<T> rsCorrectKernelSum(const std::vector<T>& h, T desiredSum)
{
  using Vec = std::vector<T>;

  int L = (int) h.size();
  int C = L/2;

  // Check some assumptions:
  rsAssert(rsIsOdd(L));
  rsAssert(h[C] == T(1));
  if(L == 0)
    return Vec();   // Should not happen. 0 is even, so we'll hit the assert anyway

  // Edge case that would lead to division by zero in the computations below:
  if(L == 1)
    return h;

  // Create a parabolic window:
  Vec w(L);
  for(int k = 0; k <= C; k++)
  {
    T x = T(k) / T(C+1);
    T y = 1 - x*x;
    w[C+k] = y;
    w[C-k] = y;
  }
  // ToDo: Make it more flexible. The caller should be able to pass the desired window.

  // Apply the window to h:
  Vec v = w * h;
  // Maybe re-use w. But maybe such optimizations should be deferred to a production version which
  // should operate on pre-allocated buffers anyway.

  // Compute the weights for h and v:
  T sh = rsSum(h);
  T sv = rsSum(v);
  sh /= desiredSum;
  sv /= desiredSum;
  T k = rsLinToLin(T(1), sv, sh, T(0), T(1));
  rsAssert(rsIsFiniteNumber(k));
  // Maybe get rid of the divisions by desiredSum and use
  // k = rsLinToLin(desiredSum, sv, sh, T(0), desiredSum); But that seems to be wrong.
  // When sh == sv == 1, we get k = inf. That may happen in an edge case when L = 1. Could it 
  // perhaps also happen when h is already perfect? Maybe try it with the L = 3 kernel
  // [1 2 1] / 2 and desiredSum = 2. That would be the linear interpolation kernel for M = 2.
  // Maybe we should check, if sh is close to the desired sum somehwere early in the function, 
  // even before doing all the many computations. But we need to figure out an appropriate 
  // threshold. Maybe some multiple of epsilon. Hmm..OK - I have preliminarily included an early
  // return when L <= 1. I'm not sure, if that's the best way to deal with it, though. Maybe there
  // could be cases where the kernel is longer but still lead to problems. More tests are needed.


  // Form the linear combination u of h and v:
  Vec u = k*h + (1-k)*v;

  // Verify result for debugging:
  //T su = rsSum(u);                 // Should be equal to desiredSum
  //rsPlotVectors(h, w, v, u);

  return u;


  // Notes:
  //
  // - In the edge case where the length L is 1, it would be wrong to produce a kernel whose only
  //   sample value is equal to the desired sum because we are not supposed to never touch the 
  //   "middle" sample because it must be 1 to satisfy the interpolation condition.
  //
  // 
  // ToDo: 
  //
  // - Make it more flexible by factoring out a function that takes an arbitrary window as 2nd
  //   input. Here, we generate a parabolic window ourselves.
}


/** Verifies that the given filter kernel h satisfies the interpolation condition which means that
the upsampled signal should pass through the original sample values. We test this by actually 
upsampling a test signal (consisting of random values) using the given kernel an check that the
upsampled signal at sample indices [M*n + L/2] matches the sample values of the original signal at
indices n. */
template<class T>
bool rsIsKernelInterpolating(const std::vector<T>& h, int M, T tol)
{
  bool ok = true;

  using Vec = std::vector<T>;

  // Create upsampled signal:
  int L = (int) h.size();                   // Length of the kernel
  int N = 4*L;                              // Length of test signal
  Vec x = rsRandomIntVector(N, +3, +9, 0);  // Create random test signal
  Vec y = rsUpSample(x, M, h);              // Upsample it by M using h

  // Verify interpolation condition:
  //T tol = 1.e-12; // Preliminary. Maybe it should be a parameter or maybe it should be based on
  //                // epsilon and L. Maybe L*eps*c for some fixed c that should be found empirically
  int shift = L/2;
  for(int n = 0; n < N; n++)
    ok &= rsIsCloseTo(y[M*n+shift], x[n], tol);

  //rsPlotVectors(x, y); // This is not a good way to plot it!
  //rsPlotVectors(y);

  return ok;

  // ToDo:
  //
  // - Maybe make a plot where the upsampled signal is plotted with lines and the original signal
  //   values are added as markers
}

//=================================================================================================
// Unit test and experimental scripts:

bool testSincUpSampler()
{
  bool ok = true;

  // For convenience:
  using Real = double;
  using Vec  = std::vector<Real>;

  auto doTest = [](int L, int M) 
  {
    Real tol = 1.e-12;                 // Preliminary - maybe use c*L*eps for some constant c

    // Create a sinc kernel corrected to give a sum of M:
    Vec h = rsSincUpSampleKernel<Real>(L, M);
    h = rsCorrectKernelSum(h, Real(M));

    // Do the checks:
    bool ok = true;
    ok &= rsIsKernelInterpolating(h, M, tol);
    if(L > 1)
    {
      Real sum = rsSum(h);
      ok &= rsIsCloseTo(sum, Real(M), tol);
      // If the length L is just 1, we cannot expect to be able to set the sum and center value. We
      // have just 1 degree of freedom - namely the value of the only sample in the kernel - and 
      // that must be 1 to satisfy the interpolation condition. 
    }

    //rsStemPlot(h);                   // Plot the generated kernel for debugging
    return ok;
  };

  // Oversample by M = 2 with different kernel lengths L:
  ok &= doTest(21,  2);
  ok &= doTest( 7,  2);
  ok &= doTest( 5,  2);
  ok &= doTest( 3,  2);
  ok &= doTest( 1,  2);

  // Oversample with kernels of length L = 41 by different factors M:
  ok &= doTest(41,   3);
  ok &= doTest(41,   4);
  ok &= doTest(41,   5);

  // Use some longer kernels and bigger factors:
  ok &= doTest(201, 10);
  ok &= doTest(513, 16);

  return ok;
}

bool testUpSample1D()
{
  // Tests if our function rsUpSample() produces the expected outputs for (scaled) linear 
  // interpolation kernels.

  bool ok = true;

  using Real = double;
  using Vec  = std::vector<Real>;

  // Create the upsampling kernels for linearly upsampling by factor 2, 3 and 6 respectively. 
  // Actually, u2 should be divided by 2 and u3 by 3 and u6 by 6, so we are actually using scaled 
  // versions of the kernels here because then the numbers are nicer (i.e. integers) and so we 
  // don't have to deal with floating point roundoff errors and can use exact equality 
  // comparisions in our test cases:
  Vec u_2 = { 1, 2, 1 };
  Vec u_3 = { 1, 2, 3, 2, 1 };
  Vec u_6 = { 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1 };

  // Verify that (linearly) upsampling by 2-then-3 or by 3-then-2 yields the same result as 
  // upsampling by 6 directly. Thanks to using integers, we can use exact equality comparisions:
  Vec u_2_3 = rsUpSample(u_2, 3, u_3); ok &= u_2_3 == u_6;
  Vec u_3_2 = rsUpSample(u_3, 2, u_2); ok &= u_3_2 == u_6;

  return ok;

  // ToDo:
  //
  // - Maybe make a couple of more tests with kernels like u_4, u_5, u_7 etc. and their products.
  //   Maybe create a helper function testUpsample(int L, int M) and call that in a double loop
  //   over l = 1..L, and m = 1..M. It should internally create the three vectors 
  //   uL = {1, 2, .., L, .., 2, 1}, uM = {1, 2, .., M, .., 2, 1}, 
  //   uLM = {1, 2, .., L*M, .., 2, 1} and then do the rsUpSample with uL,uM and uM,uL internally
  //   and compare the results to uLM.
  //
  // - We actually do not really verify that the upsampling results are the same. We just verify 
  //   that the resulting kernel looks as expected which should imply that the upsampling results 
  //   will be the same. Maybe check that directly by actually upsampling some signal. Maybe use a
  //   sequence of random integers for that. Maybe then test that also with arbitrary kernels - 
  //   maybe just containing random numbers. Maybe try using weird kernel lengths that do not 
  //   necessarily have anything to do with the numbers L and M. Try even lengths, too.
  //
  // - Maybe integrate the functions rsZeroStuff and rsUpSample into RAPT in 
  //   StandardContainerTools.h. Maybe make also lower level versions in rsArrayTools.h that 
  //   operate on raw C arrays.
  //
  // - Maybe test if the roundoff behavior is better or worse with direct upsampling by L*M or 
  //   succesive upsampling by L-then-M and M-then-L. 
  //
  // - Document the general formula for the lengths of the linear interpolation kernels. I think,
  //   it's given by 2*M-1. Also explain that this length may be very different for other types of
  //   interpolation (i.e. anti-imaging) filters such that there is no general rule. This is why 
  //   rsUpSample really needs the parameter M. The desired oversampling factor M cannot in 
  //   general be inferred from the length of h and/or x because the function doesn't assume any 
  //   particular type of anti-imaging filter.
}


bool testUpDownSample1D_1()
{
  // Some experiments with upsampling/downsampling schemes that are supposed to be an identity
  // operation when executed in sequence. A simple scheme for upsampling by 2 would be to use 
  // sample duplication for upsampling and to use the average of two successive samples for 
  // downsampling. The goal is to find schemes that provide a better upsampling quality than
  // using sample duplication. For example, linear interpolation could be used for upsampling. But
  // then we would need to do something else for the downsampling step to achieve a lossless 
  // up/down roundtrip. Of course, we could just naively downsample the linearly upsampled signal 
  // by discarding the samples that were used to fill the gaps because the original samples are 
  // still present unchanged in the upsampled signal. However, it may be better to apply some sort
  // of filtering before downsampling. It is actually possible to achieve exact reconstruction with
  // a variety of 5-point filters parametrized by a single parameter. This is detailed below.
  


  // For convenience:
  using Real = double;
  using Vec  = std::vector<Real>;
  using Mat  = RAPT::rsMatrix<Real>;
  using AT   = RAPT::rsArrayTools;

  bool ok = true;

  // Create test signal
  Vec x({0,-2,1,-6,5,-3,4,-1,0});         // First and last must currently be zero. 
  int Nx = (int) x.size();
  // ToDo: Explain why first and last samples must be zero. I think, it's because we check if the 
  // output after the roundtrip is equal to the input but this is only true when we disregard the
  // boundary values. In a realtime context, this doesn't matter anyway because in this setting,
  // we deal with a continuous stream of data and there really aren't any boundary values to deal 
  // with. We may also imagine the boundaries to be in the infinitely distant past and future.


  // Upsample by a factor of 2:
  int Ny = 2*Nx;
  Vec y  = resampleLinear(x, Ny);

  // Define the filter kernel for downsampling:
  Vec h({-0.25, 0.5, 0.5, 0.5, -0.25});   // maybe rename to d
  // This kernel has been found by considering the situation arising from upsampling an impulse by
  // a factor 2 via linear interpolation. This gives:
  //
  //   in-index:      0         1         2         3         4
  //   input:        0.0       0.0       1.0       0.0       0.0
  //   upsampled:    0.0  0.0  0.0  0.5  1.0  0.5  0.0  0.0  0.0  0.0
  //   out-index:     0    1    2    3    4    5    6    7    8    9
  //
  // Then I observed that the spike of 1 at input sample index 2 gets spread into the 3 output 
  // samples 3,4,5 with weights 0.5,1,0.5. To reconstruct the 1 from these 3 output values with 
  // some averaging, we can just add them all up and divide by 2: (0.5 + 1.0 + 0.5) / 2 = 1. That 
  // suggests to use a 3-point moving average with weights 0.5,0.5,0.5. But now the total sum of 
  // the weights is 1.5 and also with such a 3 point kernel, the reconstructed sample at index 3 
  // would be 0.25 instead of 0. But appending weights of -0.25 to both ends of the kernel fixes
  // both of these problems. What general conditions do we need? I think:
  // -(1) The total sum of weights should be 1
  // -(2) The ends must be -0.5 times the values next to the end. I think, this is because the 
  //      spike spreads with weight 0.5 into the adjacent samples int the oversampled signal.
  // -Let's express the downsampling kernel in general as [d2 d1 d0 d1 d2] where d0 is the center 
  //  coeff. I think, the conditions can now be formulated as:
  //    (1) d0 + 2*(d1+d2) = 1
  //    (2) d2 = -d1/2
  //  This should give us a 1-parametric family of filter kernels. Maybe use d0 as parameter. 
  //  Substitute d2 = -d1/2 into (1): d0 + 2*(d1 - d1/2) = 1 and solve for d1:
  //  d1 = 1 - d0. Let's try it:
  Real d0 = 0.625;                // Interesting values: 1, 0.75, 0.6875, 2/3, 0.625, 0.5
  Real d1 = 1 - d0;
  Real d2 = -d1 / 2;
  h = Vec({d2, d1, d0, d1, d2});  // Rename to d
  // Yes! It works! We now can tweak the kernel by tweaking d0. Setting d0 = 1 just takes the 
  // middle sample and does not do any filtering at all. Using d0 = 0.5 introduces a rather strong
  // filtering. Maybe we should impose the additional condition that d1 = d0/2. That leads to 
  // d0 = 2/3. That's not a very nice number in base 2 though and leads to rounding errors. Using 
  // d0 = 0.75 may be a bit to little filtering. Maybe use 11/16 = 0.6875 or 10/16 = 0.625. Maybe
  // try to figure out the 2D version, then try a couple of values of d0 and select the best 
  // downsampling filter by eye.

  // Apply the filter kernel to y and then decimate yf naively. This should give back x:
  Vec yf = filter(y, h, true);
  Vec xr(Nx);
  AT::decimate(&yf[0], Ny, &xr[0], 2);
  Vec err = x - xr;  // Error introduced in roundtrip
  ok &= rsIsAllZeros(err);
  // OK - it works, but only if the first and last samples of x are zero. Otherwise, the 
  // reconstructed first and last values are wrong. All other reconstructed values will be correct
  // though, regardless of first and last input samples in x. Maybe the kernel h is appropriate 
  // only for inner points? A simple remedy would be to just pad x with zeros before upsampling and
  // cropping after downsampling. But below, we solve it more elegantly anyway...
  
  // Now with the convenience functions:
  d0  = 0.625;
  x   = Vec({7,-2,1,-6,5,-3,4,-1,3}); 
  y   = upsampleBy2_Lin(x);
  xr  = downsampleBy2_Lin(y, d0);
  err = x - xr;
  ok &= rsIsAllZeros(err);

  // Create kernels for various values of d0 and store them in a 2D array. The 1st index is the 
  // kernel index, then 2nd is the sample index:
  Real dLo  = 0.5;        // Lowest value for d0. With 0.5, we get a flat freq response.
  Real dHi  = 1.0;        // Highest value for d0. With 1.0
  int  dNum = 7;
  Mat  kernels(dNum, 5);  // The kernels have length 5
  for(int i = 0; i < dNum; i++)
  {
    d0 = dLo + i*(dHi-dLo)/(dNum-1);
    d1 = 1 - d0;
    d2 = -d1 / 2;
    h  = Vec({d2, d1, d0, d1, d2});
    kernels.setRow(i, h);
  }
  // The response with d = 0.75 has a zero at the oversampled Nyquist limit (good!) but it features
  // a little bit of a bump somewhere below the non-oversampled Nyquist limit (bad!).

  // Create the kernel that represents the upsampler, i.e. the linear interpolator, in the 
  // oversampled domain. The mental image is that upsampling consists of zero-stuffing and then 
  // applying that FIR kernel to the zero-stuffed, upsampled signal to obtain a signal with the
  // anti-imaging filter applied:
  Vec u = { 0.5, 1.0, 0.5 };        // The "u" stands for "upsampling kernel"

  // Verify that convolving each of the rows in "kernels" leads to a transparent up -> down chain.
  // The condition for this is that the convolution of u with each kernel must have a value of 1 at 
  // the center values of zeros at all even indices (I think) when we define the origin to be at 
  // the center. ...TBC...Verify
  for(int i = 0; i < dNum; i++)
  {
    // Convolve i-th of downsampling kernel with upsampling kernel u:
    Vec d  = kernels.getRowAsVector(i);
    Vec du = rsConvolve(d, u); 

    // Verify transparency (i.e. perfect roundtrip reconstruction) conditions:
    int n0 = 3;             // Center index of du is our zero. In general, du.size()/2, I think.
    ok &= du[n0]   == 1.0;
    ok &= du[n0+2] == 0.0;
    ok &= du[n0-2] == 0.0;
    // For longer kernels, we would need to check du at n0 +- 2*j to be zero where j ranges over 
    // the integers up to (n0-1)/2, I think (verify!). For oversampling factors M other than 2, we 
    // would need to check at n0 +- M*j against zero. What the filter does at all the other samples
    // doesn't concern us because these samples will be discarded in the downsampling step. These
    // are our wiggle room that we can use to optimize the frequency response.

    // Plot the convolution result:
    //rsStemPlot(du);
    // ToDo: Maybe plot the frequency response of du. I think, it is desirable to have it as flat 
    // as possible up to fs/4 and then go down as steeply as possible from fs/4 to fs/2. And these
    // two requirements are probably conflicting such that we need to strike a trade off. But maybe
    // what we actually should try to optimize is not the frequency response of du but rather those
    // of d _and_ u. Although, here we treat u as given anyway. But in a more general setting, both
    // of them matter and both should be reasonable in their own right.

    // Note:
    //
    // - Conceptually, it would probably make more sense to call it ud rather than du and compute 
    //   it as  ud = rsConvolve(u, d)  because in practice, we apply upsampling first and then 
    //   downsampling. But it doesn't matter because convolution is commutative anyway. Maybe it 
    //   even makes more sense to write it this way because in math, function compositions also 
    //   reads from right to left such that with this mathy function composition convention, du 
    //   would actually indeed mean: first apply u, then apply d if u and d are see as functions to
    //   be applied to a signal: du(x) would mean d(u(x)).
  }


  // Plot frequency responses for the kernels:
  using Plt = SpectrumPlotter<Real>;
  Plt plt;
  plt.setFloorLevel(-100);
  plt.setFreqAxisUnit(Plt::FreqAxisUnits::normalized);
  plt.plotSpectraOfRows(kernels);
  // The plots are not normalized to 0 dB. Figure out why and fix it!


  rsAssert(ok);
  return ok;

  // Observations:
  // -All freq responses meet in the point at 0.25*fs, 0dB
  // -They seem to all have a bump at around 0.165*fs
  // -For d0 = 0.75, there seems to be a notch at the Nyquist freq 0.5*fs. That seems to be the 
  //  most desirable response.
  // -The desired freq-response is that of a halfband filter because we use it to downsample by 2.
  // -Conclusion: d0 = 0.75 seems a good overall choice for the downsampling. 
  // -The convolved du kernels become more and more triangular shaped as the center weight a0 is 
  //  increased from 0.5 to 1.0. The final one is [0 0 1/2 1 1/2 0 0]. The first one looks more 
  //  like a bump with flaps that go into the negative range at the outsides. The reason that we
  //  have some freedom of choice for the downsampling filters is that we do not care what the 
  //  filter produces at the sample indices that we intend to throw away in the subsequent 
  //  decimation step.

  // ToDo:
  // -Try to convolve the linear interpolation kernel with the h-kernel. I think, the result should
  //  be a unit impulse for the identity operation to work. Or maybe it's enough if it has a value 
  //  of 1 at the center n=0 and values of 0 at n=k*M where M is the oversampling factor (here M=2)
  //  and k is any integer. The linear interpolation kernel for M=2 is [0.5 1.0 0.5]. I can be 
  //  decomposed intto two phases (in the polyphase filtering sense) of [0.0 1.0 0.0] and 
  //  [0.5 0.0 0.5] in the oversampled time domain. Let's use index n for the original time domain
  //  and m = n*M for the oversampled domain with oversampling factor M. Then, in the n-domain, the 
  //  two phases would look like [0 1] and [1/2 1/2]. Whenever m%M == 0, we would use 
  //  y[m] = 1*x[n] + 0*x[n-1] and whenever m%M == 1, we would use y[m] = (x[n] + x[n-1])/2
  // -Try to find a pair of interpolator/decimator kernels for upsampling by a factor of M=3. That 
  //  may be easier because the interpolator kernel length is odd so it can be symmetric around its
  //  center sample. An even kernel is symmetric around the midpoint between the two center 
  //  samples which may be inconvenient.
  // -Find pairs of filters for M=4 by convolving the ones for M=2 with themselves.
  // -Find pairs of filter kernels with longer length for better quality.
  // -Figure out the z-domain transfer function of downsampling filter a. I think, it's
  //    H(z) = d0 + d1*(z^-1 + z^1) + d2*(z^-2 + z^2)
  //  Setting it to zero gives:
  //    0 = d0     + d1*(z^-1 + z^1) + d2*(z^-2 + z^2)
  //      = d0*z^2 + d1*(z+z^3)      + d2*(1 + z^4)
  //  Try to solve it explicitly as function of d0. It's a quartic, so it should be possible.
  // -Write function upsampleViaDuplication and use it together with AT::decimateViaMean. This 
  //  should also give a lossless roundtrip. However, the quality of the upsampled data will be
  //  suboptimal. Is it possible to use something other than pixel duplication for the upsampling 
  //  when we assume to decimate via mean?
  // -See also: AT::decimate, AT::decimateViaMean, AT::movingAverage3pt
  // -Maybe create some brown noise, downsample it, then upsample it and see what *looks* best
  //  visually in terms of being close to the original (down-then-up is, of course, no identity 
  //  operation due to information loss in the downsampling). 
  // -Try to create a scheme using cubic interpolation in the upsampling step. 
  // -Maybe we can somehow generalize this: given an upsampling kernel u, find a (set of) 
  //  downsampling kernels that yield a lossless roundtrip. The upsampling-by-2 kernel for linear
  //  interpolation in 1D can be written as [0.5, 1, 0.5] = [u1 u0 u1]. I think, it is the 0.5 that
  //  appears in the d2 = -0.5*d1 condition. I think, we must have u0*d2 + u1*d1 = 0? ...and in 
  //  general sum_{i,j} ui*dj = 0  where i = 2-j and the 2 is the length of the "forward wing" of 
  //  the kernel i.e. the maximum index when we assume index 0 to be at the center and let the 
  //  leftward indices be negative. We don't write d_{-1}, d_{-2} though, because they are equal
  //  to d_1, d_2 due to symmetry (using LaTeX subscript notation here for the index). 
  // -OK - let's try it with d0 = 0.75, d1 = 0.25, d2 = -0.125
  //    u = [u2 u1 u0 u1 u2] = [ 0      0.5   1     0.5    0    ]
  //    d = [d2 d1 d0 d1 d2] = [-0.125  0.25  0.75  0.25  -0.125]
  //  so we get:
  //    u0*d2 + u1*d1 + u2*d0 = 1*(-0.125) + 0.5*0.25 + 0*0.75 = -0.125 + 0.125 = 0
  //  so the formula works in this case. I'm not sure, if it's generally the right formula, though.
  //  let's try d0 = 0.8, d1 = 0.2, d2 = -0.1 and u0 = 1, u1 = 0.5, u2 = 0 as before:
  //    1*(-0.1) + 0.5*0.2 + 0*0.8 = -0.1 + 0.1 = 0
  //  OK, works in this case, too. But whether the formula with the sum holds in general for longer
  //  kernels needs to be figured out. If it does work, we have a way to produce a downsampling 
  //  kernel d when the upsampling kernel u is given (or the other way around)
  // -It would be interesting, if this could also work for IIR filters. When we have an analytic 
  //  expression for d_n, we may be able to derive an expression of u_n and then design a filter
  //  that has that u_n sequence as impulse response.
  // -Maybe approach it the other way around: assume the downsampling kernel d as given and try to 
  //  find an appropriate upsampling kernel u such that up -> down gives the identity. Maybe 
  //  assume averaging as downsampling method
  // -The eventual goal is to later make a 2D version of the found schemes to use them for image
  //  processing in upsampled images. But first things first and the first thing is the 1D version.
  // -Try also the "Magic Kernel"
  //    https://johncostella.com/magic/
  //    https://johncostella.com/magic/mks.pdf
  //  Try to use the upsampling variant and check, if upsample -> downsample is lossless. If not,
  //  take the upsampling as is and try to derive a suitable downsampling that makes the roundtrip 
  //  lossless. Some important passages from the website text:
  //  "I analytically derived the Fourier transform of the Magic Kernel in closed form, and found, 
  //  incredulously, that it is simply the cube of the sinc function.[...] An alternative way of 
  //  looking at it is that it is a “kernel” with values {1/4, 3/4, 3/4, 1/4} in each direction 
  //  (where, in this notation, the standard “replication” or “nearest neighbor” kernel is just 
  //  {0, 1, 1, 0}. [...]  the “magic” kernel is effectively a very simple approximation to the 
  //  sinc function, albeit with the twist that the new sampling positions are offset by half a 
  //  pixel from the original positions [...] the “magic” kernel can be factorized into a 
  //  nearest-neighbor upsampling to the “doubled” (or “tiled”) grid, i.e. {1, 1}, followed by 
  //  convolution with a regular {1/4, 1/2, 1/4} kernel  "
  // 
  // -Write a function that takes as input an arbitrary filter kernel h[n] and an oversampling
  //  factor M and that produces a suitable complementary kernel g[n] such that in the convolution
  //  product of h and g, we satisfy the "perfect roundtrip" condition. This condition say that:
  //  at all indices n = k*M where k is any integer, the convolution result must be 0 except for 
  //  k = 0 where it must be 1. The problem may not have unique solution so the function may need
  //  to impose some additional requirements. Maybe they can be given in terms of additional 
  //  parameters. Or maybe there is some natural way to choose a unique solution based on some 
  //  sort of optimality criterion. Maybe we could also minimize the length of the result by 
  //  imposing the constraint that soem outer coeffs need to be zero. I think, this function can 
  //  then be used to produce a downsampling kernel d[n] from a given upsampling kernel u[n] or 
  //  vice versa. Maybe the problem can be simplified by assuming symmetric kernels.
  // 
  // -Try different upsampling filter kernels derived from: Lagrange interpolation, Hermite 
  //  interpolation, B-Spline smoothing (may not interpolate, i.e. pass through orignal data 
  //  points), binomial coeffs e.g. [1 1]/2, [1 2 1]/4, [1 3 3 1]/8, [1 4 6 4 1]/16, ...,
  //  windowed sinc interpolation, etc.
  // 
  // -Maybe try to express the  upsample -> downsample  roundtrip in terms of the upsampling and 
  //  downsampling coeffs and from the roundtrip equations, derive condtions for the coeffs. Let
  //  x = [x0,x1,x2,...,xm,..., x_{M-1}], y = [y0,y1,y2,...,yn,..., y_{N-1}] where x is the 
  //  original and y the upsampled signal such that N=2*M, n=2*m. Assume to use 4 coeffs for 
  //  upsampling and that they are symmetrical, so actually only need two, and call them a1,a2.
  //  We have y[n] = a1*(x[...])...TBC...Oh! That notation is inconsistent with what was 
  //  introduced earlier (where M was used as oversampling factor, the original sample index was n
  //  and m=n*M was the oversampled index). Fix that!
  //  Maybe also have the condition that the upsampled signal must interpolate the original such 
  //  that y[2*m] = x[m]
  // 
  // - Try to design pairs of IIR upsampling/downsampling filters. The conditions for perfect 
  //   reconstruction are that the impulse response must be zero at multiples of M and 1 at 0. We 
  //   could force an IIR filter to have zero crossings at multiples of M by designing it in terms
  //   of modal filters that produce sine waves (with phase zero) at (normalized) frequencies that 
  //   happen to be multiples of 1/M (I think). GitHub Copilot also suggested to look at comb 
  //   filters. Or maybe start with a given halfband IIR filter and put it in series with an FIR 
  //   filter that produces the right zero crossings.Maybe a comb filter could be used for that?
  //   
  // 
  // -[DONE] Rename a0,a1,a2,... to d0,d1,d2, etc. and h to d. The d stands for downsampling kernel
  // -[DONE] Rename b0,b1,..  to u0,u1,... The u stands for upsampling kernel.
  // -Use consitently n as index for the original, non-oversampled signal, m as index for the 
  //  oversampled signal, M for the oversampling factor such that m = n*M. The length of the input
  //  is N and the length of the oversampled signal is N*M.
  // -Use fraction notation like 1/4 rather that decimal notation like 0.25
  // -Do all of this also in the functions below.

  // -See also:
  //  https://en.wikipedia.org/wiki/Upsampling
  //  https://en.wikipedia.org/wiki/Downsampling_(signal_processing)
}

bool testUpDownSample1D_2()
{
  bool ok = true;

  // Let's assume that we are given the downsampling kernel as [d1 d0 d1] = [0.25 0.5 0.25]. Now we 
  // want to find a corresponding upsampling kernel [u1 u0 u1] or [u2 u1 u0 u1 u2]. Consider the 
  // situation:
  //
  //   in-index:   0   1   2   3   4   5    6    7    8
  //   in-value:   2   4   8   6   4   2   -2    4    2
  //   out-value:  2      6.5      4       0.5        2
  //   out-index:  0       1       2        3         4
  // 
  // The 6.5 at out-index 1 is computed as 0.25*4 + 0.5*8 + 0.5*6, i.e. we use a 3-point averaging
  // kernel of the form [0.25 0.5 0.25]

  // Let's assume, our hypothesized condition  
  //   u0*d2 + u1*d1 + u2*d0 = 0
  // is the correct equation. Then with our given d0 = 0.5, d1 = 0.25, d2 = 0, this becomes:
  //   u0*0 + u1*0.25 + u2*0.5 = 0

  // Maybe impose additionally: u0 + 2*(u1 + u2) = 2 because this is the sum which the linear 
  // interpolation upsampling kernel gives. Maybe also require u0 = 1, also like in linear 
  // interpolation. Or maybe leave u0 as free parameter. OK, so we have 2 equations:
  //   (1)  0 = u1/4 + u2/2 = u0*d2 + u1*d1 + u2*d0
  //   (2)  2 = u0 + 2*(u1 + u2)
  // although the second is questionable. I think, the number 2 can be explained by the fact that 
  // we upsample by a factor of 2. If we would upsample by 3, then the coeffs should sum to 3, I 
  // guess. We'll see. Let's solve the 1st equation for u2:
  //   u2 = -u1/2
  // This is very reminscent of the d2 = -d1/2 that we had above. Interesting! Substituting u2 into 
  // (2) and solving for u1 gives:
  //   2 = u0 + 2*(u1 - u1/2)  
  //   2-u0 = 2*(u1 - u1/2) = 2*u1 - u1 = u1   ->   u1 = 2 - u0
  //
  // Let's solve the 1st equation for u2 assuming general d-coeffs:
  //   u2 = -(u0*d2 + u1*d1) / d0      (= -(u0*0 + u1*0.25) / 0.5 = -u1/2 for our choice of d)
  // Substitute u2 into (2):
  //   2 = u0 + 2*(u1 - (u0*d2 + u1*d1) / d0)
  // and solve for u1 using wolfram alpha with "solve 2 = u0 + 2 (u1 - (u0 d2 + u1 d1)/d0) for u1"
  // gives:
  //   u1 = (2*d0 - d0*u0 + 2*d2*u0) / (2*(d0-d1))
  // and the equation for u2 is already given above. Here it is again:
  //   u2 = -(u0*d2 + u1*d1) / d0 

  // ...OK - let's try it:

  // For convenience:
  using Real = double;
  using Vec  = std::vector<Real>;
  //using Mat  = RAPT::rsMatrix<Real>;
  //using AT   = RAPT::rsArrayTools;


  // Define coeffs of the downsampling (averaging) kernel a:
  Real d0 = 0.5;
  Real d1 = 0.25;
  Real d2 = 0.0;

  // Try some other kernels (The kernel should satisfy d0 + 2*(d1 + d2) = 1, I think):
  //d0 = 0.6; d1 = 0.2; d2 = 0.0;  // Yes. Works also.
  //d0 = 0.8; d1 = 0.1; d2 = 0.0;  // Also OK.
  //d0 = 0.4; d1 = 0.2; d2 = 0.1;  // Nope! Maybe d2 != 0 is the culprit? But why?

  // Define coeffs of the upsampling (interpolation) kernel b:
  Real u0 = 1.25;       // Try some other values
  //u0 = 1.5;
  //u0 = 1.0;
  Real u1 = (2*d0 - d0*u0 + 2*d2*u0) / (2*(d0-d1));  // Verify!
  Real u2 = -(u0*d2 + u1*d1) / d0;                   // Verify!
  // Some sets of coeffs are:
  //   u0 = 1.0,   u1 = 1.0,  u2 = -0.5;
  //   u0 = 1.5,   u1 = 0.5,  u2 = -0.25
  //   u0 = 1.25,  u1 = 0.75, u2 = -0.375
  // where we only select u0 and u1,u2 follow via the formulas. These u-coeffs resulted from 
  // choosing d0 = 0.5, d1 = 0.25, d2 = 0. Maybe plot the frequency responses for various choices
  // of u0. But the kernel for a given d.(?) Maybe first optimize d, then select b.

  // Create test signal
  //Vec x({7,-2,1,-6,5,-3,4,-1,3});
  Vec x({0,0,-2,1,5,0,0});       // preliminary
  //Vec x({0,0,0,1,0,0,0});       // preliminary
  int Nx = (int) x.size();

  // Upsample by a factor of 2 using zero stuffing:
  int Ny = 2*Nx;  // verify!
  Vec y(Ny);
  for(int i = 0; i < Nx; i++)
  {
    int  j = 2*i;
    y[j]   = x[i];
    y[j+1] = 0;
  }

  // Apply the upsampling filter:
  Vec u({u2, u1, u0, u1, u2});
  Vec yf = filter(y, u, true);
  // With this upsampling scheme, none of the original sample values from x survive in yf. The 
  // values are now all intermingled with some neighbor values. That's a bit strange because it
  // violates our intuition that the upsampled data should interpolate the original data, i.e. 
  // takes on the values of the original data at sample indices in y that correspond exactly to
  // sample indices in x. We'd expect y[6] to be equal to x[3] for example - but it isn't. But 
  // that's OK - exact interpolation might not be the most important property of an up/downsampling
  // scheme. All we care about is the lossless roundtrip and that might be possible even without
  // the interpolation property. In an interpolating upsampling scheme, we want to take over
  // the datapoints x[i] as is where the upsampled index j is given by 2*i, i.e. we want
  // y[2*i] = x[i]. It's just the y[2*i+1] that we need to fill in. With linear interpolation, 
  // the kernel in the original domain would be [0.5 0.5] such that 
  // y[2*i+1] = 0.5 * (x[i] + x[i+1]). After upsampling with zero stuffing, i.e. just 
  // preliminarily setting y[2*i+1] = 0, one could apply the kernel [0.5 1.0 0.5] = [1 2 1]/2 as 
  // post processing filter. This should give the same oversampled end result. Maybe try a 4-point
  // kernel in the original domain [1 3 3 1]/8. Linear would be [1 1]/2. Somehow these look like
  // lines of Pascal's triangle? Is that a coincidence? Maybe try also [1 4 6 4 1]/16 and
  // [1 5 10 10 5 1]/32. I think, these lines may approach a Gaussian bell curve? Try to plot it! 
  // Maybe add code for that to testGaussBlurFIR. 

  // Now do the downsampling:

  Vec xr(Nx);
  for(int i = 2; i < Nx-2; i++) {  // i is index into x
    int j = 2*i;                   // j is center index into y
    xr[i] = d0*y[j] + d1*(y[j-1] + y[j+1]) + d2*(y[j-2] + y[j+2]);
    xr[i] /= d0;   // why is this needed?
  }

  // ToDo: handle edges. Maybe the loop can go from i=1 to i < Nx-1 like in
  // downsampleBy2_Lin?

  Vec err = x - xr; 
  ok &= rsIsAllZeros(err);

  rsAssert(ok);
  return ok;

  // ToDo:
  // -Figure out why it doesn't work when d2 != 0. Maybe we still have mistakes in some of the 
  //  formulas. The xr[i] /= d0 is pretty strange anyway. I don't really understand it and have 
  //  found it by trial and error.

  
  // But how could we generalize this to a 2D kernel? ...but actually a Gaussian
  // kernel is separable, so it may work out nicely.



  // ToDo:
  // -Let the user prescribe the downsampling kernel d0,d1 or maybe even d0,d1,d2. Then compute
  //  the upsampling kernel u0,u1,u2 from:
  //    (1)  0 = u0*d2 + u1*d1 + d0*u2         (1st term is 0 bcs d2 = 0 here)
  //    (2)  2 = u0 + 2*(u1 + u2)
  // -Do also the converse: Let the user specify the upsampling kernel as u0,u1,u2 and compute from
  //  that the downsampling kernel d0,d1,d2 from:
  //    (1)  0 = d0*u2 + d1*u1 + d2*u0
  //    (2)  1 = d0 + 2*(d1 + d2)
  // -These sets of equations look nicely symmetric. The 1st is always the same, the 2nd has equal
  //  right hand sides (just with roles of a and b swapped)

  // -Generalize to arbitrary resizing rates. Here we just up/downsample by 2. The only constarint 
  //  should be that the upsampling factor is >= 1. There are some ideas for that in 
  //  MiscMathNotes.txt in the private repo.
}

bool testOverSampleAsym_M2_L3()
{
  // Under construction. 

  // The "Asym" stands for: the upsampling kernel could possibly by asymmetric (although it 
  // currently actually is symmetric - but the equations we use here would allow it to be 
  // asymmetric as well, I think). The 2x stands for: we consider the case of 2x oversampling. 
  // Maybe we should also include a "Lin" standing for: The upsampler is linear interpolation. We
  // may later want to generalize this to other upsampling filters like cubic (Langrange, Hermite, 
  // B-Spline), quintic, etc. windowed sinc, binomial, ...

  // We try to generate the downsampling filter kernel d = d[n] from a given upsampling kernel 
  // u = u[n] systematically by posing the problem as linear system of equations and solving it.
  // For some more details about what is going on here, look up the file:
  // 
  //   RS-MET-Research/Notes/DSP/TransparentOversampling.txt
  // 
  // ...TBC...

  bool ok = true;

  // For convenience:
  using Real = double;
  using Vec  = std::vector<Real>;
  using Mat  = RAPT::rsMatrix<Real>;
  using LA   = RAPT::rsLinearAlgebraNew;

  // Define oversampling factor and upsampling kernel:
  int M = 2;                           // Oversampling factor
  Vec u = { 1, 2, 1 }; u = 0.5*u;      // Upsampling kernel (linear interpolation)

  // Test: Try to avoid running into a singular matrix A by using some "random" upsampling 
  // kernel:
  //u = Vec({ 1, -1, 3 });

  int L = 5;                           // Length of downsampling kernel
  Real tol = 1.e-13;                   // Tolerance for numerical comparisons
  // ToDo: rename L to Ld (for downsample) and use the convention that L is used for the length of
  // the upsampling kernel u. Or maybe just get rid of it entirely and write 5s directly into the
  // matrix constructors

  // Below are couple of different ideas for what additional constraints we could use. From the
  // "lossless roundtrip" requirement, we get only 3 equations but for a (potentially asymmetric)
  // downsampling kernel, we have 5 coeffs so we need 2 equations more. I have some different ideas
  // for what we could use for the additional equations/constraints. The different ideas are 
  // embodied by using a bunch of different matrices A and right hand side vectors b indexed as 
  // A1,A2,... b1,b2,... and somewhere below, we then choose one of the ideas by letting, for 
  // example A = A4; b = b4; It's worth to note that not all of the different matrices work. Some 
  // turned out to be singular. The first that actually works is indeed the pair A4,b4.

  // Create coefficient matrix and right hand side of the following system of equations:
  // 
  // u1*d0 + u0*d1                         = 0    Eq. 1:  r1 = 0
  //         u2*d1 + u1*d2 + u0*d3         = 1    Eq. 2:  r3 = 1
  //                         u2*d3 + u1*d4 = 0    Eq. 3:  r5 = 0
  //    d0 +    d1 +    d2 +    d3 +    d4 = 1    Eq. 4:  Unit gain at DC
  //    d0 -    d1 +    d2 -    d3 +    d4 = 0    Eq. 5:  Zero gain at Nyquist (verify!)
  Mat A1(L, L, { u[1], u[0],  0  ,  0  ,  0  ,       // Eq. 1
                  0  , u[2], u[1], u[0],  0  ,       // Eq. 2
                  0  ,  0  ,  0  , u[2], u[1],       // Eq. 3
                  1  ,  1  ,  1  ,  1  ,  1  ,       // Eq. 4
                  1  , -1  ,  1  , -1  ,  1    });   // Eq. 5
  Vec b1({0, 1, 0, 1, 0});
  // Oh! I think, the 2nd to last row is linearly dependent of the first 3. If we just sum the 
  // first 3 rows, we get a constant row! I guess, that means the unit gain at DC condition of 
  // Eq. 4 is automatically satisfied when Eq. 1,2,3 are satisfied?
  // 
  // Maybe rename r to b because in the textfile, we use r for something else, namely for the 
  // result of the convolution of u and d. The vector b is the right hand side of our linear 
  // system. Some of its components are indeed also components of the desired RHS but calling the
  // whole vector r would nevertheless be inconsistent. Calling it b would be consistent with the
  // usual notation A x = b for linear systems. ...done!
 
  // This set of requirements (unit gain at DC, center coeff equal to 1) also leads to a singular 
  // matrix A:
  Mat A2(L, L, { u[1], u[0],  0  ,  0  ,  0  ,
                  0  , u[2], u[1], u[0],  0  ,
                  0  ,  0  ,  0  , u[2], u[1],
                  1  ,  1  ,  1  ,  1  ,  1  ,     
                  0  ,  0  ,  1  ,  0  ,  0    });  // Normalize middle sample d[2] to 1.
  Vec b2({0, 1, 0, 1, 1});
  // I think, the culprit is already line 4 (starting counting at 1 as we do in math). It seems to
  // be proportional to the sum of lines 1..3.

  // Try using the symmetry requirements d1 == d3, d0 == d4 - this also gives a singular matrix:
  Mat A3(L, L, { u[1], u[0],  0  ,  0  ,  0  ,       // Eq. 1
                  0  , u[2], u[1], u[0],  0  ,       // Eq. 2
                  0  ,  0  ,  0  , u[2], u[1],       // Eq. 3
                  0  ,  1  ,  0  , -1  ,  0  ,       // d1 - d3 = 0
                  1  ,  0  ,  0  ,  0  , -1    });   // d0 - d4 = 0
  Vec b3({0, 1, 0, 0, 0});
  // What the hell is going on? are the first 3 lines already linearly dependent? But no - that 
  // can't be the case because among them, line 1 is the only one with a nonzero 1st entry, line 2
  // the only one with a nonzero middle entry and line 3 the only one with a nonzero last entry so
  // there is no way to obtain one of the three as a linear combination of the two others.

  // Let's try to directly prescribe values for d[1] and d[3]. This should parallel the experiment
  // testUpDownSample1D_1() above where we prescribed the middle sample. But there, the notation 
  // was different - the middle sample was called d0 there, here we call it d2 (ToDo: make that 
  // consistent). When we assign d1 = d3 = 0.25, for example, we get a middle sample of 0.75 here.
  // This corresponds to the case where we assigned d0 = 0.75 in the experiment above.
  Real d1 = 0.25;
  Real d3 = 0.25;
  Mat A4(L, L, { u[1], u[0],  0  ,  0  ,  0  ,       // Eq. 1
                  0  , u[2], u[1], u[0],  0  ,       // Eq. 2
                  0  ,  0  ,  0  , u[2], u[1],       // Eq. 3
                  0  ,  1  ,  0  ,  0  ,  0  ,       // d[1] = d1
                  0  ,  0  ,  0  ,  1  ,  0    });   // d[3] = d3
  Vec b4({0, 1, 0, d1, d3});

  // Assign the middle sample d2 directly (to 0.75, say) and impose symmetry d[1] = d[3]:
  Real d2 = 0.75;
  Mat A5(L, L, { u[1], u[0],  0  ,  0  ,  0  ,       // Eq. 1
                  0  , u[2], u[1], u[0],  0  ,       // Eq. 2
                  0  ,  0  ,  0  , u[2], u[1],       // Eq. 3
                  0  ,  0  ,  1  ,  0  ,  0  ,       // d[2] = d2
                  0  ,  1  ,  0  , -1  ,  0    });   // d[3] = -d[1] -> d[1] - d[3] = 0
  Vec b5({0, 1, 0, d2, 0});

  // Like A5,b5 but imposes the symmetry using d[4] = -d[0] -> d[0] - d[4] = 0 instead of 
  // d[3] = -d[1] -> d[1] - d[3] = 0. I think that this should give the same result as A5,b5 
  // (verify that!):
  Mat A6(L, L, { u[1], u[0],  0  ,  0  ,  0  ,       // Eq. 1
                  0  , u[2], u[1], u[0],  0  ,       // Eq. 2
                  0  ,  0  ,  0  , u[2], u[1],       // Eq. 3
                  0  ,  0  ,  1  ,  0  ,  0  ,       // d[2] = d2
                  1  ,  0  ,  0  ,  0  , -1    });   // d[4] = -d[0] -> d[0] - d[4] = 0
  Vec b6({0, 1, 0, d2, 0});

  // Use a condition that requires a straight triangular shape. This can be done by requiring the
  // numeric derivatives d[1]-d[0] and d[2]-d[1] to match, so we would have:
  // d[1] - d[0] = d[2] - d[1]    ->    d[0] - 2*d[1] + d[2] = 0
  Mat A7(L, L, { u[1], u[0],  0  ,  0  ,  0  ,       // Eq. 1
                  0  , u[2], u[1], u[0],  0  ,       // Eq. 2
                  0  ,  0  ,  0  , u[2], u[1],       // Eq. 3
                  1  , -2  ,  1  ,  0  ,  0  ,       // d[0] - 2*d[1] + d[2] = 0
                  0  ,  1  ,  0  , -1  ,  0    });   // d[3] = -d[1] -> d[1] - d[3] = 0
  Vec b7({0, 1, 0, 0, 0});
  // Results in d2 = 0.71428571428571430. The exact value might be 5/7. See:
  // https://www.wolframalpha.com/input?i=0.71428571428571... Maybe compute the value symbolically
  // with SageMath. I think, the resulting downsampling kernel is:
  // d = [-1 2 5 2 -1] / 7

  // Select one combination of matrix and right hand side to use:
  Mat A = A7; Vec b = b7;
  // 1...3 do not work because their matrices A1...A3 are singular. 4...7 do work.

  // Compute downsampling kernel by solving the linear system:
  Vec d = LA::solve(A, b);

  // Verify solution:
  Vec br = A*d;
  ok &= rsIsCloseTo(br, b, tol);

  // Verify that upsampling with kernel u then downsampling with kernel d is an identity operation
  // up to roundoff error:
  Vec x = rsRandomIntVector(20, +1, +9, 0);
  ok &= testUpDownSampleRoundTrip(x, M, u, d, tol);

  // Create the combined up/down kernel:
  Vec ud = rsConvolve(u, d);

  // Plot all 3 kernels:
  rsPlotVectors(u, d, ud);

  // ToDo: Plot magnitude responses of all 3 kernels in one plot. For this, we need to zero-pad u
  // and d to the length of ud, then put all 3 into a matrix and then use code similar to in the
  // experiment testUpDownSample1D_1(). Maybe make a helper function rsToMatrix(v1, v2, v3, ...)
  // where v1,v2,v3,... etc. are vectors of possibly different length.

  return ok;


  // Notes:
  //
  // - I think, the length L of the downsampling kernel d is generally L = N + 2*(N-1) when N is 
  //   the length of the upsampling kernel u. But I'm not sure about that. -> Verify!
  // 
  // - It seems like when using d2 = 5/7, the impulse response of d looks exactly like a triangle. 
  //   But when using that value, we need a nonzero tolerance for the float comparison in 
  //   testUpDownSampleRoundTrip() and in the comparison between b and br. When using 
  //   d2 = 0.75 = 3/4, we can use exact comparisons. Maybe try approximating 5/7 with a fraction
  //   that has a denominator that is a power of two.
  //
  //
  // ToDo:
  //
  // - At the moment, we run into singular matrices for the linear system that we try to solve. I 
  //   think, we need to rethink the conditions (i.e. the equations) that we require from the 
  //   downsampling kernel d. 
  //
  // - If the symmetry conditions turn out to work, write a function that produces a symmetric 
  //   kernel d by construction, i.e. doesn't impose symmetry by two equations. I think, that 
  //   reduces the matrix of the linear system from 5x5 to 3x3 in the example above. ...well - at
  //   least when we formulate it in a way that also assumes the upsampling kernel to be symmetric.
  //   In this case, u = [u0 u1 u0], d = [d0 d1 d2 d1 d0]. 
  //
  // - Try investigating the singular matrices with a matrix oriented math software like 
  //   Matlab/Octave and or a computer algebra system like SageMath. Maybe that will give us a clue
  //   why we always get singular matrices.
  //
  // - When we use something like u = { 1, -1, 3 }; instead of u = {1, 2, 1} * 0.5, we get a 
  //   regular matrix A with the symmetry constraints. OK - good to know - but that u is, of 
  //   course, totally wrong as upsampling kernel.
  //
  // - Instead of commenting out the various attempts to set up the linear system, do things like:
  //   Mat A1 = ...; Vec b1 = ...;    Mat A2 = ...; Vec b2 = ...; and then at the bottom, select
  //   one of the systems by setting e.g. A = A2, b = b2; or something like that. This lets us 
  //   easily switch between the various attempts without having to deal with large commented out
  //   code sections.
  //
  // - Maybe instead of directly assigning a matrix A and right hand side vector b, write a class
  //   like rsLinearEquationSystem with an API that provides functions like: addEquation(), 
  //   solveDirectly(), solveIteratively(), solveLeastSquares(), solveMinNorm(), etc. The 
  //   addEquation() function should probably take a number for the right hand side and a vector
  //   of pairs of coefficients and variable indices. The class could have functions like 
  //   getNumEquations(), getNumUnknowns() or getNumVariables(). The latter could scan through the
  //   equations to figure out the highest variable index that occurs in any of the equations. The
  //   solve functions would then assemble the matrix and rhs vector and invoke the actual solvers.
  //
  // - Maybe we could then also write a similar class rsQuadraticEquationSystem in whose 
  //   addEquation() method, we would add vectors of triples of: coefficient c, var index 1 i, 
  //   var index 2 j. Each such triple would represent a term of the from c * x_i * x_j. Maybe
  //   we should assume the right hand sides to be zero and absorb them into the left hand side by
  //   adopting the convention that x0 = 1. Not sure, though. This should be consisten with the way 
  //   it is handled in rsLinearEquationSystem. This class could then perhaps be used to jointly
  //   design the two filter kernels u and d.
}

bool testOverSample_M2_L3()
{
  // In this experiment, we consider the case where the downsampling kernel d that we want to 
  // derive from the upsamling kernel u is symmetric by construction. In the experiment above, we
  // had to impose the symmetry by means of constraint equations. We had 5 unknowns and had to
  // produce 5 equations of which 3 were already fixed by the perfect roundtrip requirement. Here,
  // we only have 3 unknowns in the first place. It turns out the the perfect roundtrip conditions
  // fix only 2 degrees of freedom if we assume the upsampling kernel u to be symmetric. That means
  // we have the freedom to choose one degree of freedom just like in the symmetric case. This is
  // as it should be because the situation is fundamentally the same. It's just that here, we 
  // impose symmetry by construction whereas in the other case, we imposed it by a constraint.

  bool ok = true;

  // For convenience:
  using Real = double;
  using Vec  = std::vector<Real>;
  using Mat  = RAPT::rsMatrix<Real>;
  using LA   = RAPT::rsLinearAlgebraNew;

  // Define oversampling factor and upsampling kernel:
  int  M   = 2;                          // Oversampling factor
  Vec  u   = { 1, 2, 1 }; u = 0.5*u;     // Upsampling kernel (linear interpolation)
  Real tol = 1.e-13;                     // Tolerance for numerical comparisons

  // Using the original 3 equations that result from the perfect roundtrip conditions will produce
  // a singular matrix when u is symmetric (which is indeed the case). When u[2] = u[0] then Eq. 3
  // is the same as Eq. 1. So, the following system does not work:
  Mat A1(3, 3, { u[1], u[0]     ,  0  ,  // Eq. 1:  u1*d0 + u0*d1         = 0
                  0  , u[0]+u[2], u[1],  // Eq. 2:  u2*d1 + u1*d2 + u0*d1 = 1
                 u[1], u[2]     ,  0});  // Eq. 3:  u0*d1 + u1*d2         = 0
  Vec b1({0, 1, 0});

  // Let's replace Eq. 3 by a condition that fixes the value d[2].
  //Real d2 = 0.75;                      // d2 = 3/4 avoids roundoff errors in reconstruction
  Real d2 = 5.0 / 7.0;                   // d2 = 5/7 leads to perfect triangular shape
  Mat A2(3, 3, { u[1], u[0]     ,  0  ,  // Eq. 1:  u1*d0 + u0*d1         = 0
                  0  , u[0]+u[2], u[1],  // Eq. 2:  u2*d1 + u1*d2 + u0*d1 = 1
                  0  ,  0       ,  1});  // Eq. 3:  Fixes d2 to desired value
  Vec b2({0, 1, d2});

  // Select one combination of matrix and right hand side to use:
  Mat A = A2; Vec b = b2;
  // A1,b1 does not work because A1 is singular. A2,b2 does work.

  // Compute the left wing of the downsampling kernel by solving the 3x3 linear system:
  Vec dL = LA::solve(A, b);

  // Verify solution:
  Vec br = A*dL;
  ok &= rsIsCloseTo(br, b, tol);

  // Create the full downsampling kernel by duplicating the left wing, removing the last sample, 
  // reflecting it and then concatenating it with the left wing. The removal is because only the 
  // left wing contains the middle value. We don't want the middle value to be duplicated:
  Vec dR = dL;                           // dR will become the right wing of d.
  dR.resize(dR.size()-1);
  rsReverse(dR);
  Vec d = rsConcatenate(dL, dR);
  // Maybe factor that out into a function:  d = rsSymmetrize(dL);  but maybe the resizing should
  // be optional defaulting to false. We could have boolean parameter named removeDuplicateCenter.

  // Verify that upsampling with kernel u then downsampling with kernel d is an identity operation
  // up to roundoff error:
  Vec x = rsRandomIntVector(20, +1, +9, 0);
  ok &= testUpDownSampleRoundTrip(x, M, u, d, tol);

  // Create the combined up/down kernel:
  Vec ud = rsConvolve(u, d);

  // Plot all 3 kernels:
  rsPlotVectors(u, d, ud);  // ToDo: Plot as stem/lollipop plot

  return ok;


  // Observations:
  //
  // - We get the same results as in the experiment with the potentially asymmetric kernels where
  //   we imposed the symmetry by a constraint rather than by construction. This is totally what
  //   was expected.
  // 
  //
  // Notes:
  //
  // - In practice, it makes more sense to impose the symmetry by construction because that leads
  //   to smaller systems of equations which are easier to solve computationally (and possibly also
  //   more numerically accurate?). The case of the potentially asymmetrical kernels is more of
  //   theoretical interest. It's more general and therefore potentially more powerful but in 
  //   practice, we do not really need that kind of generality because we actually do want our 
  //   kernels to be symmetric anyway.
}

bool testOverSample_M3_L5()
{
  // _M3_L5 stands for M = 3 (oversampling factor) and L = 5 (upsampling kernel length)

  bool ok = true;

  // For convenience:
  using Real = double;
  //using Real = rsFraction<int>;             // ...would be nice to use but gives linker errors
  using Vec  = std::vector<Real>;
  using Mat  = RAPT::rsMatrix<Real>;
  using LA   = RAPT::rsLinearAlgebraNew;

  // Define oversampling factor and upsampling kernel:
  int  M = 3;                                 // Oversampling factor
  Vec  u = { 1, 2, 3, 2, 1 }; u = u/3;        // Upsampling kernel (linear interpolation)
  Real tol = 1.e-13;                          // Tolerance for numerical comparisons

  // In the file RS-MET-Research/Notes/DSP/TransparentOversampling.txt, I derived the following
  // system of equations:
  // 
  // u2*d0 +   u1*d1 +   u0*d2          = 0
  //         2*u0*d1 + 2*u1*d2 + u2*d3  = 1
  //                                d3  = p1
  //                        d2          = p2
  //
  // With some freely tweakable parameters p1,p2 which we can use to optimize the frequency 
  // response.

  // Ad hoc values:
  Real p1, p2;
  //p1 = 3./4; p2 = 2./4;
  //p1 = 2./3; p2 = 1./3;
  p1 = 3./6; p2 = 2./6;

  // Establish matrix A and right and side vector b by using the precribed values p1,p2 for d3,d2
  // resepctively:
  Mat A1(4, 4, { u[2],   u[1],   u[0],  0  ,
                  0  , 2*u[0], 2*u[1], u[2],
                  0  ,   0  ,    0   ,  1  ,
                  0  ,   0  ,    1   ,  0    });
  Vec b1({0, 1, p1, p2});

  // Require the numeric derivatives to be equal:  d1 - d0 = d2 - d1 = d3 - d2. This gives two 
  // equations which should give us a  perfect triangular kernel.
  // d1 - d0 = d2 - d1   ->   d0 - 2*d1 + d2 = 0
  // d2 - d1 = d3 - d2   ->   d1 - 2*d2 + d3 = 0
  Mat A2(4, 4, { u[2],   u[1],   u[0],  0  ,
                  0  , 2*u[0], 2*u[1], u[2],
                  1  ,  -2  ,    1   ,  0  ,
                  0  ,   1  ,   -2   ,  1    });
  Vec b2({0, 1, 0, 0});
  // Leads to:
  // 
  //   d0 = -0.15384615384615383  = -2/13
  //   d1 =  0.076923076923076927 =  1/13
  //   d2 =  0.30769230769230765  =  4/13
  //   d3 =  0.53846153846153844  =  7/13
  //
  // The numbers have been identified by Wolfram Alpha by entering them, removin one or two final
  // digits (because they may have been rounded up and my therefore be wrong) and appending 3 dots
  // i.e. entering 0.5384615384615384...  The dots are intepreted by Alpha as "this continues 
  // periodically" and it figures out the period and converts it to a fraction. I guess, we could 
  // also recover these fractions using the implementation of conversion to continued fractions 
  // where could stop the algorithm if the fractional part falls below some threshold (which 
  // should be related to the machine epsilon). 


  // Select one combination of matrix and right hand side to use:
  Mat A = A2; Vec b = b2;

  // Compute the left wing of the downsampling kernel by solving the 3x3 linear system:
  Vec dL = LA::solve(A, b);

  // Verify solution:
  Vec br = A*dL;
  ok &= rsIsCloseTo(br, b, tol);

  // Create the full downsampling kernel:
  Vec dR = dL;
  dR.resize(dR.size()-1);
  rsReverse(dR);
  Vec d = rsConcatenate(dL, dR);

  // Verify roundtrip:
  //Vec x = rsRandomIntVector(20, +1, +9, 0);       // Doesn't compile with Real = rsFraction<int>.
  Vec x({5,2,6,3,4,6,3,9,5,2,4,7,3,7,8,3,6,1,5});   // ..so let's use this instead.
  ok &= testUpDownSampleRoundTrip(x, M, u, d, tol);

  // Create the combined up/down kernel:
  Vec ud = rsConvolve(u, d);

  // Plot all 3 kernels:
  rsPlotVectors(u, d, ud);

  return ok;


  // Observations:
  //
  // - When using p1 = 3/4, p2 = 1/2, the downsampling kernel undershoots the zero line a lot. We
  //   get d1 = -0.625 and d0 = 0.25. I don't know if that undershoot is a good or a bad thing, 
  //   though. 
  //
  //
  // Notes:
  //
  // - There is some code duplication in the different experiments. In particular, the "Verify 
  //   solution", "Create the full..." ..etc. stuff. Maybe that can be factored out to get rid of 
  //   the duplication. Maybe functions verifyRoundtrip(), plotKernels() - or maybe a single 
  //   function verifyOversampleKernels(bool plot = false) or something could be used. Maybe it 
  //   should expect two left wings. That would mean that we should only produce the left wing of
  //   the u-kernel here, too.
  //
  //
  // ToDo:
  //
  // - Plot magnitude responses of u, d, ud.
  //
  // - Try to make it build when using Real = rsFraction<int> rather than double. We still get 
  //   "unresolved external symbol" linker errors related to rsPlotVectors(). We can use it when we
  //   comment out the plotting command, though. See comment in the rsPlotVectors() for an idea how
  //   to make that work.
}

bool testOverSample_M2_L5()
{
  bool ok = true;

  // For convenience:
  using Real = double;
  using Vec  = std::vector<Real>;
  using Mat  = RAPT::rsMatrix<Real>;
  using LA   = RAPT::rsLinearAlgebraNew;

  // Define oversampling factor and upsampling kernel:
  int  M = 2;                                 // Oversampling factor
  //Vec  u = { 1, 2, 3, 2, 1 }; u = u/3;        // Upsampling kernel
  Vec  u = { 1, 4, 6, 4, 1 }; u = u/4;        // Upsampling kernel
  Real tol = 1.e-13;                          // Tolerance for numerical comparisons

  // I think, using u = u/3 doesn't make sense here. I think, we need to normalize the kernel u in
  // such a way to have the sum of the values be equal to the oversampling factor M, so here we 
  // would want the sum to be equal to 2. Or wait - no - that seems to be wrong!

  // Later, we may wan to use this:
  //u = rsSincUpSampleKernel<Real>(5, M); 

  // The equations result from the prefect roduntrip contraints:
  // 
  // u1*d0 +   u0*d1                    =  0      Eq. 1
  // u1*d0 +   u2*d1 +   u1*d2 + u0*d3  =  0      Eq. 2
  //         2*u0*d1 + 2*u1*d2 + u2*d3  =  1      Eq. 3

  // Assign value to the tweakable parameter p:
  Real p;
  //p = 3./4;
  p = 1.0;

  // Establish matrix A and right and side vector b:
  Mat A1(4, 4, { u[1],   u[0],    0  ,  0  ,
                 u[1],   u[2],   u[1], u[0],
                  0  , 2*u[0], 2*u[1], u[0],
                  0  ,    0  ,    0  ,  1    });
  Vec b1({0, 0, 1, p});

  // ToDo:
  // Try to enforce a DC gain of 1 via: d0 + d1 + d2 + d3 = 1

  // Select one combination of matrix and right hand side to use:
  Mat A = A1; Vec b = b1;

  // Compute the left wing of the downsampling kernel by solving the 3x3 linear system:
  Vec dL = LA::solve(A, b);

  // Verify solution:
  Vec br = A*dL;
  ok &= rsIsCloseTo(br, b, tol);

  // Create the full downsampling kernel:
  Vec dR = dL;
  dR.resize(dR.size()-1);
  rsReverse(dR);
  Vec d = rsConcatenate(dL, dR);

  // Verify roundtrip:
  //Vec x = rsRandomIntVector(20, +1, +9, 0);       // Doesn't compile with Real = rsFraction<int>.
  Vec x({5,2,6,3,4,6,3,9,5,2,4,7,3,7,8,3,6,1,5});   // ..so let's use this instead.
  ok &= testUpDownSampleRoundTrip(x, M, u, d, tol);
  // FAILS! Ah! I think, this is because we use an upsampling kernel that doesn't interpolate the 
  // sample values. The conditions that the center-sample of the downsampling kernel must be 1 and
  // that it must have zero-crossings at distances from the center of multiples of M-1 is only 
  // correct when the upsampling kernel interpolates, i.e. the upsampled signal passes through the 
  // original data points exactly. So we really need to produce a proper interpolating upsampling 
  // kernel first. But maybe the system can be generalized to arbitrary (potentially 
  // non-interpolating) upsampling kernels. This could be an interesting direction for further 
  // research.

  // Create the combined up/down kernel:
  Vec ud = rsConvolve(u, d);

  // Plot all 3 kernels:
  rsPlotVectors(u, d, ud);

  return ok;

  // ToDo:
  //
  // - Use a different upsampling kernel u. Derive it from cubic Lagrange interpolation. At the 
  //   moment we just use the same kernel as in the M=3, L=5 case but that doesn't really make 
  //   sense here. I guess, it won't even properly interpolate. But I also think, for the algorithm
  //   here, that should not matter - except maybe when our wrong kernel somehow makes the system
  //   of equations singular. 
  //
  // - Maybe try also a kernel based on binomial coeffs, i.e. u = [1 4 6 4 1] / 16 or maybe
  //   u = [1 4 6 4 1] / 8. It also won't interpolate (I think), but it might be a nice Gaussian
  //   lowpass filter.
  //
  // - Maybe use (windowed) sinc interpolation. Maybe write a function rsSincKernel(int L, int M) 
  //   to generate it. Maybe use a window based on binomial coeffs. It has a nice Gaussian shape. 
  //   Or maybe use a true Gaussian window
}

bool testUpDownSampleFilterDilation()
{
  // We demonstrate how to produce up- and downsampling kernels for composite upsampling factors M
  // from the smaller kernels of the constituent upsampling factors. ...TBC...

  bool ok = true;

  // For convenience:
  using Real = double;
  using Vec  = std::vector<Real>;
  using Mat  = RAPT::rsMatrix<Real>;
  using LA   = RAPT::rsLinearAlgebraNew;


  // Create upsampling kernel for 2x upsampling by linear interpolation:
  Vec u2 = { 1, 2, 1 }; 
  u2 = 0.5 * u2;   

  // Produce the downsampling kernel d2  for 2x downsampling from u2 and the conditions that 
  // assign the middle sample d2[2] directly (to 0.75, say) and impose symmetry d[1] = d[3]:
  Real c = 0.75;
  Mat A(5, 5, { u2[1], u2[0],  0   ,  0   ,   0  ,       // Eq. 1
                 0  ,  u2[2], u2[1], u2[0],  0   ,       // Eq. 2
                 0  ,   0   ,  0   , u2[2], u2[1],       // Eq. 3
                 0  ,   0   ,  1   ,  0   ,   0  ,       // d[2] = c
                 0  ,   1   ,  0   , -1   ,   0    });   // d[3] = -d[1] -> d[1] - d[3] = 0
  Vec b({0, 1, 0, c, 0});
  Vec d2 = LA::solve(A, b);
  // ToDo: Maybe get rid of that computation and instead directly assign d2. We already have code
  // in the epxeriments above that does these computations.

  // Verify that upsampling with kernel u2 then downsampling with kernel d2 is an identity 
  // operation up to roundoff error:
  Vec x = rsRandomIntVector(20, +1, +9, 0);
  ok &= testUpDownSampleRoundTrip(x, 2, u2, d2, 0.0);
  // This is actually redundant with the tests in testUpDownSampleFilters() so maybe remove it 
  // here. But we still need x for the other tests down below.

  // From the up/downsampling kernels u,d for M=2, derive kernels for M=4 by convolving the kernels
  // with dilated (by M = 2) versions of themselves:
  Vec u4 = rsConvolve(u2, rsZeroStuff(u2, 2));
  Vec d4 = rsConvolve(d2, rsZeroStuff(d2, 2));
  ok &= testUpDownSampleRoundTrip(x, 4, u4, d4, 0.0);

  // Plot the kernels for oversampling with M=4:
  Vec ud4 = rsConvolve(u4, d4);
  rsPlotVectors(u4, d4, ud4);

  return ok;

  // ToDo:
  // 
  // - Explain the rationale for the dilate-convolve algorithm. The idea is to consider what 
  //   happens to the resulting filter kernel if we carry out two upsampling (or downsampling) 
  //   steps in succession (...I think).
  // 
  // - Verify that upsampling by factor 2 two times gives the same result as upsampling by factor 
  //   4 once.
  //
  // - Create an upsampling kernel u3 (by direct assignment) for 3x oversampling and a 
  //   corresponding downsampling kernel d3 by solving the appropriate linear system. Then use the
  //   dilate-convolve algorithm to produce up/downsampling kernels for 6x oversampling. Try it 
  //   both ways: 
  // 
  //     Vec u23 = rsConvolve(u2, rsZeroStuff(u3, 2));
  //     Vec d23 = rsConvolve(d2, rsZeroStuff(d3, 2));
  //
  //   and:
  // 
  //     Vec u32 = rsConvolve(u3, rsZeroStuff(u2, 3));
  //     Vec d32 = rsConvolve(d3, rsZeroStuff(d2, 3));
  //
  //   I think...verify, if these algos are correct!
  //
  // - Try upsampling a unit impulse (maybe centered at n = 10) twice and then also once with the 
  //   combined kernel
}

bool testUpDownSample1D()
{
  bool ok = true;

  // Under construction:
  //ok &= testSincUpSampler();
  ok &= testOverSample_M2_L3();
  //ok &= testOverSample_M3_L5();
  //ok &= testSincUpSampler();
  //ok &= testOverSample_M2_L5();
  //ok &= testUpDownSampleFilterDilation();

  // Unit Tests:
  ok &= testSincUpSampler();
  ok &= testUpSample1D();
  ok &= testUpDownSample1D_1();
  ok &= testUpDownSample1D_2();
  ok &= testOverSampleAsym_M2_L3();
  ok &= testOverSample_M2_L3();
  ok &= testOverSample_M3_L5();
  ok &= testOverSample_M2_L5();             // FAILS!!
  ok &= testSincUpSampler();
  ok &= testUpDownSampleFilterDilation();

  return ok;


  // Some potentially relevant resources:
  //
  // https://www.staff.ncl.ac.uk/oliver.hinton/eee305/Chapter9.pdf
  // https://course.ece.cmu.edu/~ece792/handouts/LO_Chap_Multirate.pdf  Sec. 3.4 pg. 158 ff
  // https://users.abo.fi/htoivone/courses/sbappl/asp_chapter2.pdf
}

//=================================================================================================

bool testUpDownSample2D()
{
  // Under construction. We want to create a lossless pair of upsampling/downsampling filters for
  // images, i.e. implement for 2D what we previuosly did for 1D.

  // For the 2D version, start with the assumption of bilinear interpolation for upsampling. We'll 
  // need a 5x5 kernel for downsampling. Let's express the kernel as:
  //
  //    e f c f e
  //    f d b d f
  //    c b a b c
  //    f d b d f
  //    e f c f e
  //
  // where a = d0, b = d1, c = d2 in the old notation used for the 1D case. Bilinear interpolation 
  // would spread into 4 output pixels. I think the condition that kernel must sum to 1 remains. In 
  // this case, this means: a + 4*(b+c+d+e) + 8*f = 1. Maybe the a + 2(b+c) = 1 condition should 
  // also remain valid because when we apply this to a single line, it should work as before? Maybe 
  // the d,e coeffs should be equal to a,b divided by sqrt(2). The rationale is to make the kernel
  // values dependent on distance from the center. We have 6 coeffs, so we need 5 equations if we 
  // want to treat a as free parameter as before. Maybe this is good:
  //
  //    (1) 1 = a + 2*(b+c)                   as before: d1 = 1 - d0
  //    (2) 0 = c + b/2                       as before: d2 = -d1 / 2
  //    (3) 1 = a + 4*(b+c+d+e) + 8*f         total sum of unity
  //    (4) d = b / sqrt(2)
  //    (5) e = c / sqrt(2)
  //
  // Maybe the 2D version of the kernel can be simply obtained by convolving a horizontal and a 
  // vertical version of the 1D kernel? I think, that is equivalent to to putting weighted copy of
  // the kernel into each row (or column) of the 5x5 kernel where the weight is again taken from 
  // the corresponding column (or row) index. That means, if the 1D kernel is [c b a b c]m then the 
  // 2D version would be:
  //
  //   cc  bc  ac  bc  cc                     [c]
  //   cb  bb  ab  bb  cb                     [b]
  //   ca  ba  aa  ba  ca  =  [c b a b c]  *  [a]
  //   cb  bb  ab  bb  ab                     [b]
  //   cc  bc  ac  bc  cc                     [c]
  //
  // where * denotes convolution...or does it? What we have here looks actually more like a tensor
  // product. ...figure out!
  // 
  // -For upsampling in the 2D case, see also rsImageProcessor<T>::interpolateBilinear. But maybe
  //  implement an optimized version for upsampling by 2. It should give the same result as
  //  interpolateBilinear(const rsImage<T>& img, 2, 2)
  // -I think, the upsampling kernel for linear bilinear interpolation is given by:
  //
  //    0   1/4   0                     1/4  0  1/4
  //   1/4   1   1/4   or maybe it's     0   1   0  
  //    0   1/4   0                     1/4  0  1/4


  bool ok = true;

  using TPix = float;
  using Img  = rsImage<TPix>;
  using Prc  = rsImageProcessor<TPix>;
  using IKM  = rsImageKernelMeasures<TPix>;

  // Try to repeatedly upsample a an impulse centered in a 3x3 image by a factor of 2 using 
  // Prc::interpolateBilinear. I'm interested in how the result looks like. What shape does the 
  // impulse become? A Gaussian blob?

  int numStages = 8;   // number of upsampling stages

  Img img(3, 3);
  img(1, 1) = 1.f;

  std::string name = "Impulse";
  int stage = 0;
  while(true)
  {
    // Write current image to a file:
    std::string path = name + std::to_string(stage) + ".ppm";
    //writeImageToFilePPM(img, path.c_str());

    // We interpret the interpolated image as filter kernel and take some measurments of it. The 
    // goal is to find some measurements that say something about the quality of the resampling. Of
    // particular interest is the isotropy of the filter.
    TPix mean  = IKM::mean(img);               // starts at 1/9, approaches 1/4
    TPix sumH  = IKM::centerSumHorz(img);      // 2^stage
    TPix sumV  = IKM::centerSumVert(img);      // 2^stage
    TPix sumD1 = IKM::centerSumDiagDown(img);  // 1, 1.5, 2.75, 5.375, 10.6875 -> find formula!
    TPix asRat = IKM::aspectRatio(img);        // 1
    TPix anIso = IKM::crossness(img);
    // Maybe the measurements should ignore the boundary pixel which are there only for technical 
    // reasons and are always black, so they don't actually belong to the kernel. Maybe use a 
    // cropped image to do the measurements.


    // Interpolate to next stage or leave loop:
    stage++;
    if(stage > numStages)
      break;
    else
      img = Prc::interpolateBilinear(img, 2, 2);
  }

  // Unit test for the crossness computation with crosses and (pseudo) isotropic kernels of 
  // different sizes. The 3x3 kernels for the straight and diagonal cross and the isotropic one 
  // look like:
  //
  //   0  1  0       1  0  1      k  c  k
  //   1  1  1       0  1  0      c  1  c  
  //   0  1  0       1  0  1      k  c  k
  //  
  // where c = 0.5 and k = c/sqrt(2). The isotropic kernel should give zero crossness for any value
  // of c but we use only 0.5 in the test. The straight cross should give a crossness of +1 and the 
  // diagonal cross a crossness of -1. The "isotropic" kernels of size > 3x3 are not really 
  // isotropic - they are only isotropic with respect to those pixels that actually enter the 
  // crossness computation which is good enough for this test.
  TPix tol = 1.e-6;
  for(int n = 3; n <= 11; n += 2)  // kernel sizes: 3x3, 5x5, ..., 11x11
  {
    img.setSize(n, n);
    int m = (n-1) / 2;             // m: middle

    // A straight cross:
    img.clear();
    for(int i = 0; i < n; i++) {
      img(m, i) = 1;
      img(i, m) = 1; }
    //writeImageToFilePPM(img, "Cross.ppm");
    TPix crs = IKM::crossness(img);   // 1
    ok &= rsIsCloseTo(crs, TPix(+1), tol);

    // A diagonal cross:
    img.clear();
    for(int i = 0; i < n; i++) {
      img(i, i)     = 1;
      img(i, n-1-i) = 1;   }
    //writeImageToFilePPM(img, "CrossDiag.ppm");
    crs = IKM::crossness(img); // -1
    ok &= rsIsCloseTo(crs, TPix(-1), tol);

    // An isotropic double-cross:
    img.clear();
    TPix c = 0.5;
    TPix s = 1.0/sqrt(2);
    for(int i = 0; i < n; i++) {  
      img(m, i) = c;
      img(i, m) = c; }
    for(int i = 0; i < n; i++) {
      img(i, i)     = s*c;
      img(i, n-1-i) = s*c;   }
    img(m, m) = 1;       // center (should not matter but anyway)
    //writeImageToFilePPM(img, "DoubleCross.ppm");
    crs = IKM::crossness(img); // 0
    ok &= rsIsCloseTo(crs, TPix(0), tol);
  }

  rsAssert(ok);
  return ok;


  // Observations:
  // -Using Prc::interpolateBilinear for repeatedly upsampling leads to an output image in which we
  //  don't really see a nice Gaussian circular blob. There's a clearly discernible cross of 
  //  vertical and horizontal bright pixels. See Impulse5.ppm


  // ToDo:
  // -Try to get the same result by upsampling via zero-stuffing and post filtering. I think, we 
  //  could filter vertically and horizontally by the kernel [1 2 1]/2 or use th 2D kernel
  //    1  2  1
  //    2  4  2   / 8    ...check the divisor...might be wrong
  //    1  2  1
  // -Then do the same with an isotropic 3x3 kernel. But I think, the isotropic kernel is not 
  //  separable. I think, the general condition for separability is that each row is a scalar 
  //  multiple of the same "prototype" row where the weights come from the vertical column (or vice
  //  versa, i.e. the columns are multiples of a prototype column weighted by coeffs from the 
  //  horizontal kernel).
  // -Try the magic kernel and binomial kernels.
  //
  // -Try to write an algorithm that takes a 2D kernel as input and factors it into 2 1D kernels
  //  in the least squeres sense. Define the error function:
  //    E = (1/2) sum_{i=1}^M sum_{j=1}^N (v_i h_j - k_{ij})^2
  //  where v_i, h_i are the coeffs of the vertical and horizontal 1D kernels, k_{ij} is the given
  //  2D kernel with height M and width N such that v is of length M and h is of length N. The 
  //  partial derivatives are (verify!): 
  //    d E / d v_i = sum_{j=1}^N h_j * (v_i h_j - k_ij)
  //    d E / d h_j = sum_{i=1}^M v_i * (v_i h_j - k_ij)
  //  Maybe solve the problem by a gradient based optimizer. As initial guess for v_i, h_j use
  //  the constant given by square root of the mean of the input kernel such that the initial
  //  product kernel has the same mean as the input kernel


  // Some notes for designing an isotropic fixed point kernel of size 3x3:
  // -181/128 is a decent approximation of sqrt(7) and 181/256 a decent approximation of 1/sqrt(2)
  //  ...but that may be worthless because we actually need a divisor of 255 because that's what
  //  1 maps to. Or maybe we should map 1 to 256. When the pixels are 8 bit, we will need a 16 bit
  //  accumulator anyway, so it makes sense. The kernel could look like:
  //
  //    [181  128  181]
  //    [128  256  128]   /   256
  //    [181  128  181]
}
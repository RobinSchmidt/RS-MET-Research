
//=================================================================================================
// Resampling:

// Move to RAPT, either next to resampleNonUniformLinear in Interpolation.h or into rsArrayTools:
template<class T>
void resampleLinear(const T* x, int Nx, T* y, int Ny)
{
  double dx = double(Nx) / double(Ny);
  for(int i = 0; i < Ny; i++)
    y[i] = RAPT::rsArrayTools::interpolatedValueAt(x, Nx, double(i)*dx);
}

// Convenience function to resample the given vector x to new length N
template<class T>
std::vector<T> resampleLinear(const std::vector<T>& x, int N)
{
  std::vector<T> y(N);
  resampleLinear(&x[0], (int)x.size(), &y[0], N);
  return y;
}

// Move these 2 into rsResampler:

// Upsamples the signal x by a factor of two by inserting samples in between the given samples 
// using linear interpolation which boils down to taking the average of two input samples for the
// to be inserted samples. If N is the length of, the output will have length 2*N-1. The -1 is 
// because for input N samples, there are N-1 gaps to be filled.
template<class T>
std::vector<T> upsampleBy2_Lin(const std::vector<T>& x)
{
  int Nx = (int) x.size();
  int Ny = 2*Nx-1;
  std::vector<T> y(Ny);
  for(int i = 0; i < Nx-1; i++) {
    y[2*i]   = x[i];
    y[2*i+1] = 0.5 * (x[i] + x[i+1]); }
  y[Ny-1] = x[Nx-1];
  return y;

  // ToDo:
  // -Maybe the output length should be Ny = 2*Nx. That would follow the principle of least 
  //  surprise. Then we can let the loop run to up to i < Nx and the last output y[Ny-1] would be
  //  just 0.5*x[Nx-1]. Formally, it would have to be 0.5 * (x[Nx-1] + x[Nx]) but x[Nx] is assumed 
  //  to be zero. In some sense, the y[Ny-1] would then be redundant and could actually be 
  //  discarded which is what we do here. Then the corresponding downsample function needs to be
  //  updated accordingly, too. Maybe it should be able to handle odd length inputs too. In such a
  //  case, we simply assume that the redundant last sample was discarded.

  // Notes:
  // -Note that an upsample-by-two scheme is determined by two filter kernels - one for the even 
  //  indexed output samples and one for the odd indexed output samples. In the case of simple 
  //  linear interpolation implemented here, the kernels are
  //    even: [1]
  //    odd:  [1 1]/2 = [0.5 0.5]
  // -The so called "magic kernel" (https://johncostella.com/magic/) would use [1 3 3 1]/8 for the
  //  odd indexed samples. Note sure what it would use for the even indexed ones, However...
  // -This suspiciously looks like the 4-th line of Pascal's triangle (divided by 8 = 2^3). This 
  //  suggests to try kernes of the form:
  //    even: [1 2 1]/4
  //    odd:  [1 3 3 1]/8
  //  or
  //    even: [1 4 6 4 1]/16
  //    odd:  [1 5 10 10 5 1]/32
  //  in general: use the 2n-th line of the triangle divided by 2^2n for even indices and the 
  //  (2n+1)-th line divided by 2^(2n+1) for odd indices where the index n starts at 0. For n = 0,
  //  we obtain linear interpolation. The parameter n could be some sort of smoothing parameter.
  // -I think, Pascal's triangle can also be interpreted as repeatedly convolving an initial 
  //  impulse input repeatedly by the kernel [1 1]. This should converge to a Gaussian shape in a 
  //  way that is most useful in the discrete case. A discrete approximation of a Gaussian bell 
  //  using n+1 samples may be best represented by the n-th line of the triangle divided by 2^n.
  //  It should have a sum of exactly 1. The numbers are also nice for a fixed point implementation
  //  because the are multiples of small inverse powers of 2. This can be implemented by using the
  //  unnormalized coeffs (such as [1 3 3 1]), add everything up and then do the division by 
  //  shifting. For a floating point implementation, it makes more sense to use the normalized 
  //  coeffs [1 3 3 1]/8 directly. But maybe make unit tests on 8-bit images using shifting. The 
  //  accumulator should be 16 bits wide.
  // -Implement the schemes using [1 2 1]/4, [1 3 3 1]/8 and [1 4 6 4 1]/16, [1 5 10 10 5 1]/32 and
  //  derive and implement the corresponding downsampling scheme
  // -I think, we can also express such a scheme using zero-stuffing and as post-filter use the two
  //  kernels interleaved. For example, [1 2 1]/4 = [2 4 2]/8, [1 3 3 1]/8 would give the 
  //  interleaved post-filtring kernel: [1 2 3 4 3 2 1]/8 ...but hey! That's just a linear ramp!
  //  Let's try the next: [2 8 12 8 2]/32, [1 5 10 10 5 1]/32. This gives:
  //  [1 2 5 8 10 12 10 8 5 2 1]/32. Maybe plot these interleaved kernels. The simplest one 
  //  corresponding to linear interpolation would be: [2]/2, [1 1]/2 -> [1 2 1]/2 = [0.5 1 0.5] as
  //  expected.
}

// Downsamples the signal y by a factor of two. This is the inverse operation of upsampleBy2_Lin. 
// Doing a roundtrip of upsampling and downsampling should give the original signal back exactly 
// (up to roundoff error). You can optionally pass a filter parameter that controls some filtering 
// before the downsampling. A coeff of 1 means no filtering at all, i.e. naive decimation by just 
// taking every even indexed sample and discarding the odd indexed ones. A coeff of 0.5 introduces 
// a rather strong filtering. The roundtrip identity should hold regardless of the a0 coefficient. 
// ...TBC...
template<class T>
std::vector<T> downsampleBy2_Lin(const std::vector<T>& y, T a0 = T(1))
{
  int Ny = (int) y.size();
  rsAssert(rsIsOdd(Ny));
  // We assume that y has been created via upsampleBy2_Lin and this function will always produce 
  // outputs of odd length. ToDo: Try to lift this restriction later

  int Nx = (Ny+1) / 2;  // Inverts the formula Ny = 2*Nx-1 in upsampleBy2_Lin
  std::vector<T> x(Nx);

  // Compute the other filter coeffs:
  T a1 = T(1) - a0;
  T a2 = -a1 * T(0.5);

  // Compute and return output:
  x[0] = y[0];
  for(int i = 1; i < Nx-1; i++) {  // i is index into x
    int j = 2*i;                   // j is center index into y
    x[i] = a0 * y[j] + a1 * (y[j-1]+y[j+1]) + a2 * (y[j-2]+y[j+2]);  }
  x[Nx-1] = y[Ny-1];
  return x;

  // ToDo:
  // -Maybe optimize the special case a0 = 1 (and therefore a1 = a2 = 0). Not sure, if it's worth 
  //  it. Depends on how commonly we expect this to occur. Probably not so often.
}

//=================================================================================================
// Filtering:


// Maybe move to RAPT:
template<class T>
std::vector<T> crop(const std::vector<T>& x, int first, int last)
{
  int Ny = last - first + 1;
  std::vector<T> y(Ny);
  for(int i = 0; i < Ny; i++)
    y[i] = x[i+first];
  return y;
  // We could perhaps do it in place by shifting elements leftward by "first" and then resizing
  // to "last - first + 1". See also:
  // https://stackoverflow.com/questions/421573/best-way-to-extract-a-subvector-from-a-vector
}

// Convenience function for filtering a signal vector x with an impulse response h. The output will 
// have a length of x.size() + h.size() - 1 but it may optionally be cropped to the length of the 
// original signal x.
template<class T>
std::vector<T> filter(const std::vector<T>& x, const std::vector<T>& h, 
  bool cropLength = false)
{
  int Nx = (int) x.size();                              // length of input x
  int Nh = (int) h.size();                              // length of impulse response h
  int Ny = Nx + Nh - 1;                                 // length of filtered output y
  std::vector<T> y(Ny);                                 // allocate output
  rsArrayTools::convolve(&x[0], Nx, &h[0], Nh, &y[0]);  // convolve x by h, store result to y
  if(cropLength) {
    int tail = Nh/2;  // Does this also work for even Nh? Try it!
    y = crop(y, tail, (Ny-1)-tail);
    rsAssert(y.size() == Nx); }
  return y;

  // ToDo:
  // -Check, if the cropping works also for even Nh. If not, check the formula tail = Nh/2. Maybe
  //  it needs to be tail = (Nh-1)/2 or tail = (Nh+1)/2. And/or maybe the amount to crop from start
  //  and end need to be different, i.e. we need an asymmetric crop. Not sure...
  // -When it works in all cases, maybe move the function to the library. Then add unit tests for
  //  filter length Nh = 0,1,2,3,4,5. That should suffice to be confident that it works.
}




//=================================================================================================

/** Implements a chain of identical first order filters. 

The prepareForBackwardPass function does not work correctly because only for the first stage, we
can assume that the output goes to zero at the edges - for all following stages, this assumption is
wrong because the filters that come before still produce nonzero outputs - bottom line: it doesn't
work as intended for image processing.  */

template<class TSig, class TPar>  // use TSig, TPar
class rsFirstOrderFilterChain
{


public:

  void setupFromPrototype(const rsFirstOrderFilterBase<TSig, TPar>& proto, int numStages)
  {
    stages.resize(numStages);
    for(int n = 0; n < numStages; n++)
      stages[n].setCoefficients(proto.getB0(), proto.getB1(), proto.getA1());
  }

  inline TSig getSample(TSig in)
  {
    TSig tmp = in;
    for(size_t i = 0; i < stages.size(); i++)
      tmp = stages[i].getSample(tmp);
    return tmp;
  }

  inline void reset()
  {
    for(size_t i = 0; i < stages.size(); i++)
      stages[i].reset();
  }

  inline void prepareForBackwardPass()
  {
    for(size_t i = 0; i < stages.size(); i++)
      stages[i].prepareForBackwardPass();
  }

  void applyBidirectionally(TSig* x, TSig* y, int N)
  {
    // forward pass:
    reset();
    for(int n = 0; n < N; n++)
      y[n] = getSample(x[n]);

    // backward pass:
    prepareForBackwardPass();
    for(int n = N-1; n >= 0; n--)
      y[n] = getSample(y[n]);
  }

  void applyBidirectionally(TSig* x, TSig* y, int N, int stride)
  {
    // forward pass:
    reset();
    for(int n = 0; n < N; n++)
      y[n*stride] = getSample(x[n*stride]);

    // backward pass:
    prepareForBackwardPass();
    for(int n = N-1; n >= 0; n--)
      y[n*stride] = getSample(y[n*stride]);
  }
  // optimize!


protected:

  std::vector<rsFirstOrderFilterBase<TSig, TPar>> stages;

};

//=================================================================================================

/** A nonlinear filter that implements the Shanks transformation for use in a realtime context. It 
introduces a delay of 1 sample and the first 2 produced samples may be meaningless due to the state
not yet being warmed up. */

template<class T>
class rsShanksFilter
{

public:


  T getSample(T in)
  {
    // Compute intermediates:
    T xR  = in;
    T dR  = xR - xM;
    T num = dR * dR;
    T den = dR - dL;

    // Compute output:
    T y;
    if(rsMaxNorm(den) <= tol)
      y = xM;                    // Avoid division by zero
    else
      y = xR - num / den;

    // Update state:
    xL = xM;
    xM = xR;
    dL = dR;

    // Return output:
    return y;
  }

  void reset()
  {
    xL = xM = dL = T(0);
  }

protected:

  // State:
  T xL = T(0);
  T xM = T(0);
  T dL = T(0);

  // Tolerance for avoiding division by (near) zero:
  T tol = std::numeric_limits<T>::epsilon();
  // ToDo: Maybe use 0 as default value. Maybe let it have its own type TTol. Provide a setter
  // function setDivByZeroThreshold() or something like that.

};

// ToDo: Maybe use special formulas for the 1st and 2nd call to getSample() after reset() has been
// called. Currently, the output at the first 2 samples is meaningless. The 1st sample is worse
// than the 2nd. Maybe in the very 1st call, just return the input. For the 2nd call - I'm not 
// sure - maybe try to resemble the output of the non-realtime implementation (with 1 sample 
// delay).


//=================================================================================================

/** UNDER CONSTRUCTION...I'm still trying to figure it out - it doesn't work yet.

*/

template<class TMat, class TVec> 
class rsKalmanFilter
{

public:


  rsKalmanFilter()
  {
    resetParameters();
    initState(rsZeroValue(x), rsZeroValue(P));
  }


  //-----------------------------------------------------------------------------------------------
  // \name Setup

  void setTransitionMatrix(          const TMat& newMatrix) { F = newMatrix; }

  void setMeasurementMatrix(         const TMat& newMatrix) { H = newMatrix; }

  void setTransitionNoiseCovariance( const TMat& newMatrix) { Q = newMatrix; }

  void setMeasurementNoiseCovariance(const TMat& newMatrix) { R = newMatrix; }

  void setInputControlMatrix(        const TMat& newMatrix) { B = newMatrix; }

  // Maybe instead of using assigment operators, us a (free) function rsCopyData(newMatrix, F), 
  // etc. and provide suitable explicit specializations of this function for all the relevant 
  // matrix classes. The idea is tha we want to avoid memory re-allocations in these setters, if
  // possible to make them potentially realtime safe. The idea is that for larger Kalman filters 
  // (with big matrices), the memory allocation is done once. When setting the filter object up 
  // under realtime conditions, no further (re)allocations happen because the matrxi-size doesn't
  // change in calling the setters. ...although - such an optimization is irrelevant as long as
  // the getSample function produces temporary matrices wthin its computations - which it does.
  // Maybe a production implementation should avoid that


  /** Resets all system parameters (i.e. the model matrices) to their default values. */
  void resetParameters();


  //-----------------------------------------------------------------------------------------------
  // \name Processing

  void initState(const TVec& x0, const TMat& P0)
  {
    x = x0;
    P = P0;
  }

  TVec getSample(const TVec& xIn, const TVec& u);
  // xIn is the "unfiltered" new state - i.e. "what the caller thinks" the new state should be, 
  // regardless of the model, i.e. an observation/measurement of an actual state. u is an input 
  // control vector. The output will be our updated state estimate x, i.e. "what the filter 
  // thinks" the new state should be. This estimate will be based on the old state and the new 
  // information contained in xIn. So, the mode of operation is: the caller observes some new 
  // state estimate "xIn" and asks the filter for a cleaned up version "x" of the estimate.
  // I'm not really sure, if this is the right way to operate a Kalman filter. This is all very 
  // experimental and immature. I'm still figuring this stuff out
  //
  // Nope! That is wrong! xIn is not, what the caller thinks, the state should be! It should be 
  // called measurements or observations or something. It is the raw sensor data that is used to
  // estimate the current state - the dimensionality of this vector may not even match the one of 
  // the state vector!

protected:

  // Estimates of dynamic system variables:
  TVec x;   // Estimate of state vector 
  TMat P;   // Estimate of covariance matrix of state vector

  // Static system parameters:
  TMat F;   // State transition matrix/model
  TMat H;   // State measurement matrix/model
  TMat Q;   // Covariance matrix of error/noise in state update
  TMat R;   // Covariance matrix of error/noise in state measurement
  TMat B;   // Input control matrix

  // https://en.wikipedia.org/wiki/Kalman_filter#Underlying_dynamic_system_model

  int numCalls = 0;  // for development
};


template<class TMat, class TVec>
void rsKalmanFilter<TMat, TVec>::resetParameters()
{
  F = rsUnityValue(F);
  H = rsUnityValue(H);
  Q = rsZeroValue(Q);
  R = rsZeroValue(R);
  B = rsUnityValue(B);
}


template<class TMat, class TVec>
TVec rsKalmanFilter<TMat, TVec>::getSample(const TVec& y, const TVec& u)
{
  // This implementation is very preliminary and I have no idea if this is even remotely correct.
  // I think, xIn should be renamed (maybe to y - that would be consistent with Haykin). It 
  // represents the current observation vector. ...Done!


  // Prediction:
  x = F*x + B*u;           // Predict new state from old, taking into account control vector
  P = F*P*rsTrans(F) + Q;  // Predict error covariance of x

  // Compute measurement z, innovation y and Kalman gain K:
  TVec z = H*y;                                       // Measurement of state from observations y
  TVec a = z - H*x;                                   // Innovation 
  TMat K = P*rsTrans(H) * rsInv(H*P*rsTrans(H) + R);  // Kalman gain
  // Wait no - that must be wrong! The vectors x and y do not in general have the same dimension,
  // so computing H*x and H*y cannot both make sense. Maybe it should be just: a = z - x? That 
  // would make sense: the "innovation" is the "element of surprise". If z = x, the innovation is 
  // zero which means that measured state and predicted state are exactly equal such that there's
  // no surprise at all. ....or maybe the z-vector should be our input? I think, the H matrix takes
  // a state and produces a vector of observables - I assume it the other way around, which may be 
  // wrong.

  // Correction:
  x += K*a;
  P = P - K*H*P;  // ToDo: use  P -= K*H*P;  ->  implement -= operator


  // Output is the cleaned up xIn which is equal to our current state estimate:
  numCalls++;
  return x;
}

//=================================================================================================




//-------------------------------------------------------------------------------------------------

/** Under construction.

A realtime oscillator that produces pitch-dithered sawtooth waves.

*/

template<class TFlt, class TInt> 
class rsPitchDitherSawOscOld
{

public:

  rsPitchDitherSawOscOld()
  {
    prng.setRange(0.0, 1.0);
  }

  /** Sets the period, i.e. the desired length of one cycle of the waveform. This is a floating 
  point value and it can be computed as  period = sampleRate / frequency  where frequency is the 
  desired oscillator frequency. If the period length is L and that number is not an integer but 
  has a fractional part of f, then the oscillator will produce cycles of length L1 = floor(L) and
  L2 = L1 + 1 in such a way that over time, the average length of the cycles will come out as L. */
  void setPeriod(TFlt newPeriod);
  // ..TBC... ToDo: explain deterministic and probabilistic modes

  void setRandomSeed(uint32_t newSeed)
  {
    prng.setSeed(newSeed);
  }



  //void setPhase(TFlt newPhase);
  // Should set up the current phase, i.e. the sampleCount variable


  enum class Mode
  {
    minVariance,      // Random cycle lengths with minimized variance
    equalVariance,
    equalDistance,
    minError          // Deterministic algo based on error feedback
  };

  void setMode(Mode newMode)
  {
    mode = newMode;
  }




  inline TFlt getSample();

  void reset()
  {
    sampleCount = 0;
    prng.reset();
  }

  inline void updateCycleLength();


protected:

  inline TFlt readSawValue(TInt sampleIndex, TInt cycleLength);
  // Maybe make this a static member function. It could be useful for other outside code. But maybe
  // when we optimize later, this function should use some values that are precomputed by other 
  // member functions, so maybe leave it as protected and non-static. Maybe rename to something 
  // like readSawValue..ok - done. Maybe make a function readPulseValue based on calling 
  // readSawValue twice (with phase-shifted arguments, depending on the pulseWidth parameter) and 
  // negating one of the saws. That means, we subtrac a phased-shifted version of the saw from the
  // saw. Maybe we can scale the subtracted phase-shifted version. Thereby, we can smoothly morph 
  // between supersaw and superpulse


  TInt sampleCount =   0;
  TInt floorLength = 100;    // Is the floor of the desired period (aka pitch cycle length).
  TInt cycleLength = 100;    // Is either floorLength or floorLength + 1
  TFlt fracLength  = 0.5;
  Mode mode        = Mode::minVariance;
  // ToDo: For optimization purposes, maybe we should keep all members as type TFlt in order to 
  // avoid int-to-float conversions in the per sample code. Maybe then call the type just T.

  rsNoiseGenerator<TFlt> prng;
};


template<class TFlt, class TInt>
void rsPitchDitherSawOscOld<TFlt, TInt>::setPeriod(TFlt newPeriod)
{
  floorLength = rsFloorInt(newPeriod);
  fracLength  = newPeriod - floorLength;

  //cycleLength = floorLength;
  // Maybe do not set this up here. We may get better behavior when modulating the period when we
  // defer this to getSample() which calls updateCycleLength(). When we don't set it here, we will
  // delay the update until the last cycle with the old length has been finished. Maybe in this 
  // case, we should init cycleLength to zero in reset() and maybe also here
}

template<class TFlt, class TInt>
TFlt rsPitchDitherSawOscOld<TFlt, TInt>::getSample()
{
  TFlt y = readSawValue(sampleCount, cycleLength);
  sampleCount++;
  if(sampleCount >= cycleLength)
  {
    updateCycleLength();
    sampleCount = 0;
  }
  return y;
}

template<class TFlt, class TInt>
void rsPitchDitherSawOscOld<TFlt, TInt>::updateCycleLength()
{ 
  switch(mode)
  {
  case Mode::minVariance:
  {
    cycleLength = floorLength;
    TFlt r = prng.getSample();
    if(r <= fracLength)          // Maybe use < instead of <=. See comment in rsPitchDitherProto 
      cycleLength++;
  } break;

  // ToDo: Implement the other modes

  default:
    rsError("Unknown Mode");
  }
}

template<class TFlt, class TInt>
TFlt rsPitchDitherSawOscOld<TFlt, TInt>::readSawValue(TInt n, TInt N)
{
  TFlt s = TFlt(2) / TFlt(N-1);      // Maybe precompute this and store in a member
  return (TFlt(-1) + s * TFlt(n));

  // Maybe the sampleCount variable should also be of type TFlt to avoid per sample conversion
  // from int to float. Maybe precompute s in updateCycleLength and store result in a member.
}

// ToDo:
//
// - Introduce a "mode" parameter that lets us switch between different algorithms to update the
//   cycle length. The algorithms we want to provide are: current one, deterministic (based on 
//   error feedback), another probabilistic one that uses L-1, L, L+1 with p = 0.25, 0.5, 0.25
//   respectively to make frequencies at exact integers sound similarly noisified as those at the
//   half integers.
//
// - Maybe get rid of the mode parameter again. Implement only the equal-variance algorithm. It 
//   turned out to be the right one experimentally. But maybe for experimentation and demonstration
//   purposes, an implementation that offers the different modes could be useful, so maybe 
//   implement all possible modes in this class and then make a smaller, optimized class that only
//   has the equal-variance algorithm. It could also precompute the cycle distribtuion (i.e. L1..L3
//   and p1..p3) so it doesn't need to be computed in each call to updateCycleLength().



//-------------------------------------------------------------------------------------------------

/** A class that factors out some functionality that is common to all generators that make use of 
pitch dithering. ...TBC... */

template<class T>
class rsPitchDitherHelpers
{

public:

  static void calcCycleDistribution(T period, T* midLength, T* probShort, T* probMid);
  // Maybe rename to cycleDistribEqualVariance (or ..EqVar) and maybe add functions to compute the
  // other distributions as well, although, I think, the other distributions are not really useful.
  // Not sure...maybe we shouldn't clutter the code with useless stuff. At leats not the production
  // code. For research and prototype code, it's a different story. There, we may use the useless 
  // code to demonstrate in experiments that it is indeed useless.  


};

template<class T>
void rsPitchDitherHelpers<T>::calcCycleDistribution(
  T period, T* midLength, T* probShort, T* probMid)
{
  // Compute lengths:
  T floorLength = rsFloor(period);
  T fracLength  = period - floorLength;
  T L1, L2, L3;
  if(fracLength < T(0.5))
    L1 = floorLength - T(1);
  else
    L1 = floorLength;
  L2 = L1 + T(1);
  L3 = L2 + T(1);

  // Compute intermediates:
  T e1 = L1 - period;
  T e2 = L2 - period;
  T e3 = L3 - period;
  T m1 = e1*e1;
  T m2 = e2*e2;
  T m3 = e3*e3;
  T M  = T(0.25);
  T M1 = M - m1;
  T M2 = M - m2; 
  T M3 = M - m3;
  T S  = T(1) / (e3*(m1-m2) - e2*(m1-m3) + e1*(m2-m3));

  // Compute outputs:
  *midLength = L2;
  *probShort = (M2*e3 - M3*e2) * S;
  *probMid   = (M3*e1 - M1*e3) * S;
  //*probLong  = (M1*e2 - M2*e1) * S;  // Would be redundant. See Notes
  
  // Notes:
  // 
  // - We don't have a probLong parameter because that would be redundant. It would always be given
  //   by 1 - (probShort + probMid).
}

// ToDo:
//
// - Add convenience functions to produce a whole signal vector of signals with various waveforms.
//   maybe take the waveform as std::function or some callable template type F. 
//
// - Maybe create functions to produce various wvaeforms, including additively syntehsized saw
//   waves (mayby by using trig-recursions for an optimized implementation)

//-------------------------------------------------------------------------------------------------

/** A realtime oscillator that produces pitch-dithered sawtooth waves. */

template<class T> 
class rsPitchDitherSawOsc
{

public:

  using PDH = rsPitchDitherHelpers<T>;     // Shorthand for convenience

  //-----------------------------------------------------------------------------------------------
  // \name Lifetime

  /** Default constructor. It puts the object into a valid initial state by setting up a default
  period length of 100.0 samples and triggering the appropriate computations to set up our member
  variables that control the distribution of cycle lengths. */
  rsPitchDitherSawOsc()
  {
    prng.setRange(T(0), T(1));             // We use random numbers in the interval [0,1).
    setPeriod(T(100.0));                   // Triggers computations to set up members.
    reset();                               // Assigns sampleCount.
  }

  //-----------------------------------------------------------------------------------------------
  // \name Setup

  /** Sets the period, i.e. the desired length of one cycle of the waveform. This is a floating 
  point value and it can be computed as  period = sampleRate / frequency  where frequency is the 
  desired oscillator frequency. This will immediately trigger a recomputation of the probability
  distribution of the cycle lengths and update the currently used cycle length. */
  void setPeriod(T newPeriod)
  {
    setPeriodNoUpdate(newPeriod);
    updateCycleLength();
  }

  /** Sets up a new period length just like setPeriod() does but without immediately updating the
  probability distribution and current cycle length. This results in the behavior that the new 
  period will not become effective immediately but only after finishing the currently running 
  cycle. */
  void setPeriodNoUpdate(T newPeriod)
  { 
    PDH::calcCycleDistribution(newPeriod, &midLength, &probShort, &probMid); 
  }

  /** Sets the seed for the pseudo random number generator. */
  void setRandomSeed(uint32_t newSeed)
  {
    prng.setSeed(newSeed);
  }

  //-----------------------------------------------------------------------------------------------
  // \name Inquiry

  /** Returns the average length of the cycles that are being produced. If the 3 integer cycle 
  lengths are given by L1,L2,L3 and cycles with these 3 lengths are produced with probabilities 
  p1,p2,p3 respectively, then the average cycle length P will be: P = p1*L1 + p2*L2 + p3*L3. */
  T getPeriod()
  {
    T probLong = T(1) - (probShort + probMid);
    return probShort * (midLength - T(1)) + probMid * midLength + probLong * (midLength + T(1));
  }
  // Needs tests

  //-----------------------------------------------------------------------------------------------
  // \name Processing

  /** Produces one sample at a time. */
  inline T getSample()
  {
    T y = T(-1) + sawSlope * sampleCount;  // Compute output sample.
    sampleCount += T(1);                   // Update counter. We will have produced 1 sample.
    if(sampleCount >= cycleLength)         // Is cycle finished?
    {                                      // If so..
      sampleCount = T(0);                  // ..Wrap around sample counter.
      updateCycleLength();                 // ..Compute cycleLength and sawSlope for next cycle.
    }
    return y;                              // Return output sample.
  }

  /** Resets the internal state, i.e. the sample counter and the random generator. */
  void reset()
  {
    sampleCount = T(0);
    prng.reset();
  }


protected:

  //-----------------------------------------------------------------------------------------------
  // \name Internals

  /** Updates our cycleLength member by computing a new (pseudo) random cycle length to be used 
  for the next cycle. Called from getSample() after each cycle has been completed. */
  inline void updateCycleLength()
  {
    T r = prng.getSample();                  // Random number in interval [0,1).
    if(r < probShort)
      cycleLength = midLength - T(1);        // Next cycle is short.
    else if(r < probShort + probMid)
      cycleLength = midLength;               // Next cycle is medium.
    else
      cycleLength = midLength + T(1);        // Next cycle is long.

    sawSlope = T(2) / (cycleLength - T(1));  // Slope depends on cycle length.
  }

  //-----------------------------------------------------------------------------------------------
  // \name Data

  // Members that are accessed per sample:
  T sawSlope;      // Increase of output value per sample.
  T sampleCount;   // Is always in interval [0, cycleLength).
  T cycleLength;   // Is midLength or midLength + 1 or midLength - 1.

  // Members that are accessed per cycle:
  T midLength;     // The middle one of the 3 cycle lengths to be produced.
  T probShort;     // Probability to use midLength - 1.
  T probMid;       // Probability to use midLength.

  // Embedded DSP objects:
  rsNoiseGenerator<T> prng;

  // Notes:
  //
  // - The members sampleCount, cycleLength and midLength are actually integer numbers but we let 
  //   them be of type T (which is typically float or double) anyway for optimization purposes. We 
  //   want to avoid int-to-float conversions because they are costly. The other members are indeed
  //   true floating point values that are not restricted to integers.
  // 
  // - There is no "probLong" member variable because that would be redundant. The probability to
  //   use the long cycle is always given by  1 - (probShort + probMid)  because probabilities must
  //   add up to 1.
  // 
  // - The member variables are deliberately not initialized in their declarations because some 
  //   initializations would require using some moderately complex formulas for a consistent, valid
  //   initial state. So we leave this member initialization to the constructor which calls some 
  //   functions to do the appropriate computations.
  //  
  // 
  // ToDo:
  //
  // - Replace the rsNoiseGenerator<T> member by a rsRandomGenerator<T>. This class does not exist 
  //   yet. It is supposed to factor out the integer random number generation from rsNoiseGenerator
  //   such that the object doesn't need to maintain the shift and scale members. It could provide
  //   a convience function for producing floating point outputs in the fixed range [0,1) but it 
  //   will not provide a user adjustable range. It could have a convenience function 
  //   getSampleFloat(T min, T max), though. But using that function in a hot loop is not 
  //   recommended for production code because it will need a costly division. Maybe it should have
  //   a normal getSample() function that produces floats in [0,1) and an additional getSampleRaw()
  //   or getSampleInt() function that produces the raw integer value.
  // 
  // - setRandomSeed() should probably eventually be replaced by a setRandomState function. Or 
  //   maybe complemented by it. We actually do want to have a setRandomSeed() function as well. 
  //   But when we switch to a PRNG that has no built in seed member, we will need a randomSeed 
  //   member variable here.
};

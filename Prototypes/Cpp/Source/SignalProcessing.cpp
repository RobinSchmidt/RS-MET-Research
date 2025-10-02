
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




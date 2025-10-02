
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



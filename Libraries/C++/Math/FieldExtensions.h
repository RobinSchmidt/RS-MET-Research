#pragma once

// UNDER CONSTRUCTION

namespace rema  // Rob's educational math algorithms
{


/**

The implementation roughl follows rsModularInteger

*/

template<class T>
class rsQuadraticField
{

public:


  //-----------------------------------------------------------------------------------------------
  /** \name Lifetime */

  /** Default constructor. Leaves value and square uninitialized. */
  rsQuadraticField() {}



  //-----------------------------------------------------------------------------------------------
  // \name Setup


  void set(T newCoeffA, T newCoeffB, T newSquare) { a = newCoeffA; b = newCoeffB; n = newSquare; }



protected:

  T a = T(0), b = T(0);    // Coeffs in x = a + b*r
  T n = T(0);              // Square of our root r: n = r^2, r = sqrt(n)

};



}
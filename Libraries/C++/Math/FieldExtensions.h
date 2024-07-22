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

  /** Default constructor. Leaves coeffs and square uninitialized. */
  rsQuadraticField() {}


  rsQuadraticField(const T& coeffA, const T& coeffB, const T& square)
  { a = coeffA; b = coeffB; n = square; }


  //-----------------------------------------------------------------------------------------------
  // \name Setup


  void set(T newCoeffA, T newCoeffB, T newSquare) { a = newCoeffA; b = newCoeffB; n = newSquare; }


  //-----------------------------------------------------------------------------------------------
  /** \name Operators */

  rsQuadraticField& operator=(const rsQuadraticField& rhs)
  { a = rhs.a; b = rhs.b; n = rhs.n; return *this; }


  rsQuadraticField operator-() const;

  bool operator==(const rsQuadraticField& other) const;
  bool operator!=(const rsQuadraticField& other) const;


  /*
  rsQuadraticField operator+(const rsQuadraticField& other) const;
  rsQuadraticField operator-(const rsQuadraticField& other) const;
  rsQuadraticField operator*(const rsQuadraticField& other) const;
  rsQuadraticField operator/(const rsQuadraticField& other) const;

  rsQuadraticField& operator+=(const rsQuadraticField& other);
  rsQuadraticField& operator-=(const rsQuadraticField& other);
  rsQuadraticField& operator*=(const rsQuadraticField& other);
  rsQuadraticField& operator/=(const rsQuadraticField& other);
  */



protected:

  T a = T(0), b = T(0);    // Coeffs in x = a + b*r
  T n = T(0);              // Square of our root r: n = r^2, r = sqrt(n)

};



}
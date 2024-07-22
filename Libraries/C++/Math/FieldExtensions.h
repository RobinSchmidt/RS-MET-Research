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
  // \name Inquiry


  T getCoeffA() const { return a; }

  T getCoeffB() const { return b; }

  T getSquare() const { return n; }

  bool is(const T& coeffA, const T& coeffB, const T& square) const
  { return a == coeffA && b == coeffB && n == square; }

  rsQuadraticField<T> getReciprocal() const;


  // Maybe implement:
  // T getDiscriminant() const;
  // https://en.wikipedia.org/wiki/Quadratic_field#Discriminant


  //-----------------------------------------------------------------------------------------------
  /** \name Operators */

  rsQuadraticField& operator=(const rsQuadraticField& rhs)
  { a = rhs.a; b = rhs.b; n = rhs.n; return *this; }

  bool operator==(const rsQuadraticField& other) const;
  bool operator!=(const rsQuadraticField& other) const;

  rsQuadraticField operator-() const;

  rsQuadraticField operator+(const rsQuadraticField& other) const;
  rsQuadraticField operator-(const rsQuadraticField& other) const;
  rsQuadraticField operator*(const rsQuadraticField& other) const;
  rsQuadraticField operator/(const rsQuadraticField& other) const;

  rsQuadraticField& operator+=(const rsQuadraticField& other);
  rsQuadraticField& operator-=(const rsQuadraticField& other);
  rsQuadraticField& operator*=(const rsQuadraticField& other);
  rsQuadraticField& operator/=(const rsQuadraticField& other);
 



protected:

  T a = T(0), b = T(0);    // Coeffs in x = a + b*r
  T n = T(0);              // Square of our root r: n = r^2, r = sqrt(n)

};



}
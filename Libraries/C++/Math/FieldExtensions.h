#pragma once

// UNDER CONSTRUCTION

namespace rema  // Rob's educational math algorithms
{


/** Implements the quadratic field Q(sqrt(n)) for a given integer or rational number n. The 
implementation roughly follows RAPT::rsModularInteger where we had the modulus m as parameter. Here
we have the number n as parameter. This is the square of the square-root that we want to adjoin to 
the given base field. The base field is typically the field of rational numbers Q. That means, the 
template parameter T should be some sort of rational number type such as RAPT::rsFraction. The 
represented numbers are of the form:

  x = a + b*r              where  r = sqrt(n)  and  a,b are rational
  
where r = sqrt(n) is the adjoined square-root. The resulting field is called a field extension of 
the field rational numbers. We extend it by "adjoining" r. This process of adjoining some number to
a field that previously was not present requires us to add a lot more numbers to keep the field 
closed under the arithmetic operations. It turns out that when we adjoin the square-root r of some
number n, we can express all members of the field as linear combinations of 1 and r with 
coefficients from our base field.

For n = -1, the complex (aka Gaussian) rationals result. For n = 5, we get a field that allows us to
compute Fibonacci numbers via Binet's closed form formula without resorting to inexact floating 
point arithmetic. For n = 2, we get a field that is often used in algebra textbooks as example for 
field extensions. 

References:

  (1)  A Book of Abstract Algebra  (Charles C. Pinter)



...TBC...  */

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

  double toDouble() const { return double(a) + double(b) * sqrt(double(n)); }

  //double toDouble() const { return double(a) + double(b) * sqrt(double(n)); }

  operator double() const { return toDouble(); }


  // Maybe implement:
  // T getDiscriminant() const;
  // https://en.wikipedia.org/wiki/Quadratic_field#Discriminant



  T getNorm() const { return a*a - b*b*D; }

  // see https://www.youtube.com/watch?v=0AyqablLD-A  at 1:18

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


template<class T>
rsQuadraticField<T> rsZeroValue(rsQuadraticField<T> value)
{ 
  return rsQuadraticField<T>(T(0), T(0), value.getSquare()); 
}

template<class T>
rsQuadraticField<T> rsUnityValue(rsQuadraticField<T> value)
{ 
  return rsQuadraticField<T>(T(1), T(0), value.getSquare()); 
}

// Implement rsInv

//template<class T> 
//rsQuadraticField<T> rsConstantValue(T value, rsQuadraticField<T> targetTemplate) 
//{ 
//  return rsModularInteger<T>(value, T(0), targetTemplate.n);
//}




}
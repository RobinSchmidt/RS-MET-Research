#pragma once

namespace rema  // Rob's educational math algorithms
{


// Maybe make classes rsFiniteGroup and rsFiniteRing that are based on addition- and multiplication 
// tables that the user can pass in or that can be generated by suitable functions. API-wise, they
// could work similar to the implementation of rsGeometricAlgebra. Maybe they could have functions
// like isCommutative, getCenter, getCentralizer, getNormalizer, getNumSubgroups, getCosets, etc.
// The group elements could be represented by integers (maybe unsigned, 64 bit). 

/** just a stub at the moment

In mathematics, a field is a set in which certain operations are defined and these operations
behave in the same way as addition and multiplication in rational or real numbers. That means,
besides other things that, the multiplicative inverses must exist for each element except zero.
Rational, real or complex numbers are all infinite fields but finite fields also exist. The
simplest finite fields are the modular integers when the modulus is a prime number. The only other
finite fields that exist are (isomorphic to) those, whose number of elements is an integer power
of a prime. But for these, simple modular arithmetic doesn't produce the field. Addition and
multiplication require some more elaborate algorithms. These are implemented in a naive way for
learning purposes by the classes rsFiniteFieldNaive and rsFiniteFieldElementNaive which work
together in tandem in the same way as rsGeometricAlgebra and rsMultiVector: every element holds a
pointer to the algebra object which is consulted to perform the arithmetic operation.

...tbc...

See:
https://en.wikipedia.org/wiki/Finite_field


*/

class rsFiniteFieldTables
{

public:

  rsFiniteFieldTables(int base, int exponent, const std::vector<int>& modulusCoeffs);
    /*
    : p(base), k(exponent)
  {
    //RAPT::rsAssert(rsIsPrime(p));  // todo
    //generateTables();
  }
  */

protected:

  void createOperationTables();

  int p;  // Base in p^m, should be prime
  int k;  // Exponent in p^k, a positive integer

  // Operation tables:
  std::vector<int>    mod;                // Coeffs of the modulus polynomial
  std::vector<int>    neg, rec;           // 1D tables negatives and reciprocals.
  RAPT::rsMatrix<int> add, mul, sub, div; // 2D tables for arithemetic operations.




  /*
  VecI  neg_8_2 = abstractifyTable1D(g2, neg2);    ok &= neg_8_2 == neg_8;
  VecI  rec_8_2 = abstractifyTable1D(g2, rec2);    ok &= rec_8_2 == rec_8;
  MatI  add_8_2 = abstractifyTable2D(g2, add2);    ok &= add_8_2 == add_8;
  MatI  mul_8_2 = abstractifyTable2D(g2, mul2);    ok &= mul_8_2 == mul_8;
  MatI  sub_8_2 = abstractifyTable2D(g2, sub2);    ok &= sub_8_2 == sub_8;
  MatI  div_8_2 = abstractifyTable2D(g2, div2);    ok &= div_8_2 == div_8;
  */


  // We need to form the field of polynomials of degree <= k over the modular integers with 
  // modulus p. Then we take that field modulo a specific polynomial M(x) that plays the role
  // of a modulus ...tbc...
  //using ModInt = RAPT::rsModularInteger<T>; // 
  //rsPolynomial<ModInt> M;  // M(x) is an degree k polynomial that is irreducible in Z_p = Z/pZ
  // ...so we have two levels of modular arithmetic at play here? The lower level being the usage 
  // of modular integers and the higher level being the use doing all polynomial operations modulo
  // the given M(x), requiring polynomial division with remainder?

};


// In a non-naive implementation, we should build tables for addition and multiplication in the 
// constructor. Each polynomial, i.e. each array of polynomial coeffs, maps to a unique integer
// in the range 0...p^k-1. For each pair of such integers (mapped polynomials) we need to specify
// what the result of their addition and multiplication should be - coming from the same set of 
// 0...p^k-1. The multiplication table can be turned into a 1D array rather than a full blown 2D
// matrix by a trick explained in Weitz pg. 744. I hope, a similar trick is possible for addition
// too. Weitz says nothing about that because he's only covering the case for p=2 in which addition
// reduces to xor such that no table is needed. The method there uses a primitive k-th (?) root of 
// unity, i.e. a number that, when multiplied by itself k times gives one. Maybe an analog for
// addition could be a number that when added to itself k times gives zero? And that number would 
// be just the number 1, regardless of the modulus? ...  figure this out!
//
// What happens if we use a cartesian product of modular integers, i.e. what if we take e.g.
// Z_7 x Z_7 x Z_7 with element-wise addition, multiplication and inversion? Will that produce
// the finite field F_(7^3)? In Weitz's video, he gives a rather complicated construction in terms
// polynomials over Z_7, so I suppose this (much simpler) approach will not work. But why not? 
// What happens, if we try to do things like Z_2 x Z_3 x Z_5 etc.? Maybe first try to build the 
// simplemost field F_4 as Z_2 x Z_2. Maybe this will reveal the problems. It could be implemented
// using an array of rsModularInteger and defining the arithmetic operations between such arrays. 
// But even if it doesn't produce the desired finite field - maybe it produces some other 
// interesting algebraic structure? Ah - I think, we could have (0,1) * (1,0) = (0,0), So a product
// could come out as zero even though none of the factors is zero, so it wouldn't be a field. We
// would have zero divisors. Could we repair this somehow? Maybe by disallowing all elements that 
// have some zero components and allow only those where all components are zero? But we cannot
// practically "disallow" thoses because they may result as output of some arithmetic operation.
// But maybe such a structure could still be useful enough, if the zero divisors are raer enough?


}
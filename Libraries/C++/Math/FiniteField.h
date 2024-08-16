#pragma once

namespace rema  // Rob's educational math algorithms
{


/** In mathematics, a field is a set in which certain operations are defined and these operations
behave in the same way as addition and multiplication in rational or real numbers. That means,
besides other things that, the multiplicative inverses must exist for each element except zero.
Rational, real or complex numbers are all infinite fields but finite fields also exist. The
simplest finite fields are the modular integers when the modulus is a prime number. The only other
finite fields that exist are (isomorphic to) those, whose number of elements is an integer power
of a prime. But for these, simple modular arithmetic doesn't produce the field. Addition and
multiplication require some more elaborate algorithms. These are implemented in a naive way for
learning purposes by the classes rsFiniteFieldTables and rsFiniteFieldElement which work together 
in tandem in the same way as rsGeometricAlgebra and rsMultiVector: every element holds a pointer to
the algebra object which is consulted to perform the arithmetic operation. This object contains
the operation tables. The elements of the finite fields are just represented abstractly by their
integer index which is a number in 0...n-1 where n is the number of elements in the field.

See:

https://en.wikipedia.org/wiki/Finite_field  

*/

class rsFiniteFieldTables
{

public:

  rsFiniteFieldTables(int base, int exponent, const std::vector<int>& modulusCoeffs);


  //-----------------------------------------------------------------------------------------------
  /** \name Inquiry */


  int getBase()     const { return p; }

  int getExponent() const { return k; }

  int getOrder()    const { return n; }


  const std::vector<int>&    getModulusPolynomial()   const { return mod; }


  const std::vector<int>&    getNegationTable()       const { return neg; }

  const std::vector<int>&    getReciprocationTable()  const { return rec; }

  const RAPT::rsMatrix<int>& getAdditionTable()       const { return add; }

  const RAPT::rsMatrix<int>& getSubtractionTable()    const { return sub; }

  const RAPT::rsMatrix<int>& getMultiplicationTable() const { return mul; }

  const RAPT::rsMatrix<int>& getDivisionTable()       const { return div; }




protected:

  void createOperationTables();

  // Data:
  int p;                                  // Base in p^k, should be prime
  int k;                                  // Exponent in p^k, a positive integer
  int n;                                  // Number of elements. Order of the field. n = p^k
  std::vector<int> mod;                   // Coeffs of the modulus polynomial
  std::vector<int> neg, rec;              // 1D tables for unary operations (negate, reciprocate)
  RAPT::rsMatrix<int> add, mul, sub, div; // 2D tables for binary operations (addition, ...)


  friend class rsFiniteFieldElement;

};

//=================================================================================================

/** Class for representing finite field elements. To construct such an element, you need to pass a
pointer to a rsFiniteFieldTables object that will be consulted for doing the operations. That 
pattern of implementation is similar to how rsMultiVector references a rsGeometricAlgebra object 
for doing the actual computations. */

class rsFiniteFieldElement
{

public:


  rsFiniteFieldElement() {}

  rsFiniteFieldElement(const rsFiniteFieldTables* tablesToUse)
  { 
    setTables(tablesToUse); 
  }

  rsFiniteFieldElement(int newVal, const rsFiniteFieldTables* tablesToUse)
  { 
    val = newVal;
    setTables(tablesToUse);
    // todo: check if val is in aloowed range
  }


  rsFiniteFieldElement(const rsFiniteFieldElement& b) 
  { 
    set(b); 
  }



  //-----------------------------------------------------------------------------------------------
  /** \name Setup */

  void setTables(const rsFiniteFieldTables* tablesToUse)
  {
    tables = tablesToUse;
  }

  void set(const rsFiniteFieldElement& b)
  {
    val    = b.val;
    tables = b.tables;
  }

  void set(int newVal, const rsFiniteFieldTables* tablesToUse)
  {
    val    = newVal;
    tables = tablesToUse;
    rsAssert(isOk());
  }


  //-----------------------------------------------------------------------------------------------
  /** \name Inquiry */


  int getValue() const { return val; }

  const rsFiniteFieldTables* getTables() const { return tables; }

  rsFiniteFieldElement getNegative() const 
  {
    RAPT::rsAssert(tables != nullptr);
    return rsFiniteFieldElement(tables->neg[val], tables);
  }

  rsFiniteFieldElement getReciprocal() const 
  {
    RAPT::rsAssert(tables != nullptr);
    return rsFiniteFieldElement(tables->rec[val], tables);
  }


  /** A sanity check function. */
  bool isOk() const
  {
    return tables != nullptr && val >= 0 && val < tables->n;
  }

  bool isOperationOk(const rsFiniteFieldElement& a, const rsFiniteFieldElement& b) const
  {
    bool ok = true;
    ok &= a.tables != nullptr;
    ok &= b.tables == a.tables;

    // ToDo: check if values of a and b are in the allowed range of 0..p^k-1
    // Would it actually make sense to interpret avlues outside this range as wrapped back via a
    // modulo operation?

    return ok;
  }


  rsFiniteFieldElement zero(const rsFiniteFieldTables* tablesToUse) const
  {
    return rsFiniteFieldElement(0, tablesToUse);
  }

  rsFiniteFieldElement one(const rsFiniteFieldTables* tablesToUse) const
  {
    return rsFiniteFieldElement(1, tablesToUse);
  }







  //-----------------------------------------------------------------------------------------------
  /** \name Operators */

  bool operator==(const rsFiniteFieldElement& b) const 
  { 
    return val == b.val && tables == b.tables;
  }

  bool operator!=(const rsFiniteFieldElement& b) const 
  { 
    return !(*this == b);
  }

  rsFiniteFieldElement operator+() const
  {
    return rsFiniteFieldElement(*this);
  }

  rsFiniteFieldElement operator-() const
  {
    return getNegative();
  }

  rsFiniteFieldElement operator+(const rsFiniteFieldElement& b) const
  {
    RAPT::rsAssert(isOperationOk(*this, b));
    return rsFiniteFieldElement(tables->add(val, b.val), tables);
  }

  rsFiniteFieldElement operator-(const rsFiniteFieldElement& b) const
  {
    RAPT::rsAssert(isOperationOk(*this, b));
    return rsFiniteFieldElement(tables->sub(val, b.val), tables);
  }

  rsFiniteFieldElement operator*(const rsFiniteFieldElement& b) const
  {
    RAPT::rsAssert(isOperationOk(*this, b));
    return rsFiniteFieldElement(tables->mul(val, b.val), tables);
  }

  rsFiniteFieldElement operator/(const rsFiniteFieldElement& b) const
  {
    RAPT::rsAssert(isOperationOk(*this, b));
    return rsFiniteFieldElement(tables->div(val, b.val), tables);
  }

  // ToDo: +=, -=, *=, /=


protected:

  int val = 0;
  const rsFiniteFieldTables* tables = nullptr;

};

rsFiniteFieldElement rsZeroValue(rsFiniteFieldElement value)
{ 
  return rsFiniteFieldElement(0, value.getTables()); 
}
// needs test

rsFiniteFieldElement rsUnityValue(rsFiniteFieldElement value)
{ 
  return rsFiniteFieldElement(1, value.getTables()); 
}
// needs test

rsFiniteFieldElement rsIntValue(int value, rsFiniteFieldElement targetTemplate) 
{ 
  return rsFiniteFieldElement(value, targetTemplate.getTables()); 
}
// needs test


}
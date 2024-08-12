namespace rema  // Rob's educational math algorithms
{

//-------------------------------------------------------------------------------------------------
// Helper functions
// ToDo: make them either static member functions of class rsFiniteFieldTables or generalize them
// suitably and move them into the library for polynomials - or maybe make them even more general
// to deal with arbitrary data types ...TBC...


/** Creates a std::vector of all possible polynomials over Zp (== integers modulo p) up to the
given maximum degree. These are all the possible remainders that can occur when any polynomial over
Zp is divided by a polynomial of degree k = maxDegree+1. These polynomials can be used to represent
elements of the finite field of size n = p^k where p is the (prime) modulus. */
std::vector<rsPolynomial<rsModularInteger<int>>> makeAllPolynomials(int modulus, int maxDegree)
{
  using ModInt = rsModularInteger<int>;
  using Poly   = rsPolynomial<ModInt>;

  int p = modulus;
  int k = maxDegree+1;
  int n = rsPow(p, k);

  // Helper function to increment a counter:
  auto inc = [](std::vector<int>& counter, int m)
  {
    int i = 0;
    while(i < (int)counter.size() && counter[i] >= m-1)
    {
      counter[i] = 0;
      i++;
    }
    if(i < (int)counter.size())
      counter[i] += 1;
  };
  // Maybe make this a library function. The number m is the wrap-around value where the digit 
  // wraps around to 0. In a decimal counter, this would be 10 and it would go up like:
  //   000 001 002 ... 008 009 010 011 012 ... 018 019 020 021 ... 900 901 902 ... 999 000
  // except that out counter has the digits reversed, i.e. the least important digit is leftmost

  // Helper function to create a polynomial over the modular integers with modulus m and given 
  // coefficients (as integers - they will be wrapped into the range 0..m-1 if they are beyond -
  // which they actually are not in this case):
  auto makePoly = [](const std::vector<int>& coeffs, int m)
  {
    int k = (int)coeffs.size();
    Poly poly(k-1);
    for(int i = 0; i < k; i++)
      poly.setCoeff(i, ModInt(coeffs[i], m));
    poly.truncateTrailingZeros(ModInt(0, m));
    return poly;
  };

  // Generate all the n = p^k possible polynomials of degree up to k-1 over Zp:
  std::vector<int>  counter(k);  // Content of counter is our polynomial coefficient array
  std::vector<Poly> polys(n);    // vector of polynomials
  for(int i = 0; i < n; i++)
  {
    polys[i] = makePoly(counter, p);
    inc(counter, p);
  }
  return polys;
}

rsMatrix<rsPolynomial<rsModularInteger<int>>> makeAddTable(
  const std::vector<rsPolynomial<rsModularInteger<int>>>& r,
  const rsPolynomial<rsModularInteger<int>>& m)
{
  // r: list of possible remainders
  // m: modulus polynomial

  using ModInt = rsModularInteger<int>;
  int p = m.getCoeff(0).getModulus();
  int n = (int)r.size();
  ModInt _0(0, p);
  rsMatrix<rsPolynomial<ModInt>> add(n, n);
  for(int i = 0; i < n; i++)
  {
    for(int j = 0; j < n; j++)
    {
      add(i, j) = (r[i] + r[j]) % m;       // Modulo m may be unnecessary
      add(i, j).truncateTrailingZeros(_0);
    }
  }
  return add;
}

rsMatrix<rsPolynomial<rsModularInteger<int>>> makeMulTable(
  const std::vector<rsPolynomial<rsModularInteger<int>>>& r,
  const rsPolynomial<rsModularInteger<int>>& m)
{
  // r: list of possible remainders
  // m: modulus polynomial

  using ModInt = rsModularInteger<int>;
  int p = m.getCoeff(0).getModulus();
  int n = (int)r.size();
  ModInt _0(0, p);
  rsMatrix<rsPolynomial<ModInt>> mul(n, n);
  for(int i = 0; i < n; i++)
  {
    for(int j = 0; j < n; j++)
    {
      mul(i, j) = (r[i] * r[j]) % m;
      mul(i, j).truncateTrailingZeros(_0);  // Truncation may be unnecessary
    }
  }
  return mul;
}

std::vector<rsPolynomial<rsModularInteger<int>>> makeNegTable(
  const std::vector<rsPolynomial<rsModularInteger<int>>>& r,
  const rsPolynomial<rsModularInteger<int>>& m)
{
  // r: list of possible remainders
  // m: modulus polynomial

  using ModInt = rsModularInteger<int>;
  using Poly   = rsPolynomial<ModInt>;
  int p = m.getCoeff(0).getModulus();
  int n = (int)r.size();
  ModInt _0(0, p);
  std::vector<Poly> neg(n);
  for(int i = 0; i < n; i++)
  {
    // Find additive inverse of r[i] and put it into neg[i]:
    for(int j = 0; j < n; j++)
    {
      Poly sum = (r[i] + r[j]) % m;        // Modulo m may be unnecessary
      sum.truncateTrailingZeros(_0);
      if(sum == r[0])
      {
        neg[i] = r[j];
        break;
      }
    }
  }
  return neg;
}

std::vector<rsPolynomial<rsModularInteger<int>>> makeRecTable(
  const std::vector<rsPolynomial<rsModularInteger<int>>>& r,
  const rsPolynomial<rsModularInteger<int>>& m)
{
  // r: list of possible remainders
  // m: modulus polynomial

  using ModInt = rsModularInteger<int>;
  using Poly   = rsPolynomial<ModInt>;
  int p = m.getCoeff(0).getModulus();
  int n = (int)r.size();
  ModInt _0(0, p);
  std::vector<Poly> rec(n);
  for(int i = 0; i < n; i++)
  {
    // Find multiplicative inverse of r[i] and put it into rec[i]:
    for(int j = 0; j < n; j++)
    {
      Poly prod = (r[i] * r[j]) % m;
      prod.truncateTrailingZeros(_0);      // Truncation may be unnecessary
      if(prod == r[1])
      {
        rec[i] = r[j];
        break;
      }
      else if(r[i] == _0)
      {
        // Mathematically, this makes no sense but we need to assign something well defined to the 
        // reciprocal of zero anyway, so we choose zero:
        rec[i] = _0;
        break;
      }
    }
  }
  return rec;
}

rsMatrix<rsPolynomial<rsModularInteger<int>>> makeSubTable(
  const std::vector<rsPolynomial<rsModularInteger<int>>>& r,
  const std::vector<rsPolynomial<rsModularInteger<int>>>& neg,
  const rsPolynomial<rsModularInteger<int>>& m)
{
  // r:   list of possible remainders
  // neg: list of additive inverses (aka negatives) of r
  // m:   modulus polynomial

  using ModInt = rsModularInteger<int>;
  int p = m.getCoeff(0).getModulus();
  int n = (int)r.size();
  ModInt _0(0, p);
  rsMatrix<rsPolynomial<ModInt>> sub(n, n);
  for(int i = 0; i < n; i++)
  {
    for(int j = 0; j < n; j++)
    {
      sub(i, j) = (r[i] + neg[j]) % m;     // Modulo m may be unnecessary
      sub(i, j).truncateTrailingZeros(_0);
    }
  }
  return sub;
}
// makeAddTable could perhaps be expressed as makeSubTable(r, r, n) to get rid of some code 
// duplication. But it should then be called makeAddTable but have two polynomial arrays as 
// parameters - one for the 1st and one for the 2nd operand. In general, the arrays could be of
// different size such that the table is not necessarily square - although we do not need that 
// here. These table creation methods could even be generalized and factored out. The should then
// get an operation passed as additional parameter - maybe as std::function op(p, q) where p,q
// are polynomials - and return a matrix of polynomials. These op-table generation methods could
// even be more general - the datatype does not need to be a polynomial but can be anything. But 
// for that to work, we need implicit automatic trunction of polynomials. Or do we? Maybe that 
// could be encapsulated in the op that we pass.


rsMatrix<rsPolynomial<rsModularInteger<int>>> makeDivTable(
  const std::vector<rsPolynomial<rsModularInteger<int>>>& r,
  const std::vector<rsPolynomial<rsModularInteger<int>>>& rec,
  const rsPolynomial<rsModularInteger<int>>& m)
{
  // r:   list of possible remainders
  // rec: list of multiplicative inverses (aka reciprocals) of r
  // m:   modulus polynomial

  using ModInt = rsModularInteger<int>;
  int p = m.getCoeff(0).getModulus();
  int n = (int)r.size();
  ModInt _0(0, p);
  rsMatrix<rsPolynomial<ModInt>> div(n, n);
  for(int i = 0; i < n; i++)
  {
    for(int j = 0; j < n; j++)
    {
      div(i, j) = (r[i] * rec[j]) % m;
      div(i, j).truncateTrailingZeros(_0);  // Truncation may be unnecessary
    }
  }
  return div;
}


// Under construction:
template<class TArg, class TOp>
rsMatrix<TArg> makeBinaryOpTable(
  const std::vector<TArg>& x, const std::vector<TArg>& y, const TOp& op)
{
  int M = (int)x.size();
  int N = (int)y.size();
  rsMatrix<TArg> table(M, N);
  for(int i = 0; i < M; i++)
    for(int j = 0; j < N; j++)
      table(i, j) = op(x[i], y[j]);
  return table;
}

std::vector<int> abstractifyTable1D(
  const std::vector<rsPolynomial<rsModularInteger<int>>>& x,
  const std::vector<rsPolynomial<rsModularInteger<int>>>& y)
{
  // x: 
  // y:

  int n = (int)x.size();
  rsAssert((int)y.size() == n);
  std::vector<int> t(n);
  for(int i = 0; i < n; i++)
    t[i] = rsFind(x, y[i]);
  return t;
}

rsMatrix<int> abstractifyTable2D(
  const std::vector<rsPolynomial<rsModularInteger<int>>>& x,
  const rsMatrix<rsPolynomial<rsModularInteger<int>>>& Y)
{
  // x: 
  // Y:

  int n = (int)x.size();
  rsAssert(Y.hasShape(n, n));
  rsMatrix<int> T(n, n);
  for(int i = 0; i < n; i++)
    for(int j = 0; j < n; j++)
      T(i, j) = rsFind(x, Y(i, j));
  return T;
}

//-------------------------------------------------------------------------------------------------
// class rsFiniteFieldTables

rsFiniteFieldTables::rsFiniteFieldTables(
  int base, int exponent, const std::vector<int>& modulusCoeffs)
  : p(base)
  , k(exponent)
  , mod(modulusCoeffs)
{
  n = rsPow(p, k);
  createOperationTables();
}

void rsFiniteFieldTables::createOperationTables()
{
  using ModInt = rsModularInteger<int>;
  using Poly   = rsPolynomial<ModInt>;
  using Table  = rsMatrix<Poly>;           // For operation tables for +,-,*,/
  using Array  = std::vector<Poly>;
  using VecI   = std::vector<int>;
  using MatI   = rsMatrix<int>;

  rsAssert((int) mod.size() == k+1);

  // Create the modulus polynomial:
  Poly m(k);
  for(int i = 0; i <= k; i++)
    m[i] = ModInt(mod[i], p);

  // Create the list of possible remainder polynomials:
  Array r = makeAllPolynomials(p, k-1);  // rename to r - for reaminders

  // Create the 1D operation tables for negation and reciprocation and the 2D operation tables for 
  // addition, multiplication, subtraction and division:
  Array tmp1D;
  Table tmp2D;
  tmp2D = makeAddTable(r,        m); add = abstractifyTable2D(r, tmp2D);
  tmp2D = makeMulTable(r,        m); mul = abstractifyTable2D(r, tmp2D);
  tmp1D = makeNegTable(r,        m); neg = abstractifyTable1D(r, tmp1D);
  tmp2D = makeSubTable(r, tmp1D, m); sub = abstractifyTable2D(r, tmp2D);
  tmp1D = makeRecTable(r,        m); rec = abstractifyTable1D(r, tmp1D);
  tmp2D = makeDivTable(r, tmp1D, m); div = abstractifyTable2D(r, tmp2D);



  
  // Under construction - we wnat to achieve the same with less code:
  ModInt _0(0, p);

  auto opMul = [&](const Poly& x, const Poly& y)
  {
    Poly r = (x * y) % m;
    r.truncateTrailingZeros(_0);  // Truncation may be unnecessary
    return r;
  };

  tmp2D = makeBinaryOpTable(r, r, opMul); mul = abstractifyTable2D(r, tmp2D);



  int dummy = 0;
}


}

//=================================================================================================
/*

Notes:

In the construction of the operation tables we have two levels of modular arithmetic going on: 
The lower level being the usage of modular integers and the higher level being the use doing all 
polynomial operations modulo the given modulus polynomial m = m(x).

In a non-naive implementation, we should build tables for addition and multiplication in the 
constructor. Each polynomial, i.e. each array of polynomial coeffs, maps to a unique integer
in the range 0...p^k-1. For each pair of such integers (mapped polynomials) we need to specify
what the result of their addition and multiplication should be - coming from the same set of 
0...p^k-1. The multiplication table can be turned into a 1D array rather than a full blown 2D
matrix by a trick explained in Weitz pg. 744. I hope, a similar trick is possible for addition
too. Weitz says nothing about that because he's only covering the case for p=2 in which addition
reduces to xor such that no table is needed. The method there uses a primitive k-th (?) root of 
unity, i.e. a number that, when multiplied by itself k times gives one. Maybe an analog for
addition could be a number that when added to itself k times gives zero? And that number would 
be just the number 1, regardless of the modulus? ...  figure this out!

What happens if we use a cartesian product of modular integers, i.e. what if we take e.g.
Z_7 x Z_7 x Z_7 with element-wise addition, multiplication and inversion? Will that produce
the finite field F_(7^3)? In Weitz's video, he gives a rather complicated construction in terms
polynomials over Z_7, so I suppose this (much simpler) approach will not work. But why not? 
What happens, if we try to do things like Z_2 x Z_3 x Z_5 etc.? Maybe first try to build the 
simplemost field F_4 as Z_2 x Z_2. Maybe this will reveal the problems. It could be implemented
using an array of rsModularInteger and defining the arithmetic operations between such arrays. 
But even if it doesn't produce the desired finite field - maybe it produces some other 
interesting algebraic structure? Ah - I think, we could have (0,1) * (1,0) = (0,0), So a product
could come out as zero even though none of the factors is zero, so it wouldn't be a field. We
would have zero divisors. Could we repair this somehow? Maybe by disallowing all elements that 
have some zero components and allow only those where all components are zero? But we cannot
practically "disallow" thoses because they may result as output of some arithmetic operation.
But maybe such a structure could still be useful enough, if the zero divisors are rare enough?


Maybe make classes rsFiniteGroup and rsFiniteRing that are based on addition- and multiplication 
tables that the user can pass in or that can be generated by suitable functions. API-wise, they
could work similar to the implementation of rsGeometricAlgebra. Maybe they could have functions
like isCommutative, getCenter, getCentralizer, getNormalizer, getNumSubgroups, getCosets, etc.
The group elements could be represented by integers (maybe unsigned, 64 bit). 




*/
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
  Array g2 = makeAllPolynomials(p, k-1);  // rename to r - for reaminders

  // Create the 1D operation tables for negation and reciprocation and the 2D operation tables for 
  // addition, multiplication, subtraction and division:
  Array tmp1D;
  Table tmp2D;
  tmp2D = makeAddTable(g2,        m); add = abstractifyTable2D(g2, tmp2D);
  tmp2D = makeMulTable(g2,        m); mul = abstractifyTable2D(g2, tmp2D);
  tmp1D = makeNegTable(g2,        m); neg = abstractifyTable1D(g2, tmp1D);
  tmp2D = makeSubTable(g2, tmp1D, m); sub = abstractifyTable2D(g2, tmp2D);
  tmp1D = makeRecTable(g2,        m); rec = abstractifyTable1D(g2, tmp1D);
  tmp2D = makeDivTable(g2, tmp1D, m); div = abstractifyTable2D(g2, tmp2D);

  int dummy = 0;
}


}
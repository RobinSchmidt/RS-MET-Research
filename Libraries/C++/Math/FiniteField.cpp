namespace rema  // Rob's educational math algorithms
{

//-------------------------------------------------------------------------------------------------
// Helper functions
// ToDo: make them either static member functions of class rsFiniteFieldTables or generalize them
// suitably and move them into the library for polynomials - or maybe make them even more general
// to deal with arbitrary data types. Maybe makeAllPolynomials could take as first argument a 
// std::vector<T> of all the possible values for the coefficients ...TBC...


// Creates a std::vector of all possible polynomials over Zp (== integers modulo p) up to the
// given maximum degree. These are all the possible remainders that can occur when any polynomial 
// over Zp is divided by a polynomial of degree k = maxDegree+1. These polynomials can be used to 
// represent elements of the finite field of size n = p^k where p is the (prime) modulus. 
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

// Given two arrays x,y of some type TArg and a binray operation op that takes two arguments of 
// type TArg and returns another TArg, this functions computes all the results of op(x[i], y[j])
// where i,j range over the lengths of the x,y arrays respectively and stores them into a matrix
// at position i,j.
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

// Given an arry of values of some type TArg and a predicate that takes two TArgs and returns true
// iff the second argument is the inverse of the first, this function produces the array of all the
// so defined inverses of x.
template<class TArg, class TPred>
std::vector<TArg> makeInversionTable(const std::vector<TArg>& x, const TPred& areInverses)
{
  int M = (int)x.size();
  std::vector<TArg> inv(M);
  for(int i = 0; i < M; i++)
    for(int j = 0; j < M; j++)
      if(areInverses(x[i], x[j]))
        inv[i] = x[j];
  return inv;
}

// Given two vectors x,y of elements of some datatype and assuming that y contains some permutation
// of the elements of x, this function produces the corresponding vector of permuted indices. The 
// function can be used to convert a concrete permutation of some number of objects into an 
// abstract permutaion in which each object is replaced by its index.
template<class T>
std::vector<int> abstractifyTable1D(const std::vector<T>& x, const std::vector<T>& y)
{
  int n = (int)x.size();
  rsAssert((int)y.size() == n);
  std::vector<int> t(n);
  for(int i = 0; i < n; i++)
    t[i] = rsFind(x, y[i]);
  return t;
}

// Given two vectors x and a matrix Y of elements of some datatype and assuming that the matrix 
// describes abinary operation table between elements of x, this function produces the corresponding
// abstract operation table where each element is replaced by it's corresponding index. This can be 
// used to convert an operations table that works on concrete objects of some datatype into an 
// abstract operation table that works on the indices.
template<class T>
rsMatrix<int> abstractifyTable2D(const std::vector<T>& x, const rsMatrix<T>& Y)
{
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
  rsAssert((int) mod.size() == k+1, "Modulus polynomial has wrong degree!");
  // The passed modulus polynomial does not fit together with the desired size of the field. For a
  // field of size p^k, the modulus polynomial must have a degree of k, i.e. have k+1 coefficients.

  // Some shorthands for convenience:
  using ModInt = rsModularInteger<int>;
  using Poly   = rsPolynomial<ModInt>;
  using Table  = rsMatrix<Poly>;            // Maybe use Table1D
  using Array  = std::vector<Poly>;         // Maybe use Table2D
  using VecI   = std::vector<int>;
  using MatI   = rsMatrix<int>;

  // Create the modulus polynomial:
  Poly m(k);
  for(int i = 0; i <= k; i++)
    m[i] = ModInt(mod[i], p);

  // Create the list of possible remainder polynomials:
  Array r = makeAllPolynomials(p, k-1);

  // Zero and one as modular integers with modulus p:
  ModInt _0(0, p);
  ModInt _1(1, p);

  // Predicate that returns true iff x and y are additive inverses, i.e. y is the negation of x:
  auto predNeg = [&](const Poly& x, const Poly& y)
  {
    Poly r = (x + y) % m;         // Modulo m may be unnecessary
    r.truncateTrailingZeros(_0);
    return r == _0;

    // I think, the comparison of the Poly r to the ModInt _0 works because _0 can be implictly 
    // converted to the constant polynomial with 0 as constant coeff. ...figure out!
  };

  // Predicate that returns true iff x and y are multiplicative inverses, i.e. y is the reciprocal of x:
  auto predRec = [&](const Poly& x, const Poly& y)
  {
    Poly r = (x * y) % m;         // Modulo m may be unnecessary
    r.truncateTrailingZeros(_0);
    if(x == _0)
      return y == _0;
    else
      return r == _1;

    // If x == 0, there is no reciprocal but for technical reasons, it makes sense to take 0 
    // as the reciprocal of itself in the table of reciprocals.
  };

  // Operation that performs modular addition of two polynomials:
  auto opAdd = [&](const Poly& x, const Poly& y)
  {
    Poly r = (x + y) % m;         // Modulo m may be unnecessary
    r.truncateTrailingZeros(_0);
    return r;
  };

  // Operation that performs modular multiplication of two polynomials:
  auto opMul = [&](const Poly& x, const Poly& y)
  {
    Poly r = (x * y) % m;
    r.truncateTrailingZeros(_0);  // Truncation may be unnecessary
    return r;
  };

  // Create the 1D operation tables for negation and reciprocation and the 2D operation tables for 
  // addition, multiplication, subtraction and division. We always first create the concrete 
  // polynomial representation of the table in a temporary local variable and then convert it into 
  // an abstract representation that works over the indices and store that in a member variable:
  Array t1;  // Temporary 1D table of polynomials
  Table t2;  // Temporary 2D table of polynomials
  t2 = makeBinaryOpTable( r, r,  opAdd  ); add = abstractifyTable2D(r, t2); // Addition
  t2 = makeBinaryOpTable( r, r,  opMul  ); mul = abstractifyTable2D(r, t2); // Multiplication
  t1 = makeInversionTable(r,     predNeg); neg = abstractifyTable1D(r, t1); // Negation
  t2 = makeBinaryOpTable( r, t1, opAdd  ); sub = abstractifyTable2D(r, t2); // Subtraction
  t1 = makeInversionTable(r,     predRec); rec = abstractifyTable1D(r, t1); // Reciprocation
  t2 = makeBinaryOpTable( r, t1, opMul  ); div = abstractifyTable2D(r, t2); // Division


  // ToDo: Maybe automatically make a self-test in debug builds like:
  //
  // rsAssert(isField()); 
  //
  // where isField() should check if all the field axioms are satiesfied with the given 
  // operation tables. It may fail when the user passes a reducible polynomial to the constructor 
  // by accident. We currently only check, if the polynomial has the right degree but not if its 
  // actually irreducible. I guess, such a check would be rather complicated anyway (figure out!). 
  // But what we can do with reasonable effort is checking if our resulting operation tables 
  // satisfy the field axioms. See testFiniteField() in Experiments.cpp - there we do precisely 
  // this. Well - it's O(n^3) because we need to iterate over all possible triples of elements for
  // checking associativity and distributivity. So, it's not exactly cheap, though.
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



Questions

- Do we have a "successor" operation that enumerates all elements such as the "+1" operation
  in the natural numbers or in the modular integers? I think taking literally "+1" will not work
  here because it will only cycle between p rather than p^k elements (I think). Maybe the literal
  +1 will partition the field into p^(k-1) equivalence classes. Or more generally, any element 
  that arose from a polynomial that hass all coeffs 0 except one which is 1, will do? Maybe the 
  coeff doesn't even need to be 1 but any nonzero coeff will work? Of course, the operation should
  be expressed in terms of field operations - incrementing the array index does not count!

- We know that the fields of size p^k resulting from different modulus polynomials are isomorphic.
  How can we find an isomorphism? An isomorphism can be represent by a permutation of the numbers
  0..n-1, i.e. a simple length-n array.

- Can we always find a homomorphism from GF(p^k) to GF(p^(k-1)) and therefore, by induction, to
  GF(p)? Try to write a function that produces such homomorphisms.

- Maybe one way to approach this problem fo finding isomorphisms is to take some sort of 
  "fingerprint" of each field element x and then find the element with the same fingerprint in the
  other set. The fingerprint could consist of features like: additive order, multliplicative order,
  mapped element in GF(p) in the homomorphism mentioned above (if that works out), orders of other 
  operations like (a*a + a), (a^3 + a), (a^3 + a^2), etc. Additive and multiplciative order are 
  just special cases where we use (a + a), (a * a). When the fingerprint is taken, look in the 
  other isomorphic set for the element that has the same fingerprint. That's the function value 
  f(x) of our isomorphism - or at least, it is a possible candidate for the function value.

- What set of features will give us a fingerprint that lets us uniquely determine the desired 
  function value f(x) for a given field element x? Or maybe unary features are not good enough and 
  we need to look into binray features, i.e. features of all possible pairs of elements? Maybe we 
  need to consider for each element its reciprocal and the structure by which elements are related
  to their reciprocals gives use enough information to find the isomorphism?

- Can we find a formula for the additive order of an element in terms of its polynomial coeffs?
  They give a pattern like  1 0 2 0 1 3 1. When we modularly add this pattern to itself - how long
  will it take until we are back to the original pattern?

- A naive approach to fin an isomorphsim is to try all possible permutations and check, if they are
  and isomorphism. But that's inefficient because the number of possible permutations is n!. This 
  can be done practically only in very small cases, i.e. for very small n - maybe up to n = 9. We 
  could get away with trying (n-2)! permutations because we already know that 0 and 1 must map to
  0 and 1 - but that does not help much. 

- Can it help to take the degree of the polynomial in the polynomial representation as feature?

- How about just picking one element x of the 1st representation (let's call it X) of our Galois 
  field and map it to some element y of the 2nd representation Y. Then compute x^2 and y^2 and map
  those to each other. Then compute x^3 and y^3 and map them to each other. Will that give use the
  isomorphism? ...or at least bring us closer to it - and may we can adjust it bit more to get an 
  actual isomorphism?



*/
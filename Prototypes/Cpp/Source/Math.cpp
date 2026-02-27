
// ToDo:
//
// - Move code from Tools.cpp to here - but not all math related code should go into this file. We 
//   want to have some finer granularity. For example, all the geometric algebra related stuff 
//   should get its own file, etc.


//=================================================================================================

/** Under Construction. Just a stub at the moment

A class for polynomial roots finding. It implements various methods such as the Laguerre method,
Jenkis-Traub, etc. as well as the closed form formulas for polynomials where such formuals exist,
namely polynomials of degrees 1..4. ...TBC...

*/

//template<class TCoef, class TArg>   // Maybe rename TArg to TRoot ...or maybe not
class rsPolynomialRootFinder
{

public:



   
  /** Converges to a root near "initialGuess" via the Laguerre algorithm. ...TBC...  */
  template<class TCoef, class TArg, class TTol>
  static TArg convergeLaguerre(const TCoef *a, int degree, TArg initialGuess, TTol tol, 
    int maxNumIterations);
  // Taken from RAPT::rsPolynomial::convergeToRootViaLaguerre and adapted. ToDo:
  // Maybe add parameters for tolerance (of type TTol) and maximum number of iteration (of type
  // (unsigned) int). Look upt how boost does it and be consistent with it! Also, check if boost
  // prefers to use degree or length parameters. The length is always degree+1. That could be 
  // dangerous, if the client assumes to pass a length because this may lead to acces violations.
  // The other way around is less dangerous. If the user assumes to pass a degree and we falsely 
  // interpret it as length, we will only potentially produce wrong results but will not cause an
  // access violation - so maybe it's better to do it this way. It would also be more consistent
  // with functions from rsArrayTools.



protected:


};


template<class TCoef, class TArg, class TTol>
TArg rsPolynomialRootFinder::convergeLaguerre(
  const TCoef *a, int degree, TArg initialGuess, TTol tol, int maxNumIterations)
{
  using namespace std;  // Preliminary

  const TTol eps = tol;  // Get rid - use tol directly!

  static const int numFractions      = 8;   // number of fractions minus 1 (for breaking limit cycles)
  static const int itsBeforeFracStep = 10;  // number of iterations after which a fractional step
                                            // is taken (to break limit cycles)

  // fractions for taking fractional update steps to break a limit cycles:
  static TCoef fractions[numFractions+1] =
    { TCoef(0.0),  TCoef(0.5),  TCoef(0.25), TCoef(0.75), TCoef(0.13), 
      TCoef(0.38), TCoef(0.62), TCoef(0.88), TCoef(1.0) };
  // Do they intend the numbers to be multiples of 0.125 but rounded to two decimal digits?
  // Like 0.0, 0.5, 0.25, 0.75, 0.125, 0.375, 0.625, 0.875, 1.0? But why would they round so 
  // crudely? ...why the +1? For convenience?

  TArg P[3];  // Holds P, P', P''


  // Helper function:
  // (copied from rsPolynomial<T>::evaluateWithTwoDerivativesAndError())
  auto evalP = [&](TArg z)
  {
    TCoef zeroR = rsZeroValue(real(a[0]));      // Real zero - maybe use rsReal(a[0])
    TArg  zeroC = rsZeroValue(a[0]);            // Complex zero

    P[0] = a[degree];                           // P(z)
    P[1] = zeroC;                               // P'(z)
    P[2] = zeroC;                               // P''(z)

    TTol err = rsAbs(P[0]);                     // Estimated roundoff error in evaluation
    TCoef zA = rsAbs(z);                        // Absolute value of z

    for(int j = degree-1; j >= 0; j--) 
    {
      P[2] = z * P[2] + P[1];
      P[1] = z * P[1] + P[0];
      P[0] = z * P[0] + a[j];
      err  = abs(P[0]) + zA*err;
    }

    P[2] *= rsIntValue(2, zeroR);               // P[2] *= 2
    return err;
  };

  return initialGuess;  // Preliminary

  // ToDo:
  //
  // - Copy over code from  rsPolynomial<T>::convergeToRootViaLaguerre()
  //   We first need to implement a function like
  //   rsPolynomial<T>::evaluateWithTwoDerivativesAndError()
  //   It should return something that we can compare against a tolerance type TTol, maybe the type
  //   should be called TErr in this context for error type. Maybe we should implement it as 
  //   internal helper function with a lambda function definition. We may not want to expose that
  //   function to the outside world.
}

// ToDo:
// 
// - Maybe do not let the class have template parameters, instead let the functions themselves
//   have these template params. Mabye it then makes more sense to not implement them as static
//   member functions of a class but rather as free functions in a namespace. Look up how boost
//   root finders do it and maybe follow the same pattern. See:
//   https://www.boost.org/doc/libs/1_64_0/libs/math/doc/html/rooting.html
//   These root finding algorithms are free functions in the namespace boost::math::tools.
// 
// - Maybe try to use C++20 concepts to put constraints on the template parameters TCoef, TArg.
//   For many algorithms, the concepts should express that TCoef is a real number type and TArg
//   should be either a real or the corresponding complex number type. In general TArg should 
//   often be the algebraic closure of TCoef. There may be different situation, though - for 
//   example TCoeff being scalars and TArg matrices. But I don't think, we'll have to deal with 
//   these cases in the context of roots finding
//
// - We want to factor out anything that has to do with polynomial root finding from the class
//   rsPolynomial in the RAPT library. This includes all the stuff in the "Roots (Low Level)" 
//   section. Somewhere in the experiments or prototypes, there must also already be some code 
//   where I attempted to implement the formulas for cubic (and maybe even quartic?) roots. That
//   stuff should also go into this class. As long as this class is not mature enough to integrate
//   into RAPT, let's just keep the class rsPolynomial as is and duplicate the root finding code 
//   here. It can be deleted later when this class is finished.
// 
// - The low-level API should operate on pre-allocated arrays so it can be used in real-time. We 
//   should perhaps also provide some means to limit the amount of computations with parameters 
//   like maxNumIterations, etc. The goal is to make the class realtime-safe. That means: No 
//   operations that can take an unbounded amount of time to complete. We may provie a more 
//   convenience oriented high level API on top of that.
//
// - The private repo has a textfile Documents/MathNotes/CubicAndQuarticEquations.txt where I have
//   noted some formulas, derivations, links about the closed form formulas. Look that up for help
//   for implementing them. Also, this video is very good:
//   https://www.youtube.com/watch?v=o8UNhs2OaG8  How to Solve ANY Cubic or Quartic Equation!
//
// - Maybe try to implement this class with the help from Copilot. It seems to be a suitable task
//   to tackle with AI assistance because it has a well defined and managable scope without being
//   too trivial. The AI probably knows the algorithms better than what I could figure out.
//
// - Try to avoid a direct dependency on RAPT::rsPolynomial. Maybe it should work on raw C-arrays
//   of coeffs and roots. It should also work well together with boost::polynomial:
//   https://www.boost.org/doc/libs/1_64_0/libs/math/doc/html/math_toolkit/roots/polynomials.html
//   Maybe we should have a convenience class that works directly with boost::polynomial and a raw
//   number crunching class that works on raw arrays. The webpage does not show the full 
//   implementation. To study that, see the file here in this repo:
//   RS-MET-Research/Libraries/C++/JuceModules/rs_boost/boost/math/tools/polynomial.hpp
//   It has a  std::vector<T> m_data;  and it stores the coeffs in ascending order just like
//   rsPolynomial does. 
// 
// - For the roots, it should be convenient to use TArg = std::complex<TCoef> when TCoef is float 
//   or double or long double but it should also be possible to use selfmade complex number classes
//   such as RAPT::rsComplex. Maybe, the low level API can assume that arrays of complex numbers 
//   can be interpreted as arrays of adjacent pairs of real numbers as std::complex defines it:
//   https://en.cppreference.com/w/cpp/numeric/complex.html
//   This would put some constraints on hwo the complex number class has to be implemented, though.
//   But the constraints seem to be reasonable. It's quite common to store arrays complex numbers 
//   that way. For example, many FFT libraries assume that a length N array of complex numbers is
//   stored as a length 2*N array of real numbers (i.e. of type float or double). 
// 
// - It would, however, be a nice to have if we can avoid making any assumptions about the 
//   representation and memory layout of complex numbers. If we really try to avoid that 
//   constraint, then maybe test it with a complex number class that has real and imaginary part 
//   swapped or stores the numbers in polar form.
//
// - Assume that "Real" stands for float, double or long double. Then, the most important use cases
//   for the root finder are as follows (most important on top):
// 
//     - TCoeff = std::complex<Real>, TArg = std::Complex<Real>
//     - TCoeff = Real, TArg = std::Complex<Real>
//     - TCoeff = Real, TArg = Real
// 
// - It should also be easy to replace std::complex with rsComplex or some other implementation of
//   complex numbers. In these other implementations, it should be possible use use for Real 
//   something like rsFloat32x2, rsFloat64x2, rsSimsVector<float, 4>, std::simd, see:
//   https://en.cppreference.com/w/cpp/experimental/simd.html
// 
// - It should also be possible to use it with extended precision libraries for the Real type.
//
// - It would be nice if the class would be general enough such that it could also handle 
//   polynomials with coefficients and/or roots that are: rational, integer, from a finite field
//   (aka Galois field). It would also be nice if we could use matrices for the coefficients and/or
//   the arguments. Maybe multivectors?
//
// - The problem is: Depending of the data types for TCoef and TArg, we may have to use completely
//   different algorithms for finding the roots. In a finite field, if it isn't too big, we could
//   just try all possible values. ToDo: Figure out if there is a better algorithm to find such 
//   roots. For finding real and rational roots, we could perhaps use the algorithms to find real
//   roots and then use a round-and-verify step. For "rounding" real number to rational numbers,
//   maybe we could use continued fraction expansions or Farey sequences (the latter was 
//   auto-completed by GitHub Copilot - I have no idea, if this makes sense). 
//
// - The root finding for discrete number types (integer, rational, Galois) need not to be 
//   implemented right from the start but the API should designed in such a way that these things 
//   can easily be added later in a way that feels consistent with the root-finding for real and 
//   complex numbers. We may need to think about extension fields in this context. In general, the
//   roots (i.e. TArg) will be an extension field of TCoef. For example, the complex numbers are an
//   extension field of the real numbers.
//
// - Maybe it could make sense to have different classes of root finders for the various possible
//   scenarios. For example: rsPolyRootFinderReal, rsPolyRootFinderComplex, 
//   rsPolyRootFinderInteger, rsPolyRootFinderRational, rsPolyRootFinderFinite, ....
//
// Other links that may or may not be useful:
//
// https://gitlab.liris.cnrs.fr/abatel/cornac/-/blob/54e870dc1dc7a9a06b2713f451c31ab95f562d06/cornac/utils/external/boost/math/tools/polynomial.hpp






//=================================================================================================

/** A class to represent multivariate monomials. It's purpose is to be the basic building block for
multivariate polynomials, i.e. represents a term is such a polynomial. ...TBC... */

template<class T>
class rsMultiVarMonomial
{

public:


  //-----------------------------------------------------------------------------------------------
  /** \name Lifetime.  */

  /** Default constructor. Constructs an empty term. An empty term is actually rather useless so 
  using this constructor should be accompanied by a call to setup at some point presumably rather 
  soon after construction. */
  rsMultiVarMonomial() {}

  /** Constructor that initializes our members to the given new coefficient and vector or powers.
  For example, calling it like:

    rsMultiVarMonomial t(5.0, std::vector<int>(2,3,1);

  would create the monimial term  t = 5.0 * x1^2 * x2^3 * x3^1  or  5.0 * x^2 * y^3 * z^1  if we 
  would use x,y,z as variable names in a trivariate polynomial. */
  rsMultiVarMonomial(const T& newCoeff, const std::vector<int>& newPowers)
  {
    setup(newCoeff, newPowers);
  }
  // ToDo: Provide a similar constructor that takes an initializer list for a more ergonomic API.


  //-----------------------------------------------------------------------------------------------
  /** \name Setup.  */

  void setup(const T& newCoeff, const std::vector<int>& newPowers)
  {
    coeff  = newCoeff;
    powers = newPowers;
  }

  void setCoeff(const T& newCoeff) { coeff = newCoeff; }

  void shiftCoeff(const T& amount) { coeff += amount; }

  // ToDo: setPowers(), negate(), ... see rsMonomial and implement analoguous functions 
  // here.



  //-----------------------------------------------------------------------------------------------
  /** \name Inquiry.  */

  /** Compares lhs (left hand side) and rhs (right hand side) lexicographically and returns a 
  number a number < 0 when lhs < rhs, a number > 0 when lhs > rhs and 0 when lhs == rhs. It thus 
  behaves like the good old C function strcmp(). By lexicographic, we mean that we interpret 
  monomial like x^2 * y^3 * z^1 as a string xxyyyz and we will put these strings into the order as
  they would be found in a dictionary. The coefficient plays no role in the comparison, so
  3 * x^2 * y^3 * z^1  and  5 * x^2 * y^3 * z^1  would be considered to be the same with respect to 
  that order. */
  static int compLexic(const rsMultiVarMonomial<T>& lhs, const rsMultiVarMonomial<T>& rhs);
  // Note: I think even though terms with the same powers but different coeffs are considered 
  // equivalent (neither < nor >), I think, we still establish a total order because indeed either 
  // a < b or b < a or a = b _if_ we interpret the = not as exact equality of terms but as an 
  // equivalence relation on the terms that _also_ ignores the coeff, which we probably should 
  // indeed do in this context for consistency. Verify and maybe add to the documentation.

  /** Returns true if the monomial lhs (left hand side) comes lexicographically before the monomial
  rhs (right hand side). So it implements the "lhs < rhs" operation where the lass-than relation is
  understood to be meant lexicographically. Such a comparison operation is needed in order to 
  establish a well defined order of terms in a multivariate polynomial such that we can define what
  it means for such a polynomial to be in a canonical representation because a canonical 
  representation requires us to sort the terms according to a well defined rule. */
  static bool lessLexic(const rsMultiVarMonomial& lhs, const rsMultiVarMonomial& rhs)
  { return compLexic(lhs, rhs) < 0; }

  /** Returns true, iff the "other" term has the same powers as "this". */
  bool hasSamePowersAs(const rsMultiVarMonomial& other) const
  { return powers == other.powers; }

  /** Returns a const reference to the coefficient. */
  const T& getCoeff() const { return coeff; }

  /** Returns a const reference to the vector of powers. */
  const std::vector<int>& getPowers() const { return powers; }

  /** Returns the power (aka exponent) for the variable given by varIndex where index counting 
  starts at zero. So, for example, in a trivariate polynomial p = p(x,y,z), the exponent for x 
  would correspond to varIndex = 0, the exponent for y to varIndex = 1 and the exponent for z to 
  varIndex = 2. */
  int getPower(int varIndex) const { checkVarIndex(varIndex); return powers[varIndex];  }
  // Needs tests. 


  template<class TTol>
  bool isCoeffZero(TTol tol) const
  {
    return rsIsNegligible(coeff, tol);
    //return rsMaxNorm(coeff) <= tol;
  }
  // Needs tests for T = RAPT::rsComplex, std::complex and maybe with more complicated types T

  /** Returns true iff one or more of the powers is negative. */
  bool hasNegativePowers() const;

  /** Returns the number of variables in this term. */
  int getNumVariables() const { return (int) powers.size(); }

  /** Returns the total degree of this term which is defined to be the sum of all the powers. */
  int getTotalDegree() const { return rsSum(powers); }

  /** Returns true, iff the given index i is a valid variable index. It must be >= 0 and less than 
  the number of variables. For example, in a trivariate monomial term t(x,y,z), valid indices would
  be 0,1,2 where 0 is the index corresponding to the variable x and so on. */
  bool isValidVariableIndex(int i) const { return i >= 0 && i < getNumVariables(); }

  /** Checks, if the given i is a valid variable index and triggers a debug assertion otherwise. 
  For debugging purposes only. Calls to it are supposed to be optimized away in realease builds 
  but that really depends on the implementation of rsAssert(). */
  void checkVarIndex(int i) const 
  { rsAssert(isValidVariableIndex(i), "Invalid variable index!"); }



  // ToDo: getMultiDegree() = max(powers) I think. See IVA, pg...

  //-----------------------------------------------------------------------------------------------
  /** \name Operators. */

  template<class TArg>
  TArg operator()(const std::vector<TArg>& x) const 
  { 
    rsAssert(x.size() == powers.size(), "Input vector x has wrong dimension.");
    TArg product(1);
    for(size_t i = 0; i < x.size(); i++)
      product *= rsPow(x[i], TArg(powers[i]));
    return TArg(coeff) * product;
  }
  // Maybe move implementation out of the class.

  /** Returns the negative of this monomial. */
  rsMultiVarMonomial<T> operator-() const { return rsMultiVarMonomial<T>(-coeff, powers); }

  /** Multiplies two multivariate monomials. */
  rsMultiVarMonomial<T> operator*(const rsMultiVarMonomial<T>& q) const 
  { return rsMultiVarMonomial<T>(coeff * q.coeff, powers + q.powers); }

  // ToDo: Maybe implement division as:
  // "return rsMultiVarMonomial<T>(coeff / q.coeff, powers - q.powers);
  // But this may in general produce terms that have negative powers and we also may get divisions
  // by zero.


protected:

  T coeff = T(0);
  std::vector<int> powers;

};

template<class T>
int rsMultiVarMonomial<T>::compLexic(
  const rsMultiVarMonomial<T>& lhs, const rsMultiVarMonomial<T>& rhs)
{
  rsAssert(lhs.getNumVariables() == rhs.getNumVariables(), "lhs and rhs are incompatible");
  for(size_t i = 0; i < lhs.getNumVariables(); i++)
  {
    int d = rhs.getPower(i) - lhs.getPower(i);
    if(d < 0)
      return -1;
    if(d > 0)
      return +1;
  }
  return 0;
}

template<class T>
bool rsMultiVarMonomial<T>::hasNegativePowers() const
{
  for(auto& p : powers)
    if(p < 0)
      return true;

  return false;
}



//=================================================================================================

/** Under construction.

A class to represent multivariate polynomials. The implementation follows the one of 
rsSparsePolynomial. We do not have the "Sparse" in the class name though because I do not intend to
make a different implementation for the dense case because such a dense implementation would then
probably have to look like a generalization of rsBiVariate.. and rsTriVariatePolynomial and that 
would presumably be horribly messy to implement (requiring convolutions of arbitrary dimensionality
and stuff) and I'm not convinced that this would be a good idea to do it like that! This class here
should be used for the general case, i.e. for sparse and dense multivariate polynomials.

References:

  - (IVA)  Ideals, Varieties, and Algorithms  by ...

...TBC... */

template<class T, class TTol = rsEmptyType>
class rsMultiVarPolynomial
{

public:


  using MultiPoly = rsMultiVarPolynomial<T, TTol>;  // For convenience


  //-----------------------------------------------------------------------------------------------
  /** \name Lifetime.  */

  /** Constructor. You may pass the number of variables that this polynomial expects. If you pass
  nothing, that value will default to 1. That means that by default, we'll get a polynomial in just
  a single variable. You can change that later by calling init() but such a call to init will also
  clear the array of terms because changing the number of variables may make our existing array of 
  terms incompatible with the new setting. */
  rsMultiVarPolynomial(int numVariables = 1) { init(numVariables); }


  //-----------------------------------------------------------------------------------------------
  /** \name Setup.  */


  /** Clears the array of terms. */
  void clear() { terms.clear(); }

  /** Reserves memory for the given number of terms. Can be called before calling functions like 
  addTerm() to pre-allocate the desired amount of memory beforehand when multiple terms are being
  added in a sequence. */
  void reserve(size_t numTerms) { terms.reserve(numTerms); }
  // Maybe we should loop through the terms and call terms[i].reserve() for each. Each term is an
  // object of type rsMultiVarMonomial which itself contains a std::vector for the powers. Then 
  // maybe we should move the implementation out of the class

  /** Initializes this polynomial. You need to pass the number of variables that this polynomial
  expects as inputs. For example, for a trivariate polynomial p = p(x,y,z), that number would be
  3. This will also clear our existing array of terms. This is required because changing the number
  of variables may make our existing array of terms incompatible with the new setting. A 
  re-initialization should be understood to be a disruptive operation that wipes out the state. */
  void init(int newNumVariables) { numVars = newNumVariables; clear(); }

  /** Initializes this polynomial as the zero polynomial that is compatible with the given "other"
  polynomial where compatibility is defined in the sense of the isCompatibleWith() function. */
  void initLike(const MultiPoly& other)
  {
    numVars = other.numVars;
    //termLess = other.termLess;   // Add this later
    clear();
  }
  // Maybe rename to initCompatible() or initCompatibleWith()


  /** Adds the given monomial to the polynomial. */
  void addTerm(const rsMultiVarMonomial<T>& newTerm);

  /** Convenience function that can be used to add terms via the syntax  p.addTerm(1.5,{4,2,3});  
  to add a term like  1.5 * x^4 * y^2 * z^3. */
  void addTerm(const T& newCoeff, const std::vector<int>& newPowers)
  { addTerm(rsMultiVarMonomial<T>(newCoeff, newPowers)); }


  // ToDo: setRoundoffTolerance(), subtractTerm(), scale(), negate(), 
  // But in rsSparsePolynomial, we implement scaling as a low-level method _scaleCoeffs() because
  // the scaler could be zero. Maybe a high-level scale() method without underscore should allow
  // only nonzero scalers. We could assert that the scaler is zero and then call _scaleCoeffs().


  //-----------------------------------------------------------------------------------------------
  /** \name Inquiry.  */

  /** Returns the number of variables, i.e. the dimensionality of the input x. */
  int getNumVariables() const { return numVars; }

  /** Returns the number of terms in this polynomial. */
  int getNumTerms() const { return (int) terms.size(); }

  /** Returns true iff this polynomial is the zero polynomial. */
  bool isZero() const { rsAssert(_isCanonical()); return terms.empty(); }
  
  //const std::vector<int>& getPowers(int termIndex) 
  //{ return getTerm(termIndex).getPowers(); }
  // Needs test


  /** Returns the leading term in this polynomial. */
  const rsMultiVarMonomial<T>& getLeadingTerm() const
  {
    rsAssert(_isCanonical());
    if(terms.empty())
      return zeroMonomial;
    return terms[terms.size()-1];
  }

  const rsMultiVarMonomial<T>& getTerm(int termIndex) const
  {
    checkTermIndex(termIndex);
    return terms[termIndex];
  }
  // Needs test

  /** Returns the coefficient of the term with given "termIndex". */
  const T& getCoeff(int termIndex) const 
  { 
    checkTermIndex(termIndex);
    return terms[termIndex].getCoeff();
  }

  /** Returns a const reference to the vector of powers at the given "termIndex". */
  const std::vector<int>& getPowers(int termIndex) const 
  {
    checkTermIndex(termIndex);
    return terms[termIndex].getPowers();
  }

    
  bool isValidTermIndex(int i) const { return i >= 0 && i < getNumTerms(); }

  void checkTermIndex(int i) const 
  { rsAssert(isValidTermIndex(i), "Invalid term index!"); }

  /** Checks whether the given "other" polynomial is compatible with this polynomial in the sense
  that it accepts the same number of arguments (i.e. has the same "numVars" member) 
  [ToDo: ...and uses the same monomial order (i.e. uses the same termLess function)].  */
  bool isCompatibleWith(const MultiPoly& other) const
  {
    return numVars == other.numVars; // ToDo: && termLess == other.termLess;
  }

  //-----------------------------------------------------------------------------------------------
  /** \name Operators. */

  template<class TArg>
  TArg operator()(const std::vector<TArg>& x) const
  { 
    rsAssert((int)x.size() == numVars, "Input vector x has wrong dimension.");
    TArg y(0);
    for(auto& t_i : terms)
      y += t_i(x);
    return y;
  }

  /** Adds two multivariate polynomials. */
  MultiPoly operator+(const MultiPoly& q) const 
  { MultiPoly r; add(*this, q, &r); return r; }

  /** Subtracts two multivariate polynomials. */
  MultiPoly operator-(const MultiPoly& q) const 
  { MultiPoly r; subtract(*this, q, &r); return r; }

  /** Multiplies two multivariate polynomials. */
  MultiPoly operator*(const MultiPoly& q) const 
  { MultiPoly r; multiply(*this, q, &r); return r; }


  //-----------------------------------------------------------------------------------------------
  /** \name Low level API.  */
    
  /** Computes the sum r = p + q of the polynomials p and q and stores the result in r. */
  static void add(const MultiPoly& p, const MultiPoly& q, MultiPoly* r);
  // ToDo: Document whether or not it can be used in place.

  /** Computes the difference r = p - q of the polynomials p and q and stores the result in r. */
  static void subtract(const MultiPoly& p, const MultiPoly& q, MultiPoly* r);

  /** Computes the weighted sum r = wp * p + wq * q of the polynomials p and q and stores the 
  result in r. */
  static void weightedSum(
    const MultiPoly& p, const T& wp, const MultiPoly& q, const T& wq, MultiPoly* r);

  /** Multiplies polynomials p and q and stores the result in r. It may be used in place, i.e. the
  result polynomial r can point to the memory location of the arguments p and/or q. */
  static void multiply(const MultiPoly& p, const MultiPoly& q, MultiPoly* r);


  static void divide(const MultiPoly& dividend, const std::vector<MultiPoly>& divisors,
  std::vector<MultiPoly>* quotients, MultiPoly* remainder);
  // Maybe instead of std::vectors, use C-style arrays. We would need an additional "numDivisors" 
  // parameter. Maybe it should be the 3rd parameter, directly after the "divisors" parameter. The 
  // number of quotients always equals the number of divisors and there is always a single 
  // remainder.



  /** Turns the representation of the multivariate polynomial into a canonical one. A canonical 
  representation has the following properties: (1) The terms in our array are sorted 
  lexicographically. (2) No configuration of powers appears more than once. (3) No zero 
  coefficients appear. We achieve this by first sorting the terms, then combining multiple terms
  with equal exponent configurations into single terms and finally deleting all terms that have a
  coefficient zero (up to the roundoff tolerance). */
  void _canonicalize();

  /** Removes all the terms that have a coefficient of zero (up to the roundofff tolerance). */
  void _removeTermsWithZeroCoeff();

  /** Sets the number of terms. If the new number is less than the current number, it will just 
  cut off terms from the end. If the new number is greater than the current number, it will just
  extend our vector of terms and the added terms at the end are uninitialized, i.e. may contain 
  garbage. This function should only be used if you intend to set up the new terms via e.g. 
  _setTerm() after calling _setNumTerms(). So, it's a function that needs a lot of care to be used
  properly. */
  void _setNumTerms(int newNumTerms) { terms.resize(newNumTerms); }

  /** Directly sets the coefficient and powers of the term with given index. This may 
  decanonicalize the representation in all sorts of ways: by destroying the order of terms, by 
  destroying the "powers appear at most once" property, by setting a coeff to zero. */
  void _setTerm(int index, T coeff, const std::vector<int>& powers) 
  { 
    checkTermIndex(index); 
    terms[index].setup(coeff, powers); 
  }
  // Needs tests!


  void _setTerm(int index, const rsMultiVarMonomial<T>& newTerm) 
  { 
    checkTermIndex(index); 
    terms[index] = newTerm;
  }
  // Needs tests!


  /** Directly sets the coefficient of the term with given index. It may destroy the canonical
  representation by setting a coeff to zero. */
  void _setCoeff(int index, T newCoeff)
  { 
    checkTermIndex(index); 
    terms[index].setCoeff(newCoeff); 
  }
  // Maybe rename "index" to "termIndex", pass newCoeff by const ref

  /** Shifts the coefficient with the given index by the given amount, i.e. adds the given amount 
  to the coeff. It may decanonicalize the representation by leading to a zero coeff. */
  void _shiftCoeff(int index, T amount) { _setCoeff(index, amount + getCoeff(index)); }
  // ToDo: Pass amount by const ref

  /** Finds the index in our terms array where the term at that index has the same powers as the
  given "term". If no such term is found, it will return an invalid index, i.e. one that is above
  the range of valid indices. */
  size_t _findIndexForTerm(const rsMultiVarMonomial<T>& term);
  // Maybe it should return an int and -1 for "not found"? But that would make it more complicated.
  // The implementation would be more complicated and at the call site, we would potentially also 
  // have to change the checks. But using an inte with -1 for not found would be consistent with
  // conventions used in the RAPT library.



  rsMultiVarMonomial<T>* _getTermPtr(int i)
  {
    checkTermIndex(i);
    return &(terms[i]);
  }
  // Maybe return a const pointer


  /** Checks if this polynomial is in canonical representation. A representation is canonical if it
  has no zero coefficients (up to the roundoff tolerance) and if the terms are lexicographically
  ordered. The empty polynomial is also accepted as a canonical epresentation. It represents the 
  zero polynomial. */
  bool _isCanonical() const;

  /** Returns true iff the powers of our terms are strictly ordered (lexicographically) inside of 
  our terms array. This strict order requirement also entails uniqueness of the powers. That means 
  that this function serves two purposes at the same time: Making sure that the terms are sorted 
  and that no combination of powers appears more than once. These are two of the requirements for a
  canonical representation. */
  bool _areTermsStrictlySorted() const;

  /** Returns true iff any of our terms array has a coefficient of zero (up to the roundoff 
  tolerance). In a canonical representation, this is forbidden. */
  bool _hasZeroCoeffs() const;

  /** Returns true iff any of our terms has a negative power. We currently consider this as not
  allowed, i.e. a bug - but this restriction can be lifted later, if needed. */
  bool _hasNegativePowers() const;


protected:

  std::vector<rsMultiVarMonomial<T>> terms;  // Array of terms of the form c * x0^p0 * x1^p1 * ...
  int numVars = 1;                           // Number of variables. Dimension of input argument.
  TTol tol = TTol(0);                        // Tolerance for numerical comparisons.


  static rsMultiVarMonomial<T> zeroMonomial;
  // We need a static object to represent a zero polynomial in order to be able to assign the 
  // return value of getters that return a const reference to a monomial when our terms array is
  // empty.

  // Maybe in addition to the "numVars" variable, we also need a "termLess" function pointer that 
  // can be assigned in order to be able to use different term orders. We should then add 
  // assertions like rsAssert(p.isCompatibleWith(q)); whereever this is appropriate (e.g. in 
  // adding, multiplying etc.). The isCompatibleWith() function should check that the numVars and 
  // termLess members match. termLess should by default be the less-than function that corresponds
  // to lexicographic order. Maybe the init() function should take a 2nd parameter for this 
  // function - maybe optional, defaulting to "lessLexic()".

};

template<class T, class TTol>
void rsMultiVarPolynomial<T, TTol>::addTerm(const rsMultiVarMonomial<T>& newTerm)
{
  rsAssert(newTerm.getNumVariables() == getNumVariables(), "Term has wrong number of variables");
  // Maybe use a function like isTermCompatible(newTerm);

  size_t i = _findIndexForTerm(newTerm);

  if(i >= getNumTerms())                   // Maybe use if(!isValidTermIndex(i))
  {
    terms.push_back(newTerm);
    return;
  }

  if(newTerm.hasSamePowersAs(terms[i]))
  {
    // Modify coeff at i:
    terms[i].shiftCoeff(newTerm.getCoeff());
    if(terms[i].isCoeffZero(tol))
      rsRemove(terms, i);

  }
  else
  {
    // Insert new term at position i:
    rsInsert(terms, newTerm, i);
  }

  // ToDo:
  //
  // - Implement unit tests that verify that adding a term via this function maintains a canonical
  //   representation.
}

template<class T, class TTol>
void rsMultiVarPolynomial<T, TTol>::add(
  const MultiPoly& p, const MultiPoly& q, MultiPoly* r)
{
  rsAssert(p._isCanonical());
  rsAssert(q._isCanonical());
  rsAssert(p.numVars == q.numVars);

  int Np = p.getNumTerms();      // Number of terms in left operand p
  int Nq = q.getNumTerms();      // Number of terms in right operand q
  int Nr = Np + Nq;              // Number of terms in result r (before canonicalization)

  r->init(p.numVars);
  r->tol = rsMax(p.tol, q.tol);
  r->_setNumTerms(Nr);
  for(int i = 0; i < Np; i++)
    r->_setTerm(i, p.getTerm(i));
  for(int i = 0; i < Nq; i++)
    r->_setTerm(Np + i, q.getTerm(i));

  r->_canonicalize();

  // The idea here is basically to just "concatenate" the two polynomials and then let the
  // _canonicalize() call take care of combining terms with like powers and clean up terms whose
  // coeffs cancel to zero.
  //
  // ToDo: Maybe use r->initFrom(p) in anticipation of factoring out the add() function. In 
  // rsSparsePolynomial, we have separate implementations for add(), subtract(), weightedSum() and
  // they differ only marginally. I think, doing the same here would really be too much code 
  // duplication. Maybe just implement weightedSum() and treat add() and subtract() as special 
  // cases of that. I'm not sure about that, though.
}
// Needs more tests

template<class T, class TTol>
void rsMultiVarPolynomial<T, TTol>::subtract(
  const MultiPoly& p, const MultiPoly& q, MultiPoly* r)
{
  rsAssert(p._isCanonical());
  rsAssert(q._isCanonical());
  rsAssert(p.numVars == q.numVars);

  int Np = p.getNumTerms();
  int Nq = q.getNumTerms();
  int Nr = Np + Nq;

  r->init(p.numVars);
  r->tol = rsMax(p.tol, q.tol);
  r->_setNumTerms(Nr);
  for(int i = 0; i < Np; i++)
    r->_setTerm(i, p.getTerm(i));
  for(int i = 0; i < Nq; i++)
    r->_setTerm(Np + i, -q.getTerm(i));

  r->_canonicalize();
}

template<class T, class TTol>
void rsMultiVarPolynomial<T, TTol>::weightedSum(
    const MultiPoly& p, const T& wp, const MultiPoly& q, const T& wq, MultiPoly* r)
{
  rsAssert(p._isCanonical());
  rsAssert(q._isCanonical());
  rsAssert(p.numVars == q.numVars);

  int Np = p.getNumTerms();
  int Nq = q.getNumTerms();
  int Nr = Np + Nq;

  r->init(p.numVars);                        // ToDo: use r->initLike(p)
  r->tol = rsMax(p.tol, q.tol);
  r->_setNumTerms(Nr);
  for(int i = 0; i < Np; i++)
    r->_setTerm(i, wp * p.getCoeff(i), p.getPowers(i));
  for(int i = 0; i < Nq; i++)
    r->_setTerm(Np + i, wq * q.getCoeff(i), q.getPowers(i));

  r->_canonicalize();

  // From a "clean code" perspective, it may be cleaner to avoid the code duplication between 
  // add(), subtract() and weightedSum() by just keeping the implementation of weightedSum() and
  // implementing add() and subtract() by calling the weightedSum() function with weights 1,1 and
  // 1,-1 respectively. However, from a performance perspective, that seems to be not such a good
  // idea which is why I accept this code duplication here.
  //
  // ToDo: In anticipation of factoring out the implementation here and the one in 
  // rsSparsePolynomial, we should perhaps do something like:
  //
  //   r->_setTerm(i,      wp * p.getTerm(i));
  //   r->_setTerm(Np + i, wq * q.getTerm(i));
  //
  // Rationale: In the univariate case, the function getPowers() doesn't exist. There, it's named
  // getPower() in singular. So, with this difference, duck-typing wouldn't work out. We could 
  // provide a getPower() function here too and argue that the "power" is a single vector, but 
  // that's somewhat awkward. That would require us to implement "scalar * monomial" operators for
  // reMonomial and rsMultiVarMonomial. I think, that would be the cleanest solution.
}

template<class T, class TTol>
void rsMultiVarPolynomial<T, TTol>::multiply(
  const MultiPoly& p, const MultiPoly& q, MultiPoly* r)
{
  rsAssert(p._isCanonical());
  rsAssert(q._isCanonical());

  int Np = p.getNumTerms();
  int Nq = q.getNumTerms();
  int Nr = Np * Nq;

  r->init(p.numVars);
  r->tol = rsMax(p.tol, q.tol);
  r->_setNumTerms(Nr);

  // Running through the loops backwards allows us to use it in place, i.e. the polynomial r can
  // point to the location of p and/or q:
  for(int i = Np-1; i >= 0; i--)
    for(int j = Nq-1; j >= 0; j--)
    {
      r->_setTerm(i*Nq+j, p.getTerm(i) * q.getTerm(j));
      // New, adapted.

      //r->_setTerm(i*Nq+j, p.getCoeff(i) * q.getCoeff(j), p.getPower(i) + q.getPower(j)); 
      // Original from rsSparsePolynomial 
    }

  // We may have to re-canonicalize to combine terms with equal exponent:
  r->_canonicalize();
}
// Needs tests


template<class T, class TTol>
void rsMultiVarPolynomial<T, TTol>::divide(
  const MultiPoly& f, const std::vector<MultiPoly>& fs,
  std::vector<MultiPoly>* qs, MultiPoly* r)
{
  size_t N = fs.size();

  // Sanity check the fs array:
  for(const auto & fi : fs)
  {
    rsAssert(fi.isCompatibleWith(f));
  }

  qs->resize(N);
  // ToDo: Loop through the qs and call q[i].initFrom(fs[0]) on each. This call should set up the
  // numVars and later also the termLess members of q[i]. We should also call r->initFrom(fs[0]).
  // Maybe we should have a conditional early return like if(fs.empty()). Or maybe we should use
  // f instead of fs[0] for initFrom(). Maybe we should assert that all the fs[i] are compatible
  // with f. But initFrom() is a bad name because it may be interpreted as creating a copy of the
  // input when in fact, it just initializes to the zero polynomial with compatible numVars and
  // termLess settings. But what would be a better name? 
  // 
  // AI says: Maybe "initCompatibleWith()" or "initFromCompatible()" or "initLike()". Maybe we 
  // should just call it init() and pass the input polynomial as an argument. But that would be 
  // somewhat inconsistent with the way we do initialization in other places where we just call 
  // init() without arguments. Maybe we should have two overloads of init(): One without arguments
  // that initializes to some default setting and one with a polynomial argument that initializes 
  // to a compatible setting. That seems to be the cleanest solution. We should also add assertions
  // to the add(), subtract(), weightedSum() and multiply() functions that check compatibility of 
  // the inputs.
  //
  // My ideas: zeroFrom...or maybe use the free function rsZeroValue() with a prototype argument.
  // That would be consistent with how we do it with modular integers and matrices. For example,
  // rsZeroValue(rsMatrix<T> A) creates a zero matrix with the same shape as A. We want to do 
  // something similar here just that here it's not about the shape but about numVars and termLess.
  // But: rsZeroValue creates a new object. We actually want to re-initialize an existing object.
  // Maybe estbalish a pattern for that in RAPT, too. 
  //
  // Hmmm...conidering all these options, I tend to gravitate to initLike(). 


  // Helper function to extract the leading term:
  auto lt = [](const MultiPoly& p) { return p.getLeadingTerm(); };

  // Initializations:
  MultiPoly p = f;                 // p = f
  r->initLike(f);                  // r = 0
  for(int i = 0; i < N; i++)
    ((*qs)[i]).initLike(f);        // q_i = 0  for  i = 0,...,N-1

  // This is currently an infinite loop:
  /*
  while(!p.isZero())
  {
    int i = 0;
    bool divOccurred = false;
    while(i < N && divOccurred == false)
    {
      // ...
    }
  }
  */


  int dummy = 0;

  // ToDo:
  //
  // - Implement the generalized polynomial division algorithm from IVA, pg. 65
  //
  // - Maybe rename fs to F and qs to Q.
}



template<class T, class TTol>
void rsMultiVarPolynomial<T, TTol>::_canonicalize()
{
  // In the empty case, we have nothing to do and we really *need* to return early in order to not 
  // get an access violation in the code below (in the  int p = getPower(0);  line):
  if(terms.empty())
    return;

  using Monom   = rsMultiVarMonomial<T>;
  using TermPtr = const Monom*;

  // Sort the terms by power/exponent:
  std::sort(terms.begin(), terms.end(), 
            [](const Monom& lhs, const Monom& rhs){ return Monom::lessLexic(lhs, rhs); });

  // Combine multiple terms with equal power/exponent into single terms: 
  int numTerms = getNumTerms();
  TermPtr t = _getTermPtr(0);       // Current term
  int r = 1;                        // Read index
  int w = 0;                        // Write index
  while(r < numTerms) 
  {
    if( t->hasSamePowersAs(getTerm(r)) )
      _shiftCoeff(w, getCoeff(r));
    else 
    {
      w++;
      _setTerm(w, getTerm(r));
      t = _getTermPtr(r);
    }
    r++;
  }
  _setNumTerms(w+1);                // Possibly shorten the terms array
  // This algorithm works only when the terms are sorted by exponent so it doesn't really make 
  // sense to factor it out into a function in its own right. Doing so could invite calling it on 
  // unsorted term arrays in which case we would have a bug.

  // Remove terms with coefficient zero (up to roundoff tolerance):
  _removeTermsWithZeroCoeff();

  // Check postcondition:
  rsAssert(_isCanonical(), "Canonicalization failed");
  // If this triggers, there's a bug in the canonicalization code above and/or in the 
  // implementation of _isCanonical().


  // ToDo:
  //
  // - The implementation duplicates a lot of code from rsSparsePolynomial::_canonicalize(). Try to
  //   factor out common code into a free function (maybe static member function fo some class 
  //   rsPolynomialHelpers). But they are not exactly the same. There are some adaptation that had 
  //   to be done here. This here is the more general case so if we want to unify both functions,
  //   we'll probably have to do it more similar to the way it's written here rather than in
  //   rsSparsePolynomial
}

template<class T, class TTol>
void rsMultiVarPolynomial<T, TTol>::_removeTermsWithZeroCoeff()
{
  rsRemoveIf
  (terms, 
    [this](const rsMultiVarMonomial<T>& term)
    { 
      return rsIsNegligible(term.getCoeff(), tol);
    }
  );

  // Is slightly modified copy from rsSparsePolynomial
}


template<class T, class TTol>
size_t rsMultiVarPolynomial<T, TTol>::_findIndexForTerm(const rsMultiVarMonomial<T>& t)
{
  auto less = &rsMultiVarMonomial<T>::lessLexic;
  size_t i = 0;
  while(i < terms.size() && less(terms[i], t))
    i++;
  return i;

  // ToDo:
  //
  // - Maybe use binary search rather than linear search later.
  //
}

template<class T, class TTol>
bool rsMultiVarPolynomial<T, TTol>::_isCanonical() const
{
  bool ok = true;
  ok &=  _areTermsStrictlySorted();  // Powers are sorted and don't appear more than once.
  ok &= !_hasZeroCoeffs();           // Any zero coeffs (up to roundoff) are cleaned up.
  ok &= !_hasNegativePowers();       // No negative powers allowed. May be relaxed later if needed.
  return ok;

  // Is verbatim copy from rsSparsePolynomial
}


template<class T, class TTol>
bool rsMultiVarPolynomial<T, TTol>::_areTermsStrictlySorted() const
{
  if(terms.empty())
    return true;

  using Monom   = rsMultiVarMonomial<T>;
  using TermPtr = const Monom*;

  TermPtr prev = &getTerm(0);               // Pointer to previous term
  for(int i = 1; i < getNumTerms(); i++)
  {
    const Monom& cur = getTerm(i);          // Reference to current term
    if(!Monom::lessLexic(*prev, cur))
      return false;
    prev = &cur;                            // Rebind pointer
  }

  return true;

  // ToDo:
  //
  // - Maybe change the API of getTerm tor return a const pointer. Or maybe add an additional 
  //   function getTermPtr() and/or getTermConstPtr(). Maybe if we allow to return non-const
  //   pointers, the functions should be marked with an underscore as low-level functions because
  //   they will allow the caller to manipulate the term directly which may mess up our supposed
  //   canonical representation.
}
// Needs tests

template<class T, class TTol>
bool rsMultiVarPolynomial<T, TTol>::_hasZeroCoeffs() const
{
  for(auto& t : terms)
    if( rsIsNegligible(t.getCoeff(), tol) )  // Maybe use t.isCoeffZero(tol)
      return true;

  return false;
}

template<class T, class TTol>
bool rsMultiVarPolynomial<T, TTol>::_hasNegativePowers() const
{
  for(auto& t : terms)
    if(t.hasNegativePowers())
      return true;

  return false;
}

// ToDo:
//
// - We actually get a lot of code duplication with rsSparsePolynomial in this class. Maybe think 
//   about factoring out a common baseclass or maybe some free function templates that can be used
//   by both classes. Maybe the template parameter should be the kind of term, i.e. 
//   rsMultiVarMonomial here and rsMonomial for rsSparsePolynomial and the functions should operate
//   on std::vectors of terms or maybe on C-style arrays of such terms. Maybe we'll need a bit of 
//   duck typing to make it work, i.e. rsMonomial and rsMultiVarMonomial should provide equally 
//   named member functions with the same semantics just with different data types for the powers, 
//   namely int (or const int&) in the case of rsMonomial and std::vector<int>& in the case of
//   rsMultiVarMonomial.
// 
// - Maybe we should write a class rsPolynomialHelpers into which we factor out various algorithms 
//   that are used within implementations of polynomial classes but that operate on more basic 
//   data structures like C-arrays and std::vectors. Some of the low-level static functions of 
//   class rsPolynomial and also perhaps of class rsRationalFunction could be factored out into 
//   such a "...Helpers" class as well. There we could also put the code that is common between 
//   rsSparsePolynomial and rsMultiVarPolynomial. Maybe root-finders could also be placed there. 
//   Or maybe they should go into a separate class rsPolynomialRootFinder. The key is that all 
//   these helper classesor functions should be independent of (decoupled from) our actual 
//   polynomial classes because they only operate on C/C++ primitives such as C-style arrays and 
//   (maybe) std::vectors, if needed. Peferably, we would only use C-style arrays in its API 
//   because that would make it most flexible in the ways it could be used (it wouldn't dictate 
//   use of std::vector for the coefficient- or term arrays etc.) but I'm not sure how practical 
//   that is. Maybe using std::vector makes more sense for some things, especially when we need to
//   truncate a coefficient vector because some coeffs are getting zeroed out in an operation.
//
// - Candidates for factoring out:
// 
//   - Trivial: _hasNegativePowers, _hasZeroCoeffs, _isCanonical, _removeTermsWithZeroCoeff, ...
// 
//   - Harder: _areTermsStrictlySorted, _findIndexForTerm, _canonicalize, addTerm, add, ...
// 
//   - For implmenting add(), we perhaps need a function like areCompatible() in both classes 
//     that for multivariate polynomilas verifies that numVars match. For univariate ones, it 
//     should just return true because there is no way that univariate polynomials could be 
//     incompatible. We perhaps also need a function init() in both classes that takes another
//     polynomials as reference. For MultoPolys, it should set the numVars variable from the 
//     passed prototype.

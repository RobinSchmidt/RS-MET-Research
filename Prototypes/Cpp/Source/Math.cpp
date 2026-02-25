
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


  void shiftCoeff(const T& amount) { coeff += amount; }

  // ToDo: setCoeff(), setPowers(), negate(), ... see rsMonomial and implement analoguous functions 
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
  // a < b or b < a or a = b if we interpret the = not as exact equality of terms but as an 
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
  // Needs tests.



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

  /** Clears the array of terms. */
  void clear() { terms.clear(); }

  /** Adds the given monomial to the polynomial. */
  void addTerm(const rsMultiVarMonomial<T>& newTerm);

  // ToDo: setRoundoffTolerance(), subtractTerm(), scale(), negate(), ...
  // But in rsSparsePolynomial, we implement scaling as a low-level method _scaleCoeffs() because
  // the scaler could be zero. Maybe a high-level scale() method without underscore should allow
  // only nonzero scalers. We could assert that the scaler is zero and then call _scaleCoeffs().


  //-----------------------------------------------------------------------------------------------
  /** \name Inquiry.  */

  /** Returns the number of variables, i.e. the dimensionality of the input x. */
  int getNumVariables() const { return numVars; }

  /** Returns the number of terms in this polynomial. */
  int getNumTerms() const { return (int) terms.size(); }

  
  //const std::vector<int>& getPowers(int termIndex) 
  //{ return getTerm(termIndex).getPowers(); }
  // Needs test

  const rsMultiVarMonomial<T>& getTerm(int termIndex) const
  {
    checkTermIndex(termIndex);
    return terms[termIndex];
  }
  // Needs test



    
  bool isValidTermIndex(int i) const { return i >= 0 && i < getNumTerms(); }

  void checkTermIndex(int i) const 
  { rsAssert(isValidTermIndex(i), "Invalid term index!"); }


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


  //-----------------------------------------------------------------------------------------------
  /** \name Low level API.  */
    
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

  /** Directly sets the coefficient of the term with given index. It may destroy the canonical
  representation by setting a coeff to zero. */
  void _setCoeff(int index, T newCoeff)
  { 
    checkTermIndex(index); 
    terms[index].setCoeff(newCoeff); 
  }
  // Maybe rename "index" to "termIndex"

  /** Shifts the coefficient with the given index by the given amount, i.e. adds the given amount 
  to the coeff. It may decanonicalize the representation by leading to a zero coeff. */
  void _shiftCoeff(int index, T amount) { _setCoeff(index, amount + getCoeff(index)); }


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

};

template<class T, class TTol>
void rsMultiVarPolynomial<T, TTol>::addTerm(const rsMultiVarMonomial<T>& newTerm)
{
  rsAssert(newTerm.getNumVariables() == getNumVariables(), "Term has wrong number of variables");

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
void rsMultiVarPolynomial<T, TTol>::_canonicalize()
{
  //rsMarkAsStub();

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

  // ...TBC...


  // ToDo:
  //
  // - Model the implementation after rsSparsePolynomial::_canonicalize(). 
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
//   rsMultiVarMonomial here and rsMonomial for rsSparsePolynomial. Maybe we'll need a bit of duck
//   typing to make it work.
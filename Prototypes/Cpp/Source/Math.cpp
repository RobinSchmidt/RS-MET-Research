
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







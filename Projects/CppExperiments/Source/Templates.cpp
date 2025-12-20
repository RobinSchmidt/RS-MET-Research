
// This file contains a couple of demonstrations for using templates in order to compute (discrete)
// mathematical functions (such as factorials, binomial coefficients, greatest common divisors) at 
// compile time as well as demos for usage of variadic templates to implement functions that may 
// accept an arbitrary number of arguments.


//-------------------------------------------------------------------------------------------------
// Print several lines via a variadic template - variant 1:

template <typename T>
void printLines1(T line)
{
  // This version gets instantiated when there is only a single template parameter. It serves as 
  // recursion anchor (aka base case).
  std::cout << line << "\n";
}

template<typename First,typename ... Rest>
void printLines1(First first,Rest ... rest)
{
  printLines1(first);      // Print first line
  printLines1(rest...);    // Recursive instantiation
}

//-------------------------------------------------------------------------------------------------
// Print several lines via a variadic template - variant 2:

void printLines2()
{
  // This recursion anchor gets used when the argument list is empty. It's not a template because 
  // the compiler can't deduce any template parameters from an empty argument list.
}

template<typename First,typename ... Rest>
void printLines2(First first,Rest ... rest)
{
  std::cout << first << "\n";  // Print first line
  printLines2(rest...);        // Recursive instantiation
}

//-------------------------------------------------------------------------------------------------
// Compute factorials at compile time:

template<int n> 
struct factorial 
{ 
  static const int value = n * factorial<n-1>::value; // General case: n > 0
  //enum { value = n * factorial<n-1>::value };       // ...enums work also
}; 

template<>
struct factorial<0> 
{ 
  static const int value = 1;  // Base case: n == 0
  //enum { value = 1 }; 
};
// why enums? would an int also work? ..i guess static const int would work

void testFactorial()
{
  int f0 = factorial<0>::value;  //   1
  int f1 = factorial<1>::value;  //   1
  int f2 = factorial<2>::value;  //   2
  int f3 = factorial<3>::value;  //   6
  int f4 = factorial<4>::value;  //  24
  int f5 = factorial<5>::value;  // 120
  printLines1(f0, f1, f2, f3, f4, f5);
}

//-------------------------------------------------------------------------------------------------
// Compute binomial coefficients at compile time:

template<int n, int k> 
struct binomial
{ 
  static const int value = binomial<n-1, k>::value + binomial<n-1, k-1>::value;

  //static const int value = binomial<n-1, k>::value + binomial<n, k-1>::value;
  // ..wrong formula...but is nicely symmetric
}; 

template<int n> 
struct binomial<n, 0>
{ 
  static const int value = 1;
}; 

template<int n> 
struct binomial<0, n>
{ 
  static const int value = 0;
  //static const int value = 1;     // wrong
}; 

template<> 
struct binomial<0, 0>
{ 
  static const int value = 1;
}; 

void testBinomialCoeffs()
{
  int b00 = binomial<0, 0>::value;  //   1
  int b01 = binomial<0, 1>::value;  //   0
  int b02 = binomial<0, 2>::value;  //   0
  int b03 = binomial<0, 3>::value;  //   0
  printLines1(b00);

  int b10 = binomial<1, 0>::value;  //   1
  int b11 = binomial<1, 1>::value;  //   1
  int b12 = binomial<1, 2>::value;  //   0
  int b13 = binomial<1, 3>::value;  //   0
  printLines1(b10, b11);

  int b20 = binomial<2, 0>::value;  //   1
  int b21 = binomial<2, 1>::value;  //   2
  int b22 = binomial<2, 2>::value;  //   1
  int b23 = binomial<2, 3>::value;  //   0
  printLines1(b20, b21, b22);

  int b30 = binomial<3, 0>::value;  //   1
  int b31 = binomial<3, 1>::value;  //   3
  int b32 = binomial<3, 2>::value;  //   3
  int b33 = binomial<3, 3>::value;  //   1
  printLines1(b30, b31, b32, b33);

  int b40 = binomial<4, 0>::value;  //   1
  int b41 = binomial<4, 1>::value;  //   4
  int b42 = binomial<4, 2>::value;  //   6
  int b43 = binomial<4, 3>::value;  //   4
  int b44 = binomial<4, 4>::value;  //   1
  printLines1(b40, b41, b42, b43, b44);
}
// ToDo: 
// -make a printLine function that prints them all in one line
// -try to implement a template that evaluates (x+y)^n at compile time using the binomial coeffs 
//  where x,y are also compile time constants. of course, that's a silly way of evaluating such an
//  expression - maybe implement it also in a more sensible way, namely, first evaluation z=x+y 
//  and then z^n
// -try to implement a recursive while, similar to the python code, i wrote some time ago in
//  Recursions.ipynb


// using the wrong formula and wrong binomial<0, n> and creating a matrix, we get a symmetric 
// matrix whose diagonal seems to match the "Central binomial coefficients" 
// https://oeis.org/A000984/list (at least up to 252) -> verify this

// maybe implement a general class to represent triangular arrays/matrices  and experiment with 
// other rules:
//   https://en.wikipedia.org/wiki/Triangular_array 
//   https://en.wikipedia.org/wiki/Category:Triangles_of_numbers
// the class may be used with minor modifictions for symmetric matrices, too: just instead of 
// returning 0 when j > i, return A(j,i). maybe antisymmetric matrices can also be represented by
// having a way to remove the diagonal...maybe when the use requests a size N matrix, we 
// internally use a triangular array of size N-1 or something


//-------------------------------------------------------------------------------------------------
// Compute greatest common divisors at compile time:

template<int a, int b> 
struct gcd
{ 
  static const int value = gcd<b, a%b>::value;  // General case: b > 0
}; 

template<int a> 
struct gcd<a, 0>
{ 
  static const int value = a;                   // Base case: b == 0
}; 

void testGcd()
{
  int gcd_60_21 = gcd<60, 21>::value;                // ==   3
  int gcd_60_48 = gcd<60, 48>::value;                // ==  12
  int gcd_210_1155 = gcd<2*3*5*7, 3*5*7*11>::value;  // == 105 = 3*5*7 == 210/2 == 1155/11
  printLines1(gcd_60_21, gcd_60_48, gcd_210_1155);
}
// -what if a == 0?
// -can we use a generic type T instead of int?
// -the same things can be achieved more easily with constexpr in C++17

/** Computes the greatest common divisor of a and b at compile time. Should be called like this:
int gcd_60_48 = gcd<60, 48>();  */

/*
template<int a, int b>
int gcd()
{
  if(b == 0) return a;
  else       return gcd<b, a%b>();
}
// compiles on msc but not gcc - change to the usual struct gcd { value } idiom
// needs more tests with more interesting inputs

void testGcd()
{
  int gcd_60_21 = gcd<60, 21>();
  int gcd_60_48 = gcd<60, 48>();
  int gcd_210_1155 = gcd<2*3*5*7, 3*5*7*11>();  // == 105 = 210/2 = 1155/11

  printLines1(gcd_60_21, gcd_60_48, gcd_210_1155);
}
*/

//-------------------------------------------------------------------------------------------------
// Compute means of an arbitrary number of arguments. The means are computed at runtime but the
// computation functions are templatized in order to accept an arbitrary number of arguments.

template<class T>
int numArgs(T a1)
{
  return 1;
}

template<class T, class ... Rest>
int numArgs(T first, Rest ... rest)
{
  return 1 + numArgs(rest...);
}

template<class T>
T sum(T x)
{
  return x;
}

template<class T, class ... Rest>
T sum(T first, Rest ... rest)
{
  return first + sum(rest...);
}

template<class T>
T product(T x)
{
  return x;
}

template<class T, class ... Rest>
T product(T first, Rest ... rest)
{
  return first * product(rest...);
}

/** Computes the arithmetic mean of the arguments. */
template<class T, class ... Rest>
T mean(T first, Rest ... rest)
{
  T s = sum(first, rest...);
  T n = (T)numArgs(first, rest...);
  return s / n;
}

/** Computes the geometric mean of the arguments. */
template<class T, class ... Rest>
T geoMean(T first, Rest ... rest)
{
  T p = product(first, rest...);
  T n = (T)numArgs(first, rest...);
  return pow(p, T(1)/n);
}

template<class T>
T powerSum(T p, T x)
{
  return pow(x, p);
}

template<class T, class ... Rest>
T powerSum(T p, T first, Rest ... rest)
{
  return pow(first, p) + powerSum(p, rest...);
}

template<class T, class ... Rest>
T generalizedMean(T p, T first, Rest ... rest)
{
  if(p == T(0))                          // Special case for p = 0
    return geoMean(first, rest...);      // ..we need(!) the geometric mean in this case  

  if(p == T(1))                          // Special case for p = 1
    return mean(first, rest...);         // ..we use the arithmetic mean in this case 

  // General case:
  T s = powerSum(p, first, rest...);     // Sum of the powers
  T n = (T)numArgs(first, rest...);      // Number of arguments
  T m = s / n;                           // Mean of the powers
  return pow(m, T(1)/p);                 // Generalized mean

  // Notes:
  //
  // - The p = 0 case really _needs_ special treatment because the general code would produce a 
  //   division by zero. For the p = 1 case, the special treatment is merely an optimization. The 
  //   general code would also produce the correct result but is arguably much more expensive due 
  //   to all the calls to pow().
  //
  //
  // ToDo:
  //
  // - Add tests for this function. Make sure to cover the p = 0 case in these tests. Cover also
  //   at least p = 1, p = 2, p = -1. Maybe a couple of more as well.
  //
  // - Maybe move (or copy) these implementations into the RAPT library. Maybe create a new file 
  //   AggregationFunctions.h (or just Aggregation.h) in the Math/Functions folder. Then look into
  //   class rsArrayTools for inspiration for what other aggregation functions we could possibly 
  //   need and maybe implement some of those. 
}


void testMean()
{
  bool ok = true;

  int a1 = numArgs(1.0);               ok &= a1 == 1;
  int a2 = numArgs(1.0, 2.0);          ok &= a2 == 2;
  int a3 = numArgs(1.0, 2.0, 3.0);     ok &= a3 == 3;
  int a4 = numArgs(1.0, 2.f, 5, 3.0);  ok &= a4 == 4;  // Try it with different types for the args.
  printLines1(a1, a2, a3, a4);         // Should produce 1,2,3,4

  float s1 = sum(2.f);                 ok &= s1 == 2.f;
  float s2 = sum(2.f, 3.f);            ok &= s2 == 5.f;
  float s3 = sum(2.f, 3.f, 1.f);       ok &= s3 == 6.f;
  printLines1(s1, s2, s3);             // Should produce 1,5,6

  float m1 = mean(2.f);                ok &= m1 == 2.f;
  float m2 = mean(2.f, 4.f);           ok &= m2 == 3.f;
  float m3 = mean(2.f, 4.f, 6.f);      ok &= m3 == 4.f;
  printLines1(m1, m2, m3);            // Should produce 2,3,4

  float p1 = powerSum(2.f, 2.f);      ok &= p1 == 4.f;        //  4 = 2^2
  float p2 = powerSum(2.f, 2.f, 3.f); ok &= p2 == 13.f;       // 13 = 2^2 + 3^2
  printLines1(p1, p2);


  // ToDo:
  //
  // - Implement generalized mean.
  //
  // - Try this code in compiler explorer to verify that it compiles down to what we would manually
  //   write to compute a mean of n values. We would just add them all up and divide by the number 
  //   of values.
}






// see here for more examples:
// https://en.wikipedia.org/wiki/Template_metaprogramming
// https://en.wikipedia.org/wiki/Compile-time_function_execution
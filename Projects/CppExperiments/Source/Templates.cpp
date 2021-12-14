
//-------------------------------------------------------------------------------------------------
// Printing several lines via a variadic template - variant 1:

template <typename T>
void printLines1(T line)
{
  // This version gets instantiated when there is only a single template parameter. It serves as 
  // recursion anchor.
  std::cout << line << "\n";
}

template<typename First,typename ... Rest>
void printLines1(First first,Rest ... rest)
{
  printLines1(first);      // print first line
  printLines1(rest...);    // recursive instantiation
}

//-------------------------------------------------------------------------------------------------
// Printing several lines via a variadic template - variant 2:

void printLines2()
{
  // This recursion anchor gets used when the argument list is empty. It's not a template because 
  // the compiler can't deduce any template parameters from an empty argument list.
}

template<typename First,typename ... Rest>
void printLines2(First first,Rest ... rest)
{
  std::cout << first << "\n";  // print first line
  printLines2(rest...);        // recursive instantiation
}

//-------------------------------------------------------------------------------------------------
// Computing factorials:

template<int n> 
struct factorial 
{ 
  static const int value = n * factorial<n-1>::value; // general case: n > 0
  //enum { value = n * factorial<n-1>::value };       // ...enums work also
}; 

template<>
struct factorial<0> 
{ 
  static const int value = 1;  // base case: n == 0
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
// Computing binomial coefficients:

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
// Computing greatest common divisors:

template<int a, int b> 
struct gcd
{ 
  static const int value = gcd<b, a%b>::value; // general case: b > 0
}; 

template<int a> 
struct gcd<a, 0>
{ 
  static const int value = a;  // base case: b == 0
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





// see here for more examples:
// https://en.wikipedia.org/wiki/Template_metaprogramming
// https://en.wikipedia.org/wiki/Compile-time_function_execution
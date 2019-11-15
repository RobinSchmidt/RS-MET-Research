
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
  enum { value = n * factorial<n-1>::value }; // general case: n > 0
}; 

template<>
struct factorial<0> 
{ 
  enum { value = 1 }; // base case: n == 0
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
  int gcd_60_21 = gcd<60, 21>::value;
  int gcd_60_48 = gcd<60, 48>::value;
  int gcd_210_1155 = gcd<2*3*5*7, 3*5*7*11>::value;  // == 105 = 3*5*7 == 210/2 == 1155/11
  printLines1(gcd_60_21, gcd_60_48, gcd_210_1155);
}
// what if a == 0?


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


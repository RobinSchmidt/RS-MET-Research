
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
// Experimental:

template <int n>
struct factorial 
{
  enum { value = n * factorial<n - 1>::value };
};

template <>
struct factorial<0> {
  enum { value = 1 };
};

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


/*
template<int n, int r>
int factorialHelp()
{
  if(n <= 1) return r;
  else       return factorialHelp<n-1, n*r>();
}
template<int n>
int factorial()
{
  return factorialHelp<n, 1>();
}
*/

/*
template<int n>
int factorial()
{
  if(n == 0) return 1;
  else       return factorial<n-1>() * n;
}
// VS says, the recursion is too complex - maybe it's not a tail recursion because of the 
// multiplication by n
*/

/** Computes the greatest common divisor of a and b at compile time. Should be called like this:
int gcd_60_48 = gcd<60, 48>();  */
template<int a, int b>
int gcd()
{
  if(b == 0) return a;
  else       return gcd<b, a%b>();
}
// compiles on msc but not gcc - change to the usual struct gcd { value } idiom
// needs more tests with more interesting inputs


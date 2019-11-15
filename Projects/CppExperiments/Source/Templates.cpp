
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

/** Computes the greatest common divisor of a and b at compile time. Should be called like this:
int gcd_60_48 = gcd<60, 48>();  */
template<int a, int b>
int gcd()
{
  if(b == 0) return a;
  else       return gcd<b, a%b>();
}
// needs more tests with more interesting inputs


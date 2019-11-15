#include "Common.h"
#include "ObjectLifetime.cpp"
#include "Containers.cpp"
#include "Functional.cpp"
#include "Templates.cpp"
#include "Misc.cpp"



/** Logs the call of a function to std::cout. We put the function in a class to demonstrate showing
the full path. */
#if defined(__GNUC__)
#define FUNCTION_NAME __PRETTY_FUNCTION__     // shows full path and signature with gcc
#elif defined(_MSC_VER)
#define FUNCTION_NAME __FUNCSIG__             // same with microsoft compiler
#else
#define FUNCTION_NAME __func__                // shows name only but is ISO C++
#endif
class Logger
{
public:
  void log()
  {
    std::cout << "Function: " << FUNCTION_NAME << " was called.\n"; // __FUNCTION__ exists as C macro
    //std::cout << "File:     " << __FILE__ << "\n";
    //std::cout << "Line:     " << __LINE__ << "\n";
  }
};
// move to file Introspection




class RealNumber
{

public:

  double x;

  // Constructors:
  RealNumber(double value) : x(value) {}
  RealNumber(int value) : x(value) {}

  // Assignment:
  RealNumber operator=(const int& value) { return RealNumber(value); }
  RealNumber operator=(int&& value)      { return RealNumber(value); }

  // Conversion:
  operator double() const { return x; }

  // Artihmetic:
  RealNumber operator+(const RealNumber& y) { return x + y; }
  RealNumber operator-(const RealNumber& y) { return x - y; }
  RealNumber operator*(const RealNumber& y) { return x * y; }
  RealNumber operator/(const RealNumber& y) { return x / y; }
  RealNumber operator^(const RealNumber& y) { return pow(x, y); } // this is why we do it
  // +=, =, ==, ...
};
// maybe templatize





int main()
{
  //demoObserver();
  //testReturnValueOptimization(); 
  //testFunctionShortcuts();

  int gcd_60_21 = gcd<60, 21>();
  int gcd_60_48 = gcd<60, 48>();

  //printLines();
  printLines1("Bla", 42, 3.14, 'c');
  printLines2("Blub", 2.72, 'x', 73);






  /*

  SelfDeleter* sd = new SelfDeleter;
  sd->selfDelete();
  // todo: test, if this works - check for memory leaks and/or implement destructor and put
  // breakpoint ina dn see, if it gets hit

  std::cout << "Demonstrate, how a (member) function can print its won name\n";
  Logger logger;
  logger.log();
  std::cout << "\n";

  std::cout << "Emulate multiple return value via std::array\n";
  auto a123 = get123();
  std::cout << a123[0] << a123[1] << a123[2] << "\n\n";

  std::cout << "Create a std::vector from initializer list\n";
  std::vector<int> v({ 1, 2, 3 });
  std::cout << v[0] << v[1] << v[2] << "\n\n";   // todo: wrap into function


  // move to file Functional
  std::cout << "Apply (lambda) function to each element - this has no effect on the stored vector elements\n";
  std::for_each(v.begin(), v.end(), [](int x){ return 2*x + 1; });
  std::cout << v[0] << v[1] << v[2] << "\n\n";

  std::cout << "If we want to modify the vector contents, we have to write it like that\n";
  std::for_each(v.begin(), v.end(), [&](int& x){ x = 2*x + 1; });
  std::cout << v[0] << v[1] << v[2] << "\n\n";

  // demonstrate lambda with binding by value via [=]

  // test, if std::vector initializes the memory - yes, it does
  std::vector<double> v1(10);
  printVector(v1);


  RealNumber a = 2, b = 3;
  double ab = a^b, ba = b^a; 
  // we can actually use ^ as exponentiation, if we want - may be nice for prototype code that 
  // deals with polynomials - but no - that doesn't work because the precedence rules don't match
  // mathematical usage

  */
   

  //std::cout << "Blah!";
  return 0;
}

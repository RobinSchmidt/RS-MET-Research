#include "ObjectLifetime.cpp"


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

class Statistics
{
public:
  static double normalDistribution(double x, double mu, double sigma)
  {
    static const double pi = 3.14;   // we are grossly imprecise here
    double s22 = 2 * sigma * sigma;
    double xm  = x - mu;
    return (1./sqrt(s22*pi)) * exp(-xm*xm / s22);
  }
};

/** Emulate multiple return values (of the same type) via std::array ...should this be done with
tuple instead? ...or maybe we should use structs? */
std::array<float, 3> get123()
{
  return std::array<float, 3>{ 1.f, 2.f, 3.f };
  //std::array<float, 3> a = {1.f, 2.f, 3.f}; return a; // alternative
}

// move to Tools.h
template<class T>
void printVector(const std::vector<T>& v)
{
  for(size_t i = 0; i < v.size(); i++)
    std::cout << v[i] << ", ";
  std::cout << "\n";
}




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
  testReturnValueOptimization(); // 3 constructors, 4 destructors - what?
  // implement copy- and move constructors and assignment operators - there's probably a call to
  // the default copy cosntructor somewhere


  SelfDeleter* sd = new SelfDeleter;
  sd->selfDelete();
  // todo: test, if this works - check for memory leaks and/or implement destructor and put
  // breakpoint ina dn see, if it gets hit

  std::cout << "Demonstrate, how a (member) function can print its won name\n";
  Logger logger;
  logger.log();
  std::cout << "\n";

  // use a lambda function as shortcut to an otherwise verbose function call (the compiler will
  // optimize it away):
  double mu = 5, sigma = 2;
  auto normal_5_2 = [=](double x)->double{ return Statistics::normalDistribution(x, mu, sigma); };
  double y = normal_5_2(3);


  std::cout << "Emulate multiple return value via std::array\n";
  auto a123 = get123();
  std::cout << a123[0] << a123[1] << a123[2] << "\n\n";

  std::cout << "Create a std::vector from initializer list\n";
  std::vector<int> v({ 1, 2, 3 });
  std::cout << v[0] << v[1] << v[2] << "\n\n";   // todo: wrap into function

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
   

  //std::cout << "Blah!";
  return 0;
}

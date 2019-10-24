#include <iostream>
#include <array>
#include <vector>
#include <cmath>>
#include <algorithm>
//using namespace std;




// see here https://www.youtube.com/watch?v=xGDLkt-jBJ4 at 19:50
class SelfDeleter
{
public:
  void selfDelete() { delete this; }
};
// move to file Patterns.cpp (or antipatterns?)


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

template<class T>
void printVector(const std::vector<T>& v)
{
  for(size_t i = 0; i < v.size(); i++)
    std::cout << v[i] << ", ";
  std::cout << "\n";
}



class ExpensiveToCopy
{
public:
  ExpensiveToCopy()                         { std::cout << pad << "Default Constructor\n";    }
  ExpensiveToCopy(const ExpensiveToCopy&)   { std::cout << pad << "!!!Copy Constructor!!!\n"; }
  ExpensiveToCopy(      ExpensiveToCopy&&)  { std::cout << pad << "Move Constructor\n";       }
  ~ExpensiveToCopy()                        { std::cout << pad << "Destructor\n";             }

  //ExpensiveToCopy& operator=(ExpensiveToCopy)
  //{ std::cout << pad << "!!!Copy Assignment Operator!!!\n"; }

  ExpensiveToCopy& operator=(const ExpensiveToCopy&)
  { std::cout << pad << "!!!Copy Assignment Operator!!!\n"; return *this; }

  ExpensiveToCopy& operator=(ExpensiveToCopy&&)
  { std::cout << pad << "Move Assignment Operator\n"; return *this; }

  ExpensiveToCopy operator+(const ExpensiveToCopy& rhs)
  {
    std::cout << pad << "Addition Operator\n";
    ExpensiveToCopy returnValue;
    return returnValue;
  }

  ExpensiveToCopy operator+=(const ExpensiveToCopy& rhs)
  {
    std::cout << pad << "PlusEquals Operator\n";
    ExpensiveToCopy returnValue;
    return returnValue;
  }

  //ExpensiveToCopy operator-(const ExpensiveToCopy& a)
  //{
  //  std::cout << pad << "Unary Minus Operator\n";
  //  return *this;
  //}

  ExpensiveToCopy operator-()
  {
    std::cout << pad << "Unary Minus Operator\n";
    return *this;
  }
  // calls copy constructor, returning a reference doesn't help

  std::string pad = "  ";
  // todo: let the indentation vary - when + calls constructors, they should be further indented
  // use static indent member, increase on entry, decrease on exit
};

//ExpensiveToCopy ExpensiveToCopy::operator-(const ExpensiveToCopy& a)
//{
//  std::cout << ExpensiveToCopy::pad << "Unary Minus Operator\n";
//  return *this;
//}



// https://en.cppreference.com/w/cpp/language/move_constructor
// https://en.cppreference.com/w/cpp/language/operator_arithmetic

void testReturnValueOptimization()
{
  using MyClass = ExpensiveToCopy;

  std::cout << "MyClass a, b;\n";       MyClass a, b;
  std::cout << "MyClass c = a + b;\n";  MyClass c = a + b;
  std::cout << "a = c;\n";              a = c;                // 1 copy
  std::cout << "MyClass d(c)\n";        MyClass d(c);         // 1 copy
  std::cout << "c = a + MyClass();\n";  c = a + MyClass();
  std::cout << "a = a + b + c;\n";      a = a + b + c;
  std::cout << "a = c + b + a;\n";      a = c + b + a;
  std::cout << "a += b;\n";             a += b;
  std::cout << "a = -a;\n";             a = -a;               // 1 copy
  std::cout << "b = -a;\n";             b = -a;               // 1 copy
  std::cout << "c = -(a+b)\n";          c = -(a+b);           // 1 copy

  std::cout << "End of function\n";
};


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




  //std::cout << "Blah!";
  return 0;
}

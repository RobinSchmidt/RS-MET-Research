#include <iostream>
#include <array>
#include <vector>


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
  void log() { std::cout << FUNCTION_NAME << " was called.\n";  }
};


/** Emulate multiple return values (of the same type) via std::array ...should this be done with
tuple instead? ...or maybe we should use structs? */
std::array<float, 3> get123()
{
  return std::array<float, 3>{ 1.f, 2.f, 3.f };
  //std::array<float, 3> a = {1.f, 2.f, 3.f}; return a; // alternative
}



int main()
{

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

  std::cout << "Apply (lambda) function to each element - this has no effect on the stored vector elements\n";
  std::for_each(v.begin(), v.end(), [](int x){ return 2*x + 1; }); 
  std::cout << v[0] << v[1] << v[2] << "\n\n";

  std::cout << "If we want to modify the vector contents, we have to write it like that\n";
  std::for_each(v.begin(), v.end(), [&](int& x){ x = 2*x + 1; });
  std::cout << v[0] << v[1] << v[2] << "\n\n"; 

  // demonstrate lambda with binding by value via [=]




  //std::cout << "Blah!";
  return 0;
}

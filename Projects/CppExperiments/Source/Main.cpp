#include <iostream>
#include <array>





// see here https://www.youtube.com/watch?v=xGDLkt-jBJ4 at 19:50
class SelfDeleter
{
public:
  void selfDelete() { delete this; }
};


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


// emulate multiple return values (of the same type) via std::array ...should this be done with
// tuple instead? ...or maybe we should use structs?
std::array<float, 3> get123()
{
  std::array<float, 3> a = {1.f, 2.f, 3.f};
  return a;
}

int main()
{

  SelfDeleter* sd = new SelfDeleter;
  sd->selfDelete();
  // todo: test, if this works - check for memory leaks and/or implement destructor and put
  // breakpoint ina dn see, if it gets hit

  Logger logger;
  logger.log();


  auto a123 = get123();
  std::cout << a123[0] << a123[1] << a123[2];


  //std::cout << "Blah!";
  return 0;
}

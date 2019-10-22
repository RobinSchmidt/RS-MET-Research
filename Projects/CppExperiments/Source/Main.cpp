#include <iostream>
#include <array>

// see here https://www.youtube.com/watch?v=xGDLkt-jBJ4 at 19:50
class SelfDeleter
{
public:
  void selfDelete() { delete this; }
};


/** Logs the call of a function to std::cout. */
class Logger
{
public:
  void log()
  {
    std::cout << __func__ << " was called.\n";  // ISO C++
    std::cout << __PRETTY_FUNCTION__ << "\n";  // only gcc
    //std::cout << __FUNCSIG__  << "\n";       // only msc
    //std::cout << "SelfDeleter::selfDelete";
  }
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

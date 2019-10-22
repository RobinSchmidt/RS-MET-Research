#include <iostream>>

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

int main()
{

  SelfDeleter* sd = new SelfDeleter;
  sd->selfDelete();
  // todo: test, if this works - check for memory leaks and/or implement destructor and put
  // breakpoint ina dn see, if it gets hit

  Logger logger;
  logger.log();


  //std::cout << "Blah!";
  return 0;
}

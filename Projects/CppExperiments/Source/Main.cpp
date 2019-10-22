#include <iostream>>

// see here https://www.youtube.com/watch?v=xGDLkt-jBJ4 at 19:50
class SelfDeleter
{
public:
  void selfDelete()
  {
    std::cout << "SelfDeleter::selfDelete";
    // todo: figure out, how we can infer the funtion name - i think, there's a macro
    delete this;
  }
};



int main()
{

  SelfDeleter* sd = new SelfDeleter;
  sd->selfDelete();
  // todo: test, if this works - check for memory leaks and/or implement destructor and put
  // breakpoint ina dn see, if it gets hit


  //std::cout << "Blah!";
  return 0;
}

/** Demos for managing things related to the lifetime of objects including construction, 
destruction, copy/move constsructors and -assignment operators, etc.  */

#include "Common.h"


//-------------------------------------------------------------------------------------------------

/** A class that avoids unnecessary copy operations for the return values of arithmetic operators.
The techniques can be useful for implementing a matrix class. */

class ExpensiveToCopy  // maybe rename (Arithmetic)CopyElision
{

public:

  ExpensiveToCopy()                         { std::cout << pad << "Default Constructor\n";    }
  ExpensiveToCopy(const ExpensiveToCopy&)   { std::cout << pad << "!!!COPY Constructor!!!\n"; }
  ExpensiveToCopy(      ExpensiveToCopy&&)  { std::cout << pad << "Move Constructor\n";       }
  ~ExpensiveToCopy()                        { std::cout << pad << "Destructor\n";             }

  //ExpensiveToCopy& operator=(ExpensiveToCopy)
  //{ std::cout << pad << "!!!Copy Assignment Operator!!!\n"; }
  // alternative - takes argument by value

  ExpensiveToCopy& operator=(const ExpensiveToCopy&)
  { std::cout << pad << "!!!COPY Assignment Operator!!!\n"; return *this; }

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

  ExpensiveToCopy operator-()
  {
    std::cout << pad << "Unary Minus Operator\n";
    return std::move(*this);  // avoids copy constructor call
    //return *this;           // calls copy constructor
  }


  std::string pad = "  ";
  // todo: let the indentation vary - when + calls constructors, they should be further indented
  // use static indent member, increase on entry, decrease on exit
};
// use this pattern for the rsMatrix class in RAPT
// in a unit test, use a subclass that overrides the relevant operators and constructors and counts
// up a static int that counts the number of invoked heap copy operations - and check, if it has 
// the desired value after a couple of artithmetic operations

void testReturnValueOptimization() // rename to testCopyElision
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
  std::cout << "a = -a;\n";             a = -a;
  std::cout << "b = -a;\n";             b = -a;
  std::cout << "c = -(a+b)\n";          c = -(a+b);

  std::cout << "End of function\n";
};


//-------------------------------------------------------------------------------------------------

/** Object that deletes itself in a member function - i don't know, how that could be used. Maybe 
it's  an atipattern? see here https://www.youtube.com/watch?v=xGDLkt-jBJ4 at 19:50  */
class SelfDeleter
{
public:
  void selfDelete() { delete this; } // can we aslo set this to nullptr?
};


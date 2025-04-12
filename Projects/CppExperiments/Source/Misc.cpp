//#include "Common.h"

//=================================================================================================
// Observer Pattern

class Observee;  // forward declaration

//-------------------------------------------------------------------------------------------------
/** Baseclass for objects that need to keep track of the state of an Observee object. Subclasses 
must implement the function observeeStateChanged and do whatever they need to do in response to
that state change. */

class AbstractObserver
{

public:

  /** Destructor of classes with purely virtual functions should be virtual - otherwise the 
  compiler warns. ...why? */
  virtual ~AbstractObserver() {}

  /** Callback function that must be overriden by subclasses to respond to changes of an Observee
  object. */
  virtual void observeeStateChanged(Observee* o) = 0;

};

//-------------------------------------------------------------------------------------------------
/** Class of objects whose state must be monitored by observer objects. */

class Observee
{

public:

  void registerObserver(AbstractObserver* o) 
  {  
    observers.push_back(o); // better: use some sort of appendIfNotThere function
  }

  void deregisterObserver(AbstractObserver* o)
  {
    // removeValue(observers, o);
  }

  /** Sends a notification to all our observers, that our state has changed. */
  void notifyObservers()
  {
    for(auto& o : observers)     // loop over all our observers
      o->observeeStateChanged(this);
  }

  /** Changes the state and notifies all our observers about the state change. */
  void changeSomething(int newValue)
  {
    something = newValue;
    notifyObservers();
  }

  std::vector<AbstractObserver*> observers;


  // demo facilities:
  Observee(const std::string name) : _name(name) {}
  std::string getName() const { return _name; }
  int getSomething() const { return something; }
  std::string _name;
  int something = 0;
};

//-------------------------------------------------------------------------------------------------

/** Concrete subclass of AbstractObserver that responds to changes of the state of registered
Observee objects by printing a message. */

class ConcreteObserver : public AbstractObserver
{

public:

  virtual void observeeStateChanged(Observee* o) override
  {
    handleChange(o);
  }

  void handleChange(Observee* o)
  {
    std::cout << typeid(*this).name() << "::" << __func__ << " of object " << _name;
    std::cout << " was called with observee " << o->getName();
    std::cout << ". The observee's new state is " << o->getSomething()<< ".\n\n";
  }

  // demo facilities:
  ConcreteObserver(const std::string name) : _name(name) {} 
  std::string _name;
};

//-------------------------------------------------------------------------------------------------

void demoObserver()
{
  Observee observee("Foo");

  ConcreteObserver observer1("Bar");
  ConcreteObserver observer2("Frob");

  observee.registerObserver(&observer1);
  observee.registerObserver(&observer2);

  observee.changeSomething(42);
  observee.changeSomething(73);
}

// Variations:
// -the Observer class can be defined inside the Observee class. For example, instead of having 
//  classes Button and ButtonObserver, one could have Button and Button::Observer - that's actually
//  nicer because it doesn't pollute the namespace so much.

//=================================================================================================

void demoOptional()
{
  // Demonstrates usage of std::optional, see:
  // https://en.cppreference.com/w/cpp/utility/optional

  using MaybeInt = std::optional<int>;
  bool ok = true;

  // Division of an (optional) integer by another integer may either produce another integer as
  // result or no result. The latter case occurs, when the divisor is zero or if one or both of 
  // the operands do not have a value:
  auto div = [](MaybeInt a, MaybeInt b)
  {
    if(a.has_value() && b.has_value() && b.value() != 0) 
      return MaybeInt(a.value() / b.value());
    else       
      return MaybeInt();  // Default constructor produces an optional without a value
  };

  MaybeInt c;  // c may be an integer or undefined

  c = div(15, 3);
  ok &= c.has_value() == true;
  //ok &= c             == true;   // can(?) be converted to bool -> true if it has a value
  ok &= c.value()     == 5;

  c = div(15, 0);
  ok &= c.has_value() == false; // Division by zero yields no value
  //ok &= c             == false;  // .hmm - conversion to bool doesn't seem to work
  try
  {
    // Trying to access a nonexistent value in an optional should throw std::bad_optional_access:
    ok &= c.value() == 0;
    ok &= false;  // we never get here
  }
  catch(std::bad_optional_access)
  {
    ok &= true;
    // Even when turning exceptions off (under "Code Generation"), we enter this branch. 
    // Interesting! ...but why?
  }

  MaybeInt a = 15, b = 5;
  b = div(a, c); ok &= b.has_value() == false; // Division by non-value yields no value
  b = div(c, a); ok &= b.has_value() == false; // Division of non-value yields no value

  //c = a + b;  
  // operator not defined...maybe we can define arithemtic operators for std::optional outside
  // the class


  int dummy = 0;

}

// ToDo:
// https://en.cppreference.com/w/cpp/utility/variant

//=================================================================================================


bool testAllocationLogger()
{
  bool ok = true;

  // Helper function to check if the number of allocations, deallocations, etc. that occurred 
  // matches the expected values:
  auto checkAllocState = [&](size_t expectedAllocs, size_t expectedDeallocs, size_t expectedChunks)
  {
    bool ok = true;
    ok &= heapAllocLogger.getNumAllocations()     == expectedAllocs;
    ok &= heapAllocLogger.getNumDeallocations()   == expectedDeallocs;
    ok &= heapAllocLogger.getNumAllocatedChunks() == expectedChunks;
    return ok;
  };

  // Reset the counters in the heap allocation logger:
  heapAllocLogger.reset();

  // We did not yet allocate anything, so initially, we expect all counters to be zero:
  ok &= checkAllocState(0, 0, 0);

  // Test if logger registers direct invocations of malloc and free:
  double* pDouble = (double*) malloc(10 * sizeof(double));
  ok &= checkAllocState(1, 0, 1);
  free(pDouble);
  ok &= checkAllocState(1, 1, 0);

  // Test logging for operators new and delete:
  pDouble = new double;
  ok &= checkAllocState(2, 1, 1);
  delete pDouble;
  ok &= checkAllocState(2, 2, 0);

  // Test logging for new[] and delete[]:
  pDouble = new double[10];
  ok &= checkAllocState(3, 2, 1);
  delete[] pDouble;
  ok &= checkAllocState(3, 3, 0);


  /*
  {
    //std::vector<double> vd;       // This does one allocation!   Why?
    std::vector<double> vd(10);     // This does two allocations!
    ok &= checkAllocState(5, 3, 1);
  }
  */


  // Return unit test result:
  return ok;


  // ToDo:
  //
  // - Check allocation logging of STL containers such as std::vector
  //
  // - Implement and check logging of calloc and realloc. 
}

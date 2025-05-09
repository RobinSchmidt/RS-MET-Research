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
  auto checkAllocState = [&](size_t expectedAllocs, size_t expectedDeallocs)
  {
    bool ok = true;
    ok &= heapAllocLogger.getNumAllocations()     == expectedAllocs;
    ok &= heapAllocLogger.getNumDeallocations()   == expectedDeallocs;
    return ok;
  };

  // Reset the counters in the heap allocation logger:
  heapAllocLogger.reset();

  // We did not yet allocate anything, so initially, we expect all counters to be zero:
  ok &= checkAllocState(0, 0);

  // Data type that we use for the allocation logging tests. We will allocate buffers, vectors etc.
  // of that type:
  using Data = double;


  //---------------------------------------------------------------------------
  // Test logging of low level (de)allocations (malloc, free, new, etc.)

  heapAllocLogger.reset();

  // Test if logger registers direct invocations of malloc and free:
  {
    Data* ptr = (Data*)malloc(10 * sizeof(Data));
    ok &= checkAllocState(1, 0);
    free(ptr);
  }
  ok &= checkAllocState(1, 1);

  // Test logging for operators new and delete:
  {
    Data* ptr = new Data;
    ok &= checkAllocState(2, 1);
    delete ptr;
  }
  ok &= checkAllocState(2, 2);

  // Test logging for new[] and delete[]:
  {
    Data* ptr = new Data[10];
    ok &= checkAllocState(3, 2);
    delete[] ptr;
  }
  ok &= checkAllocState(3, 3);


  //---------------------------------------------------------------------------
  // Test logging for std::unique_ptr

  heapAllocLogger.reset();

  // Test std::make_unique and automatic release:
  {
    auto p = std::make_unique<Data>();
    ok &= checkAllocState(1, 0);
  }
  ok &= checkAllocState(1, 1);


  //---------------------------------------------------------------------------
  // Test logging of (de)allocations for std::vector

  // The tests below use expectations that reflect the behavior of the MSVC compiler of Visual
  // Studio 2019. They are actually not what I would naturally expect to happen. Apparently, MSVC 
  // does some extra allocations. So, whether or not the tests below leave the ok variable in true
  // state may be compiler dependent. Well, actually, it's dependent on the implementation of the 
  // standard library (but these usually ship together with a compiler and/or IDE).

  heapAllocLogger.reset();

  // Default constructor:
  {
    std::vector<Data> v;             // This does one allocation in MSVC! Why?
    ok &= checkAllocState(1, 0);
  }
  ok &= checkAllocState(1, 1);

  // Constructor that takes a size:
  {
    std::vector<Data> v(10);         // This does two allocations in MSVC! Why?
    ok &= checkAllocState(3, 1);
  }
  ok &= checkAllocState(3, 3);

  // Copy constructor:
  {
    std::vector<Data> v(10);         // Two allocations in MSVC.
    ok &= checkAllocState(5, 3);
    std::vector<Data> v2(v);         // Another two allocations.
    ok &= checkAllocState(7, 3);
  }
  ok &= checkAllocState(7, 7);



  // Return unit test result:
  return ok;


  // Observations:
  //
  // - In MSVC, the standard constructor of std::vector does a memory allocation. That is a very 
  //   unexpected behavior. What (the fuck) does it allocate? Moreover, the constructor which takes
  //   an initial size (which is expected to allocate once), allocates twice. What is going on 
  //   there? ...well - looking at the implementation of the standard constructor of std::vector,
  //   these seems to be something strange going on. The vector has a data member _Mypair of type
  //   ... and the constructor calls this:
  //
  //     _Mypair._Myval2._Alloc_proxy(_GET_PROXY_ALLOCATOR(_Alty, _Getal()))
  //
  //   I have no idea what this is, what it does and  why. But apparently, it triggers an 
  //   allocation. Try to figure out, if this also happens in release builds. Maybe write a message 
  //   to the command line with the number of allocs and deallocs.
  //
  //
  // ToDo:
  //
  // - Check allocation logging on more STL containers like list, map, deque, etc. But their 
  //   behavior may be even less predictable than that of std::vector and may vary even more 
  //   between different standard library implementations, I guess. Because they are more complex
  //   than a simple std::vector. But it's actually pretty nice that we can learn a couple of 
  //   things about implementation details of the standard library by using this heap allocation 
  //   logging technique. Without it, I would never ever have known that std::vector does these 
  //   weird extra allocations. And knowing such things may actually be important when writing
  //   realtime safe code. We should have unit tests that make sure that we don't have any hidden
  //   allocations in code that needs to be realtime safe.
  //
  // - Test copying, moving, assigning etc. of unique pointers. Test also other smart pointers
  //   (i.e. std::shared_ptr and std::weak_ptr) and their interactions (e.g. assigning a shared_ptr
  //   to a unqiue_ptr and the other way around (if this is even possible) and so on
  //
  // - Maybe use heapAllocLogger.reset(); between the tests to decouple their expected values. 
  //   Maybe factor out some tests. ...but maybe not
  //
  // - Implement and check logging of calloc and realloc. 
  //
  // - Maybe write a custom allocator class that logs allocations such that we can pass it to the
  //   std::vector that we use in e.g. rsMatrix. It currently has this ugly instrumentation code to
  //   log the allocations which we use in the unit test to make sure that we do not miss any 
  //   unexpected extra allocations. But the disadvantage is that this complicates the API of 
  //   rsMatrix because we will then always have to pass this additional template parameter for the
  //   allocator type. This will increase the textual noise in the codebase considerably. 
  //   Alternativly, we may replace all usages of std::vector by an API compatible drop-in 
  //   replacement rsVector that does the logging in debug builds. This class may internally use 
  //   std::vector, so we can still inspect the contents of vectors in the debugger (but this will 
  //   then require one click more because the actual content of the vector will be one "folder" 
  //   level deeper into the data structure. This seems to be only a minor inconvenience but I 
  //   guess it may add up in more intense debugging sessions. But wait: We do not necessarily have
  //   to pass the allocator to every template invocation of rsMatrix. We could also use 
  //   conditional compilation: in debug builds, we always use the logging allocator and in release
  //   builds, we use the default allocator. It may uglify the implementation of rsMatrix a bit but
  //   I think, that might be acceptable. It would look like:
  //
  //     #if defined(RS_DEBUG)
  //       std::vector<T, rsLoggingAllocator> elems;
  //     #else
  //       std::vector<T> elems;
  //
  //   rather that just the else-branch. ...but actually: we ant to run the unit tests also with a
  //   build in release configuration. But when doing it like above, we would break the allocation
  //   test for rsMatrix. Maybe we should define a special macro RS_LOG_ALLOCS which we can #define
  //   also in release builds.
  //
  // - Figure out if this weird behavior of std::vector with these unexpected extra allocations 
  //   also happens with other compilers (i.e. gcc and clang).
}

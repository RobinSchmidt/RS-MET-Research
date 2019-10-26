#include "Common.h"

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
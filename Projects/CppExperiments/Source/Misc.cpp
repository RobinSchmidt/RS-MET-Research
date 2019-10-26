#include "Common.h"

//=================================================================================================
// Observer Pattern

class Observee;  // forward declaration

//-------------------------------------------------------------------------------------------------

class AbstractObserver
{

public:

  /** Destructor of classes with purely virtual functions should be virtual - otherwise the 
  compiler warns. ...why? */
  virtual ~AbstractObserver() {}

  /** Callback function that must be overriden by subclasses to respond to changes of an Observee
  object. */
  virtual void somethingChanged(Observee* o) = 0;

};

//-------------------------------------------------------------------------------------------------

class Observee
{

public:

  void addObserver(AbstractObserver* o) 
  {  
    observers.push_back(o); // better: use some sort of appendIfNotThere function
  }

  void removeObserver(AbstractObserver* o)
  {
    // removeValue(observers, o);
  }

  void notifyObservers()
  {
    for(auto& o : observers)
      o->somethingChanged(this);
  }

  void changeSomething(int newValue)
  {
    something = newValue;
    notifyObservers();
  }

  Observee(const std::string name) : _name(name) {} // demo facility
  std::string getName() const { return _name; }     // demo facility
  int getSomething() const { return something; }    // demo facility

protected:

  std::vector<AbstractObserver*> observers;

  std::string _name;  // demo facility
  int something = 0;  // demo facility
};

//-------------------------------------------------------------------------------------------------

class ConcreteObserver : public AbstractObserver
{

  virtual void somethingChanged(Observee* o) override
  {
    handleChange(o);
  }

  void handleChange(Observee* o)
  {
    std::cout << typeid(*this).name() << "::" << __func__;
    std::cout << " was called with observee " << o->getName() << ".\n";
    std::cout << "The observee's new state is " << o->getSomething()<< ".\n\n";
  }
};


//-------------------------------------------------------------------------------------------------

void demoObserver()
{
  Observee observee("Foo");
  ConcreteObserver observer;
  observee.addObserver(&observer);
  observee.changeSomething(42);
}

//=================================================================================================
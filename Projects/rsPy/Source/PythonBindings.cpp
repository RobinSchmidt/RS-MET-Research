#include "../JuceLibraryCode/JuceHeader.h"

namespace { // Avoid cluttering the global namespace.

  // A friendly class.
  class hello
  {
  public:
    hello(const std::string& country) { this->country = country; }
    std::string greet() const { return "Hello from " + country; }
  private:
    std::string country;
  };

  // A function taking a hello object as an argument.
  std::string invite(const hello& w) {
    return w.greet() + "! Please come soon!";
  }
}

// example code to create the bindings:
BOOST_PYTHON_MODULE(rsPy) // name here *must* match the name of module's dll file (rsPy.pyd)
{
  using namespace boost::python;
  class_<hello>("hello", init<std::string>())
    .def("greet", &hello::greet)   // Add a regular member function
    .def("invite", invite)         // Add invite() as a member of hello
    ;
  def("invite", invite);           // Also add invite() as a regular function to the module.
}


// examples how to create the bindings:
// https://wiki.python.org/moin/boost.python/SimpleExample







/*
hmm..i'm getting a linker error:  cannot open file 'boost_python34-vc142-mt-gd-x64-1_70.lib'
i've never said anywehere that i want to link to such a library - where does the linker get
the idea from, that it should?

this says that boost.python *must* be built seperately
https://www.boost.org/doc/libs/1_60_0/more/getting_started/windows.html
...does that mean, we can't use a juce module? ...how does the boost code determine the file 
name and that it want to link to some file at all?! such stuff is actually normally set up in 
the project settings in the IDE - it should not even be possible to set up such things in the
actualy c++ code - but for some reason, the linker know, what filename it wants to find
..maybe there's a def file somewhere?
https://docs.microsoft.com/en-us/cpp/build/reference/module-definition-dot-def-files?view=vs-2019
...nope - i found none
*/



// https://stackoverflow.com/questions/13042561/fatal-error-lnk1104-cannot-open-file-libboost-system-vc110-mt-gd-1-51-lib



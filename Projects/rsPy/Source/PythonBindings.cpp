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

BOOST_PYTHON_MODULE(extending)
{
  using namespace boost::python;
  class_<hello>("hello", init<std::string>())
    // Add a regular member function.
    .def("greet", &hello::greet)
    // Add invite() as a member of hello!
    .def("invite", invite)
    ;

  // Also add invite() as a regular function to the module.
  def("invite", invite);
}
#include "../JuceLibraryCode/JuceHeader.h"

namespace Test { // Avoid cluttering the global namespace.

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


  double rsSin(double x) { return ::sin(x); } // to disambiguate which overload should be taken

  std::complex<double> rsExp(std::complex<double> z) { return std::exp(z); }

  void mul(boost::python::numpy::ndarray& a, double scaler)
  //void mul(boost::python::numpy::ndarray a, double scaler)
  {
    // todo: make sure that it is an array of double - otherwise, raise an exception or something:
    //boost::python::numpy::dtype type = a.get_dtype();


    /*
    int numDims = a.get_nd();                   // todo: check, if this is 1
    const Py_intptr_t* sizes = a.get_shape();
    int size    = sizes[0];                     // will work only for 1D arrays - maybe should be long long?
    // todo: make it work for any dimensionality

    double* data = reinterpret_cast<double*> (a.get_data());
    for(int i = 0; i < size; i++)
      data[i] *= scaler;
    */
  }
  // https://jleem.bitbucket.io/code.html
  // https://www.boost.org/doc/libs/1_64_0/libs/python/doc/html/numpy/index.html
  // https://www.boost.org/doc/libs/1_63_0/libs/python/doc/html/numpy/tutorial/simple.html

}



// example code to create the bindings:
BOOST_PYTHON_MODULE(rsPy) // name here *must* match the name of module's dll file (rsPy.pyd)
{
  using namespace boost::python;

  numpy::initialize();
  //boost::python::numpy::initialize();
  // i think, it fills out the pointers void **PyArray_API and void** PyUFunc_API, declared as 
  // extern in boost::python and defined in rs_boost.cpp. In numpy.hpp, it is said that this 
  // function should be called before using anything in boost.numpy.


  class_<Test::hello>("hello", init<std::string>())
    .def("greet", &Test::hello::greet)   // Add a regular member function
    .def("invite", Test::invite)         // Add invite() as a member of hello
    ;
  def("invite", Test::invite);           // Also add invite() as a regular function to the module.


  // Classes:


  // Math Functions:
  def("sin", Test::rsSin);         // double -> double
  def("ellipj_cn", rosic::cn);     // double, double -> double
  def("exp", Test::rsExp);         // complex -> complex

  // String Functions:

  // NumPy Array Functions:
  def("mul", Test::mul);

}

// todo: figure out, how we can deal with numpy arrays...maybe it would be convenient, if 1D arrays 
// automatically map to std::vector
// see here for documentation:
// https://www.boost.org/doc/libs/1_66_0/libs/python/doc/html/index.html

// here is how to design a python package:
// https://www.boost.org/doc/libs/1_66_0/libs/python/doc/html/tutorial/tutorial/techniques.html
// we copuld have rsPy.audio.filters, rsPy.audio.trafos, rsPy.math.linalg, rsPy.math.calc, 
// rsPy.math.stats, rsPy.graphics, rsPy.dsp etc.

// https://www.boost.org/doc/libs/1_66_0/libs/python/doc/html/reference/high_level_components/boost_python_def_hpp.html#high_level_components.boost_python_def_hpp.functions


// examples how to create the bindings:
// https://wiki.python.org/moin/boost.python/SimpleExample

// https://stackoverflow.com/questions/10701514/how-to-return-numpy-array-from-boostpython







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



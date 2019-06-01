#include "../JuceLibraryCode/JuceHeader.h"

//extern void** PyArray_API;  // for debugging numpy issues

//void** PyArray_API = nullptr;
//void** PyUFunc_API = nullptr;

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


  double rsSin(double x) 
  { 
    return ::sin(x); 
  } // to disambiguate which overload should be taken

  std::complex<double> rsExp(std::complex<double> z) 
  { 
    return std::exp(z); 
  }

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
  // https://www.boost.org/doc/libs/1_64_0/libs/python/doc/html/numpy/index.html
  // https://www.boost.org/doc/libs/1_63_0/libs/python/doc/html/numpy/tutorial/simple.html


  // example from: https://jleem.bitbucket.io/code.html
  using namespace boost::python;
  namespace np = boost::python::numpy;
  double eucnorm(np::ndarray axis) {
    //return 0.0;  // test

    const int n = axis.shape(0);
    double norm = 0.0;
    for(int i = 0; i < n; i++) {
      double A = extract<double>(axis[i]);
      norm += A * A;
    }
    return sqrt(norm);
  }


  //double npArrayTest(np::ndarray* a) // python argument type did not macth c++ signature
  //double npArrayTest(np::ndarray a) // The debug adapter exited unexpectedly
  double npArrayTest(np::ndarray& a)
  {
    return 0.0;
  }

  // trying to create and return a numpy::ndarray
  np::ndarray npArrayCreate(int size, double value)
  {
    tuple shape = make_tuple(size);
    np::dtype dtype = np::dtype::get_builtin<double>();
    np::ndarray a = np::zeros(shape, dtype);  // maybe use  np::empty

    // todo: fill the array with "value"

    return a;
  }

  // some code for debugging and figuring out, why the numpy initialization fails:
  long long pyArrayAPI()
  {
    return (long long)(getPyArrayAPI());
  }
  void initArrayAPI()
  {
    // partially recreates _import_array(void) from __multiarray_api.h
    PyObject* numpy = PyImport_ImportModule("numpy.core.multiarray");
    PyObject* c_api = PyObject_GetAttrString(numpy, "_ARRAY_API");
    void** api = (void**)PyCapsule_GetPointer(c_api, NULL);
    setPyArrayAPI(api);

    //PyArray_API = (void**)PyCapsule_GetPointer(c_api, NULL);

  }


}



// example code to create the bindings:
BOOST_PYTHON_MODULE(rsPy) // name here *must* match the name of module's dll file (rsPy.pyd)
{
  using namespace boost::python;


  //Py_Initialize();
  // ...is done in main() here:
  // https://www.boost.org/doc/libs/1_63_0/libs/python/doc/html/numpy/tutorial/simple.html
  // but doesn't seem necessary

  numpy::initialize();
  // i think, it fills out the pointers void **PyArray_API and void** PyUFunc_API, declared as 
  // extern in boost::python and defined in rs_boost.cpp. In numpy.hpp, it is said that this 
  // function should be called before using anything in boost.numpy. but: regardless whether or not
  // we call this, we get the same "The debug adapter exited unexpectedly" errors/crashes in 
  // rsPyTest as soon as we try to do anything with numpy arrays - even just passing a reference or
  // copy to a function - does the initialization fail? or do we have to call it in a different 
  // place?
  // it seems to call _import_array(void) in __multiarray_api.h, line 1629
  // if we can't get debug breakpoints to work, maybe write functions that return the values of
  // our PyArray_API, PyUFunc_API pointers - see if they are still nullptrs even after calling 
  // initialize() - yes, that is indeed the case! ...so what can we do to figure out where it goes 
  // wrong? maybe hack into __multiarray_api.h to produce logging output, to see which path the 
  // code takes?

  //setPyArrayAPI(1);  
  // test - makes no sense, i just want to see, if we can set the pointer here - ok, this works
  // ...soo...it seems to be indeed the case that something in numpy::initialize goes wrong and our
  // PyArray_API is *not* correctly filled out at initialization -> figure out which path the code 
  // takes in _import_array! ...or maybe partially re-create the function here but *with* logging

  // or maybe this could be the solution:
  // https://docs.microsoft.com/en-us/visualstudio/python/debugging-mixed-mode-c-cpp-python-in-visual-studio?view=vs-2019


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
  def("arrayAPI", Test::pyArrayAPI); // for debug
  def("mul", Test::mul);             // doesn't work
  def("eucnorm", Test::eucnorm);
  def("npArrayTest", Test::npArrayTest);
  def("npArrayCreate", Test::npArrayCreate);
  def("initArrayAPI", Test::initArrayAPI);
 
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



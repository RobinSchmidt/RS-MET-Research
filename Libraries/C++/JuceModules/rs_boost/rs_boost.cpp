#ifdef RS_TESTING_H_INCLUDED
/* When you add this cpp file to your project, you mustn't include it in a file where you've
already included any other headers - just put it inside a file on its own, possibly with your config
flags preceding it, but don't include anything else. That also includes avoiding any automatic prefix
header files that the compiler may be using. */
#error "Incorrect use of JUCE cpp file"
#endif

#include "rs_boost.h"

//=================================================================================================
// boost::python

#include "libs//python/src/dict.cpp"
#include "libs//python/src/errors.cpp"
#include "libs//python/src/exec.cpp"
#include "libs//python/src/import.cpp"
#include "libs//python/src/list.cpp"
#include "libs//python/src/long.cpp"
#include "libs//python/src/module.cpp"
#include "libs//python/src/object_operators.cpp"
#include "libs//python/src/object_protocol.cpp"
#include "libs//python/src/slice.cpp"
#include "libs//python/src/str.cpp"
#include "libs//python/src/tuple.cpp"
#include "libs//python/src/wrapper.cpp"

#include "libs//python/src/converter/arg_to_python_base.cpp"
#include "libs//python/src/converter/builtin_converters.cpp"
#include "libs//python/src/converter/from_python.cpp"
#include "libs//python/src/converter/registry.cpp"
#include "libs//python/src/converter/type_id.cpp"

#include "libs//python/src/object/class.cpp"
#include "libs//python/src/object/enum.cpp"
#include "libs//python/src/object/function.cpp"
#include "libs//python/src/object/function_doc_signature.cpp"
#include "libs//python/src/object/inheritance.cpp"
#include "libs//python/src/object/iterator.cpp"
#include "libs//python/src/object/life_support.cpp"
#include "libs//python/src/object/pickle_support.cpp"
#include "libs//python/src/object/stl_iterator.cpp"

#include <numpy/ndarraytypes.h>
#include <numpy/__multiarray_api.h>
// without these, we get an "identifier not found" error for import_array() which is a macro
// (there's also a similar definition for import_ufunc() in __ufunc_api.h but maybe that's 
// irrelevant). maybe we are not supposed to include them and they are rather supposed
// to be included from <numpy/arrayobject.h>, <numpy/ufuncobject.h> but my python installation
// is too old?
// here is some info about that error:
// https://stackoverflow.com/questions/32899621/numpy-capi-error-with-import-array-when-compiling-multiple-modules
// https://docs.scipy.org/doc/numpy-1.10.1/reference/c-api.array.html#miscellaneous

#include "libs//python/src/numpy/dtype.cpp"
#include "libs//python/src/numpy/matrix.cpp"
#include "libs//python/src/numpy/ndarray.cpp"
#include "libs//python/src/numpy/numpy.cpp"
#include "libs//python/src/numpy/scalars.cpp"
#include "libs//python/src/numpy/ufunc.cpp"

void **PyArray_API = nullptr;
void **PyUFunc_API = nullptr;
// Thes pointers are declared as extern in 
// Anaconda3\Lib\site-packages\numpy\core\include\numpy\__multiarray_api.h, __ufunc_api.h and are 
// supposed to be defined somewhere else - which is here. They will be filled in by the 
// initialization code when the module is loaded by the python interpreter. Without these 
// definitions, we'll get "unresolved external symbol" linker errors. More information:
// https://docs.scipy.org/doc/numpy-1.13.0/reference/c-api.ufunc.html#importing-the-api 
// ...it says:
// The C-API is actually an array of function pointers. This array is created (and pointed to by a 
// global variable) by import_ufunc. The global variable is either statically defined or allowed to be 
// seen by other files depending on the state of PY_UFUNC_UNIQUE_SYMBOL and NO_IMPORT_UFUNC.

void initArrayAPI()
{
  // partially recreates _import_array(void) from __multiarray_api.h
  PyObject* numpy = PyImport_ImportModule("numpy.core.multiarray");
  PyObject* c_api = PyObject_GetAttrString(numpy, "_ARRAY_API");
  PyArray_API = (void**)PyCapsule_GetPointer(c_api, NULL);
  //PyArray_API = (void**)PyCObject_AsVoidPtr(c_api); // from other #ifdef path -> compile error
}
void initUFuncAPI()
{
  // partially recreates _import_umath(void) from __ufunc_api.h - not yet tested
  PyObject* numpy = PyImport_ImportModule("numpy.core.umath");
  PyObject* c_api = PyObject_GetAttrString(numpy, "_UFUNC_API");
  PyUFunc_API = (void**)PyCapsule_GetPointer(c_api, NULL);
  //PyUFunc_API = (void**)PyCObject_AsVoidPtr(c_api);  // other #ifdef branch
}
void initNumPy()
{
  initArrayAPI();
  initUFuncAPI(); // not yet tested
}



// some older links, collected while figuring out the linker issues - may not be relevant anymore:
// https://social.msdn.microsoft.com/Forums/vstudio/en-US/d94f6af3-e330-4962-a150-078da57ee5d0/error-lnk2019-unresolved-external-symbol-quotdeclspecdllimport-public-thiscall?forum=vcgeneral
// https://github.com/boostorg/python/issues/134
// https://stackoverflow.com/questions/45069253/boost-numpy-linker-error-in-windows-vs
// https://docs.scipy.org/doc/numpy-1.13.0/reference/c-api.html

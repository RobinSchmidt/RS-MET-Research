#ifdef RS_TESTING_H_INCLUDED
/* When you add this cpp file to your project, you mustn't include it in a file where you've
already included any other headers - just put it inside a file on its own, possibly with your config
flags preceding it, but don't include anything else. That also includes avoiding any automatic prefix
header files that the compiler may be using. */
#error "Incorrect use of JUCE cpp file"
#endif

#include "rs_boost.h"



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

// ..but with these we still get linker errors for numpy, so for the time being, all numpy stuff
// is commented out:

#include "libs//python/src/numpy/dtype.cpp"
#include "libs//python/src/numpy/matrix.cpp"
#include "libs//python/src/numpy/ndarray.cpp"
#include "libs//python/src/numpy/numpy.cpp"
#include "libs//python/src/numpy/scalars.cpp"
#include "libs//python/src/numpy/ufunc.cpp"

//BOOST_NUMPY_ARRAY_API = 0;
//BOOST_UFUNC_ARRAY_API;
void **PyArray_API = nullptr;   // ok - this fixes one of the linker errors
//declared as extern in C:\Users\Rob\Anaconda3\Lib\site-packages\numpy\core\include\numpy\__multiarray_api.h, line 807

void **PyUFunc_API = nullptr;

// hmm...these functions in question are marked as __declspec(dllimport) - so i guess that means
// they are supposed to be dynamically linked...but maybe we nevertheless need to link to some
// .lib file, see here: https://social.msdn.microsoft.com/Forums/vstudio/en-US/d94f6af3-e330-4962-a150-078da57ee5d0/error-lnk2019-unresolved-external-symbol-quotdeclspecdllimport-public-thiscall?forum=vcgeneral

// maybe this one
// C:\Users\Rob\Anaconda3\Lib\site-packages\numpy\core\lib\npymath.lib
// or
// C:\Users\Rob\Anaconda3\Lib\site-packages\numpy\core\*.pyd (for example: multiarray.pyd ...these are dlls)

// this might be relevant, too:
// https://github.com/boostorg/python/issues/134
// https://stackoverflow.com/questions/45069253/boost-numpy-linker-error-in-windows-vs

// OR: maybe we need to set a macro, such that these functions will become marked as dllexport 
// instead of import? yes - that seems to be the solution.

// we still get 2 linker errors:
// include_rs_boost.obj : error LNK2001 : unresolved external symbol BOOST_NUMPY_ARRAY_API
// include_rs_boost.obj : error LNK2001 : unresolved external symbol BOOST_UFUNC_ARRAY_API

// these are macros used in: 
//   \boost\python\numpy\internal.hpp
// and defined in:
//   C:\Users\Rob\Anaconda3\Lib\site-packages\numpy\core\include\numpy\__multiarray_api.h and __ufunc_api.h
// ...or are they define there? vs opens this file when click on "go to declaration"
// but it jumps on PyObject_HEAD ...wtf? PyObject_HEAD itself is defined in
// C:\Users\Rob\Anaconda3\include\object.h  as
// /* PyObject_HEAD defines the initial segment of every PyObject. */
// #define PyObject_HEAD                   PyObject ob_base;
// but pointing with the mouse on BOOST_NUMPY_ARRAY_API, it says void** - so does this mean, the 
// macro resolves to void**?

// here's a reference for the numpy c-api - maybe there some ifo can be found, to which library
// we are supposed to link:
// https://docs.scipy.org/doc/numpy-1.13.0/reference/c-api.html








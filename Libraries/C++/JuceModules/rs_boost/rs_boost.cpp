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

//#include <numpy/ndarraytypes.h>
//#include <numpy/__multiarray_api.h>
// without these, we get an "identifier not found" error for import_array() which is a macro
// (there's also a similar definition for import_ufunc() in __ufunc_api.h but maybe that's 
// irrelevant). maybe we are not supposed to to include them and they are rather supposed
// to be included from <numpy/arrayobject.h>, <numpy/ufuncobject.h> but my python installation
// is too old?
// here is some info about that error:
// https://stackoverflow.com/questions/32899621/numpy-capi-error-with-import-array-when-compiling-multiple-modules
// https://docs.scipy.org/doc/numpy-1.10.1/reference/c-api.array.html#miscellaneous

// ..but with these we still get linker errors for numpy, so for the time being, all numpy stuff
// is commented out:
/*
#include "libs//python/src/numpy/dtype.cpp"
#include "libs//python/src/numpy/matrix.cpp"
#include "libs//python/src/numpy/ndarray.cpp"
#include "libs//python/src/numpy/numpy.cpp"
#include "libs//python/src/numpy/scalars.cpp"
#include "libs//python/src/numpy/ufunc.cpp"
*/

/* Now that we have the dll, the next step is to figure out, how to use it in python - how do we
import it? */



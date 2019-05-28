#ifdef RS_TESTING_H_INCLUDED
/* When you add this cpp file to your project, you mustn't include it in a file where you've
already included any other headers - just put it inside a file on its own, possibly with your config
flags preceding it, but don't include anything else. That also includes avoiding any automatic prefix
header files that the compiler may be using. */
#error "Incorrect use of JUCE cpp file"
#endif

#include "rs_boost.h"

#include "libs//python/src/dict.cpp"
//#include "libs//python/src/errors.cpp"
//#include "libs//python/src/exec.cpp"
//#include "libs//python/src/import.cpp"
#include "libs//python/src/list.cpp"
#include "libs//python/src/long.cpp"
//#include "libs//python/src/module.cpp"
//#include "libs//python/src/object_operators.cpp"
//#include "libs//python/src/object_protocol.cpp"
#include "libs//python/src/slice.cpp"
#include "libs//python/src/str.cpp"
#include "libs//python/src/tuple.cpp"
#include "libs//python/src/wrapper.cpp"

#include "libs//python/src/converter/arg_to_python_base.cpp"
//#include "libs//python/src/converter/builtin_converters.cpp"
//#include "libs//python/src/converter/from_python.cpp"
//#include "libs//python/src/converter/registry.cpp"
//#include "libs//python/src/converter/type_id.cpp"

/*
// does not work yet - apparently, we also need the numpy path to the include path...
#include "libs//python/src/numpy/dtype.cpp"
#include "libs//python/src/numpy/matrix.cpp"
#include "libs//python/src/numpy/ndarray.cpp"
#include "libs//python/src/numpy/numpy.cpp"
#include "libs//python/src/numpy/scalars.cpp"
#include "libs//python/src/numpy/ufunc.cpp"
*/

//#include "libs//python/src/object/class.cpp"
#include "libs//python/src/object/enum.cpp"
//#include "libs//python/src/object/function.cpp"
#include "libs//python/src/object/function_doc_signature.cpp"
//#include "libs//python/src/object/inheritance.cpp"
//#include "libs//python/src/object/iterator.cpp"
#include "libs//python/src/object/life_support.cpp"
#include "libs//python/src/object/pickle_support.cpp"
#include "libs//python/src/object/stl_iterator.cpp"


// some .cpp files from the python folder are apparently not supposed to be included (what are they
// good for?) so they are commented out
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
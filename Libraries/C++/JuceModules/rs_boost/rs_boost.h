/*******************************************************************************
 The block below describes the properties of this module, and is read by
 the Projucer to automatically generate project code that uses it.
 For details about the syntax and how to create or use a module, see the
 JUCE Module Format.txt file.


 BEGIN_JUCE_MODULE_DECLARATION

  ID:               rs_boost
  vendor:           RS-MET
  version:          0.0.1
  name:             Rob's Boost Module
  description:      Subset of the boost libraries wrapped into a JUCE module
  website:          https://www.boost.org/
  license:          Boost

  dependencies: 
  OSXFrameworks:
  iOSFrameworks:

 END_JUCE_MODULE_DECLARATION

*******************************************************************************/

#pragma once

#define BOOST_ALL_NO_LIB
// Without this definition, the inclusion of <boost/config/auto_link.hpp>, which happens somewhere 
// deep down in boost between macro definitions and undefinitions (for example in
// <boost/python/detail/config.hpp>), will cause the linker to try to link to a static library file
// that is nowhere specified in the project settings (i really wonder, how that works). So we need
// this definition, to avoid linker errors of the type:
//   cannot open file 'boost_python34-vc142-mt-gd-x64-1_70.lib'
// what all these mt-gd-$&%§@ decorations mean, is explained here:
//   https://www.boost.org/doc/libs/1_60_0/more/getting_started/windows.html#library-naming


#define BOOST_PYTHON_SOURCE     // makes BOOST_PYTHON_DECL expand to __declspec(dllexport)
#define BOOST_NUMPY_SOURCE      // test - seems to fix some (but not all) linker errors

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION  // test - gets rid of a warning



// some tests to get rid of the numpy linker errors
//#define BOOST_PYTHON_NUMPY_INTERNAL  // test - doesn't help with numpy linker error
//#define BOOST_PYTHON_NUMPY_INTERNAL  // test - nope!
//#define _MULTIARRAYMODULE  // gives compile errors
//#undef NO_IMPORT
//#undef NO_IMPORT_ARRAY


// defining these turns the numpy liker errors into compiler errors:
//#define BOOST_NUMPY_ARRAY_API
//#define BOOST_UFUNC_ARRAY_API
// it seems they are defined as pointers-to-pointers-to-void? should they point to some global 
// object? how is this supposed to work?

// here: https://docs.scipy.org/doc/numpy-1.13.0/reference/c-api.ufunc.html#importing-the-api 
// it says:
// The C-API is actually an array of function pointers. This array is created (and pointed to by a 
// global variable) by import_ufunc. The global variable is either statically defined or allowed to be 
// seen by other files depending on the state of PY_UFUNC_UNIQUE_SYMBOL and NO_IMPORT_UFUNC.
// ...maybe try compiling boost.python with boost's own build-system and see what that does


//// should probably build a static library
//#define BOOST_PYTHON_STATIC_LIB
//#define BOOST_NUMPY_STATIC_LIB

// aggregated header for all features:
#include <boost/python.hpp>

//// separate headers for specific features:
//#include <boost/python/class.hpp>
//#include <boost/python/module.hpp>
//#include <boost/python/def.hpp>

#include <iostream>
#include <string>


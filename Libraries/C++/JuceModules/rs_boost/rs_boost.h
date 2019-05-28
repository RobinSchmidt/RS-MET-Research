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

// including any of the files from boost/python creates a linker error:
// cannot open file 'boost_python34-vc142-mt-gd-x64-1_70.lib'
// how can inclusion of a hpp file make a difference for the question, which input files the 
// linker expects? this is normally determined by the project settings - weird!

//#include <boost/python.hpp>  // aggregated header

//#include <boost/python/class.hpp>
//#include <boost/python/module.hpp>
//#include <boost/python/def.hpp>


#include <iostream>
#include <string>


// maybe rename module to RoBoSuSe (Rob's Boost Subset) or robosub
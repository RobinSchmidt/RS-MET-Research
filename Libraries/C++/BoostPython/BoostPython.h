/*******************************************************************************
 The block below describes the properties of this module, and is read by
 the Projucer to automatically generate project code that uses it.
 For details about the syntax and how to create or use a module, see the
 JUCE Module Format.txt file.


 BEGIN_JUCE_MODULE_DECLARATION

  ID:               BoostPython
  vendor:           RS-MET
  version:          0.0.1
  name:             Boost Python
  description:      Boost's python-bindings generator as JUCE module
  website:          https://github.com/boostorg/python
  license:          Boost

  dependencies: 
  OSXFrameworks:
  iOSFrameworks:

 END_JUCE_MODULE_DECLARATION

*******************************************************************************/

#pragma once

#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <iostream>
#include <string>

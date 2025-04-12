#ifndef HEAPALLOCATIONLOGGER_H_INCLUDED
#define HEAPALLOCATIONLOGGER_H_INCLUDED

#include <stdlib.h>  // For malloc, free, etc.
#include <crtdbg.h>  // For __malloc_dbg etc in MSVC


/* Under construction.

This is an attempt to track the memory allocations. Unfortunatly, this doesn't work yet. I want to 
redefine malloc etc. to do addtional logging but unfortunately, my redefined functiond never get 
called. ...this needs more research....

ToDo:

- Move this code somewhere else. Maybe into the research repo into the Projects/CppExperiments
  folder. Maybe if it's ready to use, it needs to be included as the very first thing in RAPT. 
  Maybe it should go into a folder DevTools where we collect all the stuff for debugging - like the
  allocation logging, memory leak detection, plotting, etc. The stuff there should be conditionally
  compiled only in debug builds.


*/



/** A class for logging heap allocations. ...TBC... */

class rsHeapAllocationLogger
{

public:


  void logAllocation() {  numAllocs++;  }


  size_t getNumAllocations() const { return numAllocs; }


  void logDeallocation() { numDeallocs++; }

  size_t getNumDeallocations() const { return numDeallocs; }

  size_t getNumAllocatedChunks() const { return getNumAllocations() - getNumDeallocations(); }


  void reset()
  {
    numAllocs   = 0;
    numDeallocs = 0;
  }


private:

  size_t numAllocs   = 0;
  size_t numDeallocs = 0;

};

//rsHeapAllocationLogger* rsHeapAllocationLogger::theObject = nullptr;

rsHeapAllocationLogger heapAllocLogger;



// https://stackoverflow.com/questions/1008019/how-do-you-implement-the-singleton-design-pattern

// Maybe also log the allocated size. But this requires us to keep track of all the addresses of 
// the allocated chunks and their sizes, so it would complicate the implementation a lot. That's 
// overkill at the moment. 


void* rsLoggingMalloc(size_t size)
{
  heapAllocLogger.logAllocation();
  return malloc(size);

  // See: https://en.cppreference.com/w/c/memory/malloc
}

void* rsLoggingDebugMalloc(size_t size, int blockUse, char const* fileName, int lineNumber)
{
  heapAllocLogger.logAllocation();
  return _malloc_dbg(size, blockUse, fileName, lineNumber);

  // https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/malloc-dbg?view=msvc-170
}
// This also never gets called. I'm doing something wrong. Look up how juce does it!

void rsLoggingFree(void* ptr)
{
  heapAllocLogger.logDeallocation();
  free(ptr);

  // See: https://en.cppreference.com/w/c/memory/free
}


#define malloc(size) rsLoggingMalloc(size)
//#define _malloc_dbg(size, blockUse, fileName, lineNumber) rsLoggingDebugMalloc(size, blockUse, fileName, lineNumber)
#define free(ptr)    rsLoggingFree(ptr)

// I think, this also covers new, new[], delete, delete[], because they use malloc and free 
// internally...but cant we really count on that?  ...NOPE!!!
// Also, what about realloc and calloc? 


// Code for new/delete adapted from here:
// https://learn.microsoft.com/en-us/cpp/cpp/new-and-delete-operators?view=msvc-170

void *operator new(size_t stAllocateBlock) 
{
  return malloc(stAllocateBlock);
}

void operator delete(void *pvMem) 
{
  free( pvMem );
}






// https://stackoverflow.com/questions/438515/how-to-track-memory-allocations-in-c-especially-new-delete
// https://en.wikipedia.org/wiki/Electric_Fence
// https://stackoverflow.com/questions/9702292/overriding-malloc-to-log-allocation-size
// https://stackoverflow.com/questions/262439/create-a-wrapper-function-for-malloc-and-free-in-c


// https://valgrind.org/
// https://learn.microsoft.com/en-us/cpp/c-runtime-library/crt-debug-heap-details?view=msvc-170
// https://learn.microsoft.com/en-us/cpp/cpp/new-and-delete-operators?view=msvc-170

#endif
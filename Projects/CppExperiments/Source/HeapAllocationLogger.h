#ifndef HEAPALLOCATIONLOGGER_H_INCLUDED
#define HEAPALLOCATIONLOGGER_H_INCLUDED

#include <stdlib.h>  // For malloc, free, etc.
//#include <crtdbg.h>  // For __malloc_dbg etc in MSVC



/** A class for logging heap allocations. A global object of this class is created and all calls to
malloc, free, new, delete, etc. are intercepted by redefining them (via macros in case of the 
C-functions). In these redefined versions, we log the call and then call the built in version. The 
class can be used to detect memory leaks and inadvertant heap allocations that are hidden inside 
library calls. The latter is important to verify that code that is supposed to be realtime safe 
doesn't do any unexpected heap allocations. ...TBC... */

class rsHeapAllocationLogger
{

public:

  size_t getNumAllocations()     const { return numAllocs; }
  size_t getNumDeallocations()   const { return numDeallocs; }
  size_t getNumAllocatedChunks() const { return getNumAllocations() - getNumDeallocations(); }

  bool isMemoryAllocated() const { return getNumAllocatedChunks() > 0; }

  void logAllocation()   { numAllocs++;   }
  void logDeallocation() { numDeallocs++; }

  void reset() { numAllocs = 0; numDeallocs = 0; }


private:

  size_t numAllocs   = 0;
  size_t numDeallocs = 0;

};

// A global object for logging all the heap allocations:
rsHeapAllocationLogger heapAllocLogger;
// Should we define it static?




void* rsLoggingMalloc(size_t size)
{
  heapAllocLogger.logAllocation();
  return malloc(size);
  // See: https://en.cppreference.com/w/c/memory/malloc
}
#define malloc(size) rsLoggingMalloc(size)

void rsLoggingFree(void* ptr)
{
  heapAllocLogger.logDeallocation();
  free(ptr);
  // See: https://en.cppreference.com/w/c/memory/free
}
#define free(ptr) rsLoggingFree(ptr)

// Not sure about that:
//void* rsLoggingDebugMalloc(size_t size, int blockUse, char const* fileName, int lineNumber)
//{
//  heapAllocLogger.logAllocation();
//  return _malloc_dbg(size, blockUse, fileName, lineNumber);
//
//  // https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/malloc-dbg?view=msvc-170
//}
//#define _malloc_dbg(size, blockUse, fileName, lineNumber) rsLoggingDebugMalloc(size, blockUse, fileName, lineNumber)

void *operator new(size_t size) 
{
  return malloc(size);
  // See: https://learn.microsoft.com/en-us/cpp/cpp/new-and-delete-operators?view=msvc-170
}

void operator delete(void *ptr)
{
  free(ptr);
}


// ToDo:
//
// - What about realloc and calloc? I think, we may need to redefine these, too.
//
// - Check, if the custom new/delete functions also get called for array allocations via e.g.
//   double* p = new double[10]; delete[] p;  If not, we may have to write specific versions for 
//   those as well. ..OK...done - yes, they get called.
//
// - Maybe distinguish between different forms of allocation. Log separately the calls to malloc, 
//   new, new[] and free, delete, delete[]. But maybe we should log these calls to new/delete just 
//   in addition to the calls to malloc/free. That means a call to new would register as call to 
//   new *and* as call to malloc. But maybe that's to complicated from an API point of view. I 
//   think, the client code doesn't really care about that distinction anyway. I'm not sure about 
//   that, though.
//
// - Maybe also log the allocated size. But this requires us to keep track of all the addresses of 
//   the allocated chunks and their sizes, so it would complicate the implementation a lot. That's 
//   overkill at the moment. Also, if we try to keep track of the allocated chunks, we'd probably 
//   want to use a std::vector or some other dynamically allocating data structure to store that 
//   data. But using dynamic allocations within the allocation logger itself could be problematic.
//   It could give rise to infinte recursions and therefore stack overflows.
//
// - Figure out what happens if we try to use rsHeapAllocationLogger and the Visual Studio debug
//   heap. They probably interfere such that one can use either one or the other. Maybe it should
//   be set up in some config file which strategy to use.
//
// - What about usage of the class in a mutithreded setting? It's currently not made for that 
//   purpose. Maybe the counters should be declared as thread local? Or maybe the bodies of 
//   rsLoggingMalloc/Free should be inside criticial sections? Maybe we need different 
//   implementations for different scenarios. This simple implementation here is suitable for the
//   RAPT/rosic unit tests. This is just a single threaded commandline app. Maybe the class name
//   should reflect that. rsSingleThreadedHeapAllocationLogger is quite long, though. Maybe 
//   rsHeapLoggerSingleThread or rsHeapLoggerST or rsHeapLoggerSimple. Maybe it could have some
//   self test code that makes sure that it is really used correctly? Maybe store the tread ID 
//   where it was last called and compare it to the thread ID where it is currently called and
//   trigger an error if they don't match?
//
//
// Resources:
//
// https://stackoverflow.com/questions/438515/how-to-track-memory-allocations-in-c-especially-new-delete
// https://en.wikipedia.org/wiki/Electric_Fence
// https://stackoverflow.com/questions/9702292/overriding-malloc-to-log-allocation-size
// https://stackoverflow.com/questions/262439/create-a-wrapper-function-for-malloc-and-free-in-c
//
// https://valgrind.org/
// https://learn.microsoft.com/en-us/cpp/c-runtime-library/crt-debug-heap-details?view=msvc-170
// https://learn.microsoft.com/en-us/cpp/cpp/new-and-delete-operators?view=msvc-170

#endif
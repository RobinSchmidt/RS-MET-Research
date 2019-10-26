/** Demos for using containers from the standard library. */

#include "Common.h"

/** Emulate multiple return values (of the same type) via std::array ...should this be done with
tuple instead? ...or maybe we should use structs? */
std::array<float, 3> get123()
{
  return std::array<float, 3>{ 1.f, 2.f, 3.f };
  //std::array<float, 3> a = {1.f, 2.f, 3.f}; return a; // alternative
}
// What's the difference to the alternative way? is there an additional copy? ...figure out!
// can we also emulate multiple return values of different types?
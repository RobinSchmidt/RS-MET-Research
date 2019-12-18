#pragma once

// under construction - i have no idea yet, if i'm doing the variadic template stuff for the access
// operator right
template<class T>
class MultiArray
{

public:

  MultiArray(std::vector<int> initialShape) : shape(initialShape)
  {
    // todo: allocate memory, update strides

  }

  template<typename... Rest>
  T& operator()(int i, Rest... rest) 
  {
    return &data(flatIndex((int)shape.size()-1, i, rest...));
  }
  // goal: we want to be able to use the operator like (i), (i,k), (i,j,k), ...

  // https://eli.thegreenplace.net/2014/variadic-templates-in-c/
  /*
  template<typename T, typename... Args>
  T adder(T first, Args... args) {
  return first + adder(args...);
  }
  */

  // depth is recursion-depth - maybe find better name
  int flatIndex(int depth, int index)
  {
    return index * strides[depth];
  }

  template<typename... Rest>
  int flatIndex(int depth, int i, Rest... rest)
  {
    return flatIndex(depth, i) + flatIndex(depth-1, rest...);
  }

  // i think, arithmetic operators *,/ should work element-wise - no matrix multiply - see, what
  // numpy does - the different kinds of special products (matrix-product, outer-product, 
  // inner-product, etc.) should be realized a named functions



protected:

  std::vector<int> shape;
  std::vector<int> strides;
  std::vector<T>   data;

};
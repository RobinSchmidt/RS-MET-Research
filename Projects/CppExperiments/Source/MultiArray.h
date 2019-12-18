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

    int rank = (int) shape.size();
    strides.resize(rank);

    // update strides (needs test):
    int i = rank-1;
    int s = 1;
    while(i >= 0) {
      strides[i] = s;
      s *= shape[i];
      --i;
    }

    // compute required space and allocate memory:
    s = 1;
    for(i = 0; i < rank; i++)
      s *= shape[i];
    data.resize(s);
  }

  template<typename... Rest>
  T& operator()(int i, Rest... rest) 
  {
    return data[ flatIndex((int)shape.size()-1, i, rest...) ];
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

  // factor out inot class MultiArrayView:
  std::vector<int> shape;
  std::vector<int> strides;

  // MultiArray itself should only have this additional member
  std::vector<T>   data;

};

void testMultiArray()
{
  typedef std::vector<int> VecI;
  typedef std::vector<float> VecF;
  typedef MultiArray<float> MA;

  MA a1 = MA(VecI{3});  // should create a 3D vector
  a1(0) = 1;
  a1(1) = 2;
  a1(2) = 3;


  MA a2 = MA(VecI{3,2});  // 3x2 matrix ..or 2x3? strides ar 2,1 - should be 3,1 - maybe we are 
                          // traversing the shape array in the wrong direction?
           


  int dummy = 0;
}
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
    /*
    int i = rank-1;
    int s = 1;
    while(i >= 0) {
      strides[i] = s;
      s *= shape[i];
      --i;
    }
    */

    int i = rank-1;
    int s = 1;
    while(i >= 0) {
      strides[i] = s;
      s *= shape[rank-i-1];
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
  // seems to work! test with rank 3!



  // depth is recursion-depth - maybe find better name ..perhaps dimension...but it starts
  // with the last
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
// implement it in a similar way as rsMatrix

void testMultiArray()
{
  typedef std::vector<int> VecI;
  typedef std::vector<float> VecF;
  typedef MultiArray<float> MA;

  MA a1 = MA(VecI{3});     // 3D vector
  a1(0) = 1;
  a1(1) = 2;
  a1(2) = 3;


  MA a2 = MA(VecI{3,2});  // 3x2 matrix
  a2(0,0) = 11;
  a2(0,1) = 12;
  a2(1,0) = 21;
  a2(1,1) = 22;
  a2(2,0) = 31;
  a2(2,1) = 32;

  MA a3 = MA(VecI{2,4,3});  // 2x4x3 block


  int dummy = 0;
}
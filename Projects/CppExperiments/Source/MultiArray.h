#pragma once


template<class T>
class rsMultiArrayView
{

public:

  //-----------------------------------------------------------------------------------------------
  /** \name Construction/Destruction */

  /** Default constructor. */
  rsMultiArrayView() {}

  /** Creates a multi-array view with the given shape for the given raw array of values in "data". 
  The view will *not* take ownership over the data. */
  rsMultiArrayView(std::vector<int> initialShape, T* data) : shape(initialShape)
  {
    updateStrides();
    updateSize();
    dataPointer = data;
  }
  // take initial shape by reference


  //-----------------------------------------------------------------------------------------------
  /** \name Setup */

  // setShape, fill, 


  //-----------------------------------------------------------------------------------------------
  /** \name Inquiry */

  /** Returns the number of array elements. */
  int getSize() const { return size; }

  /** Returns the number of array dimensions, i.e. the number of indices. */
  int getNumDimensions() const { return (int) shape.size(); }

  /** Returns a const reference to the shape array of the multidimensional array. The shape is 
  defined by the number of  values that each index may run over. For example, a 2x3 matrix has a 
  shape array of [2,3]. */
  const std::vector<int>& getShape() const { return shape; }

  // maybe have functions that return a pointer to a particular "row" - but the function should be 
  // a template, taking an arbitrary number of indices - for example A.getRowPointer(2, 3) would
  // return the 4th row in the 3rd slice ...or the other way around...i think so


  //-----------------------------------------------------------------------------------------------
  /** \name Element Access */

  /** Read and write access to array elements. The syntax for accessing, for example, 3D array 
  elements is: A(i, j, k) = .... */
  template<typename... Rest>
  T& operator()(int i, Rest... rest) { return dataPointer[flatIndex(0, i, rest...)]; }





protected:

  // depth is recursion-depth - maybe find better name ..perhaps dimension...but it starts
  // with the last
  int flatIndex(int depth, int index) { return index * strides[depth]; }
  // base case for variadic template instantiation

  template<typename... Rest>
  int flatIndex(int depth, int i, Rest... rest) 
  { return flatIndex(depth, i) + flatIndex(depth+1, rest...); }

  void updateStrides()
  {
    int rank = (int) shape.size();
    strides.resize(rank);
    int i = rank-1;         // last index has stride 1 -> row-major matrix storage
    int s = 1;
    while(i >= 0) {
      strides[i] = s;
      s *= shape[i];
      --i;
    }
  }
  // maybe move to cpp file

  void updateSize()
  {
    // what about edge cases resulting in zero size?

    int rank = (int) shape.size();
    size = 1;
    for(int i = 0; i < rank; i++)
      size *= shape[i];
  }

  std::vector<int> shape;
  std::vector<int> strides;
  T* dataPointer = nullptr;
  int size = 0;

};

//=================================================================================================

/** Implements an n-dimensional array. Elements of an array A can be conveniently accessed via the 
syntax: 1D: A(i), 2D: A(i,j), 3D: A(i,j,k), etc. The data is stored in a std::vector. */

template<class T>
class MultiArray : public rsMultiArrayView<T>
{

public:


  MultiArray(std::vector<int> initialShape) : rsMultiArrayView<T>(initialShape, nullptr)
  {
    data.resize(this->size);
    updateDataPointer();
  }



  // arithmetic operators *,/ should work element-wise like numpy does - the different 
  // kinds of special products (matrix-product, outer-product, inner-product, etc.) should be 
  // realized a named functions



protected:

  /** Updates the data-pointer inherited from rsMultiArrayView to point to the begin of our 
  std::vector that holds the actual data. */
  void updateDataPointer()
  {
    if(data.size() > 0)
      this->dataPointer = &data[0];
    else
      this->dataPointer = nullptr;
  }

  std::vector<T> data;

};


void testMultiArray()
{
  typedef std::vector<int> VecI;
  typedef std::vector<float> VecF;
  typedef MultiArray<float> MA;

  // 3D vector:
  MA a1 = MA(VecI{3});     
  a1(0) = 1;
  a1(1) = 2;
  a1(2) = 3;


  // 3x2 matrix:
  MA a2 = MA(VecI{3,2});  
  a2(0,0) = 11;
  a2(0,1) = 12;

  a2(1,0) = 21;
  a2(1,1) = 22;

  a2(2,0) = 31;
  a2(2,1) = 32;


  // 2x4x3 block/cuboid:
  MA a3 = MA(VecI{2,4,3}); 
  a3(0,0,0) = 111;
  a3(0,0,1) = 112;
  a3(0,0,2) = 113;

  a3(0,1,0) = 121;
  a3(0,1,1) = 122;
  a3(0,1,2) = 123;

  a3(0,2,0) = 131;
  a3(0,2,1) = 132;
  a3(0,2,2) = 133;

  a3(0,3,0) = 141;
  a3(0,3,1) = 142;
  a3(0,3,2) = 143;


  a3(1,0,0) = 211;
  a3(1,0,1) = 212;
  a3(1,0,2) = 213;

  a3(1,1,0) = 221;
  a3(1,1,1) = 222;
  a3(1,1,2) = 223;

  a3(1,2,0) = 231;
  a3(1,2,1) = 232;
  a3(1,2,2) = 233;

  a3(1,3,0) = 241;
  a3(1,3,1) = 242;
  a3(1,3,2) = 243;



  int dummy = 0;
}
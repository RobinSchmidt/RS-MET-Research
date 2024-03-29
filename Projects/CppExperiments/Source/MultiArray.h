#pragma once

// this should go into RAPT::Data...what about rsMatrix then? should this stay in rsMath 
// regardless? ...perhaps yes

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
  rsMultiArrayView(const std::vector<int>& initialShape, T* data) : shape(initialShape)
  {
    updateStrides();
    updateSize();
    dataPointer = data;
  }


  //-----------------------------------------------------------------------------------------------
  /** \name Setup */

  // setShape, fill, 


  //-----------------------------------------------------------------------------------------------
  /** \name Inquiry */

  /** Returns true, iff the two arrays A and B have the same shape. */
  static bool areSameShape(const rsMultiArrayView<T>& A, const rsMultiArrayView<T>& B)
  { return A.shape == B.shape; }

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
  // insert rsAssert flatIndex(...) < size, maybe also >= 0 - maybe have a function 
  // isFlatIndexValid(i) ...see rsMatrix, maybe add it there, too



protected:

  //-----------------------------------------------------------------------------------------------
  /** \name Index Computation */
  // maybe move to public section

  /** Used for implementing the variadic template for the () operator. Takes a recursion depth (for
  recursive template instantiation) and a variable number of indices .... */
  template<typename... Rest>
  int flatIndex(int depth, int i, Rest... rest) 
  { return flatIndex(depth, i) + flatIndex(depth+1, rest...); }

  /** Base case for the variadic template. this version will be instatiated when, in addition to 
  the recursion depth, only one index is passed. */
  int flatIndex(int depth, int index) { return index * strides[depth]; }
  // this is perhaps the best case to check, if an idex is valid, we should have:
  // 0 <= index < shape[depth] ...right? not sure

  // can we implement this without using the strides? let's try:

  /*
  int flatIndex(int stride, int index) { return index * stride; }

  template<typename... Rest>
  int flatIndex(int depth, int stride, int i, Rest... rest) 
  { 
    return flatIndex(stride, i) + flatIndex(depth, stride*shape[depth-1] , rest...); 
    //return flatIndex(depth, i) + flatIndex(depth+1, rest...); 
  }
  */

  /** Converts a C-array (assumed to be of length getNumDimensions()) of indices to a flat 
  index. */
  int flatIndex(const int* indices)
  {
    int fltIdx = 0;
    for(size_t i = 0; i < strides.size(); i++)
      fltIdx += indices[i] * strides[i];
    return fltIdx;
  }
  // needs test


  //-----------------------------------------------------------------------------------------------
  /** \name Member Updating */

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

  /** Updates our size variable according to the values in the shape array. The total size is 
  (redundantly) cached in a member variable because it's used frequently. */
  void updateSize()
  {
    if(shape.size() == 0) { size = 0; return; }   // edge case
    size = 1;
    for(size_t i = 0; i < shape.size(); i++)
      size *= shape[i];
  }


  //-----------------------------------------------------------------------------------------------
  /** \name Data */

  std::vector<int> shape;
  std::vector<int> strides;
  T* dataPointer = nullptr;
  int size = 0;

};
// maybe numIndices should be a compile-time parameter, i.e. a template-parameter and shape should 
// also be a non-owned pointer..maybe rsStaticMultiArrayView... maybe like this
/*
template<class T>
class rsMultiArrayView
{

protected:
  
  T*   dataPointer;
  int* sizesPointer; // aka shape
  int  numSizes;
  int  totalSize;    // == product(shape)

};

*/

//=================================================================================================

/** Implements an n-dimensional array. Elements of an array A can be conveniently accessed via the 
syntax: 1D: A(i), 2D: A(i,j), 3D: A(i,j,k), etc. The data is stored in a std::vector. */

template<class T>
class rsMultiArray : public rsMultiArrayView<T>
{

public:


  rsMultiArray(const std::vector<int>& initialShape) : rsMultiArrayView<T>(initialShape, nullptr)
  {
    data.resize(this->size);
    updateDataPointer();
  }



  // arithmetic operators *,/ should work element-wise like numpy does - the different 
  // kinds of special products (matrix-product, outer-product, inner-product, etc.) should be 
  // realized a named functions


  /** Adds two matrices: C = A + B. */
  rsMultiArray<T> operator+(const rsMultiArray<T>& B) const
  { 
    rsMultiArray<T> C(this->shape); 
    //this->add(*this, B, &C); //     
    return C; 
  }


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


// some idea to implement the index computation using only the shape array - which is passed along
// to the index computation function

int flatIndexRecursion(int* shape, int depth, int stride, int index)
{
  return stride * index;
}
// base case for recursion

template<typename... Rest>
int flatIndexRecursion(int* shape, int depth, int stride, int index, Rest... rest)
{
  return flatIndexRecursion(shape, depth,   stride,                index)
      +  flatIndexRecursion(shape, depth-1, stride*shape[depth-1], rest...);
}

/*
// version 2 - for test - nope - that is even wronger:
template<typename... Rest>
int flatIndexRecursion(int* shape, int depth, int stride, int index, Rest... rest)
{
  return flatIndexRecursion(shape, depth, stride*shape[depth-1],                index)
    +  flatIndexRecursion(shape, depth-1, stride, rest...);
}
*/

// rename the "depth" parameter to numIndices/numDimensions
// measure the performance of this stride-array-free computation of flat indices with the 
// stride-array based one


template<typename... Rest>
int flatIndex(int* shape, int numDimensions, int index, Rest... rest)
{
  return flatIndexRecursion(shape, numDimensions, 1, index, rest...);
}
// this is meant to be used by client code - the others above are just for internal use
// -> seems to work - move to rapt, the get rid of stride array there - it's not needed anymore

/*

int getStride(int* shape, int depth, int stride)
{
  return stride;
}

template<typename... Rest>
int getStride(int* shape, int depth, int stride, Rest...rest)
{
  return stride * getStride(shape, depth-1, rest...);
}
*/


void testMultiArray()
{
  int i;
  int shape2[5] ={ 3,5,7,11,13 };
  i = flatIndex(shape2, 5,  3,2,3,4,5);  // nope - result is wrong! should be 17503
  // https://www.wolframalpha.com/input/?i=5+%2B+4+*+13+%2B+3+*+11*13+%2B+2+*+7*11*13+%2B+3+*+5*7*11*13

  i = flatIndex(shape2, 5,  5,4,3,2,3);
  // this actually gives 17503 - i think the recursion traverses the shape array the wrong way
  // ...nope - the parameter pack is rolled up from the front - the base-case implementation gets
  // the first passed index - but we want the base-case to get the last! the indexes are traversed
  // left-to-right, but the accumulation of the strides goes right-to-left...hmm...we probably need
  // the strides array
  // ...i mean, we could implement it that way but it would imply that we get a column-major memory
  // layout...the first index would get a stride of 1 - that's counter intuitive, but we can get 
  // rid of the strides array...hmm...maybe it's better to have a strides array anyway - at least,
  // in the view-class - that makes it more flexible to use - we could write A(i,j) regardless 
  // whether a matrix is stored row- or column major - we would just need to give the view object
  // the correct strides



  // test for the recursive flat index computation function without using strides:
  int shape[3] = { 2,4,3 };                 // 2x4x3 3D array
  i = flatIndexRecursion(shape, 3, 1,   1,1,1);  // multi-index = (1,1,1) -> flat-index = 16


  i = flatIndex(shape, 3,  1,1,1);




  int shape3[4] ={ 2,4,3,2 };
  i = flatIndex(shape3, 4,  1,1,1,1); // 33 - correct
  i = flatIndex(shape3, 4,  1,1,2,1); // 39
  i = flatIndex(shape3, 4,  1,2,2,1); // 41


  //int s = getStride(shape, 3, 1)


  typedef std::vector<int> VecI;
  typedef std::vector<float> VecF;
  typedef rsMultiArray<float> MA;

  // 3D vector:
  MA a1 = MA(VecI{3});     
  a1(0) = 1;
  a1(1) = 2;
  a1(2) = 3;

  //a1 = a1 + a1;


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

  // move code over to RAPT and turn this into a unit test
  // allow the user to specify an allocator so we can unit-test the memory allocation avoidance



  int dummy = 0;
}





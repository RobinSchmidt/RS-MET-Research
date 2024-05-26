#pragma once

// Maybe rename file to SetTheory.h


namespace rema  // Rob's educational math algorithms
{

//=================================================================================================

/** Implements a set in the set-theoretic sense. Its elements can only be other sets. It's not a 
set of objects of some data type like you would expect from std::set. Instead, it's a recursive 
data structure with sets all the way down. For technical reasons, the elements must be held as 
pointers-to-elements which complicates the internal implementation but client code does not really 
need to think about this. The implementation is similar to how would one implement a tree in C++ 
with nodes where each node has an array of pointers to child nodes. 

One may think about using std::set but what should the template parameter be? If you try to use 
std::set<std::set>, it doesn't compile because now the inner set also needs a template parameter.
If we don't have any primitive datatype at some level, we'll end up in an infinite regress. This
implementation here does not have such problems. It really is a set of sets and there is no other
datatype involved. The implementation is rather naive and just for proof/demonstration of set 
theoretical concepts and entirely unpractical. It's purely educational code - basically a math 
excercise. */

class rsSetNaive
{

public:


  /** Default constructor. Creates the empty set. */
  rsSetNaive() {}


  rsSetNaive(const std::vector<rsSetNaive>& s);

  /** Copy constructor.  */
  rsSetNaive(const rsSetNaive& A);

  /** Move constructor.  */
  rsSetNaive(rsSetNaive&& A) : elements(std::move(A.elements)) 
  { 
    A.elements.clear(); 
  }
  // Needs test

  /** Copy assignment operator. */
  rsSetNaive& operator=(const rsSetNaive& A);

  /** Move assignment operator. */
  rsSetNaive& operator=(rsSetNaive&& A) 
  {
    elements = std::move(A.elements);
    A.elements.clear(); 
    return *this;
  }
  // Move out of class


  ~rsSetNaive();


  //-----------------------------------------------------------------------------------------------
  // \name Setup

  /** Adds the given set as element to this set. */
  void addElement(const rsSetNaive& a);

  // ToDo: 
  // -removeElement(size_t i);
  // -removeElement(const rsSetNaive& a); - this can actually be implemented as set-difference
  //  with the singleton { a }. Not that this would be very practical - but from a theoretical
  //  perspective, it might be an impoertant observation


  //-----------------------------------------------------------------------------------------------
  // \name Inquiry

  /** Returns the cardinality, i.e. the number of elements of this set. */
  size_t getCardinality() const { return elements.size(); }
  // maybe rename to size (or make an alias)

  /** Returns true, iff this set is the empty set. */
  bool isEmpty() const { return elements.size() == 0; }

  /** Returns true, iff this set is a singleton set, i.e. a set that has just one element. */
  bool isSingleton() const { return elements.size() == 1; }

  /** Returns true iff this set has the given set A as element. */
  bool hasElement(const rsSetNaive& a) const;
  // Maybe rename to contains - but no - that's ambiguous - it could also refer to the subset 
  // relation

  //bool isElementOf const (const rsSetNaive& A) const { return A.hasElement(*this); }

  /** Returns true iff this set has the given set A as subset. */
  bool hasSubset(const rsSetNaive& A) const;

  //bool isSubsetOf(const rsSetNaive& A) const { return A.hasSubset(*this); }
  // Needs test
  // inverse relation of hasSubset

  /** Returns true iff this set is equal to the given set A. */
  bool equals(const rsSetNaive& A) const;


  /** Returns true, iff this set is an ordered pair. This can be inferred from the structure of 
  the set. */
  bool isOrderedPair() const;

  /** Returns a string that represents this set. This is useful for debugging. */
  static std::string setToString(const rsSetNaive& A);

  /** Assumes that the set A represents an ordered pair and turns it into a string. */
  static std::string orderedPairToString(const rsSetNaive& A);


  //-----------------------------------------------------------------------------------------------
  // \name Element Access

  /** Returns the i-th element of this set. */
  rsSetNaive getElement(size_t i) const;

  /** Assuming that this set represents an ordered pair, i.e. was created by the orderedPair() 
  function, this function extracts the first element of the ordered pair. */
  rsSetNaive orderedPairFirst() const;

  /** Assuming that this set represents an ordered pair, i.e. was created by the orderedPair() 
  function, this function extracts the second element of the ordered pair. */
  rsSetNaive orderedPairSecond() const;


  //-----------------------------------------------------------------------------------------------
  // \name Element Misc

  size_t getMemoryUsage() const;


  //
  // rsSetNaive


  //-----------------------------------------------------------------------------------------------
  // \name Factory

  /** Given a set A, this function creates the set S = { A }, i.e. the singleton set that contains 
  A as its only element. One could also implement creation of singletons it via pair(A, A) but it's
  convenient to have a function to directly produce singletons. */
  static rsSetNaive singleton(const rsSetNaive& A);

  /** Given two sets A and B, this function produces the pair P = { A, B } of the two. Being able 
  to create such a pair is one of the Zermelo-Fraenkel axioms. */
  static rsSetNaive pair(const rsSetNaive& A, const rsSetNaive& B);

  /** Given two sets A,B, this function creates a set that may be used to represent the ordered 
  pair (A, B). The normal pair() function could distinguish between { A, B } and { B, A } and a 
  pair of equal values like { A, A } could not even be formed because it would just collapse to 
  { A }. To model ordered pairs uisng only sets, we use Kuratowski's definition of ordered pairs
  as (A, B) = { { A }, { A, B } }. With this definition (A, B) is distinguishable from (B, A).
  We have

    (A, B) = { { A }, { A, B } }
    (B, A) = { { B }, { A, B } }
    (A, A) = { { A } }

  With this definition, we can define n-tuples recursively as 

    (A_1, A_2, ..., A_n) = ((A_1, A_2, ..., A_{n-1}), A_n)

  Other definitions of ordered pairs are also possible but Kuratowski's seems to be the accepted 
  standard. ...TBC... */
  static rsSetNaive orderedPair(const rsSetNaive& A, const rsSetNaive& B);
  // Maybe rename to kuratowskiPair

  /** Given two sets A and B, this function produces the union of the two. Being able to create 
  such a union set is one of the Zermelo-Fraenkel axioms. */
  static rsSetNaive unionSet(const rsSetNaive& A, const rsSetNaive& B);
  // It would be nice for consistency to call the function just union, but that's not possible 
  // because union is a C++ keyword

  /** Given two sets A and B, this function produces the intersection of the two which contains 
  only those elements that are present in both A and B. */
  static rsSetNaive intersection(const rsSetNaive& A, const rsSetNaive& B);

  /** Given two sets A and B, this function produces the difference A minus B which contains only 
  those elements from A which are not in B. */
  static rsSetNaive difference(const rsSetNaive& A, const rsSetNaive& B);
  // Needs test

  /** Creates the symmetric difference of A and B. This is the union minus the intersection. */
  static rsSetNaive symmetricDifference(const rsSetNaive& A, const rsSetNaive& B);
  // Needs test

  /** Creates the set product of A and B, i.e. the set of all ordered pairs of elements from A 
  and B. */
  static rsSetNaive product(const rsSetNaive& A, const rsSetNaive& B);
  // Needs test

  /** Given a nonempty set A = { a_1, a_2, a_3, ... }, this function returns the minimum of the 
  a_i elements according to the given less-than relation. */
  static rsSetNaive minimum(const rsSetNaive& A, 
    bool (*less)(const rsSetNaive& left, const rsSetNaive& right));

  /** Given a nonempty set A = { a_1, a_2, a_3, ... }, this function returns the maximum of the 
  a_i elements according to the given less-than relation. */
  static rsSetNaive maximum(const rsSetNaive& A, 
    bool (*less)(const rsSetNaive& left, const rsSetNaive& right));

  // ToDo: Maybe implement functions for min and max that take 2 sets arguments and return the 
  // smaller or larger of the two. Then, implement the min/max that operate on the whole array of
  // elements in terms of these...but maybe not.


  /** Compares this set with rhs for equality. */
  bool operator==(const rsSetNaive& rhs) const { return equals(rhs); }

  /** Compares this set with rhs for inequality. */
  bool operator!=(const rsSetNaive& rhs) const { return !equals(rhs); }


  /** Returns a reference to the i-th element. */
  rsSetNaive& operator[](size_t i) { return *(elements[i]); }
  // -Needs test
  // -Figure out when this gets called
  // -Try re-assigning elements using this operator


  const rsSetNaive& operator[](size_t i) const { return *(elements[i]);  }
  // Is called from unionSet, for example.

  // ToDo:
  // -Implement a leastElement and greatestElement function that takes a "lessThan" comparison
  //  function as parameter. We just do a linear search through the elements with this function
  //  to find the min or max. Maybe call the functions minimum/maximum


protected:

  /** Creates a deep copy of this set via the "new" operator and returns it. The caller is 
  responsible to delete the object eventually. */
  rsSetNaive* getCopy() const;

  std::vector<rsSetNaive*> elements;

};

//=================================================================================================

/** Implements the von Neumann construction of the natural numbers based on sets. The sets are 
represented using the class rsSetNaive. In the von Neuman construction, the number zero is 
represented by the empty set and higher numbers are defined recursively via a successor function
s(n) 

  s(n) = unionSet(n, singleton(n))

This construction leads to the following situation:

  n = { 0, 1, 2, 3, ..., n-1 }

such that the cardinality of the set that represents the number n is precisely n, which is 
convenient. The first few numbers are:

  0 = { }
  1 = { 0 }       = { {} }
  2 = { 0, 1 }    = { {}, {{}} }
  3 = { 0, 1, 2 } = { {}, {{}}, { {}, {{}} }  }  ...verify this!

As can be seen, when expanding the sets fully, it gets messy rather quickly. I think, the size 
grows exponentially (verify!). This is completely useless for practical purposes. I just 
implemented this for demostration purposes as a math excercise and to clarify the concepts to 
myself.


References:

  (1) https://en.wikipedia.org/wiki/Set-theoretic_definition_of_natural_numbers  
  (2) https://cs.uwaterloo.ca/~alopez-o/math-faq/math-faq.pdf  pg 9 ff

*/

class rsNeumannNumber : public rsSetNaive
{

public:

  //-----------------------------------------------------------------------------------------------
  // \name Inquiry

  /** Checks whether or not the given set x is a well formed von Neumann number. */
  static bool isWellFormed(const rsSetNaive& x);
  // The test is rather expensive because it internally creates a reference/target set to compare
  // the input set to

  /** Checks, if the given set x represents zero. In the von Neumann construction, zero is 
  represented by the empty set. */
  static bool isZero(const rsSetNaive& x) { return x.isEmpty(); }

  /** Returns the value of a given von Neumann number represented by the set x. It is equal to the 
  cardinality of the set that represents it. */
  static size_t value(const rsSetNaive& x);

  /** Implements the less-than relation on Nuemann numbers. */
  static bool less(const rsSetNaive& x, const rsSetNaive& y);

  // An equality comparison function is not needed because we can use the set-equality comparison 
  // for that


  //-----------------------------------------------------------------------------------------------
  // \name Factory

  /** Returns the set that represents zero in the von Neumann construction. This is the empty 
  set. */
  static rsSetNaive zero() { return rsSetNaive(); }

  /** Returns the set that represents zero in the von Neumann construction. This is the set that
  contains only 0. */
  static rsSetNaive one() { return singleton(zero()); }

  /** Creates the set that represents the natural number n in the von Neumann construction. */
  static rsSetNaive create(size_t n);


  //-----------------------------------------------------------------------------------------------
  // \name Operations

  /** Given a set x representing a natural number according to the von Neumann construction, this 
  function creates its successor. */
  static rsSetNaive successor(const rsSetNaive& x);

  /** Given a set x representing a natural number strictly greater than zero, this function creates 
  its predecessor. If you feed in zero by mistake, it will trigger rsError and return the empty 
  set. */
  static rsSetNaive predecessor(const rsSetNaive& x);

  /** Computes the sum of x and y. Using s() as the successor function, it is defined as:

  x + y    = x           if y == 0
  x + s(y) = s(x + y)    if y != 0

  See (2). In order to actually implement it, we actually need a predecessor function rather than a
  successor function. */
  static rsSetNaive add(const rsSetNaive& x, const rsSetNaive& y);

  //static rsSetNaive difference(const rsSetNaive& x, const rsSetNaive& y);
  static rsSetNaive subtract(const rsSetNaive& x, const rsSetNaive& y);
  // difference is ambiguous because we already have a function with the same name in the baseclass
  // which computes the set difference. Maybe call it subtract. But this would be inconsisten with 
  // sum, product, etc. - these should them maybe renamed to add, multiply, etc - or maybe shorter
  // add, mul, div, pow. Maybe successor and predecessor should then be called inc, dec

  /** Computes the product of x and y. It is defined as:

  x * y    = 0           if y == 0
  x * s(y) = x * y + x   if y != 0 

  See (2). So, it is defined recursively using addition internally. */
  static rsSetNaive product(const rsSetNaive& x, const rsSetNaive& y);

  /** Computes the result of the integer division x/y, i.e. the integer part of the solution. */
  static rsSetNaive quotient(const rsSetNaive& x, const rsSetNaive& y);

  /** Computes the power of x^y. It is defined as:

  x ^ y    = 1           if y == 0
  x ^ s(y) = x ^ y * x   if y != 0 

  This definition in terms of multiplications is entirely analogous to the definition of 
  multiplication in terms of addition. */
  static rsSetNaive power(const rsSetNaive& x, const rsSetNaive& y);

  // ToDo:
  // -Implement logarithm and root functions
  // -Maybe implement a different construction of the naturals as well.
};



//=================================================================================================

/** Implements integers based on equivalence classes of ordered pairs of von Neumann numbers. The
pair (x, y) represents the number x - y. The Neumann naturals are embedded by letting y = 0. This
is similar to how rational numbers are defined as equivalence classes - there, a pair (x, y) would
represent x/y where x,y are integers and the integers themselves are embedded by letting y = 1. 
Inverse elements are represented by swapping the order of the components of the pair, i.e. the 
inverse of (x, y) is (y, x). That works the samein both cases. ...TBC... */

class rsNeumannInteger
{

public:

  using NN  = rsNeumannNumber;
  using Set = rsSetNaive;


  //-----------------------------------------------------------------------------------------------
  // \name Inquiry

  /** Splits the Neumann integer x = (a, b) into its two parts. */
  static void split(const rsSetNaive& x, rsSetNaive& a, rsSetNaive& b);
  // Move into rsSetNaive - it's not specific to Neumann integers

  /** Implements the equivalence relation...TBC... */
  static bool equals(const rsSetNaive& x, const rsSetNaive& y);

  /** Implements the less-than relation on Nuemann integers. */
  //static bool less(const rsSetNaive& x, const rsSetNaive& y);

  /** Returns the value that is represented by the neumann integer x. */
  static int value(const rsSetNaive& x);


  //-----------------------------------------------------------------------------------------------
  // \name Factory

  static rsSetNaive zero() { return Set::orderedPair(NN::zero(), NN::zero()); }

  static rsSetNaive one()  { return Set::orderedPair(NN::one(), NN::zero()); }

  /** Creates a (potentially non-canonical) Neumann integer represented by the ordered pair (a, b) 
  which stands for the number x = a - b */
  static rsSetNaive create(size_t a, size_t b);

  /** Creates a canonical representation of the given integer as the ordered pair
  (a, b) = (n, 0) for n >= 0  and  (a, b) = (0, -n) for n < 0. */
  static rsSetNaive create(int n);

  /** Embeds a Neumann natural number x into the Neumann integers by creating the pair (x, 0). */
  static rsSetNaive embed(const rsSetNaive& x) { return Set::orderedPair(x, NN::zero()); }
  // Needs test


  //-----------------------------------------------------------------------------------------------
  // \name Operations

  /** Turns the given Neumann integer x into its negative -x. */
  static rsSetNaive negative(const rsSetNaive& x);

  /** Computes the sum of two Neumann integers. */
  static rsSetNaive sum(const rsSetNaive& x, const rsSetNaive& y);

  /** Computes the product of two Neumann integers. */
  static rsSetNaive product(const rsSetNaive& x, const rsSetNaive& y);

  /** Turns the given Neumann integer x into its canonical representation. A tuple (a,b) represents
  the integer a-b. For example, the canonical representation of the number two is 2 = (2,0). One
  example for a non-canonical representation would be (5,3). Bringing a Neumann integer into its 
  canonical from is analoguous to reducing a representation of a rational number to lowest terms.
  The implementation of this function uses operations that use the intended semantics of the sets
  on a higher level. It's not implemented purely in terms of set operations. So, if you want to 
  simulate Peano artithmetic purely with set operations, this function cannot be used. */
  static rsSetNaive canonical(const rsSetNaive& x);
  // I think, non canonical representation can occur in a subtraction, i.e. a negation followed
  // by addition. I think, the sum of a canonical positive and negative number will give rise to 
  // a non-canonical representation. ToDo: verify and document that

};



//=================================================================================================

/** Implements rational numbers as equivalence classes of ordered pairs of Neumann integers.
...TBC... */


class rsNeumannRational
{

public:

  //using NN  = rsNeumannNumber;
  using NI  = rsNeumannInteger;
  using Set = rsSetNaive;

  //-----------------------------------------------------------------------------------------------
  // \name Inquiry


  /** Implements the equivalence relation...TBC... */
  static bool equals(const rsSetNaive& x, const rsSetNaive& y);

  /** Returns the value that is represented by the neumann integer x. */
  static std::pair<int, int> value(const rsSetNaive& x);

  static rsSetNaive numerator(const rsSetNaive& x);

  static rsSetNaive denominator(const rsSetNaive& x);



  //-----------------------------------------------------------------------------------------------
  // \name Factory

  static rsSetNaive zero() { return Set::orderedPair(NI::zero(), NI::one()); }

  static rsSetNaive one()  { return Set::orderedPair(NI::one(), NI::one()); }


  static rsSetNaive create(int num, int den);


};

bool rsNeumannRational::equals(const rsSetNaive& x, const rsSetNaive& y)
{
  rsSetNaive a, b, c, d, p, q;
  NI::split(x, a, b);                // decompose x = a/b into a, b
  NI::split(y, c, d);                // decompose y = c/d into c, d
  p = NI::product(a, d);             //   compose p = a * d
  q = NI::product(b, c);             //   compose q = b * c
  return NI::equals(p, q);
}
// Needs test

rsSetNaive rsNeumannRational::numerator(const rsSetNaive& x)
{
  rsSetNaive a, b;
  NI::split(x, a, b); 
  return a;
}
// Needs test

rsSetNaive rsNeumannRational::denominator(const rsSetNaive& x)
{
  rsSetNaive a, b;
  NI::split(x, a, b); 
  return b;
}
// Needs test

std::pair<int, int> rsNeumannRational::value(const rsSetNaive& x)
{
  rsSetNaive a, b;
  NI::split(x, a, b);
  int ai = NI::value(a);
  int bi = NI::value(b);
  return std::pair(ai, bi);
}
// Needs test



rsSetNaive rsNeumannRational::create(int num, int den)
{
  return Set::orderedPair(NI::create(num), NI::create(den));
}





}

// remo Rob's educational math objects
// remu                        utilities
// remi                        implementations
//    p                        prototypes
//    a                        algorithms
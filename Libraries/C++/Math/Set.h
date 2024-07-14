#pragma once

// Some silly Microsoft header #defines min/max as macros but we want to use these names for some
// functions, so we undefine these silly macros:
#undef min
#undef max

// Maybe rename file to SetTheory.h


namespace rema  // Rob's educational math algorithms
{

//=================================================================================================

/** Implements a set in the modern set-theoretic sense. Its elements can only be other sets. It's 
not a set of objects of some data type like you would expect from std::set, for example. I think, 
in the terminology of set theory, std::set would be considered to be a set with "urelements" or 
"atoms". But in modern set theory, there are no urelements. Instead, it's a recursive data 
structure with sets all the way down. For technical reasons, the elements must be held as pointers
to elements which complicates the internal implementation but client code does not really need to 
think about this. The implementation is similar to how would one implement a tree in C++ with nodes 
where each node has an array of pointers to child nodes. Here, each set A holds an array of pointers 
to sets. These pointees a0,a1,a2,... are the elements of the set A = {a0,a1,a3,...}.

One may think about using std::set but what should the template parameter be? If you try to use 
std::set<std::set>, it doesn't compile because now the inner set also needs a template parameter.
If we don't have any primitive datatype at some level, we'll end up in an infinite regress. This
implementation here does not have such problems. It really is a set of sets and there is no other
datatype involved. The implementation is rather naive and just for proof/demonstration of set 
theoretical concepts and entirely unpractical. It's purely educational code - basically a math 
excercise. */

class rsSetNaive  // rename to Set
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

  /** Returns true, iff this set is an ordered pair as created by the orderedPair() function. This 
  can be inferred from the structure of the set. */
  bool isOrderedPair() const;


  bool isTransitive() const;
  // Needs test




  //bool isClosedUnderComplement(const rsSetNaive& S, const rsSetNaive& A);
  // Given S, assumed to be subset of the power set P of A, this function should check, if S
  // is closed under the complement. That means: for each element s in S, check if P-s is also in S
  // wher P-s is the set difference between P = P(A) and s

  //bool isClosedUnderUnions(const rsSetNaive& S, const rsSetNaive& A);
  // Given S, assumed to be subset of of the power set of A, this function should check, if S
  // is closed under unions.

  // isClosedUnderIntersections

  // Some classifications onsubsets of power-sets:
  //bool isTopologyOn(const rsSetNaive& A);
  //bool isSigmaAlgebraOn(const rsSetNaive& A);
  //bool isChainOn(const rsSetNaive& A, const rsSetNaive& R);
    // Given partial order R on P(A), a chain is a subset of P(A) for which every two elements of
    // P(A) are comparable with respect to R. For example {},{2},{2,4},{1,2,4},{1,2,3,4} is a chain
    // on P(A) when R is the subset relation and A = {1,2,3,4}

  // See:
  // https://en.wikipedia.org/wiki/Family_of_sets

  // ToDo:
  //
  //bool isHereditarilyTransitive() const;   // or is the meaning of transitive already meant that way?
  //bool isRelationBetween(const rsSetNaive& A, const rsSetNaive& B) const;
  //bool isRelationOn(const rsSetNaive& A) const { return isRelationBetween(A, A); }

  //bool hasProperty(bool (*predicate)(const rsSetNaive& A)) const;
  //bool hasHereditaryProperty(bool (*predicate)(const rsSetNaive& A)) const;


  // isFunctionBetween, isRelationBetween, isTransitiveRelationBetween, isOrderOn, isEquivalenceOn,
  // isInjective, isSurjective



  /** Returns a string that represents this set. This is useful for debugging. */
  static std::string setToString(const rsSetNaive& A);
  // maybe rename to toString. maybe make it non-static and without argument

  /** Assumes that the set A represents an ordered pair and turns it into a string. Of course, you 
  can also apply the general setToString function to ordered pairs - but them you will get a 
  different formatting. */
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

  /** Returns the depth of the tree-like data structure, i.e. the number of nesting levels. */
  size_t getNestingDepth() const;


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
  pair (A, B). The normal pair() function could not distinguish between { A, B } and { B, A } and a 
  pair of equal values like { A, A } could not even be formed because it would just collapse into 
  the singleton { A }. To model ordered pairs uisng only sets, we use Kuratowski's definition of 
  ordered pairs as (A, B) = { { A }, { A, B } }. With this definition (A, B) is distinguishable 
  from (B, A). We have:

    (A, B) = { { A }, { A, B } }
    (B, A) = { { B }, { A, B } }
    (A, A) = { { A } }

  With this definition, we can define n-tuples recursively as 

    (A_1, A_2, ..., A_n) = ((A_1, A_2, ..., A_{n-1}), A_n)

  Other definitions of ordered pairs are also possible but Kuratowski's seems to be the accepted 
  standard. ...TBC... */
  static rsSetNaive orderedPair(const rsSetNaive& A, const rsSetNaive& B);
  // Maybe rename to kuratowskiPair and maybe implement other ways of pair creation, too.

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
  // Maybe rename to productSet or cartesianProduct to avoid ambiguity with functions that compute
  // products of Neumann numbers
  // Needs test

  /** UNDER CONSTRUCTION

  Computes the set of all functions from B to A. This is the power function or set exponentiation 
  denoted as A^B. The result will be a set of sets of ordered pairs of the form (b,a) ...TBC... */
  static rsSetNaive pow(const rsSetNaive& A, const rsSetNaive& B);
  //




  static rsSetNaive bigUnion(const rsSetNaive& A);
  // This implements the axiom of union, I think 



  /** Given a nonempty set A = { a_1, a_2, a_3, ... }, this function returns the minimum of the 
  a_i elements according to the given less-than relation. */
  static rsSetNaive min(const rsSetNaive& A, 
    bool (*less)(const rsSetNaive& left, const rsSetNaive& right));
  // Maybe use std::function for the less-relation
  // Maybe implement a binary function taking two sets as input

  /** Given a nonempty set A = { a_1, a_2, a_3, ... }, this function returns the maximum of the 
  a_i elements according to the given less-than relation. */
  static rsSetNaive max(const rsSetNaive& A, 
    bool (*less)(const rsSetNaive& left, const rsSetNaive& right));

  // Note: I wanted to call these functions min/max but some silly Microsoft header #defines
  // min/max as macros which messes up the compilation. Maybe we should #undef them?

  // ToDo: Maybe implement functions for min and max that take 2 sets arguments and return the 
  // smaller or larger of the two. Then, implement the min/max that operate on the whole array of
  // elements in terms of these...but maybe not.

  // ToDo: Implement the power set operation. We could implement it as follows: let N be the size
  // of the set A. Produce the number P = 2^N. Iterate p = 0...P-1. For each p, iterate i = 0...p, 
  // for each i, init a new subset. Iterpret the bit-pattern of p as follows 1: include subset at 
  // digit position i into current subset.

  // ToDo: Implement the set exponentiation function, i.e. the set of all functions from a set A
  // to another set B. If A has M and B has N elements, the result will have N^M elements (or is it
  // M^N?). Maybe call it pow(Set A, Set B) and/or use the ^ operator

  /** UNDER CONSTRUCTION - NEEDS TESTS */
  static rsSetNaive powerSet(const rsSetNaive& A);


  /** UNDER CONSTRUCTION - VERIFY explanation and implementation!
  
  Produces the transitive closure of the given set A. The transitive closure of a set is defined as 
  the union of the set itself with all of its elements, with their elements, etc. - recursively all 
  the way down until one eventually hits the empty sets on the lowest level. It can also be seen as
  the union of the set itself with the transitive closures of all of its elements...I think. */
  static rsSetNaive transitiveClosure(const rsSetNaive& A);
  // maybe make non-static

  // https://en.wikipedia.org/wiki/Transitive_set#Transitive_closure



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

/** UNDER CONSTRUCTION. HAS NOT YET BEEN TESTED

Collection of functions that deal with special sets that represent relations */

class rsRelation : public rsSetNaive
{

public:

  //-----------------------------------------------------------------------------------------------
  // \name Inquiry

  /** Checks if the given set R is a relation by verifying that each element is an ordered pair. */
  static bool isRelation(const rsSetNaive& R);

  /** Checks if the given set R is a relation between the domain A and the codomain B by verifying 
  that each element is an ordered pair with the first component from A and the second component from
  B. */
  static bool isRelationBetween(const rsSetNaive& R, const rsSetNaive& A, const rsSetNaive& B);

  /** Checks if the given set R is a relation on the set A, i.e. a relation between A and 
  itself. */
  static bool isRelationOn(const rsSetNaive& R, const rsSetNaive& A) 
  { return isRelationBetween(R, A, A); }

  /** Checks if the given set R is a relation with domain A. */
  static bool hasDomain(const rsSetNaive& R, const rsSetNaive& A);

  /** Checks if the given set R is a relation with codomain B. */
  static bool hasCodomain(const rsSetNaive& R, const rsSetNaive& B);


  /** Counts the number of times by which the given element a occurs on the left hand side of the 
  relation R. */
  static int numOccurencesLeft(const rsSetNaive& R, const rsSetNaive& a);

  /** Counts the number of times by which the given element b occurs on the right hand side of the 
  relation R. */
  static int numOccurencesRight(const rsSetNaive& R, const rsSetNaive& b);


  // In some of these functions, we do not need to look at all function parameters - but we want a
  // consistent API because otherwise, the caller must be careful to put the right set in (domain 
  // or codomain). We define the API such that the caller must always pass relation, domain, 
  // codomain in that order

  // VERIFY the definitions of these:

  /** Checks if every a in A occurs as left hand side in the relation R at least once. A is taken 
  to be the domain of the relation R. */
  static bool isLeftTotal(const rsSetNaive& R, const rsSetNaive& A, const rsSetNaive& B);

  /** Checks if every a in A occurs as left hand side in the relation R at most once. A is taken 
  to be the domain of the relation R. This is called right-unique because if we find a in A, it 
  will have a unique partner in B. */
  static bool isRightUnique(const rsSetNaive& R, const rsSetNaive& A, const rsSetNaive& B);

  /** Checks if every a in A occurs as left hand side in the relation R at exactly once. That is
  euqivalent of being left-total and right-unique. A function can be interpreted as a map that maps
  every element a of A to a uniquely determined element b of B. */
  static bool isFunction(const rsSetNaive& R, const rsSetNaive& A, const rsSetNaive& B);

  /** Checks if every b in B occurs as right hand side in the relation R at most once. B is taken 
  to be the codomain of the relation R. This is called left-unique because if we find b in B, it 
  will have a unique partner in A. A left-unique is also called injective. */
  static bool isLeftUnique(const rsSetNaive& R, const rsSetNaive& A, const rsSetNaive& B);

  /** Checks if every b in B occurs as right hand side in the relation R at least once. B is taken 
  to be the codomain of the relation R. A right-total relation is also called surjective. */
  static bool isRightTotal(const rsSetNaive& R, const rsSetNaive& A, const rsSetNaive& B);

  /** Checks if relation R is a function from A to B and is additionally injective (i.e. 
  left-unique) and surjective (i.e. right-total). These two additional conditions are summarized
  under the term bijective. Bijective functions als also called invertible. */
  static bool isBijectiveFunction(const rsSetNaive& R, const rsSetNaive& A, const rsSetNaive& B);

  // https://proofwiki.org/wiki/Definition:Right-Total_Relation
  // https://knowledge.anyhowstep.com/nodes/183/title/Left-unique-Relation
  // https://en.wikipedia.org/wiki/Binary_relation#Types_of_binary_relations


  /** Checks, if the relation R is trichotomic on A,B. That means, for any pair a,b from A,B 
  exactly one of the 3 conditions must me true: (a,b) in R, (b,a) in R, a == b. That models a 
  strict total order relation in which for any a,b we have exactly one of 3 cases: 
  a < b, a == b, a > b.  */
  static bool isTrichotomic(const rsSetNaive& R, const rsSetNaive& A, const rsSetNaive& B);


  // ToDo:
  // isTransitive, isReflexive, isSymmetric, isAntiSymmetric, isAsymmetric, isPartialOrder, 
  // isTotalOrder, isStrictPartialOrder, isStrictTotalOrder, isWellOrder, isEquivalence, 



  //-----------------------------------------------------------------------------------------------
  // \name Factory

  /** Creates a relation R with domain A and codomain B based on the given predicate. The pair 
  (a,b) with a in A, b in B will be included in R iff predicate(a,b) == true. */
  static rsSetNaive create(const rsSetNaive& A, const rsSetNaive& B,
    const std::function<bool(const rsSetNaive& a, const rsSetNaive& b)>& predicate);




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

  0 =   O
  1 = { 0 }       = { O }
  2 = { 0, 1 }    = { O, {O} }
  3 = { 0, 1, 2 } = { O, {O}, { O, {O} }  }  ...verify this!

As can be seen, when expanding the sets fully, it gets messy rather quickly. The size grows 
exponentially. We can really only use it for numbers up to 10 or so, before it gets too big to 
handle with reasonable resources. The class is completely useless for practical purposes. I just
implemented this for demostration purposes as a math excercise and to clarify the concepts.

The class has no mmembers. It's just a collection of functions that operate on raw sets. You need
to ensure yourself to feed in the right kinds of sets. You can use the isWellFormed() function to
verify, if a given set is actually of the right structure to represent a Neumann number.

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
  cardinality of the set that represents it. That's one of the conveniences of the von Neumann 
  construction. */
  static size_t value(const rsSetNaive& x);

  /** Implements the less-than relation on Neumann numbers. */
  static bool less(const rsSetNaive& x, const rsSetNaive& y);

  // An equality comparison function is not needed because we can use the set-equality comparison 
  // for that because Neumann numbers are unique representations of a given natural number. This
  // is not true for integers and rationals anymore that are built from Neumann numbers. They are
  // not unique because they are defined as equivalence classes. That's why there, we have specific
  // implementations of the "equals" function.
  //
  // Verify! Maybe for consistency, implement an equals function anyway and let it just delegate 
  // the call to Set::equals().



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
  static rsSetNaive sub(const rsSetNaive& x, const rsSetNaive& y);
  // difference is ambiguous because we already have a function with the same name in the baseclass
  // which computes the set difference. Maybe call it subtract. But this would be inconsisten with 
  // sum, product, etc. - these should them maybe renamed to add, multiply, etc - or maybe shorter
  // add, mul, div, pow. Maybe successor and predecessor should then be called inc, dec

  /** Computes the product of x and y. It is defined as:

  x * y    = 0           if y == 0
  x * s(y) = x * y + x   if y != 0 

  See (2). So, it is defined recursively using addition internally. */
  static rsSetNaive mul(const rsSetNaive& x, const rsSetNaive& y);

  /** Computes the result of the integer division x/y, i.e. the integer part of the solution. */
  static rsSetNaive div(const rsSetNaive& x, const rsSetNaive& y);

  /** Computes the power of x^y. It is defined as:

  x ^ y    = 1           if y == 0
  x ^ s(y) = x ^ y * x   if y != 0 

  This definition in terms of multiplications is entirely analogous to the definition of 
  multiplication in terms of addition. */
  static rsSetNaive pow(const rsSetNaive& x, const rsSetNaive& y);

  static rsSetNaive sqrt(const rsSetNaive& x);
  
  //static rsSetNaive root(const rsSetNaive& x, const rsSetNaive& n);

  static rsSetNaive log(const rsSetNaive& x, const rsSetNaive& base);




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
inverse of (x, y) is (y, x). That works the same in both cases. ...TBC... */

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
  static rsSetNaive neg(const rsSetNaive& x);

  /** Computes the sum of two Neumann integers. */
  static rsSetNaive add(const rsSetNaive& x, const rsSetNaive& y);

  /** Computes the product of two Neumann integers. */
  static rsSetNaive mul(const rsSetNaive& x, const rsSetNaive& y);

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
  p = NI::mul(a, d);             //   compose p = a * d
  q = NI::mul(b, c);             //   compose q = b * c
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

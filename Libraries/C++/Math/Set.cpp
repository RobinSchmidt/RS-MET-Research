namespace rema  // Rob's educational math algorithms
{



rsSetNaive::rsSetNaive(const std::vector<rsSetNaive>& s)
{
  for(size_t i = 0; i < s.size(); i++)
    addElement(s[i]);
}

rsSetNaive::rsSetNaive(const rsSetNaive& A)
{
  elements.resize(A.elements.size());
  for(size_t i = 0; i < A.elements.size(); i++)
    elements[i] = A.elements[i]->getCopy();
}

rsSetNaive& rsSetNaive::operator=(const rsSetNaive& A)
{
  elements.resize(A.elements.size());
  for(size_t i = 0; i < A.elements.size(); i++)
    elements[i] = A.elements[i]->getCopy();
  return *this;

  // ToDo:
  //
  // -Maybe factor out the deep copying code that is common to the copy constructor and copy 
  //  assignment operator
}

rsSetNaive::~rsSetNaive()
{
  for(size_t i = 0; i < elements.size(); i++)
    delete elements[i];
}

void rsSetNaive::addElement(const rsSetNaive& a)
{
  if(!hasElement(a))
    elements.push_back(a.getCopy());
}


bool rsSetNaive::hasElement(const rsSetNaive& a) const
{
  for(size_t i = 0; i < elements.size(); i++)
  {
    if(elements[i]->equals(a))
      return true;
  }
  return false;
}

bool rsSetNaive::hasSubset(const rsSetNaive& A) const
{
  for(size_t i = 0; i < A.elements.size(); i++)
  {
    if(!hasElement(*(A.elements[i])))
      return false;
  }
  return true;
}

bool rsSetNaive::isOrderedPair() const
{
  // We use Kuratowski pairs which have the structure (x, y) = { { x }, { x, y } } which
  // may collapse into (x, x) = { { x } } when both components are the same. 

  if(getCardinality() == 1)                     // (x, x) = { { x } }
    return isSingleton();
  else if(getCardinality() == 2)                // (x, y) = { { x }, { x, y } }
  {
    // Investigate the structure. The first element has to be a singleton and the second has to be
    // a doubleton which contains the inner element of the first element as one of its members:
    if(!elements[0]->isSingleton())
      return false;
    if(!(elements[1]->getCardinality() == 2))   // ToDo: use hasCardinality(2)
      return false;
    rsSetNaive x = elements[0]->getElement(0);
    return elements[1]->hasElement(x);

    /*
    // NEW:
    rsSetNaive x, y;
    x = elements[0]->getElement(0);
    if(getCardinality() == 2)
    y = elements[1]->getElement(1);
    else
    y =x;
    if(!elements[0]->hasElement(x))           return false;
    if(!(elements[1]->getCardinality() == 2)) return false;  // use hasCardinality(2)
    if(!elements[1]->hasElement(x))           return false;
    if(!elements[1]->hasElement(y))           return false;
    return true;
    */

    // I think, to destructure the pair, we need to use orderedPairFirst/Second. Nah! That leads
    // to an infinite mutual recursion
  }
  else
    return false;

  // Notes:
  //
  // -Set-theoretically, the order how the singleton and doubleton are stored doesn't matter. 
  //  However, in the creation of ordered pairs, we adopt the convention to insert the singleton 
  //  first. So, when ordered pairs are created via that function, we can assume that the 
  //  singleton is stored first.
}

std::string rsSetNaive::setToString(const rsSetNaive& A)
{
  if(A.isEmpty())
    return "O";
  else
  {
    std::string str = "{";
    size_t N = A.getCardinality();
    for(size_t i = 0; i < N; i++)
    {
      str += setToString(A[i]);
      if(i < N-1)
        str += ",";
    }
    str += "}";
    return str;
  }
}

std::string rsSetNaive::orderedPairToString(const rsSetNaive& A)
{
  // The ordered pair A is either of the form (x, y) = { {x}, {x,y} } or of the form
  // (x, x) = { {x}, {x,x} } = { {x}, {x} } = { {x} }.

  RAPT::rsAssert(A.isOrderedPair());
  rsSetNaive x = A.orderedPairFirst();
  rsSetNaive y = A.orderedPairSecond();
  std::string str;
  str += "( ";
  str += "{" + setToString(x) + "}" + " ; ";
  str += "{" + setToString(x) + "," + setToString(y) + "}";
  str += " )";
  return str;
}

bool rsSetNaive::equals(const rsSetNaive& A) const
{
  return hasSubset(A) && A.hasSubset(*this);

  // Notes:
  //
  // -We cannot just iterate through the elements of *this and A and compare one by one because the
  //  elements may be in a different order
}

rsSetNaive rsSetNaive::getElement(size_t i) const
{
  rsAssert(i < elements.size(), "Invalid element index");
  rsSetNaive e = *(elements[i]); // Calls copy contructor rsSetNaive(const rsSetNaive& A)
  return e;
  // Why does it not call the move-assignment operator? Or does it?
}

rsSetNaive rsSetNaive::orderedPairFirst() const
{
  rsAssert(isOrderedPair());
  return elements[0]->getElement(0);
}

rsSetNaive rsSetNaive::orderedPairSecond() const
{
  rsAssert(isOrderedPair());
  if(getCardinality() == 2)
    return elements[1]->getElement(1);
  else
    return elements[0]->getElement(0);
}

size_t rsSetNaive::getMemoryUsage() const
{
  size_t size = sizeof(rsSetNaive);
  for(size_t i = 0; i < elements.size(); i++)
    size += elements[i]->getMemoryUsage();
  return size;

  // Notes:
  //
  // -I'm not quite sure, if that algorithm computes the actual memory usage in bytes but it should 
  //  have approximately the right scaling behavior.
}

rsSetNaive rsSetNaive::singleton(const rsSetNaive& A)
{
  rsSetNaive S;
  S.addElement(A);
  return S;
}

rsSetNaive rsSetNaive::pair(const rsSetNaive& A, const rsSetNaive& B)
{
  rsSetNaive P;
  P.addElement(A);
  P.addElement(B);  // Will have no effect if B == A
  return P;
}

rsSetNaive rsSetNaive::orderedPair(const rsSetNaive& A, const rsSetNaive& B)
{
  return pair(singleton(A), pair(A, B));  // (A, B) = { { A }, { A, B } }

  // Notes:
  //
  // -Hausdorff's definition: (A, B) = { {A, 1}, {B, 2} } might be more conveniently generalize to 
  //  n-tuples because the "tags" can be generalized. But it has the potential problem that the tags 
  //  should be distiguished from the components, e.g. we can't have A == 1, for example. What if we 
  //  want to use von Neumann numbers for the tags as well as form the components?
  //
  // See:
  //
  // https://en.wikipedia.org/wiki/Axiom_of_pairing
  // https://en.wikipedia.org/wiki/Ordered_pair#Kuratowski's_definition
  // https://www.matej-zecevic.de/2022/022/kuratowski-definition-of-ordered-pairs/
  // https://math.stackexchange.com/questions/1767604/please-explain-kuratowski-definition-of-ordered-pairs
}

rsSetNaive rsSetNaive::unionSet(const rsSetNaive& A, const rsSetNaive& B)
{
  // Use copy constructor to init U to be equal to A and then add those elements from B to U that 
  // are not present in A:
  rsSetNaive U(A);
  for(size_t i = 0; i < B.getCardinality(); i++)
    U.addElement(B[i]);  // Will add B[i] only if it's not already in U
  return U;
}

rsSetNaive rsSetNaive::intersection(const rsSetNaive& A, const rsSetNaive& B)
{
  rsSetNaive I;
  for(size_t i = 0; i < A.getCardinality(); i++)
    if(B.hasElement(A[i]))
      I.addElement(A[i]);
  return I;
}

rsSetNaive rsSetNaive::difference(const rsSetNaive& A, const rsSetNaive& B)
{
  rsSetNaive D;
  for(size_t i = 0; i < A.getCardinality(); i++)
    if(!B.hasElement(A[i]))
      D.addElement(A[i]);
  return D;
}

rsSetNaive rsSetNaive::symmetricDifference(const rsSetNaive& A, const rsSetNaive& B)
{
  rsSetNaive D = difference(B, A);                 // D = B \ A
  for(size_t i = 0; i < A.getCardinality(); i++)   // Add elements from A that are not in B
    if(!B.hasElement(A[i]))
      D.addElement(A[i]);
  return D;
}

rsSetNaive rsSetNaive::product(const rsSetNaive& A, const rsSetNaive& B)
{
  rsSetNaive P;
  for(int i = 0; i < A.getCardinality(); i++)
    for(int j = 0; j < B.getCardinality(); j++)
      P.addElement(orderedPair(A[i], B[j]));
  return P;
}

rsSetNaive rsSetNaive::min(const rsSetNaive& A, 
  bool (*less)(const rsSetNaive& left, const rsSetNaive& right))
{
  if(A.isEmpty()) {
    rsError("Trying to find minimum of empty set - that's undefined");
    return rsSetNaive(); }  // ...but we need to return someting anyway

  rsSetNaive m = A[0];
  for(size_t i = 1; i < A.getCardinality(); i++)
  {
    if(less(A[i], m))
      m = A[i];
  }
  return m;
}

rsSetNaive rsSetNaive::max(const rsSetNaive& A, 
  bool (*less)(const rsSetNaive& left, const rsSetNaive& right))
{
  if(A.isEmpty()) {
    rsError("Trying to find maximum of empty set - that's undefined");
    return rsSetNaive(); }  // ...but we need to return someting anyway

  rsSetNaive m = A[0];
  for(size_t i = 1; i < A.getCardinality(); i++)
  {
    if(less(m, A[i]))
      m = A[i];
  }
  return m;
}

rsSetNaive* rsSetNaive::getCopy() const
{
  rsSetNaive* c = new rsSetNaive;
  for(size_t i = 0; i < elements.size(); i++)
    c->elements.push_back(elements[i]->getCopy());
  return c;
}

//=================================================================================================

bool rsNeumannNumber::isWellFormed(const rsSetNaive& A)
{
  //rsSetNaive target = create(value(A));  // Nope! Calling value() leads to infinite recursion!
  rsSetNaive target = create(A.getCardinality());
  return A.equals(target);
}

size_t rsNeumannNumber::value(const rsSetNaive& A) 
{ 
  rsAssert(isWellFormed(A));
  return A.getCardinality(); 
}

bool rsNeumannNumber::less(const rsSetNaive& x, const rsSetNaive& y)
{ 
  return y.hasElement(x);                           // Expensive. Uses only set operations.
  //return x.getCardinality() < y.getCardinality(); // Efficient. Relies on implementation details.

  // Notes:
  //
  // -We use the fact that for von Neumann numbers, the cardinality is equal to the represented
  //  number such that we can make use of the < operator on size_t. 
  // -Q: Can we get away without appealing to the cardinality? But maybe we don't have to. Maybe
  //  in defining relations, it's OK to use the semantics of the sets? We do this in the definition
  //  of the equivalence relation for a Neumann integer as well.
  // -Maybe we could implement a max() function based on this less that returns a maximum of two
  //  sets or a set of sets.
}
// Needs tests and clean up of the comments

rsSetNaive rsNeumannNumber::create(size_t i)
{
  if(i == 0)
    return rsSetNaive();
  else
    return successor(create(i-1));
}

rsSetNaive rsNeumannNumber::successor(const rsSetNaive& A)
{
  return unionSet(A, singleton(A));

  // Notes:
  //
  // -The operations to form a union and a singleton are allowed by the ZFC axioms. The formation of
  //  the singleton { A } can be see a special case of a pair { A, A } although it's implemented
  //  differently.
}

rsSetNaive rsNeumannNumber::predecessor(const rsSetNaive& A)
{
  if(A.isEmpty())
  {
    rsError("Zero has no predecessor!");
    return A;                              // Return the empty set - although, that's wrong
  }
  return create(value(A) - 1);

  //return maximum(A);  // This should also work - test it!

  // Notes:
  //
  // -I'm actually not sure, if/how the construction implemented here is allowed by the ZFC axioms.
  //  We need this predecessor function to implement addition, though.
  // -A more efficient way to construct the predecessor would be to create a copy of A and remove
  //  the last (or greatest) element.
  // -Even more efficient would be to start with an empty set, resize its elements vector to N-1 
  //  and copy N-1 elements from A.
  // -Maybe to implement it purely in terms of set operations, we could extract the greatest 
  //  element (finding it using the less-function), form a singleton set from it and then do a 
  //  set-difference. This algorithm: "extract element, put into singleton, form set difference of
  //  original with singleton" implements a "remove element" operation
  // -Maybe in the case of A.empty, return a special set that does not represent a Neumann number.
  //  For example, the set { 1 } =  { {0} } is not a valid Neumann number
}

rsSetNaive rsNeumannNumber::add(const rsSetNaive& x, const rsSetNaive& y)
{
  if(isZero(y))
    return x;
  else
    return successor(add(x, predecessor(y)));
}

rsSetNaive rsNeumannNumber::sub(const rsSetNaive& x, const rsSetNaive& y)
{
  rsAssert(!less(x, y), "Trying to subtract larger from smaller natural number");
  rsSetNaive c = y;       // Counts up from y to x.
  rsSetNaive d = zero();  // Counts up from 0 to x-y.
  while(less(c, x))
  {
    c = successor(c);
    d = successor(d);
  }
  return d;


  //// Alternative implementation:
  //if(isZero(y))
  //  return x;
  //else
  //  return sub(predecessor(x), predecessor(y));

  // -There are alternative implementations. We could count down from x to y, for example.
}
// Needs test


rsSetNaive rsNeumannNumber::mul(const rsSetNaive& x, const rsSetNaive& y)
{
  if(isZero(y))
    return zero();
  else
    return add(mul(x, predecessor(y)), x);
}

rsSetNaive rsNeumannNumber::div(const rsSetNaive& x, const rsSetNaive& y)
{
  rsAssert(!isZero(y), "Division by zero error!");
  rsSetNaive a = zero();  // Accumulator
  rsSetNaive q = zero();  // Quotient
  while(true)
  {
    a = add(a, y);        // Acumulate another y into a
    if(less(x, a))        // Stop accumulation when (a >= x) which means (x < a)
      return q;
    q = successor(q);     // Increment quotient by one
    //q = sum(q, one());
  }

  // ToDo:
  //
  // -Maybe implement the integer logarithm in a similar way. I think, we would just need to 
  //  accumulate multiplicatively (starting at 1)
  // -Can we also implement the modulo operation? Maybe we first need to think about implementing
  //  the difference. Maybe to compute difference, we need to use the class rsNeumannInteger? Can 
  //  we do it without it?
  // -Handle division by zero somehow
}

rsSetNaive rsNeumannNumber::pow(const rsSetNaive& x, const rsSetNaive& y)
{
  if(isZero(y))
    return one();
  else
    return mul(pow(x, predecessor(y)), x);
}

rsSetNaive rsNeumannNumber::sqrt(const rsSetNaive& x)
{
  if(isZero(x) || x == one()) return x;   // sqrt(0) == 0, sqrt(1) == 1

  rsSetNaive b = rsNeumannNumber::create(2);
  rsSetNaive b2;
  while(true)
  {
    b2 = mul(b, b);          // b^2
    if( less(x, b2) )
      break;
    b = successor(b);
  }

  if(b2 == x)
    return b;
  else 
    return predecessor(b);

  // ToDo:
  //
  // -Implement a function that takes the n-th root. The code should look the same except for the
  //  line b2 = mul(b, b) which should be replaced by bn = pow(b, n)
}


rsSetNaive rsNeumannNumber::log(const rsSetNaive& x, const rsSetNaive& b)
{
  if(isZero(x)) {
    rsError("Log of zero error");
    return zero();   }              // Mathematically wrong, but we must return something.

  rsSetNaive y = one();             // Multiplicative accumulator for powers of the base b.
  rsSetNaive p = zero();            // Additive accumulator of ones to represent current exponent.
  while(less(y, x)) {
    y = mul(y, b);
    p = successor(p); }

  if(y == x)
    return p;                       // We hitted x exactly. Our p is now the exact log.
  else
    return predecessor(p);          // We overshooted x. Subtract 1 to get floor of log.
}

// ToDo:
//
// -Implement function for n-th root. 
//  -Maybe iterate through all possible bases b starting at 2 and compute b^n. When the result is
//   greater or equal to the input x, return b. I think, we need a similar adjustment as in the log
//   function to get the floor behavior.

//=================================================================================================

void rsNeumannInteger::split(const rsSetNaive& x, rsSetNaive& a, rsSetNaive& b)
{
  a = x.orderedPairFirst();   // x = a-b
  b = x.orderedPairSecond();
}

bool rsNeumannInteger::equals(const rsSetNaive& x, const rsSetNaive& y)
{
  rsSetNaive a, b, c, d, p, q;
  split(x, a, b);                // decompose x = a - b into a, b
  split(y, c, d);                // decompose y = c - d into c, d
  p = NN::add(a, d);             //   compose p = a + d
  q = NN::add(b, c);             //   compose q = b + c
  return p.equals(q);

  // Notes:
  //
  // -The idea is analoguous to how rational numbers p = a/b, q = c/d are equal iff a*d == b*c
}

int rsNeumannInteger::value(const rsSetNaive& x)
{
  // x = (a, b) = a - b
  return int(NN::value(x.orderedPairFirst())) - int(NN::value(x.orderedPairSecond()));
}

rsSetNaive rsNeumannInteger::create(size_t a, size_t b)
{
  return Set::orderedPair(NN::create(a), NN::create(b));
}

rsSetNaive rsNeumannInteger::create(int n) 
{ 
  if(n >= 0)
    return create(size_t(n), size_t( 0));  // x = (n,  0 )
  else
    return create(size_t(0), size_t(-n));  // x = (0, |n|) = (0, -n)
}

rsSetNaive rsNeumannInteger::neg(const rsSetNaive& x)
{
  rsSetNaive a, b;
  split(x, a, b);
  return Set::orderedPair(b, a);  // -(a, b) = (b, a)
}

rsSetNaive rsNeumannInteger::add(const rsSetNaive& x, const rsSetNaive& y)
{
  rsSetNaive a, b, c, d;
  split(x, a, b);
  split(y, c, d);
  return Set::orderedPair(NN::add(a, c), NN::add(b, d)); // (a, b) + (c, d) = (a+c, b+d)
}

rsSetNaive rsNeumannInteger::mul(const rsSetNaive& x, const rsSetNaive& y)
{
  rsSetNaive a, b, c, d;
  split(x, a, b);
  split(y, c, d);
  rsSetNaive p = NN::add(NN::mul(a, c), NN::mul(b, d));  // p = a*c + b*d
  rsSetNaive q = NN::add(NN::mul(a, d), NN::mul(b, c));  // q = a*d + b*c
  return Set::orderedPair(p, q);                         // y = (p, q)

  // Questions: 
  //
  // -Does it make a difference for the representation of the set (i.e. order, duplicates)
  //  when we order the argumens for the sum and product differently? We are allowed to do this due
  //  to commutativity. Maybe test that on a lower level. Look at the string representations of
  //  2+3 and 3+2 and 2*3 and 3*2 for rsNeumannNumber
}

rsSetNaive rsNeumannInteger::canonical(const rsSetNaive& x)
{
  rsSetNaive a, b;
  split(x, a, b);
  size_t va = NN::value(a);
  size_t vb = NN::value(b);
  int    v  = int(va) - int(vb);
  return create(v);

  // Questions:
  //
  // -Can we canonicalize a number purely via set-operations, i.e. without resorting to convert the
  //  sets a,b to their intended values, doing a subtraction and the creating a canonical number 
  //  from scratch? For example, let x = +2 = (5,3) = (2,0) where 5 = { 0,1,2,3,4 } and 
  //  3 = { 0,1,2 }. We want to produce 2 = { 0,1 } only via set operations. The set difference 
  //  a-b = 5-3 would produce { 3,4 } which does not represent 2. It's not even a valid Neumann 
  //  number. Or maybe we could prove that the set { 0,1 } cannot be created from { 0,1,2,3,4 } and
  //  { 0,1,2 } purely via set operations? Or maybe wo could extract the last element from 
  //  { 0,1,2 }? Would that always work? But in sets, there is no notion of "last". But the 2 is 
  //  the last overlapping element. But maybe we could use the "largest" using our defined "less" 
  //  relation? Maybe we could implement "max"-function based on the "less" function and then do: 
  //
  //    if(b.isEmpty())
  //      return (a, b);          // (a,b) is already canonical
  //    else
  //      return (max(b), 0);
  // -Another idea: split (a,b) and form (a,0) and (0,b) and then form (a,0) + (0,b). I think, this
  //  should produce a canonical form of (a,b). ...try it! Maybe implement a function 
  //  isCanonical() that we can use in tests for that.
}














}

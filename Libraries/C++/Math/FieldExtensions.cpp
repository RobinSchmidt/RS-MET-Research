namespace rema
{

// Operators:

template<class T>
rsQuadraticField<T> rsQuadraticField<T>::operator-() const
{
  return rsQuadraticField<T>(-a, -b, n);
}

template<class T>
bool rsQuadraticField<T>::operator==(const rsQuadraticField<T>& y) const
{
  return a == y.a && b == y.b && n == y.n;
}

template<class T>
bool rsQuadraticField<T>::operator!=(const rsQuadraticField<T>& y) const
{
  return !(*this == y);
}

template<class T>
rsQuadraticField<T> rsQuadraticField<T>::operator+(const rsQuadraticField<T>& y) const
{
  rsAssert(n == y.n);
  return rsQuadraticField<T>(a + y.a, b + y.b, n);
}

template<class T>
rsQuadraticField<T> rsQuadraticField<T>::operator-(const rsQuadraticField<T>& y) const
{
  rsAssert(n == y.n);
  return rsQuadraticField<T>(a - y.a, b - y.b, n);
}

template<class T>
rsQuadraticField<T> rsQuadraticField<T>::operator*(const rsQuadraticField<T>& y) const
{
  rsAssert(n == y.n);
  T c = y.a;
  T d = y.b;
  return rsQuadraticField<T>(a*c + b*d*n, a*d + b*c, n);

  //   (a + b*r) * (c + d*r)                               expand
  // = a*c + a*d*r + b*c*r + b*d*r^2                       use: r^2 = n
  // = (a*c + b*d*n) + (a*d + b*c)*r                       final form
  //
  // ToDo: Check order or operations such that it may potentially also work correctly with a type T
  // whose multiplication is not commutative - maybe even with non-commutative addition
}

template<class T>
rsQuadraticField<T> rsQuadraticField<T>::operator/(const rsQuadraticField<T>& y) const
{
  return *this * y.getReciprocal();  // ToDo: Maybe implement division more directly
}

template<class T>
rsQuadraticField<T>& rsQuadraticField<T>::operator+=(const rsQuadraticField<T>& y)
{
  *this = *this + y;
  return *this;
}

template<class T>
rsQuadraticField<T>& rsQuadraticField<T>::operator-=(const rsQuadraticField<T>& y)
{
  *this = *this - y;
  return *this;
}

template<class T>
rsQuadraticField<T>& rsQuadraticField<T>::operator*=(const rsQuadraticField<T>& y)
{
  *this = *this * y;
  return *this;
}

template<class T>
rsQuadraticField<T>& rsQuadraticField<T>::operator/=(const rsQuadraticField<T>& y)
{
  *this = *this / y;
  return *this;
}

// Inquiry

template<class T>
rsQuadraticField<T> rsQuadraticField<T>::getReciprocal() const
{
  T d = a*a - b*b*n;
  return rsQuadraticField(a/d, -b/d, n);

  //   1 / (a + b*r)                                       multiply through by conjugate
  // = (a - b*r) / ((a+b*r)*(a-b*r))                       expand denominator
  // = (a - b*r) / ((a^2 - a*b*r + a*b*r - b^2*r^2)        use: r^2 = n, cancel +-
  // = (a - b*r) / (a^2 - n*b^2)                           let: d = a^2 - n*b^2
  // = a/d + (-b/d)*r                                      final form
}




}



/*=================================================================================================


Try to implement more algebraic field extensions as follows:

- Maybe generalize to quadratic fields that do not simply adjoin the square-root of some number but
  the solution of a0 + a1*x + a2*x^2 = 0. But this (more complicated) implemtation should go into
  its own class.

- Implement a class where we can adjoin more than one square-root. I think, if we add r = sqrt(n)
  and s = sqrt(m), the numbers are represented as a + b*r + c*s + d*r*s. In general, to adjoin
  multiple square roots, we need coeffs for all possible paiwise products. See (1), pg 297. Or 
  wait - no - it's even worse - we need coeffs for all possible products, it seems. Figure out!
  See also pg 325 - there's the case for adjoining sqrt(2) and sqrt(3) to Q. It has coeffs for
  1, sqrt(2), sqrt(3) and sqrt(6)

- For an extension by a cube-root of some number n, we would have r = cbrt(n) and, I think, our 
  numbers would have to be of the general form: a + b*r + c*r^2

  - Multiplication would be (from now on, I suppress the * for multiplication):
      (a + br + cr^2)(d + er + fr^2)
    = ad + aer + afr^2 + brd + brer + brfr^2 + cr^2d + cr^2er + cr^2fr^2
    = (ad)  +  (ae + bd)r  +  (af + be + cd)r^2  + (bf + ce)r^3 + (cf)r^4   use r^3 = n, r^4 = n*r
    = (ad + (bf+ce)n)  +  (ae + bd + (cf)n)r  +  (af + be + cd)r^2
    Verify these!

  - For reciprocation and division, I think, we would have to search for some sort of "conjugate" 
    by which we can multiply numerator and denominator. If our denominator is given by
    a + br + cr^2, we search for a number d + er + fr^2 such that the coeffs for r and r^2 in the
    product (a + br + cr^2)(d + er + fr^2) are both zero. So, we would require:
    (ae + bd + (cf)n) = 0  and  (af + be + cd) = 0. I have no idea, if that is possible, i.e. if 
    such a number d + er + fr^2 exists. Seems like we would have to solve a system of equations for
    d,e,f when we have a,b,c given. -> Figure this out!


- I think, to adjoin an m-th root of n, i.e. r = sqrt[m]{n}, we need to represent the number as
  a0 + a1*r + a2*r^2 + a3*r^3 + ... + a_{m-1} r^{m-1}

- Multiplication would have to be implemented via circular convolution of the coefficient arrays 
  (I think) ...because r^m = n ...not sure -> figure out!  ..ahh - not exactly. I think, we need
  to compute the result of linear convolution and then add the overhanging part wrapped around but
  multiplied by n. So, the wrapped around part does not *just* wrap around but is also weighted by 
  n

- I think, in full generality, if have a polynomial p(x) of degreee n given with roots r1,r2,..,rn
  then we must adjoin all the roots r1,.. to Q to get the splitting field. Sometimes, the roots are
  associated with one another such that we may get away with adjoining less than n numbers. For 
  example p(x) = x^2 - 2 has +-sqrt(2) as roots. We don't need to adjoin -sqrt(2) separately in 
  this case, +sqrt(2) is enough.. Hmm...on page 312, (1) says that any finite extension 
  F(r1,r2,...) to a field or characteristic 0 is equal to a simple extension F(r) of that field.
  That seems unbelevievable! On pg 313, the book presents a proof for why F(a,b) is equal to F(c)
  for some c. Adjoining two elements a and b can be replaced by adjoining just one element c and we
  get the same field. For example, adjoining a = sqrt(2) and b = sqrt(3) can be replaced by 
  adjoining c = a + b. In general, the number c to be adjoined is given by a + t*b for some t that 
  must satisfy certain constraints. In this case, choosing t = 1 was good enough. See (1), pg 313.
  I do not yet fully understand why that should work. Figure out!

- Maybe we also need to allow for complex elements to be adjoined? Consider this example:
  https://www.youtube.com/watch?v=QBX89S6FLMo  It adjoins a complex root w = cbrt(1). I think, it's
  a principle root of 1

- Can we also implement transcendental field extensions? How about Q-adjoin-pi? Maybe in a 
  multiplication, we would not use circular convolution but regular convolution such that the 
  polynomials would get longer in each multiplication or division. But no - I think, that doesn't 
  work for division - multiplying by the conjugate would not lead to the nice cancellation that we 
  get when r is the square root of something. That lack of cancellation may actually already pose a 
  problem for division in the m-th root of n case. But maybe we can multiply through by a 
  specifically defined conjugate 


- What about a less-than relation? Maybe we could say a + b*r < c + d*r  iff  
  (a + b*r) - (c + d*r) < 0. But does that help? I think, it may not make sense to try to define a
  general < relation. After all, for n = -1, we get the complex rationals and ordering complex 
  numbers in a way that plays nicely with the arithmetic operations isn't possible anyway.


- See:

  "One second to compute the largest Fibonacci number I can"
  https://www.youtube.com/watch?v=KzT9I1d-LlQ  

  "Complex Quadratic Integers and Primes"
  https://www.youtube.com/watch?v=eYdKx1lLagA

  "Find a Splitting Field of x^3-1 over Q"
  https://www.youtube.com/watch?v=QBX89S6FLMo



  https://en.wikipedia.org/wiki/Quadratic_field
  https://en.wikipedia.org/wiki/Quadratic_irrational_number
  https://en.wikipedia.org/wiki/Quadratic_integer
  https://mathworld.wolfram.com/QuadraticSurd.html

  https://en.wikipedia.org/wiki/Binary_quadratic_form
  https://en.wikipedia.org/wiki/Discriminant#Fundamental_discriminants
  https://en.wikipedia.org/wiki/Gaussian_rational

  https://en.wikipedia.org/wiki/Algebraic_number_field
  https://en.wikipedia.org/wiki/Cyclotomic_field

*/
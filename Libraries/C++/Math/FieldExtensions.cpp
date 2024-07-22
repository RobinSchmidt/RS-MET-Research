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


Try to implement algebraic field extensions as follows:

- Take an implementation of rational numbers like rsFraction

- To adjoin r = sqrt(n) for some integer (or rational) number n, define the the new field in terms
  of pairs of fractions (a,b) which represent a + b*sqrt(n) = a + b*r

- Addition and subtraction works elementwise, i.e. (a + b*r) + (c + d*r) = (a+c) + (b+d)*r

- Multiplication works like:
  (a + b*r) * (c + d*r) = a*c + a*d*r + b*c*r + b*d*r^2    use r^2 = n
  = (a*c + n*b*d) + (a*d + b*c)*r
 
- Reciprocation:  1 / (a + b*r) = (a - b*r) / ((a+b*r)(a-b*r)) = (a - b*r) / (a^2 - n*b^2)
  = a/d + (-b/d)*r  where  d = a^2 - n*b^2

- Division: ...

- The number n should be a member variable of the class pretty much like the modulus in the
  implementation of modular integers.

- This can be used to adjoin the sqrt(5) and then use Binnet's formula to compute Fibonacci numbers
  via efficient exponentiation in our new field. See:
  "One second to compute the largest Fibonacci number I can"
  https://www.youtube.com/watch?v=KzT9I1d-LlQ  

- Try if we can also use n = -1 and get the (rational) complex numbers. The reciprocation formula 
  looks good. It would evaluate to (a - b*i)/(a^2 + b^2)

- I think, to adjoin an m-th root of n, i.e. r = sqrt[m]{n}, we need to represent the number as
  a0 + a1*r + a2*r^2 + a3*r^3 + ... + a_{m-1} r^{m-1}

- Multiplication would have to be implemented via circular convolution of the coefficient arrays 
  (I think) ...because r^m = n ...not sure -> figure out!  ..ahh - not exactly. I think, we need
  to compute the result of linear convolution and then add the overhanging part wrapped around but
  multiplied by n. So, the wrapped around part does not *just* wrap around but is also weighted by 
  n



Can we also implement transcendental field extensions? How about Q-adjoin-pi?

- Maybe in a multiplication, we would not use circular convolution but regular convolution such 
  that the polynomials would get longer in each multiplication or division. But no - I think, that 
  doesn't work for division - multiplying by the conjugate would not lead to the nice cancellation 
  that we get when r is the square root of something. That lack of cancellation may actually 
  already pose a problem for division in the m-th root of n case. But maybe we can multiply through 
  by a specifically defined conjugate 



- See:
  "Complex Quadratic Integers and Primes"
  https://www.youtube.com/watch?v=eYdKx1lLagA


  https://en.wikipedia.org/wiki/Quadratic_field
  https://en.wikipedia.org/wiki/Quadratic_irrational_number
  https://en.wikipedia.org/wiki/Quadratic_integer
  https://mathworld.wolfram.com/QuadraticSurd.html

  https://en.wikipedia.org/wiki/Binary_quadratic_form
  https://en.wikipedia.org/wiki/Discriminant#Fundamental_discriminants
  https://en.wikipedia.org/wiki/Gaussian_rational


  https://en.wikipedia.org/wiki/Algebraic_number_field

*/
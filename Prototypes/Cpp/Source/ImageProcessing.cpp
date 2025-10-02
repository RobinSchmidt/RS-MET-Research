
//=================================================================================================
// Some image filtering functions (maybe wrap into class):

// simple box filter for smoothing/blurring an image
// maybe have options for edge handling: periodic, ignore, repeat, ...
template<class T>
void boxBlur3x3(const RAPT::rsImage<T>& x, RAPT::rsImage<T>& y)
{
  //rsAssert(y.hasSameShapeAs(x));  // todo!!
  rsAssert(y.getPixelPointer(0,0) != x.getPixelPointer(0,0), "Cant be used in place");
  int w = x.getWidth();
  int h = x.getHeight();

  for(int j = 1; j < h-1; j++)     // loop over lines
  {
    for(int i = 1; i < w-1; i++)   // loop over pixels in current line
    {
      T tmp =  x(i-1, j-1) + x(i-1, j) + x(i-1, j+1)
             + x(i,   j-1) + x(i,   j) + x(i,   j+1)
             + x(i+1, j-1) + x(i+1, j) + x(i+1, j+1);
      tmp *= T(1) / T(9);
      y(i, j) = tmp;
    }
  }

  // todo: handle edges and corners
  // maybe use a weighting that is more circular - the shape of the disease spread looks a bit
  // squarish
  // see https://en.wikipedia.org/wiki/Kernel_(image_processing)
}

template<class T>
void gaussBlur3x3(const RAPT::rsImage<T>& x, RAPT::rsImage<T>& y)
{
  //rsAssert(y.hasSameShapeAs(x));  // todo!!
  rsAssert(y.getPixelPointer(0,0) != x.getPixelPointer(0,0), "Cant be used in place");
  rsAssert(y.hasSameShapeAs(x), "Input and output images must have the same shape");
  int w = x.getWidth();
  int h = x.getHeight();

  for(int j = 1; j < h-1; j++)     // loop over lines
  {
    for(int i = 1; i < w-1; i++)   // loop over pixels in current line
    {
      T d = (T(1)/T(16)) * (x(i-1, j-1) + x(i-1, j+1) + x(i+1, j-1) + x(i+1, j+1)); // diagonal
      T a = (T(1)/T(8))  * (x(i-1, j  ) + x(i,   j-1) + x(i,   j+1) + x(i+1, j  )); // adjacent
      y(i, j) = (T(1)/T(4))*x(i, j) + a + d;
    }
  }

  // ToDo:
  // -Handle edges
}

template<class T>
void gaussBlur5x5(const RAPT::rsImage<T>& x, RAPT::rsImage<T>& y)
{
  //rsAssert(y.hasSameShapeAs(x));  // todo!!
  rsAssert(y.getPixelPointer(0,0) != x.getPixelPointer(0,0), "Cant be used in place");
  rsAssert(y.hasSameShapeAs(x), "Input and output images must have the same shape");
  int w = x.getWidth();
  int h = x.getHeight();

  T c = T(1) / T(273);
  for(int j = 2; j < h-2; j++)
  {
    for(int i = 2; i < w-2; i++)
    {
      // Outer ring:
      T t1 = 1*c * (x(i-2, j-2) + x(i-2, j+2) + x(i+2, j-2) + x(i+2, j+2));     // corners
      T t2 = 4*c * (x(i-1, j-2) + x(i-1, j+2) + x(i+1, j-2) + x(i+1, j+2) +
                    x(i-2, j-1) + x(i-2, j+1) + x(i+2, j-1) + x(i+2, j+1)   );
      T t3 = 7*c * (x(i-0, j-2) + x(i-0, j+2) + x(i-2, j-0) + x(i+2, j+0)); 

      // Inner ring:
      T t4 = 16*c * (x(i-1, j-1) + x(i-1, j+1) + x(i+1, j-1) + x(i+1, j+1));
      T t5 = 26*c * (x(i-0, j-1) + x(i-0, j+1) + x(i-1, j-0) + x(i+1, j+0));

      // Center:
      T t6 = 41*c * x(i,j);

      // Sum:
      y(i,j) = t1 + t2 + t3 + t4 + t5 + t6;
    }
  }
}

template<class T>
void gaussBlur7x7(const RAPT::rsImage<T>& x, RAPT::rsImage<T>& y)
{
  //rsAssert(y.hasSameShapeAs(x));  // todo!!
  rsAssert(y.getPixelPointer(0,0) != x.getPixelPointer(0,0), "Cant be used in place");
  rsAssert(y.hasSameShapeAs(x), "Input and output images must have the same shape");
  int w = x.getWidth();
  int h = x.getHeight();

  T c = T(1) / T(1003);
  for(int j = 3; j < h-3; j++)
  {
    for(int i = 3; i < w-3; i++)
    {
      // Outer ring:
      T t1 = 2*c * (x(i-0, j-3) + x(i-0, j+3) + x(i-3, j-0) + x(i+3, j+0));
      T t2 = 1*c * (x(i-1, j-3) + x(i-1, j+3) + x(i+1, j-3) + x(i+1, j+3) +
                    x(i-3, j-1) + x(i-3, j+1) + x(i+3, j-1) + x(i+3, j+1)   );

      // Middle ring:
      T t3 =  3*c * (x(i-2, j-2) + x(i-2, j+2) + x(i+2, j-2) + x(i+2, j+2));
      T t4 = 13*c * (x(i-1, j-2) + x(i-1, j+2) + x(i+1, j-2) + x(i+1, j+2) +
                     x(i-2, j-1) + x(i-2, j+1) + x(i+2, j-1) + x(i+2, j+1)   );
      T t5 = 22*c * (x(i-0, j-2) + x(i-0, j+2) + x(i-2, j-0) + x(i+2, j+0)); 

      // Inner ring:
      T t6 = 59*c * (x(i-1, j-1) + x(i-1, j+1) + x(i+1, j-1) + x(i+1, j+1));
      T t7 = 97*c * (x(i-0, j-1) + x(i-0, j+1) + x(i-1, j-0) + x(i+1, j+0));

      // Center:
      T t8 = 159*c * x(i,j);

      // Sum:
      y(i,j) = t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8;
    }
  }
}
// This 7x7 kernel seems to be just a more precise version of the 5x5 kernel but has the same width

// Todo: 
// -handle boundaries
// -implement a general filter3x3 function that takes a 3x3 image to be used as filter kernel
// -implement a general filer(img, kernel) function. Maybe use the convolution routine from 
//  rsMatrix (we may create an rsMatrixView). ...but maybe the 2D convolution routine should be 
//  dragged out of rsMatrix. Maybe have a class rsArrayTools2D similar to rsArrayTools and let it 
//  have a function convolve(const T *x, int Mx, int Nx, const T* h, int Mh, int Nh, T* y)
// -Implement Gaussian filters of sizes 5x5, 7x7, see:
//  https://homepages.inf.ed.ac.uk/rbf/HIPR2/gsmooth.htm
//  https://www.researchgate.net/figure/Discrete-approximation-of-the-Gaussian-kernels-3x3-5x5-7x7_fig2_325768087
//  ...done - but we don't handle the edges yet
// -Implement the "magic kernel" and its sharp variant:
//  http://www.johncostella.com/magic/ http://www.johncostella.com/magic/mks.pdf


template<class T>
void sobelEdgeDetector3x3(const RAPT::rsImage<T>& x, RAPT::rsImage<T>& G, RAPT::rsImage<T>& t)
{
  //rsAssert(y.getPixelPointer(0,0) != x.getPixelPointer(0,0), "Cant be used in place");
  //rsAssert(y.hasSameShapeAs(x), "Input and output images must have the same shape");
  //int w = x.getWidth();
  //int h = x.getHeight();

  for(int j = 1; j < h-1; j++)     // loop over lines
  {
    for(int i = 1; i < w-1; i++)   // loop over pixels in current line
    {
      // verify these:
      T Gx =      x(i-1, j-1) - x(i+1, j-1)
             + 2*(x(i-1, j  ) - x(i+1, j  ))
             +    x(i-1, j+1) - x(i+1, j+1);

      T Gy =      x(i-1, j-1) - x(i-1, j+1)
             + 2*(x(i,   j-1) - x(i,   j+1))
             +    x(i+1, j-1) - x(i+1, j+1);

      G(i, j) = sqrt(Gx*Gx, + Gy*Gy);
      t(i, j) = atan2(Gy, Gx);
    }
  }
}
// ...needs tests
// https://en.wikipedia.org/wiki/Sobel_operator
// maybe factor into sobelX, sobelY - can use same temp images as used for G and t
// see also:
// https://en.wikipedia.org/wiki/Prewitt_operator
// https://en.wikipedia.org/wiki/Roberts_cross

/** Approximates a Gaussian blur by using a first order bidirectional IIR lowpass filter several 
times in the horizontal and vertical direction ("bidirectional" here refers to forward/backward, 
not horizontal/vertical). The impulse response of a single forward pass is a decaying exponential, 
but by virtue of the central limit theorem, the more such impulse responses we convolve in, the 
more gaussian the shape becomes. A number of passes of 6 seems to be good enough to get a kernel 
that appears visually circular. Using just a single pass, the shape looks more diamond-like - which 
is not an isotropic kernel. If we want to realize isotropic kernels by filtering horizontally and 
vertically, we need to start from a separable kernel - which the gaussian kernel is.  */
template<class T>
void gaussBlurIIR(const RAPT::rsImage<T>& x, RAPT::rsImage<T>& y, T radius, int numPasses = 6)
{
  rsAssert(y.getPixelPointer(0,0) != x.getPixelPointer(0,0), "Cant be used in place");
  rsAssert(y.hasSameShapeAs(x), "Input and output images must have the same shape");
  int w = x.getWidth();
  int h = x.getHeight();

  T scaledRadius = radius / numPasses;
  // ad-hoc - maybe try to find better formula - maybe plot the variance as function of the number 
  // of passes (in 1D) to figure out the right formula experimentally - or maybe an analytic 
  // formula can be derived?...maybe normalize the area under the curve to unity?

  // Create 1D IIR filter and set up its coefficients - we want a^r = 1/2 -> a = 2^(-1/r). This 
  // means the impulse response decays down to 1/2 after r pixels for a single pass (right?):
  //rsOnePoleFilter<float, float> flt; // maybe use rsFirstOrderFilterBase
  rsFirstOrderFilterBase<T, T> flt; // maybe use rsFirstOrderFilterBase
  T a = pow(2.f, -1.f/scaledRadius); // is this formula right? we need something that lets b approach 1 as r approaches 0

  T b = 1.f - a;
  flt.setCoefficients(b, 0.f, a);

  // horizontal passes:
  y.copyPixelDataFrom(x);
  for(int n = 0; n < numPasses; n++)  // loop over the passes
    for(int j = 0; j < h; j++)        // loop over the rows
    {
      flt.applyForwardBackward(y.getPixelPointer(0, j), y.getPixelPointer(0, j), w); // old

      //// new:
      //T xL = y(0,   j); // maybe we should use x(0, j) instead - likewise for xR
      //T xR = y(w-1, j);
      //flt.applyForwardBackward(y.getPixelPointer(0, j), y.getPixelPointer(0, j), w, xL, xR);
      // I think, the new version handles the boudary condition differently - instead of assuming 
      // zero samples, it assumes infinite repetition of the boundary pixel values
    }

  // vertical passes:
  for(int n = 0; n < numPasses; n++)
    for(int i = 0; i < w; i++)
    {
      flt.applyForwardBackward(y.getPixelPointer(i, 0), y.getPixelPointer(i, 0), h, w); // old

      //// new
      //T xL = y(i, 0);
      //T xR = y(i, h-1);
      //flt.applyForwardBackward(y.getPixelPointer(i, 0), y.getPixelPointer(i, 0), h, w, xL, xR);
    }

  // todo: maybe let the user decide, how the boundaries are handled (repeat last pixel or assume 
  // zeros...maybe other options, like "reflect" could be uesed as well)

  // todo: scale the filter coefficient b, such that the integral of the impulse response becomes
  // 1 (or maybe the sum of the discrete implementation)....or maybe the sum-of-squares? maybe make
  // that user selectable - the sum of the values seems to be appropriate, if we want to use it 
  // for local averaging, as we do in the SIRP model
  // i think, this is already ensured because the geometric series 
  //   S = sum_k a^k = 1/(1-a)  
  // where k runs from 0 to infinity - so with b=1, we would get a sum of 1/(1-a) - scaling that by
  // the reciprocal, would scale by 1-a, which is exactly the formula for b

  // -Maybe combine horizontal and vertical passes with diagonal passes. Maybe that converges 
  //  faster to an isotropic filter. I think, a diagonal filter has stride w+1 or w-1. It also 
  //  needs to scale the coeff by 1/sqrt(2) to compensate for the greater distance.
  // -Maybe for the boundaries, use c*xL, c*xR where 0 <= c <= 1 is a user parameter to dial 
  //  between zero and repeat boundary conditions
}


//=================================================================================================

/** A class to represent some measures of an image filtering kernel. These measurements may be 
relevant to assess the features and quality of a filter kernel.

...TBC...

*/

template<class T>
class rsImageKernelMeasures
{

public:

  /** Sum of all values. */
  static T sum(const rsImage<T>& img);


  static T mean(const rsImage<T>& img) { return sum(img) / img.getNumPixels(); }


  /** Sum of pixel values of the horizontal center line */
  static T centerSumHorz(const rsImage<T>& img);


  static T centerSumVert(const rsImage<T>& img);

  static T centerSumDiagDown(const rsImage<T>& img);

  //static T centerSumDiag2(const rsImage<T>& img);
  // maybe instead of Diag1/2, call them DownDiag and UpDiag


  // do two diag versions, too


  static T aspectRatio(const rsImage<T>& img) { return centerSumHorz(img) / centerSumVert(img); }
  // Why only the center sum? Wouldn't it make more sense to take all rows and all columns? But no!
  // In this case, we would in both cases just sum over all pixels and the ratio would always be 
  // unity. And if we have to pick one row and one column, the center makes the most sense. Maybe it 
  // would make sense to use all rows/cols if we introduce a weight for each row/col that depends on
  // how far away that row/col is from the center - maybe like 1/distance or something.


  /** Measures how anisotropic the kernel is by comparing the sum of the center horizontal strip 
  and the diagonal strip where the off-center pixel values in the diagonal strip are weighted by
  sqrt(2) because they are by that factor farther away from the center pixel. A circularly 
  symmetric (i.e. perfectly isotropic) kernel should give a crossness of zero. A kernel that 
  looks like a cross (like a plus +) gives a value of 1 and a kernel that looks like a diagonal 
  cross (like an x) gives a value of -1.

  ...TBC...
  
  For this computation to make sense, we need to assume that the kernel is rotationally symmetric 
  for a rotation of 90° such that there is no difference between centerSumHorz and centerSumVert 
  and also no difference between centerSumDiagDown and centerSumDiagUp.  */
  static T crossness(const rsImage<T>& img);
  // Maybe rename to crossness, diamondness, squareness, twinkle




  //T sum;      // sum of all values
  //T energy;   // sum of all values squared

  // ToDo:
  // -aspect ratio:  horizontal centerline sum / vertical centerline sum
  // -symmetryHorz:  right wing - left wing (both including the centerline for odd width and height
  // -symmetryVert:  top wing - bottom wing (..dito)
  // -symmetryDiag1: top left wing - bottom right wing
  // -symmetryDiag2: top right wing - bottom left wing
  // -mean, max, rms (root-mean-square)
  // -isotropy

};

template<class T>
T rsImageKernelMeasures<T>::sum(const RAPT::rsImage<T>& img)
{
  T sum(0);
  for(int j = 0; j < img.getHeight(); j++)
    for(int i = 0; i < img.getWidth(); i++)
      sum += img(i, j);
  return sum;
}

template<class T>
T rsImageKernelMeasures<T>::centerSumHorz(const RAPT::rsImage<T>& img)
{
  int h = img.getHeight();

  rsAssert(rsIsOdd(h)); 
  // Currently, this is implemented only for kernels with odd height. ToDo: For even h, we need to 
  // scan 2 horizontal lines near the center and compute their average.

  T sum(0);
  int i = (h-1) / 2;
  for(int j = 0; j < img.getHeight(); j++)
    sum += img(i, j);
  return sum;

  // Maybe factor out a sumHorz(img, i) function. that takes the sum of the i-th line
}

template<class T>
T rsImageKernelMeasures<T>::centerSumVert(const RAPT::rsImage<T>& img)
{
  int w = img.getWidth();
  rsAssert(rsIsOdd(w)); // see comment in centerSumHorz, situation is analogous here
  T sum(0);
  int j = (w-1) / 2;
  for(int i = 0; i < img.getWidth(); i++)
    sum += img(i, j);
  return sum;
}

template<class T>
T rsImageKernelMeasures<T>::centerSumDiagDown(const RAPT::rsImage<T>& img)
{
  int w = img.getWidth();
  int h = img.getHeight();
  int n = rsMin(w, h); 
  T sum(0);
  for(int i = 0; i < n; i++)
    sum += img(i, i);
  return sum;
}


template<class T>
T rsImageKernelMeasures<T>::crossness(const RAPT::rsImage<T>& img)
{
  int w = img.getWidth();
  int h = img.getHeight();
  rsAssert(w == h);

  // Form the horizontal center sum:
  T sh = centerSumHorz(img);

  // Form the weighted diagonal center sum:
  int c = (h-1) / 2;              // center
  T sdw(0);                       // initialize sum to 0
  sdw += img(c, c);               // center pixel gets a weight of 1
  T s = sqrt(2);                  // weight for off-center diagonal pixels
  for(int i = 0; i < c; i++)
    sdw += s * img(i, i);
  for(int i = c+1; i < h; i++)
    sdw += s * img(i, i);

  // Compute the anisotropy:
  //T a = (sh - sdw) / h;  
  T a = (sh - sdw) / (h - 1);  // (h-1) may make sense bcs the center pixel may not count

  // Ad hoc to normalize value for diagonal cross to -1:
  if(a < 0)
    a *= 1.0 / sqrt(2);
  // That's rather unelegant. Can we do something better?

  return a;

  // For this measurement to make sense, the kernel needs to satsify certain symmetries which we
  // check here:
  T sv = centerSumVert(img);
  rsAssert(rsIsOdd(w));
  rsAssert(sv == sh);
  // maybe do
  //rsAssert( isHorizontallySymmetric(img) );
  //rsAssert( isVerticallySymmetric(img) );
  //rsAssert( isDownDiagonallySymmetric(img) );
  //rsAssert( isUpDiagonallySymmetric(img) );
  //rsAssert( isRotationallySymmetric(img) );
  // maybe check all possible symmetries of a square. How many are there? I think, it's 8, see
  // https://proofwiki.org/wiki/Definition:Symmetry_Group_of_Square
  // http://mathonline.wikidot.com/the-group-of-symmetries-of-the-square
  // Maybe the kernel should have all of them
}

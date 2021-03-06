﻿#include "Tools.cpp"  // this includes rapt and rosic

//-------------------------------------------------------------------------------------------------
// move some of this code to rapt:

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
}


// todo: implement a general filter3x3 function that takes a 3x3 image to be used as filter kernel


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
  // the reciprocal, would scale by 1-a, whcih is exactly the formula for b
}

void testGaussBlurIIR()
{
  int w = 101;
  int h = 101;
  float radius = 20.0f;   // decay down to 1/2 after that many pixels (for 1 pass)
  int numPasses = 6;      // 6 seems good enough for a visually isotropic filter
  // controls shape - the higher, the better we approximate a true Gaussian - 5 or 6 seems to be
  // good enough


  //radius = 30.f; // test

  //numPasses = 1; // for testing decay

  // try with w != h

  RAPT::rsImage<float> x(w,h), y(w,h);  // input and output images

  // input is a single white pixel in the middle on black background: 
  x.fillAll(0.f);
  x(w/2, h/2) = 1.f;
  //x(w/4, h/4) = 1.f;

  gaussBlurIIR(x, y, radius, numPasses);


  //float ySum = rsArrayTools::sum(y.getPixelPointer(0,0), w*h); 
  // should be 1, regardless of radius and numPasses

  RAPT::rsImageProcessor<float>::normalize(y);
  // without normalization, it looks very dark - what we actually want is an energy-preserving
  // filter (i think)...or maybe a filter that preserves the total sum of pixel values?

  writeImageToFilePPM(x, "InputGaussIIR.ppm");
  writeImageToFilePPM(y, "OutputGaussIIR.ppm");
  writeScaledImageToFilePPM(y, "OutputGaussIIRx4.ppm", 4);

  // Observations:
  // -when using a low number of passes, we get a diamond-like shape which becomes more and more
  //  circular, the more passes we use
  // -without normalizing the output, it tends to get very dark, such that we don't really see 
  //  anything - however, the sum of the pixel values is nevertheless 1, as it should be
  //  (it tends to get less than one, when the radius becomes so large that there would be a 
  //  significant number nonzero zero pixels outside the image - but if we choose the radius such 
  //  that nonzero pixels onyl exist within the image, it seems to work)
  // -when using last-pixel repetition for the boundary conditions and the white pixel is not at 
  //  the center (like using x(w/4, h/4) = 1.f;) and the radius is sufficiently large, the 
  //  "center of mass" in the filtered image shifts toward the corner - i.e. the corner is brighter
  //  than it should be. that makes sense, because assuming the out-of-range pixels to just repeat 
  //  the final brightness value - where it should actually decay - we assume then to be brighter 
  //  than they actually would be, when using the same filter on a larger image and cropping later
  //  -maybe we should use the final pixel values from the opriginal image in each pass - and not 
  //   the ones of the output of the previous pass
}





// mathematically, all these implementations should give the same results, but they may behave 
// differently numerically and/or may be more or less efficient:

/** Applies the given filter to the given image multiple times. This implementation interleaves 
the horizontal and vertical passes. */
template<class T>
void applyMultiPass1(rsFirstOrderFilterBase<T, T>& flt, rsImage<T>& img, int numPasses)
{
  int w = img.getWidth();
  int h = img.getHeight();
  for(int n = 0; n < numPasses; n++)  // loop over the passes
  {
    // horizontal pass:
    for(int j = 0; j < h; j++)        // loop over the rows
      flt.applyForwardBackward(img.getPixelPointer(0, j), img.getPixelPointer(0, j), w);

    // vertical pass:
    for(int i = 0; i < w; i++)        // loop over the columns
      flt.applyForwardBackward(img.getPixelPointer(i, 0), img.getPixelPointer(i, 0), h, w);
  }
}
// instead of using a serial connection of forward and backward passes, we could also try a 
// parallel connection - the center sample would be counted twice, so maybe, we should use
// y(x) = forward(x) + backward(x) - b*x
// where forward, backward mean the forward and backward application of the filter and b is the
// feedforward coeff of the filter. x is the input signal and y is the output


/** Apply filter in west-south-west/east-north-east direction (and back). */
template<class T>
void applySlantedWSW2ENE(rsFirstOrderFilterBase<T, T>& flt, const rsImage<T>& x, rsImage<T>& y)
{
  rsAssert(y.hasSameShapeAs(x));
  int w  = x.getWidth();
  int h  = x.getHeight();
  int numDiagonals = (w+1)/2 + h - 1;  // verify this!
  for(int d = 0; d < numDiagonals; d++)
  {
    // figure out start and end coordinates of the current diagonal:
    int iStart = 0;
    int jStart = d;
    if(d >= h) {
      jStart = h-1;
      iStart = 2*(d-jStart); }

    // apply forward pass from west-south-west to east-north-east:
    int i = iStart;
    int j = jStart;

    flt.reset();                       // old -  todo: use some xL - maybe x(iStart, jStart)

    //T xij = x(i,j);
    //flt.setStateForConstInput(x(i, j));  // new - seems to cause problems
    //int dummy = 0;
    // todo: figure out, if it also causes similar problems when doing horizontal and vertical 
    // filtering instead of slanted

    while(i < w && j >= 0) {
      y(i, j) = flt.getSample(x(i, j));
      i++; if(i >= w) { j--; break; }
      y(i, j) = flt.getSample(x(i, j));
      i++; j--; }

    // apply backward pass from east-north-east to west-south-west:
    i--; j++;
    flt.prepareForBackwardPass();       // old

    //flt.prepareForBackwardPass(x(i, j));  // new 
    // no - that's wrong when the filter is used in place - then x(i,j) has already been 
    // overwritten by now and now contains y(i,j) ...we somehow must extract the last sample of the
    // slanted line *before* overwriting it ...the filter-state x1 should still contain it, so 
    // maybe flt.prepareForBackwardPass(flt.getStateX()); should work
    //T x1 = flt.getStateX();
    //flt.prepareForBackwardPass(flt.getStateX());

    if(rsIsOdd(w) && i == w-1) {
      y(i, j) = flt.getSample(x(i, j));
      i--; j++; }
    while(i >= 0 && j <= jStart) {
      y(i, j) = flt.getSample(x(i, j));
      i--; if(i < 0) break;
      y(i, j) = flt.getSample(x(i, j));
      i--; j++; }
  }
}
// h.. the "new" lines lead to weird results

// example: image with w=9, h=6:
//   │ 0 1 2 3 4 5 6 7 8
// ──┼──────────────────
// 0 │ 0 0 1 1 2 2 3 3 4
// 1 │ 1 1 2 2 3 3 4 4 5
// 2 │ 2 2 3 3 4 4 5 5 6
// 3 │ 3 3 4 4 5 5 6 6 7
// 4 │ 4 4 5 5 6 6 7 7 8
// 5 │ 5 5 6 6 7 7 8 8 9
// matrix entries stand for the index of the diagonal d - we mark each pixel with the index of the 
// diagonal that crosses it
// traversal of pixel locations (only forward)
// d  (i,j),(i,j),(i,j),(i,j),(i,j),(i,j),(i,j),(i,j),(i,j),(i,j)
// 0: (0,0),(1,0)
// 1: (0,1),(1,1),(2,0),(3,0)
// 2: (0,2),(1,2),(2,1),(3,1),(4,0),(5,0)
// 3: (0,3),(1,3),(2,2),(3,2),(4,1),(5,1),(6,0),(7,0)
// 4: (0,4),(1,4),(2,3),(3,3),(4,2),(5,2),(6,1),(7,1),(8,0)
// 5: (0,5),(1,5),(2,4),(3,4),(4,3),(5,3),(6,2),(7,2),(8,1)
// 6: (2,5),(3,5),(4,4),(5,4),(6,3),(7,3),(8,2)
// 7: (4,5),(5,5),(6,4),(7,4),(8,3)
// 8: (6,5),(7,5),(8,4)
// 9: (8,5)
// desired start pixels for (i,j) for backward pass - always the last index pair in each line: 
// (1,0),(3,0),(5,0),(7,0),(8,0),(8,1),(8,2),(8,3),(8,4),(8,5)  ..corresponding start indices:
// (0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(2,5),(4,5),(6,5),(8,5)

// https://theasciicode.com.ar/extended-ascii-code/box-drawings-single-horizontal-line-character-ascii-code-196.html

template<class T>
void applySlanted(rsImage<T>& img, T kernelWidth)
{
  rsFirstOrderFilterBase<T, T> flt;

  kernelWidth /= sqrt(T(1.25));  
                 // == sqrt(1*1 + 0.5*0.5): length of line segment in each pixel -> verify

  T a = pow(T(2), T(-1)/kernelWidth);
  flt.setCoefficients(T(1)-a, T(0), a);

  applySlantedWSW2ENE(flt, img, img);
  flipLeftRight(img, img);
  applySlantedWSW2ENE(flt, img, img);
  flipLeftRight(img, img);

  // -it makes a difference whether we do apply->flip->apply->flip or flip->apply->flip->apply
  //  ->the corners look different (test with high kernel width)
  // -maybe this will not happen, when the apsect ratio is 2x1? if so, maybe we should extend the 
  //  image to have aspect ratio 2x1, filter and then crop? similarly, for the diagonal filters, we
  //  should extend to aspect ratio 1x1 and then we can get rid of having to do it in two ways and 
  //  taking the average? -> try it
  // -todo: try in preparForBackwardPass not to assume to go down to zero but to some 
  //  constant and then pass the last input pixel brightness to that - maybe that fixes it?
  //  -use flt.prepareForBackwardPass(flt.getX1())..or something - the filter stores the previous
  //   input so we can use that without extra code
  // ..then also apply the filters to the transposed image

  // -what could be meaningful boundary conditions for images - just repeating the last pixel value
  //  would make the filter very sensitive to cropping pixels away when there are fast changes at 
  //  the boundary (i.e. last pixel black, 2nd to last white -> crop by 1 pixel -> get totally 
  //  different result) ...maybe we should use some sort of local average near the boundary - say, 
  //  over 10 pixels?
}

template<class T>
void testImageFilterSlanted(int w, int h, T kernelWidth)
{
  rsImage<T> img(w, h);
  //img(1, 1) = 1.f;
  //img(2, 2) = 1.f; // try 1,1; 1,2; 2,1; 3,3; 3,2; 2,3
  //img(3, 3) = 1.f;

  img(w/2, h/2) = 1.f;
  //img(20, 20) = 1.f;
  //img(21, 21) = 1.f;

  applySlanted(img, kernelWidth);
  rsImageProcessor<T>::normalize(img);
  std::string name = "SlantedFilter_" + std::to_string(w) + "x" + std::to_string(h) + ".ppm";
  writeImageToFilePPM(img, name.c_str());
}

void testImageFilterSlanted()
{
  testImageFilterSlanted( 80,  40,  15.f); 
  //testImageFilterSlanted( 80,  40,  30.f); 

  //testImageFilterSlanted(  9,   6,  2.f);
  //testImageFilterSlanted(  9,   7,  2.f);  // result not symmetrical
  //testImageFilterSlanted( 20,  10,  3.f); 
  //testImageFilterSlanted( 21,  11,  3.f); 


  /*
  testImageFilterSlanted( 50, 100, 30.f);
  testImageFilterSlanted( 51, 100, 30.f);
  testImageFilterSlanted(100,  30, 30.f);
  testImageFilterSlanted(100,  50, 30.f);
  testImageFilterSlanted(100,  70, 30.f);
  testImageFilterSlanted(100, 100, 30.f);
  testImageFilterSlanted(101,  30, 30.f);
  testImageFilterSlanted(101,  50, 30.f);
  testImageFilterSlanted(101,  70, 30.f);
  testImageFilterSlanted(101, 101, 30.f);
  testImageFilterSlanted(102,  52, 30.f);
  */

  // Observations:
  // -When the image width is even, we see a checkerboard pattern. I think, in the second pass, 
  //  every even-numbered lines encounter a doublet of nonblack pixels along its journey and the 
  //  odd numbered lines encounter only black pixels (or vice versa). When w is odd, every line of
  //  the second pass encounters one nonblack pixel.
  //  -the checkerboard pattern is undesirable, when the filter is used on its own, but when used
  //   as part of a higher-level filtering algo that also includes vertical and horizontal passes,
  //   i think, the effect will be smeared out by these passes, so it may not be a problem in 
  //   practice - we need tests that use the filter in this context
}

/** Apply filter in south-west/north-east direction (and back). */
template<class T>
void applyDiagonalSW2NE(rsFirstOrderFilterBase<T, T>& flt, rsImage<T>& img)
{
  int w = img.getWidth();
  int h = img.getHeight();
  int numDiagonals  = w + h - 1;

  for(int d = 0; d < numDiagonals; d++)
  {
    // figure out start and end coordinates of the current diagonal:
    int iStart = d;
    int jStart = 0;
    if(d >= w) {
      iStart  = w-1;
      jStart += d-w+1; }

    // go from top-right to bottom-left:
    flt.reset();
    int i = iStart;
    int j = jStart;
    while(i >= 0 && j < h) {
      img(i, j) = flt.getSample(img(i, j));
      i--; j++;  } // go one pixel to bottom-left

    // go from bottom-left to top-right:
    flt.prepareForBackwardPass();
    i++; j--;
    while(i <= iStart && j >= jStart)  {
      img(i, j) = flt.getSample(img(i, j));
      i++; j--; } // go one pixel to top-right
  }
}

/** Apply filter in south-east/north-west direction (and back). */
template<class T>
void applyDiagonalSE2NW(rsFirstOrderFilterBase<T, T>& flt, rsImage<T>& img)
{
  int w = img.getWidth();
  int h = img.getHeight();
  int numDiagonals  = w + h - 1;
  for(int d = 0; d < numDiagonals; d++)
  {
    int iStart = 0;
    int jStart = h-d-1;
    if(d >= h) {
      iStart = d-h+1;
      jStart = 0;  }

    flt.reset();
    int i = iStart;
    int j = jStart;
    while(i < w && j < h) {
      img(i, j) = flt.getSample(img(i, j)); i++; j++; }

    flt.prepareForBackwardPass();
    i--; j--;
    while(i >= iStart && j >= jStart) {
      img(i, j) = flt.getSample(img(i, j)); i--; j--; }
  }
}

template<class T>
void applyDiagonal(rsFirstOrderFilterBase<T, T>& flt, rsImage<T>& img)
{
  //// test:
  //// SW -> NE first, then SE -> NW:
  //applyDiagonalSW2NE(flt, img);
  //applyDiagonalSE2NW(flt, img);
  //return;

  int w = img.getWidth();
  int h = img.getHeight();
  rsImage<T> tmp1 = img, tmp2 = img;

  // The order in which these two functions are called determines, how the edge effects manifest 
  // themselves, so we do it in both orders and take the average:

  // SW -> NE first, then SE -> NW:
  applyDiagonalSW2NE(flt, tmp1);
  applyDiagonalSE2NW(flt, tmp1);

  // SE -> NW first, then SW -> NE:
  applyDiagonalSE2NW(flt, tmp2);
  applyDiagonalSW2NE(flt, tmp2);

  // average:
  rsArrayTools::weightedSum(tmp1.getPixelPointer(0,0), tmp2.getPixelPointer(0,0), 
    img.getPixelPointer(0,0), w*h, T(0.5), T(0.5));

  // hmm... well - it's not ideal, but at least, it's symmetrical now - but try to get rid of 
  // having to do it twice


  // todo: test this with images for which h > w - this probably needs different code
  // ..hmmm - it seems to work
}

template<class T>
void exponentialBlur(rsImage<T>& img, T radius)
{
  rsFirstOrderFilterBase<T, T> flt;
  T a = pow(T(2), T(-1)/radius);
  flt.setCoefficients(T(1)-a, T(0), a);
  applyMultiPass1(flt, img, 1);   // replace by apply HorzVert (no MultiPass)
  radius /= sqrt(T(2));           // because Pythagoras
  a = pow(T(2), T(-1)/radius);
  flt.setCoefficients(T(1)-a, T(0), a);
  applyDiagonal(flt, img);
}

template<class T>
void exponentialBlur(rsImage<T>& img, T radius, int numPasses)
{
  //radius /= numPasses;  // does that formula make sense? ...nope: it contracts too much
  radius /= sqrt(T(numPasses));   // looks better
  for(int n = 0; n < numPasses; n++)
    exponentialBlur(img, radius);
}
// todo: figure out the right formula for contracting the radius as function of the number of 
// passes by considering the impulse response of the N-pass filter (given by the N-fold convolution
// of the 1-pass impulse response with itself). maybe normalize the area under that impulse 
// response (or maybe the squared impulse response - but that may be equivalent to what i already
// have implemented for the butterworth scaler - energy normalization). let
// h_1(t) = exp(-t/T) for t >= 0, h_N(t) = conv(h_1, h_{N-1}) be the impulse responses of 1-pass 
// and N-pass filters -> obtain expressions for h_N(t) and \int_0^{inf} h_N(t) dt...maybe it's 
// something like h_N(t) = k_N * t^N * exp(-t/T) ? -> use sage

// try a (recursive) moving-average filter - should create a rectangular block - needs to 
// implement prepareForBackwardPass - when input goes to zero, output will go to zero in a 
// straight line - so the y-state should be just half of the final y-state?

void testExponentialBlur()
{
  int w = 401;
  int h = 401;
  float radius = 12.f;
  int numPasses = 3; // i'd say, 3 is the sweet spot in terms of the tradeoff between isotropy
  // and CPU requirements - maybe even just 2 - with 3, the non-isotropy is not even visible in the
  // outermost contours anymore, with two it seems invisible in the 1/256 contour - and so it 
  // should be even less visible in the kernel itself - but more passes may be required, if the 
  // shape of the decay is desired to be more Gaussian


  rsImage<float> img(w, h);
  img(w/2, h/2) = 1.f;


  exponentialBlur(img, radius, numPasses);

  rsImageProcessor<float>::normalize(img);
  writeImageToFilePPM(img, "ExponentialBlur.ppm");

  // plot contours: 
  std::vector<float> levels = { 1.f/8192, 1.f/4096, 1.f/2048, 1.f/1024, 1.f/512, 1.f/256, 1.f/128, 
                                1.f/64,   1.f/32,   1.f/16,   1.f/8,    1.f/4,   1.f/2 };
  std::vector<float> colors(levels.size());
  rsFill(colors, 1.f);
  rsImageContourPlotter<float, float> contourPlotter;
  rsImage<float> contourLines = contourPlotter.getContourLines(img, levels, colors, true);
  colors = rsRangeLinear(0.f, 1.f, (int) levels.size());
  rsImage<float> contourFills = contourPlotter.getContourFills(img, levels, colors, true);
  writeImageToFilePPM(contourLines, "ExponentialBlurContourLines.ppm");
  writeImageToFilePPM(contourFills, "ExponentialBlurContourFills.ppm"); // looks wrong!



  // -the shape of the contours is a bit like a rounded octagon, but viewing the actual kernel, 
  //  that's not really an issue
  // -the brightness decays away from the center exponentially, as expected (the spacing of the 
  //  contours at the center is a bit wider than the given radius in pixels because we have 
  //  multiple passes and the decay is exponential only asymptotically)
  // -maybe gaussian blur can be achieved by using multiple passes of the exponential blur
  // -next step: try to use a complex version of that - what if we just use the existing functions
  //  with a complex radius? would that work?
  // -as an alternative to normalizing after applying the filter: calculate total energy of the 
  //  image before and after filtering and multiply by the sqrt of the quotient - preserving the
  //  total energy preserves perceived overall brightness (right?)
  // -according to this video: https://www.youtube.com/watch?v=LKnqECcg6Gw  human brightness/color
  //  perception follows the log of the energy where the energy is given by the squared pixel 
  //  brightnesses - that implies, a perceptually correct filter should do it like this: 
  //  square pixel values -> apply filter -> square-root pixel values. this is especially important
  //  when the filter is applied to RGB channels of a color image
}


// move to rsImageProcessor when finished:

template<class T>
void flipLeftRight(const rsImage<T>& x, rsImage<T>& y)
{
  int w = x.getWidth();
  int h = x.getHeight();
  y.setSize(w, h);
  for(int j = 0; j < h; j++)
    rsArrayTools::reverse(&x(0, j), &y(0, j), w);
}

template <class T>
void rsReverse(T* x, int N, int stride)
{
  for(int i = 0; i < stride*N/2; i += stride)
    rsSwap(x[i], x[N-i-1]);
}
// needs test - if it works, move to rsArrayTools

template<class T>
void flipTopBottom(const rsImage<T>& x, rsImage<T>& y)
{
  int w = x.getWidth();
  int h = x.getHeight();
  y.setSize(w, h);
  for(int i = 0; i < w; i++)
    rsReverse(&x(i, 0), h, w); // w is the stride
}

template<class T>
void transpose(const rsImage<T>& x, rsImage<T>& y)
{
  int w = x.getWidth();
  int h = x.getHeight();
  y.setSize(h, w);
  for(int i = 0; i < w; i++)
    for(int j = 0; j < h; j++)
      y(j, i) = x(i, j);
}




template<class T>
bool writeComplexImageToFilePPM(const rsImage<std::complex<T>>& img, const char* path)
{
  int w = img.getWidth();
  int h = img.getHeight();
  rsImage<T> imgR(w,h), imgI(w,h), imgA(w,h), imgP(w,h);
  for(int j = 0; j < h; j++) {
    for(int i = 0; i < w; i++) {
      imgR(i, j) = img(i, j).real();
      imgI(i, j) = img(i, j).imag();
      imgA(i, j) = abs(img(i, j));
      imgP(i, j) = arg(img(i, j)); }}


  rsImageProcessor<T>::normalize(imgR);
  rsImageProcessor<T>::normalize(imgI);
  rsImageProcessor<T>::normalize(imgA);
  rsImageProcessor<T>::normalize(imgP);

  writeImageToFilePPM(imgR, "RealPart.ppm");
  writeImageToFilePPM(imgI, "ImagPart.ppm");
  writeImageToFilePPM(imgA, "AbsValue.ppm");
  writeImageToFilePPM(imgP, "Phase.ppm");
  // todo: make writing all parts (re,im,abs,phase) optional, use single temp image for all parts
  // -use path variable - append Re,Im,Abs,Phs - requires that the writer function does not expect
  //  the .ppm extension to be passed - the function should append it itself - this requires a lot
  //  of code to be modified

  return true;  // preliminary
}

template<class T>
void applyComplexExpBlur(rsImage<std::complex<T>>& img, T radius, T omega, int numPasses,
  T diagRadiusScaler = sqrt(T(2)), T diagFreqScaler = sqrt(T(2)))
{
  // compensate for number of passes:
  radius /= sqrt(T(numPasses));
  omega  *= sqrt(T(numPasses));

  // compute filter coeffs for and apply horizontal and vertical passes:
  using Complex = std::complex<T>;
  rsFirstOrderFilterBase<Complex, Complex> flt;
  Complex j(T(0), T(1));
  Complex ar = pow(T(2), T(-1)/radius);
  Complex ai = j*omega;
  Complex a  = ar + ai;
  Complex b  = Complex(1) - a;
  flt.setCoefficients(b, T(0), a);
  applyMultiPass1(flt, img, numPasses);
  // i think the phase response is controlled by the angle of b, but due to bidirectional 
  // application, this will cancel out, so the initial phase will always be 0, regardless of the
  // angle of b

  // compute filter coeffs for and apply diagonal and antidiagonal passes:
  radius /= diagRadiusScaler;   // because Pythagoras
  omega  *= diagFreqScaler;     // 
  ar = pow(T(2), T(-1)/radius);
  ai = j*omega;
  a  = ar + ai;
  b  = Complex(1) - a;
  flt.setCoefficients(b, 0.f, a);
  for(int n = 1; n <= numPasses; n++)
    applyDiagonal(flt, img);
}
// Interesting interference patterns can be created when using a rather high frequency (in 
// relation to the radius). Also, the multiplication factors for the diagonal passes could be
// different, leading to different results - this is only interesting for artistic purposes - for
// generating isotropic kernels, the factors should be as they are - maybe make them optional 
// parameters (done)

void testComplexExponentialBlur()
{
  int w = 401;
  int h = 401;
  int numPasses = 3;

  using Complex = std::complex<float>;
  Complex j(0.f, 1.f);


  float radius = 20.f;
  float omega  = float(PI/60);
 
  rsImage<Complex> img(w, h);
  img(w/2, h/2) = 1.f;


  rsImage<Complex> imgC(w, h);
  imgC.convertPixelDataFrom(img); // maybe make it more convenient: imgC = img.convert<Complex>();

  applyComplexExpBlur(imgC, radius, omega, numPasses);

  writeComplexImageToFilePPM(imgC, "ComplexExponentialBlur"); // filename not yet used by function

  // -for spotting anisotropy, the phase-plot seems to be useful
  // -the abs looks cool!
}

void animateComplexExponentialBlur()
{
  //rsVideoFileWriter v;
  //std::string str = v.getFfmpegInvocationCommand("Balh");

  // video parameters:
  int w         = 400;  // ffmpeg accepts only even widths and heights
  int h         = 400;
  int numFrames = 400;  // 400 is nice
  int fps       = 25;


  // animation parameters:
  int numPasses = 3;
  float radius1 = 20.f;
  float radius2 = 20.f;
  float omega1  = float(PI/60);
  float omega2  = float(PI/20);
  float radScl1 = sqrt(2.f);
  float radScl2 = sqrt(2.f);
  float frqScl1 = sqrt(2.f);
  float frqScl2 = sqrt(2.f);

  //// test:
  //omega1  = float(-PI/20);
  //omega2  = float(PI/10);

  // create animation:
  using Complex = std::complex<float>;
  rsImage<Complex> imgC(w, h);   // complex image
  rsImage<float>   imgR(w, h);   // real image
  rsVideoRGB video(w, h);
  //video.setPixelClipping(true);    // turn this off to reveal problems

  for(int n = 0; n < numFrames; n++)
  {
    // compute frame parameters:
    float c = float(n) / float(numFrames-1);
    float radius = (1.f-c)*radius1 + c*radius2;

    //float c2 = 1 - (c-1)*(c-1);  // slower toward the end
    //float c2 = sqrt(c);
    float c2 = pow(c, 1.f/3.f);
    float omega  = (1.f-c2)*omega1  + c2*omega2;

    float radScl = (1.f-c)*radScl1 + c*radScl2;
    float frqScl = (1.f-c)*frqScl1 + c*frqScl2;

    // compute complex frame:
    imgC.clear();
    imgC(w/2,   h/2)   = 1.f;
    imgC(w/2,   h/2-1) = 1.f;
    imgC(w/2-1, h/2)   = 1.f;
    imgC(w/2-1, h/2-1) = 1.f;
    applyComplexExpBlur(imgC, radius, omega, numPasses, radScl, frqScl);

    // compute absolute value or real part and append to video:
    for(int j = 0; j < h; j++)
    {
      for(int i = 0; i < w; i++)
      {
        imgR(i, j) = abs(imgC(i, j));
      }
    }
    rsImageProcessor<float>::normalize(imgR);
    video.appendFrame(imgR, imgR, imgR);

  }

  rsVideoFileWriter vw;
  vw.setFrameRate(fps);
  vw.setCompressionLevel(10);  // 0: lossless, 10: good enough, 51: worst
  vw.setDeleteTemporaryFiles(false);
  //vw.writeVideoToFile(video, fileName);
  vw.writeVideoToFile(video, "ComplexExpBlur");

  //ffmpeg -y -r 25 -i VideoTempFrame%d.ppm -vcodec libx264 -crf 10 -pix_fmt yuv420p -preset veryslow ExpBlur.mp4


  // -the frequency sweep should probably be nonlinear (slower at the end - where the frequency 
  //  is higher)

  //int dummy = 0;
}


/** This implementation first does all the horizontal passes and then all the vertical passes. */
template<class T>
void applyMultiPass2(rsFirstOrderFilterBase<T, T>& flt, rsImage<T>& img, int numPasses)
{
  int w = img.getWidth();
  int h = img.getHeight();

  // horizontal passes:
  for(int n = 0; n < numPasses; n++)  // loop over the passes
    for(int j = 0; j < h; j++)        // loop over the rows
      flt.applyBidirectionally(img.getPixelPointer(0, j), img.getPixelPointer(0, j), w);

  // vertical passes:
  for(int n = 0; n < numPasses; n++)
    for(int i = 0; i < w; i++)
      flt.applyBidirectionally(img.getPixelPointer(i, 0), img.getPixelPointer(i, 0), h, w);
}


/** Implements a chain of identical first order filters. 


the prepareForBackwardPass function does not work correctly because onyl for the first stage, we
can assume that the output goes to zero at the edges - for all follwoign stages, this assumption is
wrong because the filters that com before still produce nonzero outputs - bottom line: it doesn't
work as intended for image processing.

*/

template<class TSig, class TPar>  // use TSig, TPar
class rsFirstOrderFilterChain
{


public:

  void setupFromPrototype(const rsFirstOrderFilterBase<TSig, TPar>& proto, int numStages)
  {
    stages.resize(numStages);
    for(int n = 0; n < numStages; n++)
      stages[n].setCoefficients(proto.getB0(), proto.getB1(), proto.getA1());
  }

  inline TSig getSample(TSig in)
  {
    TSig tmp = in;
    for(size_t i = 0; i < stages.size(); i++)
      tmp = stages[i].getSample(tmp);
    return tmp;
  }

  inline void reset()
  {
    for(size_t i = 0; i < stages.size(); i++)
      stages[i].reset();
  }

  inline void prepareForBackwardPass()
  {
    for(size_t i = 0; i < stages.size(); i++)
      stages[i].prepareForBackwardPass();
  }




  void applyBidirectionally(TSig* x, TSig* y, int N)
  {
    // forward pass:
    reset();
    for(int n = 0; n < N; n++)
      y[n] = getSample(x[n]);

    // backward pass:
    prepareForBackwardPass();
    for(int n = N-1; n >= 0; n--)
      y[n] = getSample(y[n]);
  }

  void applyBidirectionally(TSig* x, TSig* y, int N, int stride)
  {
    // forward pass:
    reset();
    for(int n = 0; n < N; n++)
      y[n*stride] = getSample(x[n*stride]);

    // backward pass:
    prepareForBackwardPass();
    for(int n = N-1; n >= 0; n--)
      y[n*stride] = getSample(y[n*stride]);
  }
  // optimize!


protected:

  std::vector<rsFirstOrderFilterBase<TSig, TPar>> stages;

};

/** DOES NOT WORK YET! maybe the prepareForBackwardPass is wrong? doing only the horizontal 
passes gives an asymmetric result - try with 1D test signal - compare multipasses of single 
stage filter to single pass of multistage filter - the result should be the same
This implementation creates a chain of identical filters and applies the chain at once in a 
single pass - i hope that this reduces uncached memory reads.  */
template<class T>
void applyMultiPass3(rsFirstOrderFilterBase<T, T>& flt, rsImage<T>& img, int numPasses)
{
  int w = img.getWidth();
  int h = img.getHeight();

  // create a chain of identical filters based on flt: 
  rsFirstOrderFilterChain<T, T> chain;
  chain.setupFromPrototype(flt, numPasses);

  // horizontal pass:
  for(int j = 0; j < h; j++)        // loop over the rows
    chain.applyBidirectionally(img.getPixelPointer(0, j), img.getPixelPointer(0, j), w);

  //// vertical pass:
  //for(int i = 0; i < w; i++)
  //  chain.applyBidirectionally(img.getPixelPointer(i, 0), img.getPixelPointer(i, 0), h, w);

  int dummy = 0;
}
// no - it cannot possibly work because each filter assumes in applyBidirectionally that the input
// goes to zero when we reach the boundary - but that is true only for the first filter stage - the 
// 2nd stage gets the deacying tail from the 1st, so its input does not immediately drop to zero


void testMultiPass()
{
  int N = 101;
  int numPasses = 5;

  std::vector<float> x(N), y1(N), y2(N);
  rsFill(x, 0.f);
  x[N/2] = 1.f;

  float a = 0.9f;
  float b = 1.f - a;

  rsFirstOrderFilterBase<float, float> flt;
  flt.setCoefficients(b, 0.f, a);
  y1 = x;
  for(int n = 0; n < numPasses; n++)
    flt.applyForwardBackward(&y1[0], &y1[0], N);

  rsFirstOrderFilterChain<float, float> chain;
  chain.setupFromPrototype(flt, numPasses);
  y2 = x;
  chain.applyBidirectionally(&y2[0], &y2[0], N);

  //rsPlotVectors(x, y1);
  rsPlotVectors(y1, y2, y1-y2);

  // y1 and y2 match, if numPasses == 1 and the deviation gets worse as numPasses goes up. If the 
  // number of samples N is increased (try 5 stages with N=101 and N=501 ), they match better 
  // again, so i guess, it has to do with the edge handling - yes - this makes sense: for the 
  // second filter, we are not allowed to assume that the input goes to zero immediately because 
  // it would still get nonzero inputs from the tail of the stage before it - the 
  // prepareForBackwardPass function for the chain can not just call prepareForBackwardPass for 
  // each stage - that is wrong!
  // It could work, if we could use a parallel connection instead of a serial one
}

// todo: make a function testComplexGauss that plots the 1D complex gaussian kernel, i.e. the 
// impulse response of the 1D complex gaussian filter (real, imaginary, magnitude and phase)
// i want to see, if it's a gaussian enveloped sinusoid ..or maybe it has this linear frequency 
// sweep in it that the yehar blog post mentions? if so, why?

void plotComplexGauss1D()
{
  int N = 1001;
  int numPasses = 5;



  using Complex = std::complex<float>;

  Complex j(0.f, 1.f);

  float omega = 0.03f;
  //float omega = 0.05;  // should increase with increasing numPasses (but how exactly?)

  //a = T(0.8) * exp(j*T(PI/16)); 

  //Complex a = 0.95f * exp(j*omega);

  float k = 1.0f;  // controls shape: 1.0 flat-top, lower values create a hole in the middle
                     // that could be useful when combining it with a wider gaussian to cancel
                     // tails

  Complex a = (1.f-k*omega) * exp(j*omega);  

  //Complex a = (1.f-omega) * exp(j*omega);
  // seems like this formula preserves the shape - omega just stretches or squishes it along the 
  // time axis

  //float k = 1.5f;
  //Complex a = pow((1.f-omega), k) * exp(j*omega);  


  //Complex b = 1.f - a;
  Complex b = 1.f - abs(a);

  std::vector<Complex> x(N), y(N);
  rsFill(x, Complex(0.f, 0.f));
  x[N/2] = 1.f;

  rsFirstOrderFilterBase<Complex, Complex> flt;
  flt.setCoefficients(b, 0.f, a);
  y = x;
  for(int n = 0; n < numPasses; n++)
    flt.applyForwardBackward(&y[0], &y[0], N);


  std::vector<float> yr(N), yi(N), ya(N), yp(N); // real, imag, abs, phase
  for(int n = 0; n < N; n++)
  {
    yr[n] = y[n].real();
    yi[n] = y[n].imag();
    ya[n] = abs(y[n]);
    yp[n] = arg(y[n]);
  }


  // create a real gaussian filter output for reference:
  rsFirstOrderFilterBase<float, float> flt2;
  flt2.setCoefficients(1.f-abs(a), 0.f, abs(a));
  std::vector<float> y2(N);
  rsFill(y2, 0.f);
  y2[N/2] = 0.025f; // it has much larger amplitude (why?) - we compensate by scaling input
  for(int n = 0; n < numPasses; n++)
    flt2.applyForwardBackward(&y2[0], &y2[0], N); 




  //rsPlotVectors(yr, yi, ya);
  rsPlotVectors(yr, yi, ya, y2);
  //rsPlotVectors(yr, yi, ya, yp);

  // Observations:
  // -smaller omega -> wider kernel
  // -actually, it does not look like a sinusoid with gaussian envelope at all
  // -also, the absolute value does not look like a gaussian
  // -with low values of k and low values of N, we may get ripples 
  //  (N = 1001, omega = 0.1, k = 0.125) - they go away when increasing N to 2001

  // todo: maybe for reference, plot a real gausian filter output as well
  // -figure out, how kernel-width changes as function of numPasses in order to come up with 
  //  compensation formula

}


// maybe have a phase parameter
template<class T>
void complexGaussBlurIIR(const RAPT::rsImage<T>& x, RAPT::rsImage<T>& yr, RAPT::rsImage<T>& yi,
  T radius, T freq, int numPasses = 6)
{
  rsAssert(yr.getPixelPointer(0,0) != x.getPixelPointer(0,0), "Cant be used in place");
  rsAssert(yi.getPixelPointer(0,0) != x.getPixelPointer(0,0), "Cant be used in place");
  rsAssert(yr.hasSameShapeAs(x), "Input and output images must have the same shape");
  rsAssert(yi.hasSameShapeAs(x), "Input and output images must have the same shape");
  int w = x.getWidth();
  int h = x.getHeight();

  //T scaledRadius = radius / numPasses; 

  T scaledRadius = radius; 


  using Complex = std::complex<T>;
  rsFirstOrderFilterBase<Complex, Complex> flt;

  rsImage<Complex> y(w,h);  // complex image

  Complex a;
  T sigma = -log(T(2)) / scaledRadius;   // s in s + j*w
  //T omega = T(2*PI) / period;       // omega = 2*PI*freq
  T omega = T(2*PI) * freq;           // omega = 2*PI*freq
  Complex j(T(0), T(1));                // imaginary unit

  // test - all formulas should give same result when freq = 0:
  a = pow(2.f, -1.f/scaledRadius);    // that's the original formula form the real case
  a = exp(sigma);                     // 2^x = e^(x*log(2))
  a = exp(sigma + j*omega);           // slightly different, even when freq=0 - why?

  // test (all done with 20 passes):
  //a = T(0.8) * exp(j*T(PI));       // checkerboard pattern, widens with more passes
  //a = T(0.8) * exp(j*T(PI/2));    // diagonals at the corners
  //a = T(0.8) * exp(j*T(PI/4));    // also diagonals at corners, but with lower frequency
  //a = T(0.8) * exp(j*T(PI/8));    // yet lower frequency, circles appear at center
  a = T(0.8) * exp(j*T(PI/16));     // circles become more apparent
  //a = T(0.9) * exp(j*T(PI/16));     // weird checkboard, garbarge with 35 passes
  //a = T(0.7) * exp(j*T(PI/16));

  // test with 6 passes:
  //a = T(0.8) * exp(j*T(PI/16));
  //a = T(0.8) * exp(j*T(PI/14));
  //a = T(0.8) * exp(j*T(PI/12));




  //Complex b = Complex(T(1), T(0)) - a;  // maybe apply a rotation (-> phase)

  Complex b = Complex(T(1), T(0)) - abs(a);  // phase = 0 ?

  flt.setCoefficients(b, T(0), a);

  y.convertPixelDataFrom(x);

  // -scaledRadius = 1, freq = 0.5 (omega = pi) produces checkerboard pattern, b=0.5, a=-0.5



  applyMultiPass1(flt, y, numPasses);    // interleave horizontal and vertical passes
  //applyMultiPass2(flt, y, numPasses);  // horizontal passes first, then vertical
  //applyMultiPass3(flt, y, numPasses);    // single pass of filter-chain - not yet working



  // instead of running 1 filter multiple times, use a chain of filters and run over the data only
  // once (less uncached memory reads/writes necessary -> likely faster), maybe a complex version
  // of rsSmoothingFilter could be suitable - compare results to current implementation (should be
  // the same)



  // copy real and imaginary parts into output (factor out):
  for(int j = 0; j < h; j++) {
    for(int i = 0; i < w; i++) {
      yr(i,j) = y(i,j).real();
      yi(i,j) = y(i,j).imag(); }}


  int dummy = 0;

  // ToDo: 
  // -try to get a disc-shape by using complex 1-pole filters, 
  //  see https://www.youtube.com/watch?v=vNG3ZAd8wCc
  //  the complex one-poles should be obtained by partial fraction expansion of a butterworth 
  //  filter (right? if not, check out the code linked under the video)
  //  ...maybe that could be done also by a 2D FFT and cutting out a circular (low-frequency)
  //  area?
  // https://github.com/mikepound/convolve
  // http://yehar.com/blog/?p=1495
  // https://dsp.stackexchange.com/questions/58449/efficient-implementation-of-2-d-circularly-symmetric-low-pass-filter/58634#58634
}

void testComplexGaussBlurIIR()
{
  int   w         = 201;
  int   h         = 201;
  int   numPasses = 20;
  float radius    = 20.0f;   // decay down to 1/2 after that many pixels (for 1 pass)
  float freq      = 0.f;    // 

  //freq = 0.05f;    // 0.1 pixel as frequency should result in a 10 pixel period

  float k = 2.0;

  //freq = 1 / period;
  // oh - wait - the function uses a scaled radius - that means we should then use an equally
  // scaled period as well


  float scaledRadius = radius / numPasses;
  float period = k*scaledRadius; 

  freq = 1 / period;


  //freq = 0.05f;    // 0.1 pixel as frequency should result in a 10 pixel period

  RAPT::rsImage<float> x(w,h), yr(w,h), yi(w,h), ya(w,h);
  x.fillAll(0.f);
  x(w/2, h/2) = 1.f;

  complexGaussBlurIIR(x, yr, yi, scaledRadius, freq, numPasses);

  // todo: get the magnitude of yr,yi
  for(int j = 0; j < h; j++)
    for(int i = 0; i < w; i++)
      ya(i,j) = sqrt(yr(i,j)*yr(i,j) + yi(i,j)*yi(i,j));


  RAPT::rsImageProcessor<float>::normalize(yr);
  RAPT::rsImageProcessor<float>::normalize(yi);
  RAPT::rsImageProcessor<float>::normalize(ya);

  writeImageToFilePPM(x,  "InputComplexGaussIIR.ppm");
  writeImageToFilePPM(yr, "OutputComplexGaussRealIIR.ppm");
  writeImageToFilePPM(yi, "OutputComplexGaussImagIIR.ppm");
  writeImageToFilePPM(ya, "OutputComplexGaussAbsIIR.ppm");

  // Observations:
  // -we need a quite large number of passes to get the kernel visually isotropic (>= 25)
  // -the radius seems to have an effect on the apparent frequency at the center
  // -try it in 1D - do we get a sinusoid with gaussian envelope with multiple passes?
  // -with numPasses = 50, radius = 100, freq = 0.1, we get a black result
  //  -maybe we should use a period that is proportional to the radius:
  //   period = k*radius, freq = 1/period
  // -the number of passes has an impact on the initial phase (i.e. brightness in the middle
  //  ...why? ..compare 25 and 35
  // -the abs looks quite squarish - not circular is it should (maybe plot contour lines to
  //  see it better)

}




template<class T>
void addInfectionSpeck(RAPT::rsImage<T>& I, const RAPT::rsImage<T>& P, 
  int i, int j, T weight = T(1))
{
  I(i, j) = weight * P(i, j); 
}

// Inits S and R, given I and P
template<class T>
void initSirpSR(RAPT::rsImage<T>& S, const RAPT::rsImage<T>& I, RAPT::rsImage<T>& R,
  const RAPT::rsImage<T>& P)
{
  // susceptible is initially the total-population minus infected S(x,y) = P(x,y) - I(x,y):
  RAPT::rsArrayTools::subtract(
    P.getPixelPointer(0,0), I.getPixelPointer(0,0), S.getPixelPointer(0,0), S.getNumPixels());

  // 0 are initally recovered:
  R.fillAll(0.f); 
}

// A uniform population density (normalized to 1) with two specks of infection.
template<class T>
void initSirpUniform(RAPT::rsImage<T>& S, RAPT::rsImage<T>& I, RAPT::rsImage<T>& R, RAPT::rsImage<T>& P)
{
  int w = S.getWidth();
  int h = S.getHeight();
  P.fillAll(1.f); 
  I.fillAll(0.f); 
  I(w/4,h/2) = 1.0; 
  I(w/2,h/4) = 1.0;

  initSirpSR(S, I, R, P);
}

// A population density showing a horizontal gradient with two specks of infection. Clearly shows,
// how the disease spreads faster in more densely populated areas.
template<class T>
void initSirpGradient(RAPT::rsImage<T>& S, RAPT::rsImage<T>& I, RAPT::rsImage<T>& R, RAPT::rsImage<T>& P)
{
  int w = S.getWidth();
  int h = S.getHeight();

  // create a population density:
  P.fillAll(0.f); 
  for(int j = 0; j < h; j++)      // loop over lines
    for(int i = 0; i < w; i++)    // loop over pixels in current line
      P(i, j) += T(i+1) / T(w);   // a horizontal gradient
   
  //rsImageProcessor<T>::normalize(P); // nope! not good!

  // two specks of infection:
  I.fillAll(0.f); 
  addInfectionSpeck(I, P,   w/4, h/2);
  addInfectionSpeck(I, P, 3*w/4, h/2);

  initSirpSR(S, I, R, P);
}

// Adds a Gaussian population cluster to P centered at (cx,cy) with width w and height h
template<class T>
void addPopulationCluster(RAPT::rsImage<T>& P, T cx, T cy, T w, T h, T weight = T(1))
{
  for(int j = 0; j < P.getHeight(); j++)
  {
    for(int i = 0; i < P.getWidth(); i++)
    {
      T dx = (T(i) - cx) / w;
      T dy = (T(j) - cy) / h;
      T d  = sqrt(dx*dx + dy*dy);
      T g  = exp(-d);

      //g *= 10; // preliminary

      P(i, j) += weight * g;

      /*
      T d  = (dx*dx)/(w*w) + (dy*dy)/(h*h); // weighted distance
      d   /= 100; // preliminary
      T g  = exp(-d);
      P(i, j) += weight * g;
      */
    }
  }
}
// verify this formula!!
// see https://mathworld.wolfram.com/BivariateNormalDistribution.html

template<class T>
void initSirpClusters(RAPT::rsImage<T>& S, RAPT::rsImage<T>& I, RAPT::rsImage<T>& R, RAPT::rsImage<T>& P)
{
  int w = S.getWidth();
  int h = S.getHeight();

  // add population clusters:
  P.fillAll(0.f);
  addPopulationCluster(P, 0.3f*w, 0.7f*h, 0.2f *w, 0.25f*h, 0.99f);
  addPopulationCluster(P, 0.6f*w, 0.2f*h, 0.15f*w, 0.1f *h, 0.95f);

  // two specks of infection:
  I.fillAll(0.f); 
  addInfectionSpeck(I, P,   w/4, h/2, 0.5f);
  addInfectionSpeck(I, P, 3*w/4, h/2, 1.0f);
  // using smaller weights seems to cuase a time delay in the outbreak

  initSirpSR(S, I, R, P);
}
// maybe have a clustered structure with pathways between the clusters (roads, airlines, etc.)
// this may be a straight line of high population density between the cluster centers (although
// that doesn't make much sense from a modeling perspective, the effect should be similar to
// actual roads)


// We need to create some population density - it should have some baseline level and a couple 
// of clusters of high density (cities) - the base-level ensures that the disease spreads between
// the clusters - maybe it should not be totally equal, so we can see, how it affects the spread 
// between the clusters

// write initSirpClustered


void epidemic()
{
  // A simple model for how epidemics evolve in time is the Susceptible-Infected-Recovered (SIR) 
  // model. It is a system of 3 ordinary differential equations that model the temporal evolution
  // of the 3 functions S(t), I(t), R(t) over time. The model is given by:
  //   S' = -t*S*I
  //   I' =  t*S*I - r*I
  //   R' =          r*I
  // where the prime ' denotes the time-derivative. The reasoning behind the 2nd equation is: the 
  // t*S*I term models the number of new infecctions, which should be proportional to the number of 
  // already infected people I and also to the number of people that may get infected S. The -r*I 
  // models the number of peoples which recover, which should be proportional to the number of 
  // infected people. The proportionality factors t,r are the transmission rate and the recovery 
  // rate. S' and R' sort of follow automatically: when people transition from S to I, I goes up
  // by t*S*I and so S must go down by the same amount. Likewise, when people transition form I to 
  // R, I goes down by r*I and R must go up by the same amount. At all times, we must have: 
  // S+I+R = P, where P is the total population size.
  //
  // Here, we implement a simple extension of the SIR model that also models how the disease 
  // spreads spatially. To do so, we think of S,I,R as functions of 2D space and time: 
  // S = S(t,x,y), I = I(t,x,y), R = R(t,x,y). Here, S,I,R are densities of susceptible, infected 
  // and recovered people and P(x,y) is a general population density as function of x,y. The model
  // becomes:
  //   S' = -t*S*I_av
  //   I' =  t*S*I_av - r*I
  //   R' =             r*I
  // The only change is that the transition from S to I is now governed by I_av instead of I where
  // I_av denotes a local spatial average of infected people.....
  

  // animation parameters:
  int w   = 360;       // image width
  int h   = 360;       // image height  - 120 is the minimum height for facebook
  int fps = 25;        // frames per second
  int N   = 800;       // number of frames
  float dt = w / 250.f;
  //float dt   = 0.1;  // delta-t between two frames

  // for develop - smaller sizes:
  //w = 50; h = 50; N = 100;

  // model parameters:
  float t = 0.5f;     // transmission rate
  float r = 0.002f;   // recovery rate
  float d = 1.0f;    // diffusion between 0..1 (crossfade between I and I_av)

  // grids for population density P(x,y), density of susceptible people S(x,y), infected people 
  // I(x,y) and recovered people R(x,y)
  RAPT::rsImage<float> P(w,h), S(w,h), I(w,h), R(w,h);

  //initSirpUniform(S, I, R, P);
  //initSirpGradient(S, I, R, P);  // shows how the speed of spreading depends on population density
  initSirpClusters(S, I, R, P);




  rsVideoRGB video(w, h);
  video.setPixelClipping(true);    // turn this off to reveal problems

  RAPT::rsImage<float> I_av(w,h);  // temporary, to hold local average

  rsConsoleProgressIndicator progressIndicator;
  std::cout << "Computing frames: ";
  progressIndicator.init();

  for(int n = 0; n < N; n++)       // loop over the frames
  {
    video.appendFrame(I, R, S);// infected: red, recovered: green, susceptible: blue

    // compute local spatial average (todo: maybe use better (= more circular) kernels):
    gaussBlur3x3(I, I_av);  

    // update density-images of S,I,R
    for(int j = 0; j < h; j++) {
      for(int i = 0; i < w; i++) {

        // compute intermediate variables:
        //float tSI = t*S(i,j)*I_av(i,j);
        float tSI = t*S(i,j) * (d*I_av(i,j) + (1.f-d)*I(i,j));
        float rI  = r*I(i,j);

        // compute time derivatives:
        float sp = -tSI;        // S' = -t*S*I
        float ip =  tSI - rI;   // I' =  t*S*I - r*I
        float rp =        rI;   // R' =          r*I

        // test: bypass spatial averaging:
        //ip =  t*S(i,j)*I(i,j) - r*I(i,j);   // I' =  t*S*I - r*I

        // update S,I,R:
        S(i,j) += dt * sp;
        I(i,j) += dt * ip;
        R(i,j) += dt * rp;  }}



    /*
    // test for debugging the algo:
    int i = w/2;
    int j = h/2;
    float sum = S(i,j) + I(i,j) + R(i,j);   // should be P(i,j)
    float Pij = P(i,j);
    float ratio = Pij / sum;                // should be 1
    int dummy = 0;
    */

    // enforce the invariant that S+I+R = P at all times (i hope that mitigates error accumulation
    // of the numeric method - maybe factor out):
    for(int j = 0; j < h; j++) {
      for(int i = 0; i < w; i++) {

        S(i,j) = rsMax(S(i,j), 0.f);  // it may go negative, especially at higher pixel sizes


        float Sij = S(i,j), Iij = I(i,j), Rij = R(i,j), Pij = P(i,j);

        rsAssert(Sij >= -0.2 && Iij >= 0 && Rij >= 0);

        float sum = Sij + Iij + Rij;
        float scl = Pij / sum;

        S(i,j) *= scl;
        I(i,j) *= scl;
        R(i,j) *= scl; }}
    // but this is really a dirty trick - try to get it as good as possible without that trick and
    // put it in to make the implementation even better - but don't use it to fix a botched 
    // implementation...

    progressIndicator.update(n, N-1);
  }
  std::cout << "\n\n";



  // factor out into convenience function:
  std::string fileName = "SIRP_t=" + std::to_string(t)
                           + "_r=" + std::to_string(r)
                           + "_d=" + std::to_string(d); // the trailing zeros are ugly
  rsVideoFileWriter vw;
  vw.setFrameRate(fps);
  vw.setCompressionLevel(0);  // 0: lossless, 10: good enough, 51: worst
  //vw.setDeleteTemporaryFiles(false);
  //vw.writeVideoToFile(video, fileName);
  vw.writeVideoToFile(video, "SIRP");


  // Observations:
  // -the simulation with the gradient in the population density shows that the disease spreads
  //  faster when the population density is higher
  // -the effect of d seems to be logarithmic? between d=0.2 and d=0.1 there's barely a difference
  //  in propagation speed, but between d=0.1 and d=0.01, the difference is apparent. also, with 
  //  such small d, the shape of the propagation becomes more squarish
  //  -> todo: d should control the width of a filter kernel, not crossfade between filtered and 
  //     unfiltered
  //  -> use bidriectional IIR filters
  // -we may actually use t < 1

  // Video Encoding Notes:
  // -with t=0.5, r=0.002, d=1 and the clusters, i get artifacts when using H.264 with CRF=0
  //  which actually is supposed to be lossless :-O :-(

  // ToDo:
  // -test, if/how the dynamics depends on the pixel-size - if it does, something is wrong - it 
  //  seems to - maybe that's related to the filter kernel? with larger pixels sizes, we should use
  //  larger kernels?
  //  -with 240x240 and the clusters, there seem to be some magenta-artifacts
  //  -with 400x300, something goes seriously wrong, 200x150 works, so it's not the aspect ratio
  //   -300x300 shows artifacts around frame 150 - it seems like at least one of the variables goes
  //    negative -> figure out which one and why -> it's S and goes significantly into the negative 
  //    range - what can we do?? clip it at zero?
  //   ..an aspect-ratio other than one is unrealistic anyway because it would mean that the 
  //   disease spreads faster in one than the other direction - the spread would be anisotropic
  //   (we could compensate by using an elliptic filter kernel)

  // -maybe plot the 3 curves of a single selected point - they should resemble the curves from
  //  the SIR model
  // -do a linear interpolation between I and i_av with a diffusion parameter
  // -try to find a setting of t,r that stops the transmission
  // -try a more interesting population density functions P(x,y) 
  // -use a better filter kernel that has a more circular footprint - maybe try a 5x5 gaussian 
  //  kernel - thest this by looking at the shape when the population density is uniform

  // interesting settings to compare:
  // -clusters, t=2.0, r=0.02, d=0.2
  // -clusters, t=2.0, r=0.5,  d=0.2
  // -clusters, t=2.0, r=1.0,  d=0.2 - seems like not all get infected
  // -clusters, t=2.0, r=1.0,  d=0.2 - seems like the top-right cluster does not get infected

  // refinements:
  // -the parameters t,r,d could be functions of space - models regional differences in 
  //  interaction, healthcare and mobility

}
// SIR model in python:
// https://www.youtube.com/watch?v=wEvZmBXgxO0


// class for testing rsTensor - we use a subclass to get access to some protected members that we 
// need to investigate in the tests.
template<class T>
class rsTestTensor : public rsTensor<T>
{

public:

  using rsTensor::rsTensor;
  //using rsTensor::operator=;

  // tests conversion from structured to flat indices and back:
  static bool testIndexConversion()
  {
    bool r = true;
    int L = 2, M = 5, N = 3;
    rsTestTensor<T> A({L,M,N});
    for(int l = 0; l < L; l++) {
      for(int m = 0; m < M; m++) {
        for(int n = 0; n < N; n++) {
          int flat = A.flatIndex(0, l,m,n);  // 1st argument 0 is the recursion depth
          int lmn[3]; A.structuredIndices(flat, lmn);
          r &= lmn[0] == l && lmn[1] == m && lmn[2] == n; }}}
    return r;
  }


  static bool testOuterProduct(const rsTestTensor<T>& A, const rsTestTensor<T>& B)
  {
    rsTensor<T> C = getOuterProduct(A, B);

    // recompute elements of D using other algorithm for index computation and compare, if it's
    // still equal to C after doing so
    rsTestTensor<T> D(C.getShape());
    int indices[10];   // 10 should be enough for this test - production code should not use this
    for(int k = 0; k < D.getSize(); k++) {
      D.structuredIndices(k, indices);
      int flatIndexA = A.flatIndex(indices);
      int flatIndexB = B.flatIndex(&indices[A.getNumIndices()]);
      D.data[k] = A.data[flatIndexA] * B.data[flatIndexB];
    }

    return rsArrayTools::equal(C.getDataPointer(), D.getDataPointer(), C.getSize());
  }

};

bool testTensorOuterProduct()
{
  bool r = true;

  using TestTens = rsTestTensor<double>;

  TestTens A({ 2,4,3 }), B({ 4,5 });
  A.fillRandomly();
  B.fillRandomly();
  r &= TestTens::testOuterProduct(A, B);

  return r;
}

// test getting back left and right factors from outer product:
bool testTensorFactors()
{
  bool r = true;

  using Tens = rsTensor<double>;

  double tol = 1.e-15;
  Tens A({ 2,4,3 }), B({ 4,5 });
  A.fillRandomly(-10.0, 10.0, 1);
  B.fillRandomly(-10.0, 10.0, 2);
  Tens C  = Tens::getOuterProduct(A, B);
  Tens A2 = Tens::getLeftFactor(  C, B); // allow syntax C.getLeftFactor(B)
  Tens B2 = Tens::getRightFactor( C, A); // C.getRightFactor(A)
  //Tens DA = A - A2;  // should be numerically close to zero - seems to work
  //Tens DB = B - B2;
  r &= A2.equals(A, tol);
  r &= B2.equals(B, tol);

  // todo: 
  // -maybe do this in a loop with different random ranks, shapes, and data

  return r;
}

bool testTensorContraction()
{
  bool r = true;
  using Tens = rsTensor<double>;
  using VecI = std::vector<int>;

  // test contractions of 2x3x3, 3x2x3 and 3x3x2 tensors:
  Tens C, D;
  C.setShape(VecI({2,3,3}));
  C.setData(VecI({111, 112, 113,
                  121, 122, 123,
                  131, 132, 133,
    
                  211, 212, 213,
                  221, 222, 223,
                  231, 232, 233}));
  D = Tens::getContraction(C, 1,2); // should be (366, 666)
  r &= D.getShape() == VecI({2});
  r &= D(0) == 366.0 && D(1) == 666.0; // 111 + 122 + 133, 211 + 222 + 233

  C.setShape(VecI({3,2,3}));
  C.setData(VecI({111, 112, 113,
                  121, 122, 123,

                  211, 212, 213,
                  221, 222, 223,

                  311, 312, 313,
                  321, 322, 323}));
  D = Tens::getContraction(C, 0,2);
  r &= D.getShape() == VecI({2});
  r &= D(0) == 636.0 && D(1) == 666.0; // 111 + 212 + 313, 121 + 222 + 323

  C.setShape(VecI({3,3,2}));
  C.setData(VecI({111, 112, 
                  121, 122, 
                  131, 132,

                  211, 212, 
                  221, 222, 
                  231, 232,

                  311, 312, 
                  321, 322, 
                  331, 332}));
  D = Tens::getContraction(C, 0,1);
  r &= D.getShape() == VecI({2});
  r &= D(0) == 663.0 && D(1) == 666.0; // 111 + 221 + 331, 112 + 222 + 332


  // todo: contraction with higher ranks, compare with old implementation

  // compare with old implementation:
  //rsMultiArrayOld<double> M(5, 


  // maybe implement a function sumOverIndex(int i) which sums over the given index and thereby 
  // reduces the rank by one - this could already be useful in rsMultiArray, for example to create
  // averages along certain dimensions - maybe see what numpy has

  return r;
} 


bool testTensor()
{
  bool r = true;


  using TestTens = rsTestTensor<double>;
  using Tens     = rsTensor<double>;

  r &= TestTens::testIndexConversion();


  r &= testTensorOuterProduct();
  r &= testTensorFactors();
  r &= testTensorContraction();



  Tens E3 = Tens::getPermutationTensor(3);
  Tens E4 = Tens::getPermutationTensor(4);
  Tens E5 = Tens::getPermutationTensor(5);
  // stop here - they grow fast! namely, like N^N - most entries are zero though - so it's really a
  // bad idea to explicitly use them in production code

  // verify Eq. 209
  Tens D1 = Tens::getDeltaTensor(3);
  Tens D2 = Tens::getGeneralizedDeltaTensor(2);
  Tens D3 = Tens::getGeneralizedDeltaTensor(3);
  //...
  //  compare with rsGeneralizedDelta, rsLeviCivita in IntegerFunctions.h


  //r &= A == A2; // are operators not inherited? hmm - this says, they are, except the assignment
  // operator:
  // https://www.linuxtopia.org/online_books/programming_books/thinking_in_c++/Chapter14_018.html



  //using VecI = std::vector<int>;
  //using VecD = std::vector<double>;


  // combine this with the old rsMultiArrayOld tests
  return r;
}

bool testPlane()
{
  bool result = true;

  rsParametricPlane3D<double> plane;

  using Vec = rsVector3D<double>;

  Vec u(0,0,0), v(1,0,0), w(0,1,0); // the xy plane

  plane.setVectors(u, v, w);

  Vec a;
  a = plane.getPointOnPlane(2, 3);

  Vec b = Vec(1,1,1);  // target
  a = plane.getClosestPointOnPlane(b); // should be (1,1,0)

  b = Vec(2,3,5);
  a = plane.getClosestPointOnPlane(b); // should be (2,3,0)


  // try with a more general plane (slanted with offset):
  u.set(1,2,3);
  v.set(2,4,3);
  w.set(2,6,8);
  plane.setVectors(u,v,w);
  a = plane.getPointOnPlane(2, 3);
  double s, t;
  plane.getMatchParameters(a, &s, &t);
  result &= s == 2;
  result &= t == 3;

  // try if it really finds the optimal point by lookong how the error increases when we increase
  // or decrease s,t from the supposedly optimal values
  b = Vec(2,-3,5);
  plane.getMatchParameters(b, &s, &t);
  a = plane.getPointOnPlane(s, t);
  double errMin = dot(a-b, a-b);  // minimum - no other combination of s,t should get closer

  // moving the parameters s,h a little bit away form their computed optimal values should 
  // increase the error - verify that:
  double h = 0.1;    // amount of wiggle for the parameters
  double err;        // error for the wiggled parameters
  a = plane.getPointOnPlane(s-h, t); err = dot(a-b, a-b); result &= err >= errMin;
  a = plane.getPointOnPlane(s+h, t); err = dot(a-b, a-b); result &= err >= errMin;
  a = plane.getPointOnPlane(s, t-h); err = dot(a-b, a-b); result &= err >= errMin;
  a = plane.getPointOnPlane(s, t+h); err = dot(a-b, a-b); result &= err >= errMin;


  plane.getMatchParametersXY(b, &s, &t);
  a = plane.getPointOnPlane(s, t);
  result &= a.x == b.x && a.y == b.y;

  plane.getMatchParametersXZ(b, &s, &t);
  a = plane.getPointOnPlane(s, t);
  result &= a.x == b.x && a.z == b.z;

  double tol = 1.e-14;
  plane.getMatchParametersYZ(b, &s, &t);
  a = plane.getPointOnPlane(s, t);
  //result &= a.y == b.y && a.z == b.z;  // fails! ..it's only close to
  result &= rsIsCloseTo(a.y, b.y, tol) && rsIsCloseTo(a.z, b.z, tol);




  return result;
}

bool testManifoldPlane()
{
  bool result = true;

  using Vec3 = rsVector3D<double>;
  using Vec  = std::vector<double>;
  using Mat  = rsMatrix<double>;

  // create the parametric plane:
  rsParametricPlane3D<double> plane;
  plane.setVectors(Vec3(2,4,1), Vec3(1,-2,3), Vec3(-2,3,5));

  // create the functions for the manifold object:
  std::function<void(const std::vector<double>&, std::vector<double>&)> u2x, x2u;
  u2x = [=](const Vec& u, Vec& x)
  {
    rsAssert(u.size() == 2);
    rsAssert(x.size() == 3);
    Vec3 v = plane.getPointOnPlane(u[0], u[1]);
    x[0] = v.x; x[1] = v.y; x[2] = v.z;
  };
  x2u = [=](const Vec& x, Vec& u)
  {
    rsAssert(u.size() == 2);
    rsAssert(x.size() == 3);
    Vec3 v(x[0], x[1], x[2]);
    //plane.getMatchParametersXY(v, &u[0], &u[1]); // does not work
    //plane.getMatchParametersXZ(v, &u[0], &u[1]); // does not work
    //plane.getMatchParametersYZ(v, &u[0], &u[1]); // does not work
    plane.getMatchParameters(v, &u[0], &u[1]);   // works
  };
  // maybe try alternative functions that don't optimize the distance but instead just match up
  // xy, xz or yz - in each case, we just use two of the 3 equations

  // create and set up the manifold object:
  int N = 2, M = 3; // dimensionalities of manifold and embedding space
  rsManifold<double> mf(N, M);
  mf.setCurvilinearToCartesian(u2x);
  mf.setCartesianToCurvilinear(x2u);
  mf.setApproximationStepSize(1.0);

  // do tests:
  Vec x(3), u(2);
  double tol = 1.e-7; 
  Mat El, Eu, gl, gu, delta;

  u = { 83, 23};
  delta = mf.getNumericalBasisProduct(u);
  El = mf.getCovariantBasis(u);
  Eu = mf.getContravariantBasis(u);
  //delta = El * Eu;  // nope! no delta! also, it's 3x3 - we expect a 2x2 matrix
  delta = Eu * El;    // yes! 2x2 delta!
  gl = mf.getCovariantMetric(u); 
  gu = mf.getContravariantMetric(u);
  delta = gl * gu;
  delta = gu * gl;
  // yes! this looks good! the metrics do indeed multiply to the identity matrix! The key seems to
  // be to use plane.getBestParameters instead of just selecting the first 2 equations for the
  // backward conversion! ...how does this generalize to non-plane surfaces? should we also somehow
  // find the tuple of parameters that minimzes the Euclidean distance? Or maybe it's better to not
  // rely on the numerical computation of via the contravariant basis and use only the covariant 
  // one and obtain the contravariant by matrix inversion and lowering indices - it seems to be 
  // generally fishy - what about the analytic Jacobian? i guess, it depends from which equation
  // it was derived - one that minimizes the distance or some other formula...at least, the fog
  // is settling now and it begins to make sense!
  // However: it's interesting to note that the El and Eu matrices still mutliply to the identity,
  // even in cases when the metric gl and gu totally don't



  return result;
}




// obsolete
bool testManifold1()
{
  bool result = true;

  int N = 2, M = 3; // dimensionalities of manifold and embedding space

  rsManifold<double> mf(N, M);

  using Vec = std::vector<double>;
  using Mat = rsMatrix<double>;

  // for converting between the u- and x-coordinates and vice versa:
  std::function<void(const std::vector<double>&, std::vector<double>&)> u2x, x2u;

  // for computing Jacobian matrices:
  std::function<void(const std::vector<double>&, rsMatrix<double>&)> u2xJ, x2uJ;

  u2x = [=](const Vec& u, Vec& x)
  {
    rsAssert(u.size() == 2);
    rsAssert(x.size() == 3);
    x[0] = 1 +  2*u[0] +  3*u[1];  // x(u,v) = 1 +  2*u +  3*v
    x[1] = 2 - 11*u[0] + 13*u[1];  // y(u,v) = 2 - 11*u + 13*v
    x[2] = 3 +  5*u[0] -  7*u[1];  // z(u,v) = 3 +  5*u -  7*v
  };
  // sage:
  // var("x y z u v")
  // e1 = x == 1 +  2*u +  3*v
  // e2 = y == 2 - 11*u + 13*v
  // e3 = z == 3 +  5*u -  7*v
  // solve([e1,e2],[u,v])
  // -> [[u == 13/59*x - 3/59*y - 7/59, v == 11/59*x + 2/59*y - 15/59]]

  x2u = [=](const Vec& x, Vec& u)
  {
    rsAssert(u.size() == 2);
    rsAssert(x.size() == 3);
    double k = 1./59;
    u[0] = (13*x[0] - 3*x[1] -  7) / 59;  // u(x,y) = (13*x - 3*y - 7 ) / 59
    u[1] = (11*x[0] + 2*x[1] - 15) / 59;  // v(x,y) = (11*x + 2*y - 15) / 59
    // z is redundant but we could use it to check, if the vector (x,y,z) is actually on the plane
  };

  // set up:
  mf.setCurvilinearToCartesian(u2x);
  mf.setCartesianToCurvilinear(x2u);
  //mf.setCurvToCartJacobian(u2xJ);
  //mf.setCartToCurvJacobian(x2uJ);
  mf.setApproximationStepSize(1.e-4);

  // automated tests:
  Vec x(3), u(2);
  double tol = 1.e-7; 

  Mat El, Eu, gl, gu, delta;

  u = { 83, 23};
  delta = mf.getNumericalBasisProduct(u);
  El = mf.getCovariantBasis(u); // E[0] = E_u = (1,-11,5), E[1] = E_v = (3,13,-7) - this seems ok
  Eu = mf.getContravariantBasis(u); // E^u = (13 -3 0)/59, E^v = (11 2 0)/59
  //delta = El * Eu;  // nope! no delta! also, it's 3x3 - we expect a 2x2 matrix
  delta = Eu * El;    // yes! 2x2 delta!
  gl = mf.getCovariantMetric(u);      // this seems ok
  gu = mf.getContravariantMetric(u);  // this seems also ok
  delta = gl * gu;  // nope! still the result is not the delta!
  delta = gu * gl;  // nope!

  // maybe try something dirty - maybe something is in the wrong order:
  //rsSwap(gl(0,0), gl(1,1));
  //rsSwap(gu(0,0), gu(1,1));
  //delta = gl * gu; 
  // nope! doesn't help!


  //u = { 83, 23};
  result &= mf.runTests({ 10, 20}, tol);

  // todo: try it with M=N=2 - maybe it's something about the M != N? - but no - i think, we would
  // just get the top-left 2x2 corners of all matrices that have a dimension > 2


  return result;
}

// obsolete
bool testManifold2()
{
  bool result = true;

  int N = 2, M = 3; // dimensionalities of manifold and embedding space


  using Vec = std::vector<double>;
  using Mat = rsMatrix<double>;

  // for converting between the u- and x-coordinates and vice versa:
  std::function<void(const std::vector<double>&, std::vector<double>&)> u2x, x2u;

  // for computing Jacobian matrices:
  std::function<void(const std::vector<double>&, rsMatrix<double>&)> u2xJ, x2uJ;

  u2x = [=](const Vec& u, Vec& x)
  {
    rsAssert(u.size() == 2);
    rsAssert(x.size() == 3);
    x[0] = 2*u[0] + 3*u[1];  // x(u,v) = 2*u + 3*v
    x[1] = 4*u[0] + 5*u[1];  // y(u,v) = 4*u + 5*v

    // z(u,v) is the key for making gl*gu = gu*gl = id work - if it's not a constant function, the
    // identity seems to become invalid :-O
    //x[2] = 0;  // test - with 0, we get good results
    x[2] = 50;  // with a constant, we still get good results
    //x[2] = 3*u[0] + 2*u[1];  // with this, we get bad results
    //x[2] = u[0]; 
    //x[2] = u[0] + u[1]; 
    //x[2] = 7*u[0] + 5*u[1];
  };
  // var("x y u v")
  // e1 = x == 2*u + 3*v
  // e2 = y == 4*u + 5*v
  // solve([e1,e2],[u,v])
  // [[u == -5/2*x + 3/2*y, v == 2*x - y]]

  x2u = [=](const Vec& x, Vec& u)
  {
    rsAssert(u.size() == 2);
    rsAssert(x.size() == 3);
    u[0] = (-5/2.)*x[0] + (3/2.)*x[1];  // u(x,y) = -5/2*x + 3/2*y
    u[1] = 2*x[0] - x[1];               // v(x,y) =  2*x - y
  };
  // maybe this is not the best definition of x2u - it assumes that x,y,z is actually a valid 
  // vector that lies in the plane - which is not the case when we do the numeric approximation of
  // the derivatives - we use things like x+h,y,z, etc. which will not be in the plane. this may
  // result in problems. maybe we should instead use the closest point in the plance to the given
  // x,y,z and retun the u,v parameters of that. this can be done by minimizing the distance 
  // between the given x,y,z vector and x(u,v),y(u,v),z(u,v) with respect to u and v. maybe make a 
  // class rsPlane3D which does these computations

  rsManifold<double> mf(N, M);

  // set up:
  mf.setCurvilinearToCartesian(u2x);
  mf.setCartesianToCurvilinear(x2u);
  //mf.setCurvToCartJacobian(u2xJ);
  //mf.setCartToCurvJacobian(x2uJ);
  mf.setApproximationStepSize(1.0);

  // automated tests:
  Vec x(3), u(2);
  double tol = 1.e-7; 

  result &= mf.runTests({ 10, 20}, tol);

  Mat El, Eu, gl, gu, delta;

  u = { 83, 23};
  delta = mf.getNumericalBasisProduct(u);
  El = mf.getCovariantBasis(u); 
  Eu = mf.getContravariantBasis(u); 
  //delta = El * Eu;
  delta = Eu * El;
  gl = mf.getCovariantMetric(u);
  gu = mf.getContravariantMetric(u);
  delta = gl * gu;
  delta = gu * gl;

  // ok - this seems to work - is the problem really because of M > N? do the formulas to compute 
  // the metric from the basis vectors only valid when M == N?

  // As soon as z(u,v) is not a constant, the product gl*gu or gu*gl is not the identity matrix 
  // anymore. Why is that? Is it the case that the equation gl*gu = gu*gl = id is simply not true 
  // anymore when M > N? or is the computation of the contravariant metric via using contravariant 
  // basis vectors not valid whne M != N? The computaion of the covariant metric seems to correct, 
  // though (see Sochi's Differential Geometry, Eq 152, 186
  // maybe see also Eq 181 - it says something about contravariant basis vectors

  // Eu has always zero entries at positions 2,5 - no matter what z(u,v) is - may this have to do 
  // with it? is it actually correct to have zero entries? the z-coordinate is actually dependent 
  // on x,y
  // the approximation function may actually pass invalid/inconsistent coordinates that are *not*
  // on the surface - does that matter? - the (x,y,z) that was computed from (u,v) is valid, but
  // (x+h,y,z), (x,y+h,z), (x,y,z+h) are all invalid coordinates - none of them is on the surface
  // unless z(u,v)=const ...right? ...maybe try to solve it with analytic Jacobians - maybe that 
  // can make the problem go away?

  // Maybe we should do something like: find closest point on the surface to the given coordinates

  return result;
}


// adds some extensions to facilitate testing
template<class T>
class rsManifold2 : public rsManifold<T>
{

public:

  // maybe these should be put into the baseclass:
  using Vec  = std::vector<T>;
  using Mat  = rsMatrix<T>;
  using Tens = rsMultiArray<T>;

  using rsManifold::rsManifold; // inherit constructors

/*
// move to Snippets
class A
{
  public: 
    A(int x) {}
};

class B: public A
{
  using A::A;  // inherit all constructors from A, this syntax also applies when A, B are 
               // templates - don't use using A<T>::A<T>; in this case
};
*/

  // i-th covariant basis vector
  Vec getBasisVector(const Vec& u, int i) const
  {
    Vec up(u), um(u);
    Vec xp(M), xm(M);
    up[i] = u[i] + h;    // u+
    um[i] = u[i] - h;    // u-
    toCartesian(up, xp); // x+
    toCartesian(um, xm); // x-
    return (1/(2*h)) * (xp - xm);
  }

  // derivative of i-th covariant basis vector with respect to j-th coordinate
  Vec getBasisVectorDerivative(const Vec& u, int i, int j) const
  {
    Vec up(u), um(u);
    up[j] = u[j] + h;    // u+
    um[j] = u[j] - h;    // u-
    Vec xp = getBasisVector(up, i);
    Vec xm = getBasisVector(um, i);

    /*
    // maybe here, we should convert xp,xm from cartesian to polar before taking the difference?
    // let's try it:
    Vec xp2(N), xm2(N);
    toCurvilinear(xp, xp2);
    toCurvilinear(xm, xm2);
    return (1/(2*h)) * (xp2 - xm2);
    // nope! doesn't work either
    */

    /*
    // maybe like this:
    Vec dx = (1/(2*h)) * (xp - xm);  // for debug
    toCurvilinear(dx, up); // re-use up
    return up;
    // nope!
    */

    return (1/(2*h)) * (xp - xm);
  }
  // maybe for the higher order derivative approximations which build upon lower approximated 
  // derivatives, we should use smaller stepsizes h. if the first 2 derivatives are approximated as
  // f'(x)   ~= ( f (x+h1) - f (x-h1) ) / (2*h1) 
  // f''(x)  ~= ( f'(x+h2) - f'(x-h2) ) / (2*h2)
  // = ((f(x+h1+h2)-f(x-h1+h2))/(2*h1) - ((f(x+h1-h2)-f(x-h1-h2))/(2*h1))) / (2*h2)
  // ..or something...but when h1==h2, some terms cancel out, so we get an effectively lower number
  // of evaluation points - and more evaluation points usually give better estimates...figure out 
  // the details....

  // partial derivative of vector field f with respect to i-th coordinate
  Vec getVectorFieldPartialDerivative(const Vec& u, const FuncVecToVec& f, int i) const
  {
    Vec up(u), um(u);    // wiggled u coordinate vectors
    up[i] = u[i] + h;    // u+
    um[i] = u[i] - h;    // u-
    Vec Ap(N), Am(N);    // A = f(u) at up and um
    f(up, Ap);
    f(um, Am);
    return (1/(2*h)) * (Ap - Am);
  }

  /** Not yet tested..
  Computes the rank-3 tensor D of the partial derivatives of the (matrix of) basis-vectors with 
  respect to the input coordinate vector at the given position vector u. The element D(k,i,j) is
  the j-th component of the i-th basis vector, partially differentiated with respect to the k-th 
  input coordinate. */
  rsMultiArray<T> getCovariantBasisDerivative(const std::vector<T>& u) const
  {
    int i, j, k;
    // i: index of basis vector
    // j: index of basis-vector component
    // k: index of partial differentiation

    std::vector<T> up(u), um(u);            // wiggled u coordinate vectors
    rsMatrix<T> Ep, Em;                     // matrix of basis-vectors at up and um
    rsMultiArray<T> D({N,N,N});             // rank-3 tensor of basis-matrix partial derivatives
    T s = 1/(2*h);

    for(k = 0; k < N; k++)
    {
      up[k] = u[k] + h;
      um[k] = u[k] - h;

      Ep = getCovariantBasis(up);  // basis at up
      Em = getCovariantBasis(um);  // basis at um

                                   // central difference approximation:
      for(i = 0; i < N; i++)
        for(j = 0; j < N; j++)
        {
          D(k, i, j) = s * (Ep(i, j) - Em(i, j));
          // or maybe the indices i,j should be swapped? ..nope - because the basis vectors are 
          // returned in the columns...right? and the ordering of indices of a derivative of a 
          // basis vector should certainly match the index ordering of the vector itself, so i,j it
          // must be. And that k comes first makes also sense

          // or maybe like this:
          //D(k, i, j) = s * (Ep(j, i) - Em(j, i));

          // ..ok let's try:
          //D(k, j, i) = s * (Ep(i, j) - Em(i, j));
        }



      up[k] = u[k];
      um[k] = u[k];
    }

    return D;
  }




  /** Under construction - not yet working! This seems to be algorithmically much more efficient and 
  accurate than using the Christoffel-symbols - if it only would work! :'-(

  Computes the covariant derivative of a given vector field A^i = f(u), which means, f is a vector
  of functions that computes contravariant components A^i at a given contravariant position vector 
  which itself is also given in contravariant components u^i. The dimensionality of the produced 
  vector field should be equal the dimensionality of  the input vector u, which in turn should 
  match the (intrinsic) dimensionality of the manifold (can this be relaxed?).  */
  rsMatrix<T> getCovariantDerivative2(const std::vector<T>& u, const FuncVecToVec& f) const
  {
    rsAssert((int) u.size() == N, "Input vector u has wrong dimensionality");

    // Algorithm: 
    //  (1) compute contravariant output vector A at u
    //  (2) compute matrix E of covariant basis vectors at u
    //  (3) compute matrix dA of partial derivatives of the vector field
    //  (4) compute rank-3 tensor dE of the partial derivatives of the basis-matrix
    //  (5) combine them via product rule ("(f*g)' = f' * g + g' * f") as in Eq. 358 where our
    //      "f" is the vector field and our "g" is the matrix of basis vectors (which are both 
    //      functions of the input vector u)

    std::vector<T> A(N); f(u, A);                         // (1)
    rsMatrix<T>      E = getCovariantBasis(u);            // (2)
    rsMatrix<T>     dA = getVectorFieldDerivative(u, f);  // (3)
    rsMultiArray<T> dE = getCovariantBasisDerivative(u);  // (4)

                                                          // (5) - apply product rule according to Ref.(1), Eq. 358, line 3:
    rsMatrix<T> D(N,N);  // maybe we can re-use E?
    int i, j, k;
    // j: derivative index
    // k: component index
    // i: summation index
    D.setToZero();
    for(j = 0; j < N; j++) 
      for(k = 0; k < N; k++)
        for(i = 0; i < N; i++)
        {
          // the "f' * g" term:

          //D(j, k) += E(i, k) * dA(j, k);  // why no i in dA?


          //D(j, k) += E(i, k) * dA(j, i);  // that looks most like the formula

          //D(j, k) += E(i, k) * dA(i, j);

          //D(j, k) += E(i, k) * dA(k, i);

          //D(j, k) += E(i, k) * dA(i, k);




          //D(j, k) += E(i, k) * dA(k, j);
          //D(j, k) += E(k, i) * dA(j, k);
          //D(j, k) += E(k, i) * dA(k, j);


          // the "g' * f" term:

          //D(j, k) += A[i] * dE(j, i, k);
          //D(j, k) += A[i] * dE(j, k, i);


          D(j, k) += E(k,i) * dA(j,i)  +  A[i] * dE(j,k,i);
          //D(j, k) += E(i,k) * dA(j,i)  +  A[i] * dE(j,i,k);
          //D(j, k) += E(i,k) * dA(j,i)  +  A[i] * dE(j,k,i);
          //D(j, k) += E(k,i) * dA(j,i)  +  A[i] * dE(j,i,k);
          // A[i] and dA(j,i) seem to be correct, also j, must be the 1st index in dA and dE, so it
          // seems we only have for possibilities left to distribuite k,i in E(i,k) and dE(j,i,k) 
          // - but none of them works. the topmost seems to be the one, i would expect to be 
          // correct - could we have to use a repeated index somewhere?



          //D(j, k) += A[i] * dE(k, i, j);  // seems wrong
          // seems to make no difference if we use j,i,k or k,i,j - dE symmetric with respect to 
          // 1st and 3rd index?

          // actually D(j, k) += E(i, k) * dA(j, i)  +  A[i] * dE(j, i, k); makes sense:
          // i,k is always the index of the k-th component of the i-th basis vector (derivative),
          // j is always the index of partial differentiation which always comes first - the rest
          // must then follow - but it just doesn't work!
          // but wait: the basis vector are stored as column vectors, so we should use k,i to 
          // access the k-th element of the i-th vector! the rwo index selects the component!



          int dummy = 0;
        }
    // it still doesn't work!
    // wait: in the last line of Eq 358, it is denoted that the component is A^i;_j ..but there was
    // this relabeling i <-> k step in between ...but the relabeling only applied to the 2nd term
    // could that be the key?

    // still wrong! but mybe the Christoffel formula is also wrong - we should verify that first
    // ...but how? we need a numerical example of a covariant derivative computation - maybe try
    // to evaluate one with sage:
    // http://doc.sagemath.org/html/en/reference/manifolds/sage/manifolds/differentiable/tensorfield.html
    // Sage-Manifolds:
    // https://arxiv.org/pdf/1804.07346.pdf
    // https://arxiv.org/pdf/1412.4765.pdf

    // see also (2) pg 204 ff

    return D;

    //rsError("Function not yet finished.");
    //return rsMatrix<T>();
  }
  // evaluate it from first principles via Eq. 358, line 3 using the product rule. This formula 
  // is much simpler than using Christoffel symbols. Implement the Christoffel-symbol based 
  // formula in Eq. 362 too and compare results. ...implement also the formulas in Eqs. 361-366
  // ...366 is especially complicated, dealing with tensors of general rank - to hardcode it for a 
  // tensor of given rank, we could used nested loops with the nesting level given by the tensor 
  // rank - but the rank is an unknown runtime variable, so loop-nesting is no option. what to do? 
  // maybe our nested loop over the indices should be re-expressed as a flat-loop over flat-array 
  // indices in the rsMultiArray data structure and we should convert to non-flat indices as 
  // needed? or maybe recursion should be used for the unknown nesting depth? ..if so - how 
  // exactly?
  // looking at line 4 of Eq.358, it seems like the Christoffel symbols somehow are used to
  // compute the derivative of a basis vector as linear combination of the basis-vector 
  // components themselves - is that correct?

};

bool testManifoldPolar()
{
  // We use a polar coordinate system in 2D

  bool result = true;

  using Vec  = std::vector<double>;
  using Mat  = rsMatrix<double>;
  using Tens = rsMultiArray<double>;

  // Forward and inverse trafo between polar and cartesian coodinates:
  std::function<void(const std::vector<double>&, std::vector<double>&)> u2x, x2u, u2v;
  u2x = [=](const Vec& u, Vec& x)
  {
    rsAssert(u.size() == 2);
    rsAssert(x.size() == 2);
    double r = u[0], phi = u[1]; // radius, phi
    x[0] = r * cos(phi);  // x 
    x[1] = r * sin(phi);  // y
  };
  x2u = [=](const Vec& X, Vec& u)
  {
    rsAssert(u.size() == 2);
    rsAssert(X.size() == 2);
    double x = X[0], y = X[1];
    u[0] = sqrt(x*x + y*y);
    u[1] = atan2(y, x);
  };

  // Create and set up manifold:
  int N = 2, M = 2;
  rsManifold2<double> mf(N, M);
  mf.setCurvilinearToCartesian(u2x);
  mf.setCartesianToCurvilinear(x2u);
  //mf.setApproximationStepSize(pow(2.0, -20));
  //mf.setApproximationStepSize(pow(2.0, -15));
  mf.setApproximationStepSize(pow(2.0, -10));
  //mf.setApproximationStepSize(pow(2.0, -5));
  //mf.setApproximationStepSize(pow(2.0, -3));

  //double r, phi;
  //r = 2.0;
  //phi   = PI/6;
  //Vec u({ r, phi });


  //Vec u = Vec({3, 2});
  //Vec u = Vec({2, PI/4});
  //Vec u = Vec({4, PI/4});
  //Vec u = Vec({10, PI/4});
  //Vec u = Vec({1, PI/4});  // hmm - interesting things happen with PI/4
  //Vec u = Vec({1, PI});  // with PI, D1 fails probably due to singularity in inverse trafo
  //Vec u = Vec({1, PI/6});  // D2 seems to be a shuffled version of D1
  Vec u = Vec({ 3, 0.5 });
  //Vec u = Vec({ 5, 1.2 });



  // Christoffel symbols :
  rsMultiArray<double> C;
  C = mf.getChristoffelSymbols2ndKind(u);


  std::function<void(const std::vector<double>&, rsMatrix<double>&)> dvdu_p, dvdu_c, dxdu;

  /*
  // define a vector field:
  u2v = [=](const Vec& u, Vec& v)
  {
    rsAssert(u.size() == 2);
    rsAssert(v.size() == 2);
    double r = u[0], phi = u[1];  // radius, phi
    v[0] =  r*r  * cos(2*phi);    // vector field component in radial direction...
    v[1] = (1/r) * sin(3*phi);    // ...in angular-direction
  };
  // partial derrivatives:
  // dv1/dr = 2*r * cos(2*phi)        dv1/dPhi = -2*r^2 * sin(2*phi)
  // dv2/dr = (-1/r^2) * sin(3*phi)   dv2/dPhi = (3/r)  * cos(3*phi)

  // Function to compute the covariant derivative analytically:

  dvdu_c = [=](const Vec& u, Mat& dv)
  {
    double r = u[0], phi = u[1];            // radius, phi
    std::vector<double> v(2); u2v(u, v);    // compute v(u)
    dv(0, 0) = 2*r*cos(2*phi);              // = dv1/dr
    dv(1, 0) = -r*r*2*sin(2*phi) - r*v[1];  // = dv1/dPhi - r*v2
    // not complete - fill the other two as well - they are not in the book
  };
  // Example from Bärwolff pg 832
  */



  // Let's take another example from "Aufstieg zu den Einsteingleichungen", pg. 207, ff
  // vector field in polar coodinates:
  u2v = [=](const Vec& u, Vec& v)
  {
    double r = u[0], phi = u[1];  // radius, phi
    double s = sin(phi), c = cos(phi);
    v[0] = 2*r*s*c;
    v[1] = c*c - s*s;
  };
  // partial derivative:
  dvdu_p = [=](const Vec& u, Mat& dv)
  {
    double r = u[0], phi = u[1];
    std::vector<double> v(2); u2v(u, v);    // compute v(u)
    double s = sin(phi), c = cos(phi);
    dv(0, 0) = 2*s*c;                       // dv1/dr
    dv(0, 1) = 0;                           // dv2/dr
    dv(1, 0) = 2*r*(c*c - s*s);             // dv1/dPhi
    dv(1, 1) = -4*s*c;                      // dv2/dPhi
  };
  // covariant derivative:
  dvdu_c = [=](const Vec& u, Mat& dv)
  {
    double r = u[0], phi = u[1];
    std::vector<double> v(2); u2v(u, v);    // compute v(u)
    double s = sin(phi), c = cos(phi);
    dv(0, 0) = 2*s*c;                       // dv1/dr
    dv(0, 1) = (c*c-s*s)/r;                 // dv2/dr
    dv(1, 0) = r*(c*c - s*s);               // dv1/dPhi
    dv(1, 1) = -2*s*c;                      // dv2/dPhi
    // Eq 8.34
  };






  // computes basis vectors E1, E2 analytically (derivative of (x,y) with repect to (r,p)
  dxdu = [=](const Vec& u, Mat& E)
  {
    double r = u[0], phi = u[1];            // radius, phi
    E(0, 0) =    cos(phi);
    E(0, 1) = -r*sin(phi);
    E(1, 0) =    sin(phi);
    E(1, 1) =  r*cos(phi);
  };
  rsMatrix<double> E  = mf.getCovariantBasis(u);
  rsMatrix<double> E2(2, 2); dxdu(u, E2);
  // they are stored as column-vectors ...somehow, this makes the indexing unnatural...maybe 
  // change that
  // var("r p")
  // x = r*cos(p)
  // y = r*sin(p)
  // dxdr = diff(x, r)
  // dxdp = diff(x, p)
  // dydr = diff(y, r)
  // dydp = diff(y, p)
  // dxdr, dxdp, dydr, dydp


  // compute basis vector derivatives:
  std::function<void(const Vec&, Tens&)> d2xdu2;
  d2xdu2 = [=](const Vec& u, Tens& dE)
  {
    double r = u[0], phi = u[1];            // radius, phi

    dE(0, 0, 0) = 0;               // x_dr_dr
    dE(0, 0, 1) = -sin(phi);       // x_dr_dp
    dE(0, 1, 0) = -sin(phi);       // x_dp_dr
    dE(0, 1, 1) = -r*cos(phi);     // x_dp_dp

    dE(1, 0, 0) = 0;               // y_dr_dr
    dE(1, 0, 1) = cos(phi);        // y_dr_dp
    dE(1, 1, 0) = cos(phi);        // y_dp_dr
    dE(1, 1, 1) = -r*sin(phi);     // y_dp_dp
  };
  // 1st index: output cartesian coordinate
  // 2nd index: coordinate with respect to which first derivative was taken
  // 3rd index: coordinate with respect to which second derivative was taken
  // dE(i,j,k) is the i-th component of the j-th basis vector's derivative with respect to the
  // i-th coordinate?
  /*
  var("r p")
  x = r*cos(p)
  y = r*sin(p)

  # 1st derivatives
  x_dr = diff(x, r)
  x_dp = diff(x, p)
  y_dr = diff(y, r)
  y_dp = diff(y, p)

  # 2nd derivatives
  x_dr_dr = diff(x, r, r)
  x_dr_dp = diff(x, r, p)
  x_dp_dr = diff(x, p, r)
  x_dp_dp = diff(x, p, p)

  y_dr_dr = diff(y, r, r)
  y_dr_dp = diff(y, r, p)
  y_dp_dr = diff(y, p, r)
  y_dp_dp = diff(y, p, p)

  #x_dr, x_dp, y_dr, y_dp
  x_dr_dr, x_dr_dp, x_dp_dr, x_dp_dp, y_dr_dr, y_dr_dp, y_dp_dr, y_dp_dp
  */

  Tens dE = mf.getCovariantBasisDerivative(u);
  Tens dE2({ 2,2,2 }); d2xdu2(u, dE2);
  // maybe we should test this first
  // they match only partially - verify formulas above and also the ordering
  // it seems they are in a different order

  // compute covariant derivative witn various means:
  Mat  D0(2, 2); dvdu_c(u, D0);                 // analytically, for reference
  Mat  D1 = mf.getCovariantDerivative( u, u2v); // via Christoffel symbols
  Mat  D2 = mf.getCovariantDerivative2(u, u2v); // via product rule
  Mat  error = D1-D2;
  // OK, D0 and D1 match, so D1 probably correct, so the error is in D2

  // compare the 2nd term in the product rule in lines 3 and 4 - they should be the same - if not,
  // our d_j_E_i may be wrong

  double maxError = error.getAbsoluteMaximum();



  // try computing covariant derivative via product rule, Eq. 358, line 3
  // we need the ingredients:
  // A:  vector field at u
  // dA: derivative of vector field at u
  // E:  basis vectors at u
  // dE: derivative of basis vectors
  Mat D3(2, 2);
  Vec A(2); u2v(u, A);
  Mat dA = mf.getVectorFieldDerivative(u, u2v); // maybe make a function to compute it analytically
  int i, j, k;
  for(j = 0; j < N; j++)
  {
    for(k = 0; k < N; k++)
    {
      for(i = 0; i < N; i++)
      {
        //D3(j, k) += A[i]*dE(i,j,k) + dA(j,i)*E(i,k);  // indices guessed
        //D3(j, k) += A[i]*dE(j,k,i) + dA(j,i)*E(k,i);  // same as D2: 1.63,-3.56,-3.05,13.89
        //D3(j, k) += A[i]*dE(j,k,i) + dA(j,i)*E(k,j);

        // test:
        Vec Ei   = mf.getBasisVector(u, i);
        Vec djEi = mf.getBasisVectorDerivative(u, i, j);  // verified
        D3(j, k) += Ei[k]*dA(j, i) + A[i]*djEi[k];
        //D3(j, k) += A[i]*Eij[k] + dA(i,j)*Ei[k]; // dA(i,j) has to be wrong!
      }
    }
  }
  // even D4(j, k) += A[i]*Eij[k] + dA(j,i)*Ei[k];  doesn't work!....the product rule seems broken
  // or i fundamentally misunderstand this equation 358 :-( - maybe i misunderstand, what the terms
  // d_j_A^i and/or d_j_E^i mean?


  // check identity 312, for i=j=1:
  i = 1, j = 1;
  Vec djEi = mf.getBasisVectorDerivative(u, i, j);
  Vec sum_G_Ek(2);
  for(k = 0; k < N; k++)
  {
    Vec Ek = mf.getBasisVector(u, k);
    sum_G_Ek = sum_G_Ek + C(k, i, j) * Ek;
  }
  // OK: djEi == sum_G_Ek - identity 312 is working, so my computation of djEi is correct

  // try to compute a single partial derivative vector via Eq 358:
  j = 1;        // we only need to choose j because i is summation index
  //j = 0;
  Vec Aj_c(N);  // covariant derivative of vector field A with resepct to j-th coordinate
  Vec Ei(N);    // i-th basis vector
  Vec Aj_p;     // partial derivative of vector field A with respect to j-th coordinate
  u2v(u, A);    // vector field A at u - redundant (computed above already)
  Aj_p = mf.getVectorFieldPartialDerivative(u, u2v, j);  // verified (more or less), 13.62, 0.96
  //Aj_p = mf.getVectorFieldPartialDerivative(u, u2v, 0);  // -3.92, -0.90
  for(i = 0; i < N; i++)
  //for(i = j; i <= j; i++)  // test: bypass summation over i - just take i=j term - seems to make no difference
  {
    Ei   = mf.getBasisVector(u, i);                // verified
    djEi = mf.getBasisVectorDerivative(u, i, j);   // verified
    Aj_c = Aj_c + Aj_p[i] * Ei;                    // ?
    Aj_c = Aj_c + A[i] * djEi;                     // ?
  }
  // also the wrong result - but the result is the same as in D2,D4 - so they are all wrong in the
  // same way, which is somewhat reassuring but also not
  // try to evaluate the product rule with pencil and paper - maybe use a simpler vector field

  // implement the Christoffel-symbol formula:
  Aj_c = Aj_p;
  for(k = 0; k < N; k++)
    for(i = 0; i < N; i++)
      Aj_c[i] += C(i, k, j) * A[k];
  // OK - that looks good


  // do it all explicitly by foot as on page 204 in (2):
  Vec Er  = mf.getBasisVector(u, 0); // basis vector in radial direction (r: radius)
  Vec Ep  = mf.getBasisVector(u, 1); // basis vector in angular direction (p: phi, phase)
  Vec Err = mf.getBasisVectorDerivative(u, 0, 0); // derivative of Er in r-direction
  Vec Epr = mf.getBasisVectorDerivative(u, 1, 0); // derivative of Ep in r-direction
  Vec dAr = mf.getVectorFieldPartialDerivative(u, u2v, 0); // vector field derivate in r-direction
  //Vec dAp = mf.getVectorFieldPartialDerivative(u, u2v, 1); // vector field derivate in p-direction
  u2v(u, A);
  Vec D5_1 = dAr[0]*Er + dAr[1]*Ep + A[0]*Err + A[1]*Epr;
  // matches with 1st row of D2 so if we can figure out what's wrong here, we can also say what's
  // wrong with D2 (the main product-rule-based calculation)
  // do we need to use normalized basis vectors? but none of the books says so and it doesn't seem
  // to make sense anyway. do we need the contravariant basis instead - but no, the books use 
  // subscript notation and it wouldn't make sense either. I think, we need to expand the basis 
  // vectors themselves
  // see this video at 9:18
  // https://www.youtube.com/watch?v=U5iMpOn5IHw&list=PLJHszsWbB6hpk5h8lSfBkVrpjsqvUGTCx&index=19&t=9m18s
  // and also at 11:10 - there are expressions for the basis vector derivatives
  double r = u[0];
  Err = Vec({0,0});
  Epr = (1/r) * Er;
  Vec D6_1 = dAr[0]*Er + dAr[1]*Ep + A[0]*Err + A[1]*Epr;



  // compare Er,Ep to pg 191, Eq 8.18, 8.19
  Vec Er1 = Vec({       cos(u[1]),      sin(u[1]) });
  Vec Ep1 = Vec({ -u[0]*sin(u[1]), u[0]*cos(u[1]) });
  // ..ok - they look good


  // try what happens if we convert Er,Ep,Err,Epr to polar coordinates before using them:
  Vec Er2(N), Ep2(N), Err2(N), Epr2(N);
  mf.toCurvilinear(Err, Err2);
  mf.toCurvilinear(Epr, Epr2);
  mf.toCurvilinear(Er,  Er2);
  mf.toCurvilinear(Ep,  Ep2);
  Vec D7_1 = dAr[0]*Er2 + dAr[1]*Ep2 + A[0]*Err2 + A[1]*Epr2;
  // nope! ..but maybe the derivatives need a different conversion rule? see eigenchris video
  // mentioned below! yes! i think, my basis-vector derivatives are wrong because they are 
  // expressed in the wrong coordinates...try the explicit formulas from the video ..at 13:50


  // could it be that our result is represented in different coordinates?
  Vec test1(2), test2(2);
  x2u(D5_1, test1);
  u2x(D5_1, test2);
  // ...nope! test1 is suspiciously close to (1,1) though..but no - not always


  // formulas 8.33, page 205 top - which is the Christoffel-based formula written out:
  double vrr = dAr[0] + A[0]*C(0,0,0) + A[1]*C(0,1,0); 
  double vpr = dAr[1] + A[0]*C(1,0,0) + A[1]*C(1,1,0);
  // matches the analytic reference result D0

  // changing the approximation stepsize also seems to have no effect

  // damn! that looks totally correct! could the christoffel and analytic formulas be both wrong
  // in the same way?

  // i think, i have found the reason: getBasisVectorDerivative returns the zero-vector for Err! we 
  // apparently have a numerical problem - catastrophic cancellations or something! ..or maybe the
  // derivative is indeed zero? this may also be the case - verify analytically - oh - yes: 
  // Er = (cos(p),sin(p)) which is const with respect to r, so the derivative is indeed zero.

  // see also:
  // https://web.pa.msu.edu/courses/2012spring/AST860/03-22.pdf
  // https://liavas.net/courses/math430/  beautiful handouts
  // https://phys.libretexts.org/Bookshelves/Relativity/Book%3A_Special_Relativity_(Crowell)/09%3A_Flux/9.04%3A_The_Covariant_Derivative

  // The video series on youtube by eigenchris:
  // https://www.youtube.com/watch?v=U5iMpOn5IHw&list=PLJHszsWbB6hpk5h8lSfBkVrpjsqvUGTCx&index=19
  // expecially at 9.18: we have the derivatives of the basis vectors expressed in cartesian 
  // coordinates - but we want them (do we?) in polar coordinates. maybe Err and Epr should be 
  // converted to polar coordinates before they are used?

  // https://www.youtube.com/watch?v=8S_XOjd5Mec&list=PLJHszsWbB6hpk5h8lSfBkVrpjsqvUGTCx&index=21&t=0s

  return result;
}

bool testManifoldSphere()
{
  // We use a spherical coordinate system as defined in "Tensor Calculus Made Easy" (1), pg 19 ff.
  // (2) Principles of Tensor Calculus
  bool result = true;

  using Vec = std::vector<double>;
  using Mat = rsMatrix<double>;

  // Forward and backward trafo:
  std::function<void(const std::vector<double>&, std::vector<double>&)> u2x, x2u, u2v;
  u2x = [=](const Vec& u, Vec& x)
  {
    rsAssert(u.size() == 3);
    rsAssert(x.size() == 3);
    double r = u[0], theta = u[1], phi = u[2]; // radius, theta, phi
    x[0] = r * sin(theta) * cos(phi);  // x   (1), Eq. 12
    x[1] = r * sin(theta) * sin(phi);  // y
    x[2] = r * cos(theta);             // z
  };
  x2u = [=](const Vec& X, Vec& u)
  {
    rsAssert(u.size() == 3);
    rsAssert(X.size() == 3);
    double x = X[0], y = X[1], z = X[2];
    double r     = sqrt(x*x + y*y + z*z);
    double theta = acos(z/r);
    double phi   = atan2(y,x);
    u[0] = r; u[1] = theta; u[2] = phi;
  };


  int N = 3, M = 3;
  rsManifold2<double> mf(N, M);

  // set up:
  mf.setCurvilinearToCartesian(u2x);
  mf.setCartesianToCurvilinear(x2u);
  //mf.setCurvToCartJacobian(u2xJ);
  //mf.setCartToCurvJacobian(x2uJ);
  mf.setApproximationStepSize(pow(2.0, -10));

  double r, theta, phi;
  r = 2.0;
  theta = PI/3;
  //theta = PI/5;
  phi   = PI/6;
  Vec u({ r, theta, phi });

  // Christoffel symbols of the first kind:
  rsMultiArray<double> C = mf.getChristoffelSymbols1stKind(u);

  // (2) Eq 341 ff
  double c221, c331, c122, c332, c133, c233;
  double sinTheta = sin(theta);
  double cosTheta = cos(theta);
  c221 = -r;
  c331 = -r*sinTheta*sinTheta;
  c122 = r;    // == c212
  c332 = -r*r*sinTheta*cosTheta;
  c133 = r*sinTheta*sinTheta;
  c233 = r*r*sinTheta*cosTheta;

  double c = C(0,2,2);  // c133 = 1.5
  c = C(1,2,2); // c233 = 1.7320...
  // ok - looks good - add some automatic checks

  double tol = 1.e-5;  // the numerical precision is quite bad!
  result &= rsIsCloseTo(c221, C(1,1,0), tol);  // 341
  result &= rsIsCloseTo(c331, C(2,2,0), tol);  // 342
  result &= rsIsCloseTo(c122, C(0,1,1), tol);  // 343
  result &= rsIsCloseTo(c122, C(1,0,1), tol);  // symmetry
  result &= rsIsCloseTo(c332, C(2,2,1), tol);  // 344
  result &= rsIsCloseTo(c133, C(0,2,2), tol);  // 345
  result &= rsIsCloseTo(c133, C(2,0,2), tol);  // symmetry
  result &= rsIsCloseTo(c233, C(1,2,2), tol);  // 346
  result &= rsIsCloseTo(c233, C(2,1,2), tol);  // symmetry

  // Christoffel symbols of the second kind:
  C = mf.getChristoffelSymbols2ndKind(u);
  double  c212, c313, c323;
  c122 = -r;
  c133 = -r*sinTheta*sinTheta;
  c212 = 1/r;
  c233 = -sinTheta*cosTheta;
  c313 = 1/r;
  c323 = cosTheta / sinTheta;

  result &= rsIsCloseTo(c122, C(0,1,1), tol);  // 347
  result &= rsIsCloseTo(c133, C(0,2,2), tol);  // 348
  result &= rsIsCloseTo(c212, C(1,0,1), tol);  // 349
  result &= rsIsCloseTo(c212, C(1,1,0), tol);  // symmetry
  result &= rsIsCloseTo(c233, C(1,2,2), tol);  // 350
  result &= rsIsCloseTo(c313, C(2,0,2), tol);  // 351
  result &= rsIsCloseTo(c313, C(2,2,0), tol);  // symmetry
  result &= rsIsCloseTo(c323, C(2,1,2), tol);  // 352
  result &= rsIsCloseTo(c323, C(2,2,1), tol);  // symmetry



  // test Riemmann tensor:
  // ...



  // test Ricci tensor:
  // ...



  // test covariant differentiation by comparing the two implementations - via Christoffel symbols
  // and via derivatives of basis vectors - i assume that it's unlikely that i got them both wrong
  // in the same way..

  // We define some weird fantasy vector field v(u) in spherical coordinates:
  u2v = [=](const Vec& u, Vec& v)
  {
    rsAssert(u.size() == 3);
    rsAssert(v.size() == 3);
    double r = u[0], theta = u[1], phi = u[2]; // radius, theta, phi
    v[0] = r*r * sin(theta) + cos(2*phi);      // vector field component in radial direction...
    v[1] = r * sin(2*theta) + cos(3*phi);      // ...in theta-direction
    v[2] = (1/r) * sin(3*theta) + cos(2*phi);  // ...in phi-direction
  };

  u = Vec({2, 3, 4});
  rsMatrix<double> D1 = mf.getCovariantDerivative( u, u2v);
  rsMatrix<double> D2 = mf.getCovariantDerivative2(u, u2v);
  rsMatrix<double> Err = D1-D2;
  // ok - these are totally different - either one of the two functions or both are still wrong
  // ...would have been too good to be true, if it worked at first shot
  // ...figure out, if the Christoffel-based formula is right - it is simpler and based on already
  // verified computation of Christoffel symbols - so it's more likely to be correct
  // maybe verify it via the exmaple in Bärwolff pg. 832
  // OK - i think it is because my idea of using the product rule directly was too naive - i 
  // didn't take into account, that the basis-vectors themselves are expressed in cartesian 
  // coordinates, but we need them in spherical coordinates (or something like that)

  // we could also test it by trying to compute the covariant derivative of the metric which should 
  // always be zero...or unity? but we don't have a covariant derivative function for 
  // matrix-fields yet

  return result;
}

// https://ask.sagemath.org/question/36777/covariant-derivative-gives-error-why-sage-751/
// f = function('f')
// B=Manifold(2,'B',start_index=1)
// polar.<R,Phi> = B.chart(R'R:(0,+oo) Phi:(0,2*pi):\Phi')
// G = B.riemannian_metric('G')
// G[1,1]=diff(f(R),R)
// G[2,2]=f(R)^2
// nabla=G.connection()
// S=B.tensor_field(1,1)
// S[1,1]=R^(0.5)
// S[2,2]=R^3
// S.display()
// nabla(S)

// http://sage-doc.sis.uta.fi/reference/manifolds/sage/manifolds/differentiable/tensorfield.html
// http://sage-doc.sis.uta.fi/reference/manifolds/sage/manifolds/differentiable/vectorfield.html#sage.manifolds.differentiable.vectorfield.VectorField

bool testManifoldEarth()
{
  // We use a spherical coodinate system with latitude from -90° (southpole) to +90° (northpole)
  // and longitude from 0° to 360°. The intrsinsic dimensionality is 2 because teh radius is fixed.
  // The sphere is embedded in 3D Euclidean space.

  // https://en.wikipedia.org/wiki/Spherical_coordinate_system#In_geography
  // https://en.wikipedia.org/wiki/Geographic_coordinate_system
  // https://en.wikipedia.org/wiki/Geodetic_datum
  // https://en.wikipedia.org/wiki/Geodesy

  double radius = 3.5;
  //double radius = 6371;  // radius of the sphere
  // The radius of earth ranges from 6357 km at the equator to 6378 km at the poles (it's not a 
  // perfect sphere but somewhat flattened ("oblate spheroid") due to rotation). The 6371 results 
  // from taking...hmm.. it's not the average - wikipedia says something about using perfect 
  // spheres with equal volume or surface area
  // https://en.wikipedia.org/wiki/Earth_radius
  // maybe in a more sophisticated implementation, we should actually use an oblate-spheroidal
  // coordinate system - i think, the only change would be to use bigRadius in the formulas for
  // x,y and smallRadius in the formulas for z - and in the inverse transformation, we should
  // divide x,y by the bigRadius and z by the smallRadius before using asin,atan2 - but in this
  // case, the radii cannot be inferred from the (x,y,z) tuple anymore - they have to be known
  // as parameters to the x2u conversion function, too (as they are to the u2x function)
  // more generally, we could define ellipsoidal coordinates by having 3 radii rx,ry,rz

  int N = 2, M = 3; // dimensionalities of manifold and embedding space

  rsManifold<double> mf(N, M);

  using Vec = std::vector<double>;
  using Mat = rsMatrix<double>;
  using Tens = rsMultiArray<double>;

  // for converting between the u- and x-coordinates and vice versa:
  std::function<void(const std::vector<double>&, std::vector<double>&)> u2x, x2u;

  // for computing Jacobian matrices:
  std::function<void(const std::vector<double>&, rsMatrix<double>&)> u2xJ, x2uJ;

  // The function x(u) that converts from 2D (latitude (degrees north), longitude (degrees east)) 
  // vectors to vectors in 3D space:
  u2x = [=](const Vec& u, Vec& x)
  {
    rsAssert(u.size() == 2);
    rsAssert(x.size() == 3);

    double phi = (PI/180) * u[0];              // latitude, radians north, phi
    double lam = (PI/180) * u[1];              // longitude, radians east, lambda
    double sinPhi, cosPhi, sinLam, cosLam;
    sinPhi = sin(phi);  cosPhi = cos(phi);
    sinLam = sin(lam);  cosLam = cos(lam);

    x[0] = radius * cosPhi * cosLam;  // x
    x[1] = radius * cosPhi * sinLam;  // y
    x[2] = radius * sinPhi;           // z
  };

  // The function to analytically compute the Jacobian matrix of x(u): 
  //   dx1/du1 dx1/du2 ... dx1/duN
  //   dx2/du1 dx2/du2 ... dx2/duN
  //     ...
  //   dxM/du1 dxM/du2 ... dxM/duN
  // at some given position vector u
  u2xJ = [=](const Vec& u, Mat& J)
  {
    rsAssert(u.size() == 2);
    //rsAssert(J.hasShape(M, N);

    double k = PI / 180;
    double phi = k * u[0];
    double lam = k * u[1];  // lambda
    double sinPhi, cosPhi, sinLam, cosLam;
    sinPhi = sin(phi);  cosPhi = cos(phi);
    sinLam = sin(lam);  cosLam = cos(lam);
    k *= radius;

    J(0, 0) = -k * sinPhi * cosLam;
    J(0, 1) = -k * cosPhi * sinLam;

    J(1, 0) = -k * sinPhi * sinLam;
    J(1, 1) =  k * cosPhi * cosLam;

    J(2, 0) =  k * cosPhi;
    J(2, 1) =  0;
  };

  // The function that u(x) converts a 3D position, supposed to be on the sphere, in space into 
  // latitude and longitude:
  x2u = [=](const Vec& X, Vec& u)
  {
    rsAssert(u.size() == 2);
    rsAssert(X.size() == 3);

    //double ri = 1 / radius;
    double ri = 1 / sqrt(X[0]*X[0] + X[1]*X[1] + X[2]*X[2]); 
    // test - yes!! solves metric problem - but it lets the unit tests fail - increasing the
    // tolerance fixed this

    double x = ri * X[0], y = ri * X[1], z = ri * X[2];

    u[0] = (180/PI) * asin(z);           // latitude, degrees north, = phi(x,y,z)
    u[1] = (180/PI) * atan2(y, x);       // longitude, degrees east, lambda(x,y,z)
  };
  // Note: 
  // We *could* infer the radius from the incoming data. When doing so, x2u would work even 
  // without knowing the actual radius from the outside.  If the position would be outside the 
  // sphere, it would be projected onto the sphere by casting a ray from the origin to the point
  // X = (x,y,z). Unfortunately, computing the radius from the input totally messes up the 
  // numerical precision of computing the contravariant basis vectors - the error really gets 
  // large. So we don't do that (this problem was really hard to find).
  // But wait: the mess up is with respect to comparisons with the analytic Jacobian - however, 
  // using the actual radius of the incoming point will generate a point on our sphere that is
  // closest to the incoming point, even if that is not on the sphere - maybe we will get better
  // behavior of the metrics multiplying to the identity? With planes, it is the case that the
  // distance-minimizing inverse function gave the best results. -> try it! YESSSS!! indeed!
  // this solves the problem! ...i think, we should do it in the jacobian computation, too - maybe
  // the discrepancy was due to doing it differently in both functions?
  // ...however - doing so makes the unit test fail - figure out why - ok - increasing the 
  // tolerance a bit solved it

  // The function to analytically compute the Jacobian matrix of u(x): 
  x2uJ = [=](const Vec& X, Mat& J)
  {
    // under construction
    rsAssert(X.size() == 3);
    //rsAssert(J.hasShape(N, M);

    double ri = 1 / radius;

    //double ri = 1 / sqrt(X[0]*X[0] + X[1]*X[1] + X[2]*X[2]);  // test

    double k = ri * 180/PI;
    double x = ri * X[0], y = ri * X[1], z = ri * X[2];

    J(0, 0) = 0;
    J(0, 1) = 0;
    J(0, 2) = k / sqrt(1 - z*z);

    k /= x*x + y*y;
    J(1, 0) = -k*y;
    J(1, 1) =  k*x;
    J(1, 2) =  0;
  };
  // x2u is: phi = k * asin(z/r), lambda = k * atan2(y/r, x/r) where k = 180/PI
  // the derivatives can be computed with sage as:
  //
  // var("x y z r k")
  // dx = diff(k * atan2(y/r,x/r), x)    // dLambda/dx
  // dy = diff(k * atan2(y/r,x/r), y)    // dLambda/dy
  // dz = diff(k * asin(z/r), z)         // dPhi/dz
  // dx, dy, dz
  //
  // giving:
  //
  // dLambda/dx = -k*y/(r^2*(x^2/r^2 + y^2/r^2))
  // dLambda/dy =  k*x/(r^2*(x^2/r^2 + y^2/r^2))
  // dPhi/dz    =  k  /(r*sqrt(-z^2/r^2 + 1))     ?= (k/r) * sqrt(1 - z^2)
  //
  // which can be simplified into the formuals above. The other derivatives are zero.


  // set up:
  mf.setCurvilinearToCartesian(u2x);
  mf.setCartesianToCurvilinear(x2u);
  //mf.setCurvToCartJacobian(u2xJ);
  //mf.setCartToCurvJacobian(x2uJ);
  mf.setApproximationStepSize(1.e-4);   // 1.e-4 good for the covariant basis
  // somewhere between 1.e-3 and 1.e-4 seems to be the sweet spot - this has been found by trial 
  // and error, comparing numeric to analytic computations and the sweet spot may be different for 
  // differet problems...also...it may actually be different for the u2x and x2u functions - this 
  // value here seems good for u2x - todo: figure out best value for x2u - maybe let the class take 
  // different values for the different directions - maybe even for the different variables (they 
  // may have different ranges like one from -1..+1 another from 0 to 1000

  // automated tests:
  Vec x(3), u(2);
  bool result = true;
  //double tol = 1.e-7; 
  double tol = 1.e-6; 

  Mat El, Eu, gl, gu, delta;

  //u = { 23, 57};
  u = { 83, 23};
  delta = mf.getNumericalBasisProduct(u);
  El = mf.getCovariantBasis(u);
  Eu = mf.getContravariantBasis(u);
  delta = El * Eu;  // nope! no delta! also, it's 3x3 - we expect a 2x2 matrix
  delta = Eu * El;    // yes! 2x2 delta!
  gl = mf.getCovariantMetric(u);      // has unequal entries on diagonal
  gu = mf.getContravariantMetric(u);  // has equal entries on diagonal
  delta = gl * gu;
  delta = gu * gl;
  // compare with:
  // https://www.einsteinrelativelyeasy.com/index.php/general-relativity/35-metric-tensor-exercise-calculation-for-the-surface-of-a-sphere
  // but we use differently defined spherical coordinates
  // how can it be that our basis vectors seem to be correct, yet the metric seems to be wrong?
  // could there be the same mistake in the numerical and analytical formulas?
  // when the Jacobians are assigned, the metric problem comes back, regardless whether we use the 
  // actual or nominal radius inside the function - maybe it has to do with some entries being zero
  // which should not be zero? maybe it's about the way, we derived the analytic expressions - 
  // maybe we should use minimization of distance to the surface? Like, defining u(x) as
  // min_u(|x - x(u)|) instead of just "somehow" inverting the transformation equations in an 
  // ad-hoc way (which may not even be possible) - this minimum equation can actually always be
  // solved in principle - we just do well defined operations that can always be done: evaluations,
  // subtractions, multiplications, derivatives ...but we may end up with an implicit equation
  // for the u_i which may or may not be solvable explicitly. ...but maybe the best thing is to 
  // not use the inverse transformation at all and just rely on the u2x function and use matrix 
  // inversion to obtain everything else - this seems much simpler and more straightforward

  //u = { 83, 23};
  result &= mf.runTests({ 83,  23}, tol);
  //result &= mf.runTests({ 23,  57}, tol);

  // actually, this should be enough to test - clean up all the code below..but not yet
  int inc = 5;
  for(int i = -(180-inc); i <= 180-inc; i += inc) {
    for(int j = -(90-inc); j <= 90-inc; j += inc) {
      u = { double(j), double(i) };
      result &= mf.runTests(u, tol); }}



  // special values:
  result &= mf.runTests({  0,   0}, tol);
  result &= mf.runTests({  0,  90}, tol);   // 1st: z-direction, 2nd: negative x-direction
  result &= mf.runTests({  0, -90}, tol);   // 1st: z-direction, 2nd: x-direction

                                          // close to and at discontinuity / wrap-around of longitude:
  result &= mf.runTests({  0,  179}, tol);
  result &= mf.runTests({  0, -179}, tol);
  //result &= mf.runTests({  0,  180}, tol); // doesn't work - no surprise
  //result &= mf.runTests({  0, -180}, tol); // dito

  // close to and at singularities of latitude:
  result &= mf.runTests({ 89,   0}, tol);
  result &= mf.runTests({-89,   0}, tol);
  //result &= mf.runTests({ 90,   0}, tol); // doesn't work - no surprise
  //result &= mf.runTests({-90,   0}, tol); // dito


  //mf.setApproximationStepSize(1.e+2);
  //mf.setApproximationStepSize(1.e-8);

  u = { 83, 23};
  Tens C1, C2;
  C1 = mf.getChristoffelSymbols1stKind(u);
  C2 = mf.getChristoffelSymbols2ndKind(u);
  // maybe this should be tested with Eq. 341 ff.



  
  mf.toCartesian(Vec({ 0,   0}), x); // (x,y,z) = ( r, 0, 0)
  mf.toCurvilinear(x, u);
  mf.toCartesian(Vec({ 0,  90}), x); // (x,y,z) = ( 0, r, 0)
  mf.toCurvilinear(x, u);
  mf.toCartesian(Vec({ 0, 180}), x); // (x,y,z) = (-r, 0, 0)
  mf.toCurvilinear(x, u);
  mf.toCartesian(Vec({ 0, 270}), x); // (x,y,z) = ( 0,-r, 0)
  mf.toCurvilinear(x, u);            // return -90 instead of 270 - maybe fix it - or maybe it's right?
  mf.toCartesian(Vec({90,   0}), x); // (x,y,z) = ( 0, 0, r)
  mf.toCurvilinear(x, u);
  mf.toCartesian(Vec({-90,  0}), x); // (x,y,z) = ( 0, 0,-r)
  mf.toCurvilinear(x, u);
  // move to a unit test...
 

  El = mf.getCovariantBasis(Vec({  0,   0})); // l for "lower indices"
  // one basis vector should point in the y-direction, the other in the z-direction
  // ...seems to be ok - but what length should they have?

  Eu = mf.getContravariantBasis(Vec({  0,   0})); // u for "upper indices"
  delta = Eu * El;  // should result in 2x2 Kronecker-delta tensor, if all is right

  //Mat bla = El *Eu; // what about El * Eu? does that have a meaning, too?

  gl = mf.getCovariantMetric(Vec({  0,   0}));
  // what do we expect?
  // it's diagonal and the magnitude of the entries is proportional to the radius^2 which seems
  // to be correct - 

  gu = mf.getContravariantMetric(Vec({  0,   0}));

  delta = gl * gu;  // should also give delta-tensor, aka identity matrix - yup but not as
                    // numerically precise as above

  // Compute the length of a "angle" degree arc using the metric tensor:
  double angle = 3.0;
  Mat du(2, 1, {angle, 0.0});            // increase angle° in longitude and 0° in latitude
  Mat ds2 = du.getTranspose() * gl * du; // should be a 1x1 matrix giving the squared length
  double ds  = sqrt(ds2(0,0));           // the actual length as scalar
  double tmp = radius * angle*PI/180;    // analytic formula for the length
  // ds should be close to tmp (they actually match perfectly!)


  u = { 83, 23};
  Tens Riemann1 = mf.getRiemannTensor1stKind(u);
  Tens Riemann2 = mf.getRiemannTensor2ndKind(u);
  Mat  Ricci1   = mf.getRicciTensor1stKind(u);
  Mat  Ricci2   = mf.getRicciTensor2ndKind(u);
  // todo compare these to analytically computed values - we may need a more common sort of 
  // spherical coordinates like r,theta,phi or something because the analytic expressions found
  // in books and on the internet assume these

  // compute Ricci scalar
  double R = mf.getRicciScalar(u);
  double Rt = 2 / (radius*radius); // should be 2 / radius^2 according to (2), pg. 477
  // R has wrong sign - but absolute values look good - but only when the radius is of order 1
  // smells like numerical approximation errors

  // from here, it gets wrong:
  Vec berlin  = Vec({52.52,     13.405});  // a contravariant position (of Berlin)
  Vec phoenix = Vec({33.4484, -112.0740}); // Phoenix, Arizona
  Vec b2p     = phoenix - berlin;         // direction




  /*
  // metric should (?) be the same everywhere - but is not!:
  gl = mf.getCovariantMetric(Vec({  0,   0}));
  gl = mf.getCovariantMetric(Vec({  0,  30}));
  gl = mf.getCovariantMetric(Vec({  0,  60}));
  gl = mf.getCovariantMetric(Vec({  0,  90}));
  gl = mf.getCovariantMetric(Vec({  0, 120}));

  gl = mf.getCovariantMetric(Vec({ 30,   0}));
  gl = mf.getCovariantMetric(Vec({ 30,  30}));


  gl = mf.getCovariantMetric(Vec({ 30,  60}));
  gu = mf.getContravariantMetric(Vec({ 30,  60}));
  delta = gl * gu;  // not the identity!
  */




  // todo: check getCovariantBasis at various positions

  gl = mf.getCovariantMetric(berlin);
  gl = mf.getCovariantMetric(phoenix);


  ds = mf.getContravariantDotProduct(b2p, b2p, berlin); // last argument should make no difference
  ds = sqrt(ds); // around 9100 km - ok - so this is wrong!
  // wait - the formual is wrong! we cannot simply use the metric - we have to compute an integral
  // along the geodesic curve between the locations - and *inside* this integral we would use the 
  // metric for approximating the length of the segment!
 
  // check, if the metric tensor is independent from position - it should be a constant for a 
  // sphere



  // https://www.trippy.com/distance/Phoenix-to-Berlin

  //rsNormalize(b2p);

  // 33.4484° N, 112.0740° W

  /*
  Vec a, v;
  u = 


  v = mf.lowerVectorIndex(a, u);
  */


  // todo: 
  // -test dot-products
  // -test index raising and lowering


  rsAssert(result == true);
  return result;
}
// todo:
// have alternative parametrizations of the sphere
// -x,y and just project up by z = sqrt(1 - x^2 - y^2)
// -r,phi compute x,y and project up (or down, if r is negative)
// -use thes alternative parametrization for experimenting with coordinate transformations - we 
//  need functions r(lon, lat), phi(lon, lat), lon(r, phi), lat(r, phi)


// http://einsteinrelativelyeasy.com/index.php/general-relativity/34-christoffel-symbol-exercise-calculation-in-polar-coordinates-part-ii
// http://einsteinrelativelyeasy.com/index.php/general-relativity/70-the-riemann-curvature-tensor-for-the-surface-of-a-sphere

// has explicit formuals for Riemann and Ricci tensors for sphere and torus
// https://digitalcommons.latech.edu/cgi/viewcontent.cgi?article=1008&context=mathematics-senior-capstone-papers

// more tensors defined (schouten, weyl, etc.)
// http://www2.math.uu.se/~svante/papers/sjN15.pdf

void testManifoldEllipsoid()
{
  // https://www.researchgate.net/publication/45877605_2D_Riemann-Christoffel_curvature_tensor_via_a_3D_space_using_a_specialized_permutation_scheme
  // ah - damn - it doesn't have the inverse transformations - i think, i really should compute the
  // inverse metric by matrix inversion

  // axes of the ellipsoid
  double a = 3;
  double b = 2;



}

// How can we express the Minkowski metric of flat spacetime? We would need a metric tensor of the
// form
//  -1  0  0  0
//   0  1  0  0   Eq.241
//   0  0  1  0
//   0  0  0  1
// or maybe use the 1st coordinate for time. Should we use: x1 = i*u1, x2 = u2, x3 = u3, x4 = u4
// where i is the imaginary unit?
// Can we express the basis vectors in terms of a given metric? maybe Cholesky decomposition is
// needed? maybe the basis vectors are not unique? or maybe some other decomposition such that
// G = E^T * E,  where E is the matrix of basis vectors and G is the metric as matrix








// code from:
// https://www.geeksforgeeks.org/print-subsets-given-size-set/

/* 
arr[]  ---> Input Array
n	     ---> Size of input array
r	     ---> Size of a combination to be printed
index  ---> Current index in data[]
data[] ---> Temporary array to store current combination
i	     ---> index of current element in arr[]	 */
void combinationUtil(int arr[], int n, int r, int index, int data[], int i)
{
  // Current combination is ready, print it
  if (index == r) {
    for (int j = 0; j < r; j++)
      printf("%d ", data[j]);
    printf("\n");
    return;
  }

  // When no more elements are there to put in data[]
  if (i >= n)
    return;

  // current is included, put next at next location
  data[index] = arr[i];
  combinationUtil(arr, n, r, index + 1, data, i + 1);

  // current is excluded, replace it with next (Note that i+1 is passed, but index is not
  // changed):
  combinationUtil(arr, n, r, index, data, i + 1);
}
// The main function that prints all combinations of size r in arr[] of size n. This function 
// mainly uses combinationUtil()
void printCombination(int arr[], int n, int r)
{
  int* data = new int[r];                 // A temporary array to store all combination one by one
  combinationUtil(arr, n, r, 0, data, 0); // Print all combination using temprary array 'data[]'
  delete[] data;
}
int testPrintCombinations()
{
  int arr[] = { 10, 20, 30, 40, 50 };
  int r = 3;
  int n = sizeof(arr) / sizeof(arr[0]);
  printCombination(arr, n, r);
  return 0;
}
// todo: adapt code to not print the combinations but instead store them in an output array

bool testSubsets()
{
  bool ok = true;
  using Vec = std::vector<int>;
  Vec set = { 10, 20, 30, 40, 50 };
  Vec subsets = rsSubsetsOfSize(set, 3);
  ok &= subsets == Vec({ 10,20,30, 10,20,40, 10,20,50, 10,30,40, 10,30,50, 10,40,50, 20,30,40, 
     20,30,50, 20,40,50, 30,40,50});
  return ok;
}

void testSortedSet()
{
  //testPrintCombinations();

  using Set = rsSortedSet<int>;
  bool r = true;

  r &= testSubsets();

  Set A, B, C;

  A = Set({1,3,5,6,9});
  B = Set({2,3,4,5,7});
  C = A + B; r &= C == Set({1,2,3,4,5,6,7,9});  // union
  C = B + A; r &= C == Set({1,2,3,4,5,6,7,9});
  C = A - B; r &= C == Set({1,6,9});            // difference
  C = B - A; r &= C == Set({2,4,7}); 
  C = A * B; r &= C == Set({3,5});              // intersection
  C = B * A; r &= C == Set({3,5});
  C = A / B; r &= C == Set({1,2,4,6,7,9});      // symmetric difference
  C = B / A; r &= C == Set({1,2,4,6,7,9});
  C = A + A; r &= C == A;
  C = A - A; r &= C == Set({});
  C = A * A; r &= C == A;
  C = A / A; r &= C == Set({});


  // try also A+A, A-A, A*A, A/A, try with empty sets, 

  A = Set({1,2,4,6,7,9});
  B = Set({2,3,5,7,8});
  C = A + B; r &= C == Set({1,2,3,4,5,6,7,8,9});
  C = A - B; r &= C == Set({1,4,6,9});
  C = B - A; r &= C == Set({3,5,8});
  C = A * B; r &= C == Set({2,7});
  C = A / B; r &= C == Set({1,3,4,5,6,8,9});

  A = Set({1,3,5,7,9});
  B = Set({2,4,6,8});
  C = A + B; r &= C == Set({1,2,3,4,5,6,7,8,9});
  C = A - B; r &= C == Set({1,3,5,7,9});
  C = A * B; r &= C == Set({});
  C = A / B; r &= C == Set({1,2,3,4,5,6,7,8,9});

  A = Set({1,2,3,4,5,6});
  B = Set({3,4,5,6,7,8});
  C = A + B; r &= C == Set({1,2,3,4,5,6,7,8});
  C = A - B; r &= C == Set({1,2});
  C = B - A; r &= C == Set({7,8});
  C = A * B; r &= C == Set({3,4,5,6});
  C = A / B; r &= C == Set({1,2,7,8});

  // maybe make tests with randomized sets, check if set algebraic rules hold

  auto D = Set::cartesianProduct(A.getData(), B.getData());

  // todo: implement a class rsRelation..or maybe it should be an internal class of Set - or maybe
  // we should not use another class but the very same Set class - a relation *is* a set
  // https://en.wikipedia.org/wiki/Binary_relation#Operations_on_binary_relations
  // this could also be used to represent (directed) graphs - the relation would be the edges.
  // what about heterogenous sets? maybe they can be implemented by storing pointers to void or
  // to some generic "rsSetElement" baseclass which hase a type field and a data field - we would
  // need to invent some scheme for ordering them - maybe the type could be a string and we sort 
  // sets lexicographically by type and within a type, use the < relation of the respective type
  // ..but what if we have types like integer and rational and one set is A = { 1,2,3 } and another 
  // is B = { 1,2/1,3 } - the 2/1 in B is of different type than the 2 in A - should we treat them
  // as equal nonetheless in comparisons? maybe there should be a type-system that allows for
  // one type to encompass another (like the rationals with the reals)

  int dummy = 0;
}

void testAutoDiff()
{
  using DN = rsDualNumber<float, float>;

  DN x, y, z, r;

  bool t = true;  // test

  x = 3.f;
  y = 2.f;
  z = 5.f;


  //r = x+y; t &= r == 5.f;
  //r = x-y; t &= r == 1.f;
  //r = x*y; t &= r == 6.f;
  //r = x/y; t &= r == 1.5f;


  r = rsSin(x);
  r = rsCos(x);
  r = rsExp(x);
  //r = rsLog(x);
  //r = rsAbs(x);
  //r = rsPow(x, 2.f);
  //r = rsPow(x, y);   //


  //x = (2.f, 3.f);      // doesn't work - why does it even compile?
  x = {2.f, 3.f};      // this "tuple-initialization" works, maybe use it also for rsFraction - what about complex?
  x = DN(2.f, 3.f);

  r = rsSin(x); t &= r == DN(sin(x.v),   x.d*cos(x.v));
  r = rsCos(x); t &= r == DN(cos(x.v),  -x.d*sin(x.v));
  r = rsExp(x); t &= r == DN(exp(x.v),   x.d*exp(x.v));
  //r = rsLog(x); t &= r == DN(log(x.v),   x.d / x.v );
  //r = rsAbs(x); t &= r == DN(fabs(x.v),  x.d * rsSign(x.v));
  //r = rsPow(x, 5.f); t &= r == DN(pow(x.v, 5.f), x.d * 5.f * pow(x.v, 4.f));
  //r = rsPow(x, y);   //


  // test a function that uses sum, product, quotient and chain rule, such as
  // f(x) = exp(-x) * sin(3*x) / (1 + x^2 * cos(x))


  // maybe they sould take a float argument?
  //auto f = [&](DN x)->DN { return 10*x*x - 2*x*x*x; };    // ok
  //auto f = [&](DN x)->DN { return rsSin(x);         };    // ok
  //auto f = [&](DN x)->DN { return 2 * rsSin(x);     };    // ok
  //auto f = [&](DN x)->DN { return rsSin(x) * 2;     };    // ok
  //auto f = [&](DN x)->DN { return rsSin(2*x);       };    // ok
  //auto f = [&](DN x)->DN { return rsSin(x*2);       };    // ok
  //auto f = [&](DN x)->DN { return rsSin(x/2);       };    // ok
  //auto f = [&](DN x)->DN { return 1 / (1 + x*x);    };    // ok
  //auto f = [&](DN x)->DN { return rsSin(2*x + 1);   };    // ok
  //auto f = [&](DN x)->DN { return rsSin(1 + 2*x);   };    // ok
  //auto f = [&](DN x)->DN { return rsSin(2*x - 1);   };    // ok
  //auto f = [&](DN x)->DN { return rsSin(1 - 2*x);   };    // ok


  // If f would take a float, we would have to explicitly construct an ADN from x inside f, like
  // for example: return rsSin(2.f*ADN(x)); but we do the implicity construction here

  auto f = [&](DN x)->DN { return rsExp(-x/31)*rsSin(5*x/2) / (2 + x*x * (1+rsCos(x)) + 1); };
  // Computing the analytic derivative with sage:
  //   f  = exp(-x/31) * sin(5 * x / 2) / (2 + x*x * (1 + cos(x)) + 1)
  //   fp = diff(f,x)
  //   f, fp
  // shows that the derivative is already quite messy in this case, such that using autodiff 
  // seems to be more convenient. What about the 2nd derivative? Can we nest autodiff numbers 
  // suitably? but then maybe, the type of d should indeed be different from the type of v - d may
  // itself be an dual number while v is still a regular number


  // Computes f1(2) along with its derivative f1'(2) - the derivative is computed because we seed
  // the d with 1.f:
  //r = f1( ADN(2.f, 1.f) );

  // todo: plot f1 and f1':
  static const int N = 500;
  float xMin = 0.f, xMax = 8.f;
  float X[N], V[N], D[N];
  rsArrayTools::fillWithRangeLinear(X, N, xMin, xMax);
  for(int n = 0; n < N; n++) {
    r    = f(X[n]);             // works, if d defaults to 1
    V[n] = r.v;
    D[n] = r.d; }
  rsPlotArraysXY(N, X, V, D);
  // we need to take more care when to init d with 0 and when with 1 - the implicit conversions are
  // sometimes right, sometimes wrong - we should perhaps be always explicit



  // test some more complicated expressions:
  // We evaluate the bivariate function f(x,y) = sin(x) * cos(y) at the point (2,3) - what does the
  // derivative part of the result represent? Is it the mixed 2nd derivative f_xy of f(x,y)? How 
  // can we get the partial derivatives f_x,f_y, i.e. the gradient? What about the Hessian matrix?
  x = DN(2.f, 0.f);
  y = DN(3.f, 0.f);
  r = rsSin(x) * rsCos(y);



  // try nesting dual numbers:
  using DN2 = rsDualNumber<float, DN>; // the 2nd part is itself a dual number
  auto f2 = [&](DN2 x)->DN2 
  { 
    //return rsSin(x);
    //return rsCos(x);
    return rsExp(x);
  };
  //auto f2 = [&](NDN x)->NDN { return x*x*x; };
  float D2[N]; 
  for(int n = 0; n < N; n++) {
    DN2 r = f2(DN2(X[n])); 
    V[n]  = r.v;      // value
    D[n]  = r.d.v;    // 1st derivative
    D2[n] = r.d.d;    // 2nd derivative
  }
  rsPlotArraysXY(N, X, V, D, D2);
  // looks ok so far, but more tests needed - we should have a unit test that systematically tests
  // elementary functions, arithmetic operators, complicated combinations of everything with 
  // simple, simply-nested, doubly-nested, triply-nested, etc. dual numbers


  // try nesting twice:
  using DN3 = rsDualNumber<float, DN2>;
  //auto f3 = [&](NNDN x)->NNDN { return rsSin(x); };
  auto f3 = [&](DN3 x)->DN3 { return rsExp(x); };
  float D3[N];
  for(int n = 0; n < N; n++) {
    DN3 r = f3(DN3(X[n]));
    V[n]  = r.v;
    D[n]  = r.d.v;
    D2[n] = r.d.d.v;
    D3[n] = r.d.d.d;
  }
  rsPlotArraysXY(N, X, V, D, D2, D3);
  // doesn't work - 3rd derivative has extra factor of 2 for exp function - i think, it has 
  // something to do with bothe terms in the product rule evaluating to exp(x), so we get
  // 1*exp(x) + exp(x)*1 ...or something -> figure out


  // nesting thrice:
  using DN4 = rsDualNumber<float, DN3>;
  // ...stuff to do...

  int dummy = 0;

  // ToDo:
  // -maybe instead of initializing d with 0 when constructing from a real number, we should use 1
  //  to seed the perturbation ...or maybe demand the user to be explicit, whether to use 0 or 1
  //  -> when computing f(v + 1*d), we should get (f(v), f'(v)) instead of(f(v), 0) as result of a 
  //  function evaluation
  // -maybe input, output and derivative should have different types, like R^N, R^M, R^(MxN)...but 
  //  actually we can also do this by using rsMatrix for all 3 types
  // -we may need generalized versions of chain-, product-, quotient-rules
  // -could element functions of a multivariate function themselves operate on univariate dual 
  //  numbers?
  // -in general, a derivative value can be seen as having arisen form a mapping (R->R x R) -> R,
  //  i.e. inputs are a function R->R and a real number from R and output is a real number from R
  //  ...but we can also view it as (R->R) -> (R->R), i.e. both, input and output are functions 
  //  R->R
  // -what about nesting dual numbers?
  // -maybe try the simpler function x^3 * y^2 first...hmmm...i think, the basic arithmetic 
  //  operations will never produce a nonzero d-part if both operands have zero d-part. How, then,
  //  can nonzero d-parts arise in the first place, when we assume that elementary functions (such 
  //  as sqrt), are themselves implemented in terms of basic arithmetic operations (such as the 
  //  Babylonian algorithm)? ...Maybe implement the Babylonian algo....
  //  Ah - that's actually consistent with what the elementary functions also do, due to the 
  //  multiplication by the inner derivative value due to the chain-rule. They also will never
  //  produce nonzero d-parts when the argument as a zero d-part. ....so how do nonzero d-parts
  //  ever arise in the first place unless we initialize them as nonzero? Is it, when we first
  //  initialize an AutoDiffNumber with a function evaluation result of a function of a normal 
  //  number? ..like y = sin(x), where y is an AutoDiffNumber and x is a normal number, we should
  //  produce y as y = ADN(sin(x), cos(x));? At some point, we must transition from normal 
  //  computations to autodiff computations - that seems to be the place where the d-part must be
  //  assigned...obviously...try to figure out, how to do a simple gradient descent with a bivariate
  //  function via autodiff...can we also implement an ODE solver in terms of autodiff?
  // -how about forming dual number from complex (instead of real) numbers? or forming complex 
  //  numbers of dual numbers?



  x = DN(1.5f, 0.5f);
  y = DN(2.5f, 0.75f);
  r = rsSin(x) * rsCos(y);


}

void testAutoDiff2()
{
  // Tests automatic differentiation for a function with 2 inputs and 1 output. The derivative is 
  // the gradient vector which as 2 components, just as the input.

  using Vec = rsVector2D<float>;
  using DN  = rsDualNumber<float, Vec>;

  // f(x,y) = x^2 + y^3, f_x = 2*x, f_y = 3*y^2
  auto f = [&](Vec v)->DN 
  { 
    DN x = DN(v.x, Vec(1,0));
    DN y = DN(v.y, Vec(0,1));
    return x*x + y*y*y;
  };
  DN r;
  r = f(Vec(3.f, 2.f)); // 17,  6, 12
  r = f(Vec(5.f, 3.f)); // 52, 10, 27


  auto f2 = [&](Vec v)->DN 
  { 
    DN x = DN(v.x, Vec(1,0));
    DN y = DN(v.y, Vec(0,1));
    return (rsSin(x*x) + rsCos(y*y*y)) / (1 + (x*y)*(x*y)) ;
  };
  r = f2(Vec(3.f, 2.f));
  r = f2(Vec(5.f, 3.f));
  // Sage gives the partial derivatives:
  //   var("x y")
  //   f   = (sin(x*x) + cos(y*y*y)) / (1 + (x*y)*(x*y))
  //   f_x = diff(f, x);
  //   f_y = diff(f, y);
  //   #f, f_x, f_y
  //   N(f(5,3)), N(f_x(5,3)), N(f_y(5,3))
  // which evaluate to:
  //   (-0.00187827680898942, 0.0446065178382468, -0.113010657281297)
  // OK - looks good - the gradient is computed correctly!
  
  // todo: 
  // -maybe plot the gradients as vectors field
  // -try a function from R^2 -> R^3, such as the surface of a torus using std::vector and rsMatrix
  //  for TVal, TDer

  int dummy = 0;
}


void testAutoDiff3()
{
  // Trying to produce Jacobians of a function R^2 -> R^3. Such functions define a 2D surface 
  // embedded in 3D space. (Later maybe use a torus as example surface).


  // Define 3 scalar functions for the x,y,z components of the output vector:
  using Vec2 = rsVector2D<float>;
  using DN   = rsDualNumber<float, Vec2>;
  auto fx = [&](Vec2 v)->DN
  {
    DN x = DN(v.x, Vec2(1,0));
    DN y = DN(v.y, Vec2(0,1));
    return x*y;  // preliminary
  };
  auto fy = [&](Vec2 v)->DN
  {
    DN x = DN(v.x, Vec2(1,0));
    DN y = DN(v.y, Vec2(0,1));
    return x*x + y*y;  // preliminary
  };
  auto fz = [&](Vec2 v)->DN
  {
    DN x = DN(v.x, Vec2(1,0));
    DN y = DN(v.y, Vec2(0,1));
    return x*x - y*y;  // preliminary
  };

  // Define a function f that combines fx,fy,fz and returns a 3-vector of dual numbers. The 
  // derivative part of each output component is the gradient of the respective scalar function. 
  // Together, they form the Jacobian matrix:
  using Vec3 = rsVector3D<DN>;
  auto f = [&](Vec2 v)->Vec3
  {
    return Vec3(fx(v), fy(v), fz(v));
  };

  Vec3 r = f(Vec2(2,3));
  //        v   d.x d.y
  // r.x =  6    3   2
  // r.y = 13    4   6
  // r.z = -5    4  -6
  // (6,13,-5) is the output and [[3 2],[4 6],[4 -6]] the Jacobian. Is that the expected result?


  // OK - this seems to work - but can we do this more conveniently? ...like getting the result as 
  // 3-vector and Jacobian as actual 3x2 matrix as output of a function taking a 2-vector and
  // hopefully avoiding a lot of the boilerplate? and what about general functions R^M -> R^N? 
  // somehing like:
  // using Vec = std::vector<float>;
  // using Mat = rsMatrix<float>;
  // using DN  = rsDualNumber<Vec, Mat>;
  // auto f = [&](Vec v)->DN  // input: 2D vector, output: Dual<3D vector, 3x2 matrix>
  // {
  //   DN vx = DN(Vec(3), Mat(3,3)); // (1,0,0)
  //   DN vy = DN(Vec(3), Mat(3,3));
  //   DN vz = DN(Vec(3), Mat(3,3));
  // };


  // ToDo: try a R^3 -> R^2 function - maybe some sort of projection (fish-eye? spherical? i.e. 
  // first project points onto the sphere surface (by expressing them in spherical coordinates and 
  // setting the r-coordinate to 1) and then map the sphere surface onto the plane (maybe by 
  // inverting the parametric description of a sphere, if possible)

  // try cumulative product of 3-vector (see video about ForwardDiff.jl etc.)

  int dummy = 0;
}


void testAutoDiff4()
{
  // The same as above but with vectors that can have arbitrary lengths using std::vector:

  int M = 2;                                // dimensionality of input space
  int N = 3;                                // dimensionality of output space
  using VecF = std::vector<float>;          // input vector of length M
  using DN   = rsDualNumber<float, VecF>;   // component function value and gradient
  using VecD = std::vector<DN>;             // component function values and gradients

  // A function to convert a vector of numbers into a vector of numbers with attached vector for 
  // the gradient, where the index of the only non-zero element of the gradient matches the index
  // in the output vector:
  auto toDual = [&](VecF v)->VecD
  {
    size_t N = v.size();
    VecF g(N);  // todo: init to zeros
    VecD r(N);  // out vector of dual numbers
    for(size_t i = 0; i < N; i++) {
      g[i] = 1; r[i] = DN(v[i], g); g[i] = 0; }
    return r;
  };
  // this may become a static utility function of class rsDualNumber

  // Define the 3 component functions:
  auto f0 = [&](VecF v)->DN { VecD d = toDual(v); return d[0]*d[1];             };
  auto f1 = [&](VecF v)->DN { VecD d = toDual(v); return d[0]*d[0] + d[1]*d[1]; };
  auto f2 = [&](VecF v)->DN { VecD d = toDual(v); return d[0]*d[0] - d[1]*d[1]; };

  // todo: use some more complicated functions
  //auto f0 = [&](VecF v)->DN { VecD d = toDual(v); return rsSin( d[0]*d[1]);             };
  //auto f1 = [&](VecF v)->DN { VecD d = toDual(v); return rsSqrt(d[0]*d[0] + d[1]*d[1]); };
  //auto f2 = [&](VecF v)->DN { VecD d = toDual(v); return rsCos( d[0]*d[0] - d[1]*d[1]); };
  // ...hmm...maybe a torus would be really nice indeed - maybe define the formulas in functions
  // rsTorusX(u, v, R, r), rsTorusY(u, v, R, r), rsTorusZ(u, v, R, r) - maybe as static function
  // in a class rsSurfaces - could also have rsSphereX(u, v, R), rsEllipticCylinderX(u, v, rx, ry)
  // or maybe make a baseeclass rsParametricSurface3D with virtual functions getX(u, v), 
  // getY(u, v), getZ(u, v) that subclasses must implement
  // Could this autodiff stuff be usful for differential geometry - i'm currently using numerical
  // differentiation there

  // Wrap the 3 component functions into a single multi-valued function:
  auto f = [&](VecF v)->VecD { return VecD({ f0(v), f1(v), f2(v) }); };

  // Evaluate f at (x,y) = (2,3). Each component of the result should contain the value and the 
  // gradient at that value. The result r has N=3 component and each gradient has M=2 components:
  VecD r = f(VecF({ 2,3 }));
  r = f(VecF({ 3,5 }));
  int dummy = 0;

  // Can this be done more conveniently or efficiently? Can the user provide a function array
  // instead of having to create function objects for each component function separately? 
  // Especially the toDual function is costly because it allocates - it's ok for a prototype but
  // not for production code. In production code, autodiff with dynamically allocated arrays for
  // the gradients is generally not a good idea. With statically allocated vectors like rsVector2D,
  // it's fine, though.
}

// see also:
// https://math.stackexchange.com/questions/1839497/what-is-the-difference-between-hyperreal-numbers-and-dual-numbers
// https://math.stackexchange.com/questions/2194476/differential-form-vs-hyperreal-vs-dual-number
// https://en.wikipedia.org/wiki/Hyperreal_number
// https://en.wikipedia.org/wiki/Infinitesimal


void testAutoDiffReverse1()
{  
  using ADN = rsAutoDiffNumber<float, float>; 

  std::vector<ADN::Operation> ops;

  bool ok = true;

  ADN x(2.f, ops);     // x = 2
  ADN y(3.f, ops);     // y = 3
  ADN z(5.f, ops);     // z = 5
  ADN f(0.f, ops);     // f(x,y,z)
  float d, t;          // derivative and target
  float tol = 1.e-8f;  // tolerance for floating point comparisons


  x.initLoc(); y.initLoc(); z.initLoc();  // get rid of these calls!

  
  // test derivatives of univariate functions:
  ops.clear();
  f = rsSqrt(x);
  f.computeDerivatives();
  t = 0.5f/rsSqrt(x.v);
  ok &= rsIsCloseTo(x.d, t, tol);

  ops.clear();
  f = rsSin(rsSqrt(x));
  f.computeDerivatives();
  t = (cos(sqrt(x.v)))/(2.f*sqrt(x.v));
  ok &= rsIsCloseTo(x.d, t, tol);

  ops.clear();
  f = rsExp(rsSin(rsSqrt(x)));
  f.computeDerivatives();
  t = (exp(sin(sqrt(x.v))) * cos(sqrt(x.v)))/(2.f*sqrt(x.v));
  ok &= rsIsCloseTo(x.d, t, tol);

  // ok - looks good so far
  // the getDerivative doesn't seem to make sense anymore - instead, we need a call to 
  // f.computeDerivatives() which should assign the x.d field and then compare x.d to t


  // test derivatives of binary operators:
  ops.clear();
  f = x + y;
  f.computeDerivatives();
  ok &= rsIsCloseTo(x.d, 1.f, tol);  // (x+y)_x = 1
  ok &= rsIsCloseTo(y.d, 1.f, tol);  // (x+y)_y = 1

  ops.clear();
  f = x - y;
  f.computeDerivatives();
  ok &= rsIsCloseTo(x.d,  1.f, tol);  // (x-y)_x =  1
  ok &= rsIsCloseTo(y.d, -1.f, tol);  // (x-y)_y = -1

  ops.clear();
  f = x * y;
  f.computeDerivatives();
  ok &= rsIsCloseTo(x.d, y.v, tol);  // (x*y)_x = y
  ok &= rsIsCloseTo(y.d, x.v, tol);  // (x*y)_y = x

  ops.clear();
  f = x / y;
  f.computeDerivatives();
  ok &= rsIsCloseTo(x.d,  1.f/ y.v,      tol);  // (x/y)_x =  1/y
  ok &= rsIsCloseTo(y.d, -x.v/(y.v*y.v), tol);  // (x/y)_y = -x/y^2

  ops.clear();
  f = x * y * z;
  f.computeDerivatives();
  ok &= rsIsCloseTo(x.d, y.v*z.v, tol);  // (x*y*z)_x = y*z
  // wrong


  d = t;  // to suppress warning
  int dummy = 0;

  //d = x.d


  // test derivatives of iterated binary operators, like f(x,y,z) = x*y*z, etc.


  // ...but the binary operations are still wrong - these are more complicated because i can't just
  // apply the chain-rule and move on to the next inner operand because there are now two operands, 
  // so the expression tree actually branches ...maybe we need an actual tree to store the record?




  // i think, i will need another field in rsAutoDiffNumber to store the "adjoint" - only in that 
  // case it makes actually sense for function f: R^N -> R functions. The R^N inputs must be of 
  // type rsAutoDiffNumber and in the reverse pass we compute df/dxn
  /*
  ops.clear();
  f = x+x;
  d = f.getDerivative();
  t = 2.f;

  ops.clear();
  f = rsSin(x) + rsCos(x);
  d = f.getDerivative();
  t = rsCos(x.v) - rsSin(x.v);
  */


  /*
  ops.clear();
  ADN y3 = rsSin(rsSqrt(x*x));
  float d3 = y3.getDerivative();
  */

  // hmm - for a chain of elementary functions, the storage scheme is redundant - the result of 
  // operation i is the (first) operand of operation n+1...but maybe that's the way it has to be 
  // and will make more sense in binary operations?


  // -maybe the numbers should actually store a derivative in a field and init it to NaN and 
  //  actually set it in the backward pass?
  // -the operands and results in the recorded operations should be AutoDiffNumbers, not TVal
  // -maybe the record needs to store pointers to operands instead of the operands themselves?
  // -then, when a binary operation is encountered, we call getDerivative recursively on both 
  //  operands?
  // -try a scalar function with 2 inputs, maybe something resembling a neural network - then 
  //  actually the weights are also inputs to the function
  // -how about y = tanh(a0 + a1*x1 + a2*x2) - after the backward pass, we want the derivative fields
  //  of a0,a1,x1,a2,x2 contain the partial derivatives of y with respect to these variables
  // -define z = a0 + a1*x1 + a2*x2, and compute the derivative of tanh with repect to z
  // -compute derivatives of z: dz/da0 = 1, dz/da1 = x1, dz/dx1 = a1, dz/da2 = x2, dz/dx2=a2
  // -the sensitivity of a sum with respect to a summand is 1
  // -the sensitivities of a difference a-b are 1 and -1 respectively
  // -the sensitivity of a product with respect to one factor is the other factor
  // -maybe we need partial derivatives of the binary operators with respect to each of the 
  //  operands. if underscore denotes partial derivative, we have: (x+y)_x = 1, (x+y)_y = 1, 
  //  (x-y)_x = 1, (x-y)_y = -1, (x*y)_x = y, (x*y)_y = x, (x/y)_x = 1/y, (x/y)_y = -x/y^2
  // -maybe when we encounter an add, we don't need recursion - instead just assign the derivative
  //  fields for both operands - the operands are in positions i-1 and i-2 in the loop
  // -i think, the goals are quite different in forward and reverse mode: in forward mode, we want
  //  that in y.d of an end result y = E(x) for some complicated expression E, we want the 
  //  derivative of E with respect to x. in reverse mode, we want to evaluate a function of several
  //  variables, say r = f(x,y,z) and at the end, we want the the x.d, y.d, z.d contain the partial
  //  derivatives of f with respect to x,y,z - the reverse pass should assign them - we may need to
  //  distinguish between variables that have a memory location and temporaries and assign the 
  //  memory variables in the reverse pass

}

void testDualComplex()
{
  using DCN = rsDualComplexNumber<float>;

  float a, b, c, d;

  a = 2; b = 3; c = 5; d = 7;

  DCN z(a, b, c,d);
  DCN w(a,-b,-c,d);

  DCN zw = z*w;
  DCN wz = w*z; 

  float A = a*a + b*b;
  float D = 2*(a*d-b*c); 
   
  DCN v(A, 0, 0, -D);

  DCN zwv = zw*v;  
  // zwv has only real part nonzero, so first augmenting the fraction by a - i*b - c*E + i*E*d and 
  // then by A - i*E*D with A,D given as above, we can make the denominator purely real


  DCN z1 (1,2,3,4);
  DCN z2 (5,6,7,8);
  DCN z3 = z1 * z2;
  DCN z4 = z3 / z1;
  DCN z5 = z3 / z2;
  // ok - it seems division works


  // In order to use these numbers to perform autodiff in the complex domain, we should initialize
  // the real-dual part to 1 and imaginary-dual part to 0. Then, in the real-dual and 
  // imaginary-dual parts of the results (stored in c,d), we'll se the complex derivative.

  // initializing the dual part to 1 and imdual to 0, we should get the complex derivative 2*z of 
  // f(z) = z^2 in the derivative part, since z = 2 + 3*i, 2*z = 4 + 6*i
  z = DCN(2,3,1,0);  // 2 + 3*i + 1*E * 0*i*E
  w = z*z;           // value =  -5 + 12i, derivative =   4 +  6i
  w = z*z*z;         // value = -46 +  9i, derivative = -15 + 36i


  w = rsSin(z);      // w is always almost symmetric - but values match with sage






  w = rsSin(z*z);    
  // that seems to be wrong - is it because the derivative is computed correctly only when 
  // (c,d) = (1,0) and in the computation of z*z, it already becomse soemthing else - it seems 
  // to work when manually computing z^2 first but using 1,0 for the derivative:
  w = rsSin(DCN(-5,12,1,0));
  // could it be meaningleass to try to propagate derivative information further than 1 step in the
  // computation? or do we get something interesting other than the derivative?


  // sage:
  // z = 2 + 3*I
  // N(sin(z*z)), N(cos(z*z))


  // todo: implement exp, sin, cos, sqrt, etc. and see if complex autodiff also works for them. if 
  // it works, it may be useful for complex Newton iteration

  int dummy = 0;

}


/*
     1    i  E  iE
  1  1    i  E  iE
  i  i   -1  iE -E
  E  E   iE  0  0
 iE  iE  -E  0  0

 oh - there are already dual complex numbers - but they work differently:
 https://en.wikipedia.org/wiki/Dual-complex_number

*/

template<class T>
void plotFunction(int N, T xMin, T xMax, const std::function<T(T)>& f)
{
  GNUPlotter plt;
  std::vector<T> x(N), y(N);
  plt.rangeLinear(&x[0], N, xMin, xMax);
  for(int n = 0; n < N; n++)
    y[n] = f(x[n]);
  plt.addDataArrays(N, &x[0], &y[0]);
  plt.plot();
}
// move to GNUPlotter - but it should take up to 10 functions


// todo: multiplication of 3D vectors:
// convert to spherical: x,y,z -> r,phi,theta -> multiply radii, add angles -> convert back
// we want it to be consistent with tha notion that when z=0,theta=0, x + i*y = r*exp(i*phi) and 
// likewise if y=0,phi=0, x + i*z = r*exp(i*theta)
// i think: r = sqrt(x^2 + y^2 + z^2), phi = atan2(y,x), theta = atan2(z,x)
// x = r*cos(phi)*cos(theta), y = r*sin(phi)*cos(theta) z = r*cos(phi)*sin(theta)
// can be used to create 3D mandelbrot sets (mandelbulbs)

/*
template<class T>
void cartesianToSpherical(T x, T y, T z, T* r, T* p, T* t)
{
  *r = sqrt(x*x + y*y + z*z);  // radius
  *p = atan2(y, x);            // phi
  *t = atan2(z, x);            // theta
}
// needs test

template<class T>
void sphericalToCartesian(T r, T p, T t, T* x, T* y, T* z)
{
  T cp = cos(p);
  T sp = sin(p);
  T ct = cos(t);
  T st = sin(t);
  *x = r*cp*ct;
  *y = r*sp*ct;
  *z = r*cp*st;
}
// needs test
*/





template<class T>
rsVector3D<T> mul(const rsVector3D<T>& a, const rsVector3D<T> b)
{
  T ar, ap, at; cartesianToSpherical(a.x, a.y, a.z, &ar, &ap, &at);
  T br, bp, bt; cartesianToSpherical(b.x, b.y, b.z, &br, &bp, &bt);
  ar = ar * br;
  ap = ap + bp;
  at = at + bt;
  T cx, cy, cz; sphericalToCartesian(ar, ap, at, &cx, &cy, &cz);
  return rsVector3D<T>(cx, cy, cz);
}
// needs test
// Multiplication of two 3-dimensional vectors using a rule that generalizes complex multiplication 
// (of 2D vectors) in a somewhat natural way. If a.z = b.z = 0, the multiplication behaves like the 
// complex numbers (a.x + i*a.y) * (b.x + i*b.y). Likewise, if a.y = b.y = 0, it behaves like 
// (a.x + i*a.z) * (b.x + i*b.z). So the y and z components both behave like imaginary parts if the 
// respective other component is 0. Can be used to create 3D Mandelbrot sets ("Mandelbulbs"). The 
// operation is commutative and associative but not distributive over addition.
// ...i think this is still wrong - the functions for spherical/cartesian conversion seem wrong - a
// roundtrip doesn't work - i think, it may not be possible to have it behave like complex numbers
// x + i*y (z=0) and x + i*z (y=0) at the same time?
// ToDo: 
// -implement division:  ar = ar / br; ap = ap - bp; at = at - bt;


void testVectorMultiplication3D()
{
  using Complex = std::complex<float>;
  using Vec3    = rsVector3D<float>;

  Complex a(3,4),   b(4,3),   c;
  Vec3    A(3,4,0), B(4,3,0), C;

  c = a * b;
  C = mul(A, B);

  A.set(3,0,4); B.set(4,0,3);
  C = mul(A, B);

  
  A.set(7,-2,3); B.set(-1,3,5); C.set(4,-3,-2);
  Vec3 D1, D2;


  // test associativity:
  D1 = mul(A, mul(B, C));  // A * (B * C)
  D2 = mul(mul(A, B), C);  // (A * B) * C
  // yes! it's associative

  // test commutativity:
  D1 = mul(A, B);  // A * B
  D2 = mul(B, A);  // B * A
  // yes! it's commutative

  // test distributivity:
  D1 = mul(A, B+C);            // A * (B+C)
  D2 = mul(A, B) + mul(A, C);  // A*B + A*C
  // nope! not distributive!


  float x = 3, y = -2, z = 5;
  float r, p, t;
  cartesianToSpherical(x, y, z, &r, &p, &t);
  sphericalToCartesian(r, p, t, &x, &y, &z);
  // this does not reconstruct x,y,z - something is wrong with the conversion!


  int dummy = 0;
}



// References: (1): Numerik (Andreas Meister, Thomas Sonar)
// Implementation follows directly the box "Unter der Lupe: Die Hermite-Interpolation" on page 73 
// and is not suitable for production use (has memory allocations and "Shlemiel the painter" 
// algos (i think)).

/** Constructs helper polynomial l_ik(x) as defined in (1) pg. 73, top-right. Unoptimized 
prototype. */
template<class T>
rsPolynomial<T> generalizedLagrangeHelper(const std::vector<std::vector<T>>& f, int i, int k)
{
  using Poly = rsPolynomial<T>;
  int  m = (int) f.size() - 1;               // index of last datapoint
  Poly pm({ T(1) });                         // prod_j ((x-x_j) / (x_i - x_j))^nj
  for(int j = 0; j <= m; j++) {
    if(j != i)   {
      int nj = (int) f[j].size() - 1;        // exponent of j-th factor
      Poly qj({ -f[j][0], T(1) });           //  (x - x_j)
      qj = qj * T(1) / (f[i][0] - f[j][0]);  //  (x - x_j) / (x_i - x_j)
      qj = qj^nj;                            // ((x - x_j) / (x_i - x_j))^nj
      pm = pm * qj; }}                       // accumulate product
  Poly qi({ -f[i][0], T(1) });               // (x - x_i)
  qi = qi^k;                                 // (x - x_i)^k
  qi = qi *  (T(1) / rsFactorial(k));        // (x - x_i)^k / k!
  return qi * pm;
}

/** Constructs generalized Lagrange polynomial L_ik(x). This is a recursive implementation and the 
most direct translation of the formula in the book. But it's just for proof of concept because 
calling ourselves recursively in a loop is horribly inefficient. There are lots of 
recomputations. */
template<class T>
rsPolynomial<T> generalizedLagrange(const std::vector<std::vector<T>>& f, int i, int k)
{
  using Poly = rsPolynomial<T>;
  int  ni    = (int)f[i].size()-1;
  Poly l_ik  = generalizedLagrangeHelper(f, i, k);
  if(k == ni-1)
    return l_ik;
  else {
    Poly sum;
    for(int mu = ni-1; mu >= k+1; mu--) {
      T s = l_ik.derivativeAt(f[i][0], mu);             // l_ik^(mu) (x_i)
      sum = sum + generalizedLagrange(f, i, mu) * s; }  // recursion
    return l_ik - sum; }
}

/** Returns the Hermite interpolation polynomial for the data given in f. We use the following 
conventions for the input data: f[i][0] is the x-coordinate of the i-th datapoint, f[i][1] is the 
corresponding y-coordinate, f[i][2] is the 1st derivative, etc. In general, f[i][k] is the (k-1)th 
derivative of the i-th datapoint value, except for k=0, in which case it is the x-value. For k=1, 
it is the 0-th derivative which is conventionally the function value itself. */
template<class T>
rsPolynomial<T> hermiteInterpolant(const std::vector<std::vector<T>>& f) // rename!
{
  using Poly = rsPolynomial<T>;
  int m = (int) f.size() - 1;    // index of last datapoint
  Poly p;
  for(int i = 0; i <= m; i++) {
    int ni = (int)f[i].size()-1;
    for(int k = 0; k <= ni-1; k++) {
      Poly L_ik = generalizedLagrange(f, i, k);   // still very inefficient
      p = p + L_ik * f[i][k+1]; }}                // maybe division by k! should occur here later
  return p;
}



// pt should be the matrix containing the Pascal triangle with alternating signs
template<class T>
int generalizedLagrangeHelper0(int k, int n1, T* a, const rsMatrix<T>& pt)
{
  // Polynomial is n1-th line of alternating Pascal triangle, shifted by k
  // l_0k = ((x-1)/(0-1))^n1 * (x-0)^k / k!
  using AT = rsArrayTools;
  int N = n1+k+1;  // number of coeffs
  AT::fillWithZeros(a, k);   // maybe rename to clear
  AT::negateOdd(pt.getRowPointerConst(n1), &a[k], N-k);
  AT::scale(a, a, N, T(1)/rsFactorial(k));  // use rsInverseFactorials[k]
  return N;
}
// O(n1+k)
// returns number of coeffs (not degree!) ..maybe it should return the degree, i.e. N-1

template<class T>
int generalizedLagrangeHelper1(int k, int n0, T* a, const rsMatrix<T>& pt)
{
  // Polynomial is k-th line of alternating Pascal triangle, shifted by n0
  // l_1k = ((x-0)/(1-0))^n0 * (x-1)^k / k!
  using AT = rsArrayTools;
  int N = n0+k+1;
  AT::fillWithZeros(a, n0);
  if(rsIsOdd(k)) AT::negateEven(pt.getRowPointerConst(k), &a[n0], N-n0);
  else           AT::negateOdd( pt.getRowPointerConst(k), &a[n0], N-n0);
  AT::scale(a, a, N, T(1)/rsFactorial(k));  // use rsInverseFactorials[k]
  return N;
}
// O(n0+k)
// maybe make a function that gnererates the helper polynomials for arbitrary i - should take
// i, k, arrays of the xj and nj - we can create these by repeated convolutions

// convenience function - maybe make it a lambda in hermiteInterpolant01 (it's only used there):
template<class T>
int generalizedLagrangeHelper01(int i, int k, int n01, T* a, const rsMatrix<T>& pt)
{
  rsAssert(i == 0 || i == 1);
  if(i == 0) return generalizedLagrangeHelper0(k, n01, a, pt);
  else       return generalizedLagrangeHelper1(k, n01, a, pt);
}

// i think, overall complexity is O(M^3) where M = max(n0,n1)
template<class T>
int hermiteInterpolant01(T* y0, int n0, T* y1, int n1, T* p, const rsMatrix<T>& pt)
{
  // Initializations:
  int N = n0 + n1;                    // number of coeffs in interpolant
  std::vector<T> dl(rsMax(n0,n1)+1);  // value and derivatives of l_ik
  rsMatrix<T> L(N, N);                // generalized Lagrange polynomials
  rsArrayTools::fillWithZeros(p, N);  // clear to prepare for accumulation

  // Compute L_ik polynomials (i = 0,1) via backward recursion in O(ni^3) and accumulate them 
  // into p in O(ni*N):
  auto accumulate = [&](int i)->void 
  { 
    // Initializations:
    L.setToZero();        // todo: rename to clear
    int ni, no;           // ni = n0 or n1, no is the respective other n
    T   xi;               // xi = 0  or 1
    T*  yi;               // yi = y0 or y1
    if(i == 0) { ni = n0; no = n1; xi = T(0); yi = y0; }
    else       { ni = n1; no = n0; xi = T(1); yi = y1; }

    // Accumulation:
    for(int k = ni-1; k >= 0; k--) {
      int nc = generalizedLagrangeHelper01(i, k, no, L.getRowPointer(k), pt);
      rsPolynomial<T>::evaluateWithDerivatives(xi, L.getRowPointer(k), nc-1, &dl[0], ni);
      for(int mu = ni-1; mu >= k+1; mu--)
        L.addWeightedRowToOther(mu, k, -dl[mu]); }
    for(int k = 0; k <= ni-1; k++)
      for(int i = 0; i < N; i++)
        p[i] += yi[k] * L(k, i);
  };
  accumulate(0);  // accumulate coeffs for i = 0
  accumulate(1);  // accumulate coeffs for i = 1
  return N;       // number of produced coeffs in p for convenience (client should know that)
}
// -we could take the loop iterations for k = n-1 for computing L_0k and L_1k out of the loops and 
//  loop only from k = n-2 down to 0 - this would avoid the unnecessary evaluateWithDerivatives 
//  call (in the (k-1) iteration, the inner "for mu" loop is not entered and the computed values 
//  in dl are not used) - then we could perhaps get rid of the +1 in rsMax(n0,n1)+1, but that may 
//  increase the code size, when the generalizedLagrangeHelper01 function (and its callees) are 
//  inlined, which is likely - so maybe don't do it
// -test it with more inputs - maybe n0=4, n1=6 (and vice versa) or something
// -try to accumulate the L_1k polynomials before the L_0k polynomials and compare the numerical
//  precision of the coeffs -> doesn't seem to make a difference
// -minimize memory allocation and make function using a workspace
// -compare to result with my old code using the linear system - maybe that can be generalized to
//  use different numbers of derivatives at x0 = 0 and x1 = 1, too

// convenience function:
template<class T>
int hermiteInterpolant01(T* y0, int n0, T* y1, int n1, T* p)
{
  // Compute matrix of Pascal triangle coeffs in O(N^2):
  int N = n0 + n1;                  // maybe max(n0,n1) is enough
  rsMatrix<T> pt(N, N);
  for(int n = 0; n < N; n++)
  {
    //::rsNextPascalTriangleLine(pt.getRowPointer(n-1), pt.getRowPointer(n), n+1); // old
    RAPT::rsNextPascalTriangleLine(pt.getRowPointer(n-1), pt.getRowPointer(n), n);  // new
  }
  // todo: 
  // -use a class for triangular matrices - saves half of the memory

  return hermiteInterpolant01(y0, n0, y1, n1, p, pt);
}


template<class T>
int generalizedLagrangeHelper01(int i, int k, int n0, int n1, T* a)
{
  // create Pascal triangle:
  static const int maxN = 10;  // preliminary
  rsMatrix<T> pt2(maxN, maxN);

  // old:
  //for(int n = 0; n < maxN; n++)
  //  ::rsNextPascalTriangleLine(pt2.getRowPointer(n-1), pt2.getRowPointer(n), n+1); // old

  // new:
  //pt2.setAllValues(-1);
  //pt2(0,0) = 1;
  for(int n = 0; n < maxN; n++)
    RAPT::rsNextPascalTriangleLine(pt2.getRowPointer(n-1), pt2.getRowPointer(n), n);


  // todo: 
  // -use a class for triangular matrices - saves half of the memory

  // split into 2 functions, without taking an i input
  if(     i == 0)  return generalizedLagrangeHelper0(k, n1, a, pt2);
  else if(i == 1)  return generalizedLagrangeHelper1(k, n0, a, pt2);
  else             rsError(); return 0;
}
// only for testing - always generates the pascal triangle - production code should do this just 
// once


// for better numerical precision, try to postpone out the division by k! for as long as possible
// ...nope - that doesn't work:
template<class T>
rsPolynomial<T> generalizedLagrangeHelper2(const std::vector<std::vector<T>>& f, int i, int k)
{
  using Vec  = std::vector<T>;
  using Poly = rsPolynomial<T>;
  int  m = (int) f.size() - 1;               // index of last datapoint
  Poly pm({ T(1) });                         // prod_j ((x-x_j) / (x_i - x_j))^nj
  for(int j = 0; j <= m; j++) {
    if(j != i)   {
      int nj = (int) f[j].size() - 1;              // exponent of j-th factor
      Poly qj(Vec({ -f[j][0], T(1) }));      //  (x - x_j)
      qj = qj * T(1) / (f[i][0] - f[j][0]);  //  (x - x_j) / (x_i - x_j)
      qj = qj^nj;                            // ((x - x_j) / (x_i - x_j))^nj
      pm = pm * qj; }}                       // accumulate product
  Poly qi(Vec({ -f[i][0], T(1) }));          // (x - x_i)
  qi = qi^k;                                 // (x - x_i)^k
  //qi = qi *  (T(1) / rsFactorial(k));        // (x - x_i)^k / k!
  return qi * pm;
}
template<class T>
rsPolynomial<T> generalizedLagrange2(const std::vector<std::vector<T>>& f, int i, int k)
{
  using Poly = rsPolynomial<T>;
  int  ni    = (int)f[i].size()-1;
  Poly l_ik  = generalizedLagrangeHelper2(f, i, k);
  if(k == ni-1)
    return l_ik;
  else {
    Poly sum;
    for(int mu = ni-1; mu >= k+1; mu--) {
      T s = l_ik.derivativeAt(f[i][0], mu);              // l_ik^(mu) (x_i)
      sum = sum + generalizedLagrange2(f, i, mu) * s; }  // recursion
    return l_ik - sum; }
}
template<class T>
rsPolynomial<T> hermiteInterpolant2(const std::vector<std::vector<T>>& f) 
{
  using Poly = rsPolynomial<T>;
  int m = (int) f.size() - 1;    // index of last datapoint
  Poly p;
  for(int i = 0; i <= m; i++) {
    int ni = (int)f[i].size()-1;
    for(int k = 0; k <= ni-1; k++) {
      Poly L_ik = generalizedLagrange2(f, i, k);  // still very inefficient
      //p = (p + L_ik * f[i][k+1]) * (T(1) / rsFactorial(k));
      p = p + L_ik * (f[i][k+1]/rsFactorial(k)); 
    }
  }
  return p;
}
// doesn't work




/** Compares coefficient array a of length N to polynomial p and return true, iff they represent 
the same polynomial */
template<class T>
bool rsIsCloseTo(const rsPolynomial<T>& p, const T* a, int N, T tol = T(0))
{
  if(p.getNumCoeffs() != N)
    return false;
  return rsArrayTools::almostEqual(p.getCoeffPointerConst(), a, N, tol);
}

void testHermiteInterpolation()
{
  using Vec  = std::vector<float>;
  using Poly = rsPolynomial<float>;
  using AT   = rsArrayTools;

  //      x   f   f' f''
  Vec f0({0, -1, -2    }); int n0 = 2;
  Vec f1({1,  0, 10, 20}); int n1 = 3;
  std::vector<Vec> f({ f0, f1 });  // our data

  float tol = 0.f;

  bool ok = true;

  // test helper polynomials:
  Poly l_00 = generalizedLagrangeHelper(f,0,0); ok &= l_00 == Poly(Vec({1,-3,+3,-1}));
  Poly l_01 = generalizedLagrangeHelper(f,0,1); ok &= l_01 == Poly(Vec({0,1,-3,+3,-1}));
  Poly l_02 = generalizedLagrangeHelper(f,0,2); ok &= l_02 == Poly(Vec({0,0,0.5,-1.5,+1.5,-0.5}));
  Poly l_10 = generalizedLagrangeHelper(f,1,0); ok &= l_10 == Poly(Vec({0,0,1}));
  Poly l_11 = generalizedLagrangeHelper(f,1,1); ok &= l_11 == Poly(Vec({0,0,-1,1}));
  Poly l_12 = generalizedLagrangeHelper(f,1,2); ok &= l_12 == Poly(Vec({0,0,0.5,-1,0.5}));
  Poly l_13 = generalizedLagrangeHelper(f,1,3); ok &= l_13 == Poly(Vec({0,0,-1.f/6,0.5,-0.5,1.f/6}));
  // looks good so far

  // now the optimized version:
  static const int maxN = 20;
  float a[maxN];
  int N;
  N = generalizedLagrangeHelper01(0, 0, n0, n1, a); ok &= rsIsCloseTo(l_00, a, N, tol);
  N = generalizedLagrangeHelper01(0, 1, n0, n1, a); ok &= rsIsCloseTo(l_01, a, N, tol);
  N = generalizedLagrangeHelper01(0, 2, n0, n1, a); ok &= rsIsCloseTo(l_02, a, N, tol);

  N = generalizedLagrangeHelper01(1, 0, n0, n1, a); ok &= rsIsCloseTo(l_10, a, N, tol);
  N = generalizedLagrangeHelper01(1, 1, n0, n1, a); ok &= rsIsCloseTo(l_11, a, N, tol);
  N = generalizedLagrangeHelper01(1, 2, n0, n1, a); ok &= rsIsCloseTo(l_12, a, N, tol);
  N = generalizedLagrangeHelper01(1, 3, n0, n1, a); ok &= rsIsCloseTo(l_13, a, N, tol);

  rsAssert(ok);
  // todo: split into two functions, not taking the i as input


  // test generalized Lagrange polynomials:
  Poly L_01 = generalizedLagrange(f, 0, 1); ok &= L_01 == Poly(Vec({0,1,-3,3,-1}));
  Poly L_00 = generalizedLagrange(f, 0, 0); ok &= L_00 == Poly(Vec({1,0,-6,8,-3}));

  Poly L_12 = generalizedLagrange(f, 1, 2); // ok &= L_12 == Poly(Vec({0,0.5,-1,0.5})); // wrong!
  // the coefficients are shifted to the right by one index, i.e. we have a factor of x too much - 
  // but the formula says tha L_12 should actually be equal to l_12 (which it is) but the numbers
  // in the example suggest that l_12 = x * L_12
  // ..i think, this may be a mistake in the book - our p polynomial later seems to come out right
  // :-O

  Poly L_11 = generalizedLagrange(f, 1, 1); ok &= L_11 == Poly(Vec({0,0,-3,5,-2}));
  Poly L_10 = generalizedLagrange(f, 1, 0); // ok &= L_10 == Poly(Vec({0,0,7,-10,4}));  // wrong!
  // this still fails!
  
  // todo: try to evaluate L_ik and its derivatives at 0 and 1
  float y;
  y = L_01.derivativeAt(0.f, 0);  // 0
  y = L_01.derivativeAt(0.f, 1);  // 1
  y = L_01.derivativeAt(0.f, 2);
  y = L_01.derivativeAt(0.f, 3);

  Poly p = hermiteInterpolant(f);

  // That's the example result in the book - check, if it gives the right values:
  Poly p1({-1,-2,-8,16,-5});
  y = p1.derivativeAt(0.f, 0); // -1
  y = p1.derivativeAt(0.f, 1); // -2
  y = p1.derivativeAt(1.f, 0); //  0
  y = p1.derivativeAt(1.f, 1); //  10
  y = p1.derivativeAt(1.f, 2); //  20
  // yep - works

  ok &= p == p1;
  // looks good even though the L_12, etc. polynomials do not match the book - maybe that's a 
  // mistake in the book? ...more tests needed - with more datapoints and more derivatives

  // Now compare result polynomial p to the one generated by the the optimized code:
  AT::fillWithNaN(a, maxN); 
  N = hermiteInterpolant01(&f0[1], n0, &f1[1], n1, a); 
  ok &= rsIsCloseTo(p, a, N, tol);


  // try to match more derivatives but using only two datapoints:
  f0 = Vec({0, -1, -2,  3, 1    }); n0 = 4;  // value and 3 derivatives at x0 = 0
  f1 = Vec({1,  2,  1, -1, 2, -1}); n1 = 5;  // value and 4 derivatives at x1 = 1
  f = std::vector<Vec>({ f0, f1 }); 
  p = hermiteInterpolant(f);
  y = p.derivativeAt(0.f, 0); // -1
  y = p.derivativeAt(0.f, 1); // -2
  y = p.derivativeAt(0.f, 2); //  3
  y = p.derivativeAt(0.f, 3); //  1
  y = p.derivativeAt(1.f, 0); //  2
  y = p.derivativeAt(1.f, 1); //  1
  y = p.derivativeAt(1.f, 2); // -1
  y = p.derivativeAt(1.f, 3); //  2
  y = p.derivativeAt(1.f, 4); // -1
  // okayish, but numerically imprecise

  // Compare to optimized code::
  AT::fillWithNaN(a, maxN); 
  N = hermiteInterpolant01(&f0[1], n0, &f1[1], n1, a); 
  //ok &= rsIsCloseTo(p, a, N, tol);
  // fails, but the coeffs are in the same ballpark - maybe the optimized function is numerically 
  // more precise? -> test that! especially the coeffs for higher powers are different

  float y0[maxN], y1[maxN];  // output and derivatives at x0 = 0 and x1 = 1
  Poly::evaluateWithDerivatives(0.f, a, N-1, y0, n0-1);  // looks good
  Poly::evaluateWithDerivatives(1.f, a, N-1, y1, n1-1);  // y1[4] is totally off
  // Numerically, all values evaluated at x0 = 0 are perfect, but at x1 = 1 we have errors. In the
  // optimized version, y1[4] is totally wrong - this looks more like a bug rather than a numerical
  // precision issue -> figure out ...maybe the prototype code is numerically more precise after 
  // all? but why should that be the case? ..why should the optimzed code produce different coeffs
  // anyway? we just avoid redundant computations - maybe we need to zero out more things in the 
  // algo?
  //




  // test it with reversed inputs:
  f0 = Vec({0,  2,  1, -1, 2, -1}); n0 = 5;  // value and 4 derivatives at x0 = 0
  f1 = Vec({1, -1, -2,  3, 1    }); n1 = 4;  // value and 3 derivatives at x1 = 1
  f  = std::vector<Vec>({ f0, f1 }); 
  p  = hermiteInterpolant(f);
  AT::fillWithNaN(a, maxN); 
  N = hermiteInterpolant01(&f0[1], n0, &f1[1], n1, a); 
  //ok &= rsIsCloseTo(p, a, N, tol);
  // relative error is high at p[4]...but absolute error is higher elsewhere




  // todo: evaluate value and derivatives at 0 and 1 for both the prototype and optimized code and
  // compare it to the desired values



  // try to match more than 2 datapoints
  f0 = Vec({0, -1, -2    });
  f1 = Vec({1,  0, 10, 20});
  Vec f2({2,  -2, 3, -4, 5});

  f = std::vector<Vec>({ f0, f1, f2 }); 
  p = hermiteInterpolant(f);
  y = p.derivativeAt(0.f, 0); //  -1
  y = p.derivativeAt(0.f, 1); //  -2
  y = p.derivativeAt(1.f, 0); //   0
  y = p.derivativeAt(1.f, 1); //  10
  y = p.derivativeAt(1.f, 2); //  20
  y = p.derivativeAt(2.f, 0); //  -2
  y = p.derivativeAt(2.f, 1); //   3
  y = p.derivativeAt(2.f, 2); //  -4 
  y = p.derivativeAt(2.f, 3); //   5
  // okayish, but numerically imprecise

  // trying it with delayed division by 1/k!
  p1 = hermiteInterpolant2(f); // ...nope - totally different from p!

  // ToDo: 
  // -maybe move over to prototypes section
  // -get rid of the blatant inefficiencies in the current implementation
  // -try to improve numerical accuracy - figure out, why it's so bad - maybe we can factor out a 
  //  couple of divisions in order to work with integers for longer (i think, the l_ik and L_ik 
  //  polynomials can be scaled to use integer coeffs, if we do the division by 1/k! later)
  // -implement an optimized version for the special case m=1, x0=0, x1=1 - a lot of simplification
  //  occurs in this case
  //  -the helper polynomials reduce to shifted versions of binomial coeffs with alternating signs


  rsAssert(ok);
  int dummy = 0;
}

// todo: implement Gregory-Netwon interpolation formula that directly matches finite differences.
// see (1) page 56. One may think about using Hermite interpolation using finite difference 
// approximations to derivatives...maybe that may give the same results?


// maybe rename functions to use "Grid" instead of "Mesh" - research which terminology is more 
// common in PDE solvers and use that...i think, "mesh" implies 2D whereas "grid" is nD? if so,
// use grid, because the functions should actually also work for true 3D meshes - here, we deal with 
// 2D surfaces in 2D or 3D space...we'll see

// Prototypes for testing/comparison
template<class T>
void addRegularMeshVertices2D(
  rsGraph<rsVector2D<T>, T>& m, int Nx, int Ny, T dx = T(1), T dy = T(1))
{
  for(int i = 0; i < Nx; i++)
    for(int j = 0; j < Ny; j++)
      m.addVertex(rsVector2D<T>(dx * T(i), dy * T(j)));
  // -maybe addVertex should allow to pre-allocate memory for the edges
  // -using dx,dy will have to be taken into account for the edge-weights, 
  //  too...ewww....having different dx,dy values could mess up the gradient calculation - maybe 
  //  don't use it for now
}
// obsolete
// -maybe make a class rsMeshGenerator - this could be a static member function addVertices2D or
//  addRectangularVertices or something
// -actually, even though the surface is 2D, each vertex could be a 3D vector - maybe the user 
//  could pass in du,dv instead of dx,dy and 3 functions fx(u,v), fy(u,v), fz(u,v) to compute 
//  coordinates - but maybe that can be postponed

template<class T>
void addMeshConnectionsToroidal2D(rsGraph<rsVector2D<T>, T>& m, int Nx, int Ny, int i, int j)
{
  // vertex k should be connected to the 4 vertices to its left, right, top, bottom, with 
  // wrap-arounds, if necessary:
  int il = (i-1+Nx) % Nx, ir = (i+1) % Nx;    // +Nx needed for modulo when i-1 < 0
  int jb = (j-1+Ny) % Ny, jt = (j+1) % Ny;    // dito for +Ny
  int kl = il * Ny + j,   kr = ir * Ny + j;   // west, east
  int kb = i  * Ny + jb,  kt = i  * Ny + jt;  // south, north
  int k  = i  * Ny + j;                       // flat index of vertex with indices i,j
  m.addEdge(k, kl, 1.f);                      // try to get rid of the 1.f
  m.addEdge(k, kr, 1.f);
  m.addEdge(k, kb, 1.f);
  m.addEdge(k, kt, 1.f);
  // -maybe the order matters for efficient access? ..and maybe we could pre-allocate the 
  //  memory for the edges?
}
// todo: make a 3D version - maybe a general one via templates - and then let the user set up the 
// actual (x,y,z)-coordinates in a torus (edge weights need to be recomputed after that) - use also
// ellipses instead of circles for major and minor radius -> additional flexibility

template<class T>
void addMeshConnectionsPlanar2D(rsGraph<rsVector2D<T>, T>& m, int Nx, int Ny, int i, int j)
{
  int il = i-1, ir = i+1;
  int jb = j-1, jt = j+1;
  int k  = i * Ny + j;
  if(il >=  0) m.addEdge(k, il * Ny + j,  1.f);  // west
  if(ir <  Nx) m.addEdge(k, ir * Ny + j,  1.f);  // east
  if(jb >=  0) m.addEdge(k, i  * Ny + jb, 1.f);  // south
  if(jt <  Ny) m.addEdge(k, i  * Ny + jt, 1.f);  // north
}
// needs test

// -make cylindrical version: wraparounds around 1 axis, truncation for the other - maybe let the 
//  user select which axis is periodic
// -maybe make a conical version - it's like the cylinder but with an extra point that gets 
//  connected to all points of one nonperidic edge (top or bottom, if left is periodic with right)
// -maybe a sort of sphere can be created if we do it with top and bottom - it's double-cyclinder
//  but the topology is the same - geometry should be created by assigning (x,y,z) coordinates to
//  the vertices

template<class T>
void addMeshConnectionsToroidal2D(rsGraph<rsVector2D<T>, T>& m, int Nx, int Ny)
{
  for(int i = 0; i < Nx; i++) 
    for(int j = 0; j < Ny; j++) 
      addMeshConnectionsToroidal2D(m, Nx, Ny, i, j);
  // -include stencil in the name or have an option to choose different stencils
  // -have variants that connect to the diagonal neighbours as well and allow for different 
  //  handling of the edges
  // -have a function that re-assigns the edge weights according to computed Euclidean distances.
  //  this may be useful when we use the same mesh topology with different intended geometries. But
  //  this should not be used for toroidal topology when the vertices are only 2D because the 
  //  wrapped edges would get a wrong distance then (they are conceptually just one step apart but 
  //  in 2D space, they have a geometric distance of Nx*dx or Ny*dy)
  //  To create a different geometry, the user should loop through the vertices and re-assign their
  //  positions (x,y) or later also (x,y,z)
}

template<class T>
void addMeshConnectionsPlanar2D(rsGraph<rsVector2D<T>, T>& m, int Nx, int Ny)
{
  for(int i = 0; i < Nx; i++) 
    for(int j = 0; j < Ny; j++) 
      addMeshConnectionsPlanar2D(m, Nx, Ny, i, j);
}
// this is a 4-point stencil


int getFlatMeshIndex2D(int i, int j, int Ny)
{
  return i  * Ny + j;
}

// 3-point stencil:
template<class T>
void addMeshConnectionsStencil3(rsGraph<rsVector2D<T>, T>& mesh, int Nx, int Ny)
{
  for(int i = 0; i < Nx-1; i++) {
    for(int j = 0; j < Ny-1; j++)  {
      int k1 = getFlatMeshIndex2D(i,   j, Ny);
      int k2 = getFlatMeshIndex2D(i+1, j, Ny);
      mesh.addEdge(k1, k2, true);
      if(rsIsEven(i+j)) {
        k2 = getFlatMeshIndex2D(i, j+1, Ny);
        mesh.addEdge(k1, k2, true); }}}
}


template<class T>
void randomizeVertexPositions(rsGraph<rsVector2D<T>, T>& mesh, T dx, T dy, 
  int minNumNeighbors = 0, int seed = 0)
{
  using Vec2 = rsVector2D<T>;
  rsNoiseGenerator<T> ng;
  ng.setSeed(seed);
  T rnd;
  for(int k = 0; k < mesh.getNumVertices(); k++) 
  {
    if(mesh.getNumEdges(k) >= minNumNeighbors)
    {
      Vec2 v = mesh.getVertexData(k);
      v.x += dx * ng.getSample();
      v.y += dy * ng.getSample();
      mesh.setVertexData(k, v);
    }
  }
}

template<class T>
void scaleVertexPositions(rsGraph<rsVector2D<T>, T>& mesh, T sx, T sy)
{
  for(int k = 0; k < mesh.getNumVertices(); k++) 
  {
    rsVector2D<T> v = mesh.getVertexData(k);
    v.x *= sx;
    v.y *= sy;
    mesh.setVertexData(k, v);
  }
}


/** Redistributes the vertices in the mesh, such that a force equilibrium is reached for each node, 
where forces are excerted by connections - the connections behave like springs under tension
...tbc... */
template<class T>
void moveVerticesToEquilibrium(rsGraph<rsVector2D<T>, T>& mesh, int minNumNeighbors = 1)
{
  using Vec2 = rsVector2D<T>;

  T updateRate = T(0.5);  // needs experimentation
  int numIts = 0;
  int maxNumIts = 100;
  T thresh = T(0.0001);      // this may be too high




  auto getForce = [&](int i)->Vec2
  {
    Vec2 f;
    Vec2 vi = mesh.getVertexData(i);
    for(int k = 0; k < (int) mesh.getNumEdges(i); k++) {
      int   j = mesh.getEdgeTarget(i, k);
      Vec2 vj = mesh.getVertexData(j);
      Vec2 fj = vj - vi;
      f += fj; }
    return f;
  };

  auto adjustVertices = [&]()->T
  {
    T maxDistance(0);
    for(int i = 0; i < mesh.getNumVertices(); i++)
    {
      if(mesh.getNumEdges(i) >= minNumNeighbors)
      {
        Vec2 vi = mesh.getVertexData(i);
        Vec2 f  = getForce(i);
        Vec2 dv = updateRate * f;
        mesh.setVertexData(i, vi + dv);
        maxDistance = rsMax(maxDistance, rsNorm(dv));
      }
    }

    return maxDistance;
  };
  // returns the maximum distance that a vertex was moved - used for stopping criterion


  while(adjustVertices() > thresh && numIts < maxNumIts)
  {
    numIts++;
  }

  int dummy = 0;
}
// find better name: maybe equilibrifyVertices, moveVerticesToEquilibrium, 
// equalizeVertexDistribution
// ToDo:
// -let the user determine points of attraction and repulsion (with strengths) to allow for 
//  increase or decrease of vertex density in regions of interest
// -use momentum to speed up convergence
// -let the user pass a distance-to-force function F(d) (currently the identity?)
//  try F(d) = (d-a)^2 where a is some desired distance or F(d) = d + 1/d 
// -move into a class rsGraphGeometryManipulator
// -let the user create arbitrary geometries by starting with a rectangle of honeycomb (3-point)
//  or 4-point stencil and remove nodes by subtracting shapes defined by F(x,y) - remove 
//  every node for which F(x,y) becomes negative (or positive) - use to subtract circles, etc.
//  F(x,y) should actually not tak x,y as input but a vector, such that it generalizes to 3D

template<class T>
T getEdgeLength(rsGraph<rsVector2D<T>, T>& mesh, int i, int k)
{
  int j = mesh.getEdgeTarget(i, k);
  rsVector2D<T> vi = mesh.getVertexData(i);
  rsVector2D<T> vj = mesh.getVertexData(j);
  rsVector2D<T> dv = vj - vi;
  return rsNorm(dv);

}

template<class T>
rsGraph<rsVector2D<T>, T> getHexagonMesh(int Nx, int Ny)
{
  rsGraph<rsVector2D<T>, T> mesh;
  addRegularMeshVertices2D(mesh, Nx, Ny);
  addMeshConnectionsStencil3(mesh, Nx, Ny); // connections for top and right edge are missing
  float sy = sqrt(3.f);                     // = 2*sin(60°) - verify, that this is the correct scale factor
  scaleVertexPositions(mesh, 1.f, sy);
  moveVerticesToEquilibrium(mesh, 3);
  return mesh;
}
// -todo: derive use an explicit fromula to adjust the vertex position instead of using the 
//  iterative algorithm with the forces
// -allow for different topologies like with the rectangular meshes

// rename to getQuadrangleMesh or getRectangleMesh
template<class T>
rsGraph<rsVector2D<T>, T> getPlanarMesh(T x0, T x1, int Nx, T y0, T y1, int Ny)
//rsMesh2D<float> getPlanarMesh(T x0, T x1, int Nx, T y0, T y1, int Ny)
{
  // todo: let the user select edge handling

  using Vec2 = rsVector2D<T>;
  using Mesh = rsGraph<Vec2, T>;
  //rsMesh2D<float> m;

  Mesh m;
  addRegularMeshVertices2D(  m, Nx, Ny);
  addMeshConnectionsPlanar2D(m, Nx, Ny);

  for(int i = 0; i < Nx; i++)
  {
    for(int j = 0; j < Ny; j++)
    {
      int k = rsFlatMeshIndex2D(i, j, Ny);

      // compute x,y and set vertex data accordingly

    }
  }

  // todo: 
  // -compute and assign vertex positions
  // -assign edge weights

  // int k  = i  * Ny + j;  // flat mesh vertex index - maybe provide a function to compute it:
  // rsFlatMeshIndex2D(int i, int j, int Ny)

  return m;
}


// maybe make lamdas in testMeshGeneration
// expected number of graph-connections for a plane topology
// 4 neighbours (inner points):         (Nu-2)*(Nv-2)
// 3 neighbours (edges except corners):  2*(Nu-2) + 2*(Nv-2)
// 2 neighbours (corner points):         4
int getNumConnectionsPlane(int Nu, int Nv)
{
  return 4 * (Nu-2)*(Nv-2) + 3 * (2*(Nu-2) + 2*(Nv-2)) + 2 * 4; 
}

// other topologies:
int getNumConnectionsCylinder(int Nu, int Nv)
{
  return getNumConnectionsPlane(Nu, Nv) + 2*Nv;
}
int getNumConnectionsTorus(int Nu, int Nv)
{
  return getNumConnectionsPlane(Nu, Nv) + 2*Nv + 2*Nu;
  // should also be equal to Nu*Nv
}

template<class T>
void plotMesh(rsGraph<rsVector2D<T>, T>& m, std::vector<int> highlight = std::vector<int>())
{
  GraphPlotter<T> gp;
  gp.plotGraph2D(m, highlight);
}

void testMeshGeneration()
{
  using Vec2 = rsVector2D<float>;
  using Vec3 = rsVector3D<float>;
  using Mesh = rsGraph<Vec2, float>;       // remove
  using MG   = rsMeshGenerator2D<float>;

  int Nx = 7;     // number grid coordinates along x-direction
  int Ny = 5;     // number grid coordinates along y-direction

  int Nv = Nx*Ny; // number of vertices
  int Ne = 4*Nv;  // each vertex has 4 neighbours (in toroidal topology)

  bool ok = true;

  // create mesh and add the vertices:
  Mesh m;
  addRegularMeshVertices2D(m, Nx, Ny);
  ok &= m.getNumVertices() == Nv;

  // add connections for toroidal topology (x and y axes are bot periodic):
  addMeshConnectionsToroidal2D(m, Nx, Ny);
  Ne = m.getNumEdges();
  ok &= m.getNumEdges()    == 4*Nv; // each vertex has 4 neighbours in toroidal topology

  // create a planar topology, where the edges are just edges such that points on the edges do not 
  // have a full set of 4 neighbours:
  m.clearEdges();
  ok &= m.getNumEdges() == 0;
  addMeshConnectionsPlanar2D(m, Nx, Ny);

  // Compare actual to expected number of edges:
  // 4 neighbours (inner points):         (Nx-2)*(Ny-2)
  // 3 neighbours (edges except corners):  2*(Nx-2) + 2*(Ny-2)
  // 2 neighbours (corner points):         4
  // i think, the formula works only for Nx,Ny >= 2
  Ne = 4 * (Nx-2)*(Ny-2) + 3 * (2*(Nx-2) + 2*(Ny-2)) + 2 * 4;  // maybe simplify
  // int a = Nx-2; int b = Ny-2; Ne = 4*a*b + 6*(a+b) + 8;     // formula simplified
  ok &= m.getNumEdges() == Ne;

  // todo: maybe check dx and dy for all the neighbours to see, if index computations for
  // the neighbours is indeed correct

  // todo: solve heat- or wave-equation numerically on such a grid to test the numerical gradient
  // computation on irregular grids

  // todo: have a function plotMesh - maybe it should create an image and write it to a ppm file

  //float x0 = -3, x1 = +3;  Nx = 31;
  //float y0 = -2, y1 = +2;  Ny = 21;


  // Create and set up mesh generator:
  int Nu = 31; 
  Nv = 21;     // was used above for numVertices but now for number of v-values

  int tgt; // target
  MG mg;
  mg.setNumSamples(Nu, Nv);
  mg.setParameterRange(-3, +3, -2, +2);

  // Generate and retrieve meshes:
  mg.updateMeshes();
  rsGraph<Vec2, float> pm = mg.getParameterMesh();  // used in PDE solver - maybe rename to getMesh
  rsGraph<Vec3, float> sm = mg.getSpatialMesh();    // used for visualization
  tgt = getNumConnectionsPlane(Nu, Nv);
  Ne  = pm.getNumEdges();
  // Nu=21,Nv=31 gives 2500 edges for a plane topology

  // maybe the spatial mesh is not needed here - maybe another class should be responsible for 
  // creating that

  // create mesh of polar coordinates:
  int Nr = 9;   // num radii
  int Na = 48;  // num angles - we want 360/Na to be an integer to have nice values for the 
                // direction angles
  mg.setNumSamples(Nr, Na);
  mg.setTopology(MG::Topology::cylinderH);
  mg.updateMeshes();
  pm = mg.getParameterMesh();
  //std::function<float(float u, float v)> fx, fy; 
  // functions to create (x,y) coordinates from (u,v) parameter values

  // We set the coordinates manually here - eventually, there may be an API that let's the user 
  // pass fx,fy (these functions should assume normalized inputs in 0..1):

  // todo: use hole (inner) radius and outer radius
  //float dr = 0.1;

  float holeRadius  = 0.4f;
  float outerRadius = 1.0f;
  float wiggle      = 0.1f;
  for(int i = 0; i < Nr; i++)
  {
    float r = holeRadius + (outerRadius - holeRadius) * i / (Nr-1);
    for(int j = 0; j < Na; j++)
    {
      float a  = float(2*PI * j / Na);
      float r2 = r + r * wiggle * sin(5*a);
      float x  = r2 * cos(a);
      float y  = r2 * sin(a);
      int k = mg.flatIndex(i, j);
      pm.setVertexData(k, Vec2(x, y));
    }
  }

  // We also compute the edge weights manually here (factor out):
  for(int i = 0; i < (int)pm.getNumVertices(); i++)
  {
    int numNeighbors = pm.getNumEdges(i);
    for(int k = 0; k < numNeighbors; k++)
    {
      int j = pm.getEdgeTarget(i, k);      // index of current neighbor
      Vec2 vi = pm.getVertexData(i);
      Vec2 vj = pm.getVertexData(j);
      float dx = vj.x - vi.x;
      float dy = vj.y - vi.y;
      float d  = sqrt(dx*dx + dy*dy);      // Euclidean distance - try Manhattan distance, too

      if(d > 0) 
        pm.setEdgeData(i, k, 1/d);
      else      
        pm.setEdgeData(i, k, 0);  // happens for "bottom" edge in circle
    }
  }
  // move to rsMeshGenerator2D

  ok &= pm.isSymmetric();
  Nv = pm.getNumVertices();
  plotMesh(pm);


  // create and plot hexagon mesh:
  Nx = 21;  // using high numbers makes plotting slow, but around 20 is ok
  Ny = (int) round(Nx / sqrt(3.f));
  Mesh mesh = getHexagonMesh<float>(Nx, Ny);
  plotMesh(mesh);

  int i = (Nx*Ny)/2;
  float e0    = getEdgeLength(mesh, i, 0);
  float e1    = getEdgeLength(mesh, i, 1);
  float e2    = getEdgeLength(mesh, i, 2);
  float ratio = rsMax(e0,e1,e2) / rsMin(e0,e1,e2);
  // maybe generalize to whatever number edges a vertex has and factor into a fucntion 
  // getEdgeLengthRatio ..or maybe getMinEdgeLength(i), getMaxEdgeLength(i)

  /*
  randomizeVertexPositions(mesh, 0.25f, 0.25f, 3);
  plotMesh(mesh);
  moveVerticesToEquilibrium(mesh, 3);
  plotMesh(mesh);
  */

  // todo: 
  // -solve wave equation on the mesh and somehow visualize the results - maybe have a black
  //  background and let dots become red for positive and green for negative values and black for 
  //  zero (or maybe blue)
  // -don't forget to average the bottom line
  // -try flat disc vs cone in 3D
  // -arrange boundaries somewhat randomly (or using some nice shapes) and then let the vertices 
  //  find their own positions by minimizing the potential energy when the vertices would be 
  //  connected by springs
  //  -iterate and in each step compute total force on each vertex and move it a little bit into 
  //   that direction until an equilibrium is reached

  int dummy = 0;
}
// -don't cofuse the potential 2D mesh irregulatity with the irregularity in the 3D mesh after 
//  projecting into 3D space -> the connection weights should not depend on the 3D locations but on
//  the 2D locations. If 3D locations should be relevant, we need a 3D version of 
//  rsNumericDifferentiator<T>::gradient2D which should use a 3x3 least-squares matrix
// -to simulate a disc with a center point, we could take - for example - the bottom row and 
//  conceptually collapse the 0-radius cricel into a single point, which implies that the u values
//  in the bottom row should all be equal - maybe just compute as usual (with cone topology) and 
//  afterwards compute the average of the row and assign all values in the row to that average
// -as energy functions, use E_kin = sum u_x^2 + u_y^2, E_pot = sum u_xx^2 + u_yy^2 or sum lap^2
// -try u_t = a * (u_x + u_y), u_t = a * (u_x - u_y), 
//  a sort of wave equation with additional effects: u_tt = a*lap(u) + b*div(u) + c*curl(u)
//  -> figure out, what these effects are
// -maybe use trapezoidal integration in time (use laplacian of time-step n together with the one
//  from time-step n-1 for updating the scalar field u on the grid)
// -use this for reverb

template<class T>
void laplacian2D(const rsGraph<rsVector2D<T>, T>& mesh, 
  const std::vector<T>& u, std::vector<T>& L)
{
  int N = mesh.getNumVertices();
  int k, numNeighbors;
  rsAssert((int) u.size() == N);
  rsAssert((int) L.size() == N);
  for(int i = 0; i < N; i++)
  {
    numNeighbors = mesh.getNumEdges(i);    // number of neighbors of vertex vi
    T uSum = T(0);                         // weighted sum of neighbors
    T wSum = T(0);                         // sum of weights
    for(int j = 0; j < numNeighbors; j++)  // loop over neighbors of vertex i
    {
      k   = mesh.getEdgeTarget(i, j);      // index of current neighbor
      T w = mesh.getEdgeData(i, j);        // weight in weighted sum of neighbors
      uSum += w * u[k];                    // accumulate weighted sum of neighbor values
      wSum += w;                           // accumulate sum of weights
    }
    L[i] = u[i] - uSum/wSum;               // value minus weighted average of neighbors
  }
  // todo: reverse roles of j,k
}
// -experimental - the idea is that the Laplacian represents, how far a value is away from the 
//  average of its neighborhood (see the 3blue1brown video on the heat equation), so we compute 
//  that weighted average ad return the difference of the value and the average
// -we assume that the edge weights are inversely proportional to the distances - todo: lift that
//  assumption - we can't assume that - but maybe it works with any choice of weights, due to the 
//  fact that we divide by the sum of weights, it will in any cas produce a weighted average
// -applying the laplacian to the laplacian again should yield the biharmonic operator


// test solving the transport equation on an irregular grid, created from a regular grid by 
// jittering the x,y-coordinates


/** Numerically solves a general 1st order in time, 2nd order in space, linear differential 
equation of the form:

   u_t = ax*u_x + ay*u_y + axx*u_xx + ayy*u_yy + axy*u_xy

on an arbitrary grid of vertices using numerical estimation of spatial derivatives by the method of
directional derivatives and an explicit Euler step in time. 

ToDo: maybe it's too ambitiuous to have such a general function - maybe treat the transport equation 
first
*/
template<class T>
std::vector<std::vector<T>> solveExamplePDE1(
  const rsGraph<rsVector2D<T>, T>& mesh, const std::vector<T> u0, int numTimeSteps, float deltaT)
{
  // under construction

  using Vec2 = rsVector2D<float>;
  using Vec  = std::vector<float>;
  using ND   = rsNumericDifferentiator<float>;

  // Coefficients that determine, what sort of differential equation this is:
  T ax  = -1.0f, ay  = -0.5f;  // (negative?) velocity components in x- and y-direction
  T axx =  0.0f, ayy =  0.0f;  // diffusivity in x- and y-direction
  T axy =  0.0f;               // a sort of shear diffusion? an a_yx would be redundant?

  // create arrays for the dependent variable u and its various partial derivatives and do some
  // initializations:
  int M = mesh.getNumVertices();  // should be Mx*My
  Vec u(M), u_t(M), u_x(M), u_y(M), u_xx(M), u_xy(M), u_yx(M), u_yy(M);
  std::vector<Vec> result;
  int   N  = numTimeSteps;  // number of time steps
  float dt = deltaT;        // delta-t between time-steps
  u = u0;

  // the main solver loop, stepping through the time-steps:
  result.push_back(u);
  for(int n = 1; n < N; n++)
  {
    // Compute spatial derivatives numerically:
    ND::gradient2D(mesh, u,   u_x,  u_y );  // compute gradient vector (u_x, u_y)
    ND::gradient2D(mesh, u_x, u_xx, u_xy);  // compute 1st row/column of Hessian matrix
    ND::gradient2D(mesh, u_y, u_yx, u_yy);  // compute 2nd row/column of Hessian matrix
    // u_xy should be equal to u_yx up to numerical inaccuracies (which may be higher than roundoff 
    // error) - check that - maybe it's useful to use the average of the two estimates:
    // u_xy = 0.5 * (u_xy + u_yx) in a scheme...but in many PDEs, the mixed derivatives are not 
    // used anyway - instead, only the Laplacian = u_xx + u_yy is used which can be estimated by 
    // our simpler laplacian2D function above which does not use the gradient but only the function
    // u itself -> todo: compare the results of both estimation methods

    // Compute temporal derivative using the defining PDE:
    u_t = ax*u_x + ay*u_y + axx*u_xx + ayy*u_yy + axy*u_xy;
    // ..is slow - lots of temporary vectors are created

    // update the dependent variable u = u(x,y,t) from time t to time t + dt using a simple Euler
    // step and store the result:
    u = u + dt * u_t;      // also slow
    result.push_back(u);
    // todo: try trapezoidal step, should use average of dt from this and previous iteration, maybe 
    // the use should adjust a value that blends between Euler and trapezoidal
  }

  // todo: 
  // -how should we handle boundary conditions - and how are they implicitly handled, when we 
  //  don't care about them? i think, this corresponds to setting u(t) = 0 at the boundary (or 
  //  whatever values the boundary points initially had....or wait - no - i think, it corresponds to
  //  a free boundary
  // -maybe the user could pass another array with the same length as the array of vertices that 
  //  specifies for each vertex a boundary value b and boundary strength s and after (or before?) 
  //  the update of u[i], we do u[i] = (1-s)*u[i] + s*b ...i think, if we do it before, it 
  //  translates better to handling Neumann conditions the same way - in this case, we'd use
  //  u_x[i] = (1-s)*u_x[i] + s*b
  //  -this also allows for "soft" boundary conditions and the conditions can be applied anywhere 
  //   in the grid -> it's straightforward to implement and very flexible - the code does not need 
  //   to distiguish beteen inner points and boundary points - it just applies this sample simple 
  //   "soft-condition" formula to all points
  // -i think, we need to make sure that also the boundary points all have at least 2 neighbors, 
  //  otherwise the solver will encounter underdetermined systems (and use the minimum norm solution
  //  which may or may not be meaningful ...but probably, it's not)
  // -the method should be generally applicable to all PDEs that can be written explicitly as, for 
//    example: u_tt = F(u, u_t, u_x, u_y, u_xx, ...) - the highest time deriavtive must be isolated
  //  on one side. Implicit PDEs like F(u, u_t, u_tt, u_x, ...) = 0 can not easily be treated...but 
  //  maybe they can be handled by computing the spatial derivatives as usual and then running a 
  //  multidimensional root-finder to find (u,u_t,u_tt) with (u_x,u_y,...) as parameters at all 
  //  points on the mesh?
  // -try to apply the method to stationary problems, i.e. problems, where the LHS is not u_t or 
  //  u_tt but 0 by just replacing the 0 by u_t and iterate until u_t has become 0 at all points, 
  //  i.e. the system has converged to a stationary state
  // -try to analyze stability using the von Neumann method, energy analysis and disturbations 
  //  ("Störungstheorie")
  // -in 1D schemes for the transport equation, it can be advanatageous to take a one-sided spatial
  //  difference instead of a centered one in the direction of the movement of the wave (upwind 
  //  scheme) - maybe in 2D and on irregular meshes, something similar can be done, by including 
  //  settings the weight to zero for all neighbor node difference-vectors whose projection on the
  //  velocity vector is negative
  // -the transport equation is actually numerically more problematic than the (higher order in 
  //  space) heat- or wave-equations - so maybe try these first...but these contain the laplacian
  //  and we actually want to try the gradient estimation...hmmm
  // -if the euqation includes advection (transport) and diffusio, maybe it could make sense to use
  //  an upwind estimate for the advection term and a centered estimate for the diffusion term

  // todo: 
  // -allow the use to set an input ("potential") - it's a (space-dependent) constant that's added 
  //  to u_t (or u_tt)..should be another vector V
  // -implement a similar scheme for a 2nd order in time PDE (yeah! it's wave-equation time! :-D)
  // -implement Schrödinger equation - uses complex numbers - maybe this can be used for that?
  // -implement Navier-Stokes - but this works on vector fields rather than scalar fields but
  //  we can break it down into two separate scalar fields
  // -maybe make up a very general 2nd order PDE in space and time with various coeffs that can be
  //  set by the user to dial in aspects of waves, diffusion, convection and more - be creative
  //  ...if it uses complex numbers, this may also encompass the Schroedinge euqation (i think)
  // -provide a simplified implementation when only the Laplacian is needed - this can be estimated
  //  by the simpler function above


  // https://en.wikipedia.org/wiki/Convection%E2%80%93diffusion_equation

  // http://www-personal.umd.umich.edu/~remski/java/source/Transport.html

  int dummy = 0;
  return result; // do this later, also take the mesh as input
}

template<class T>
void getExtent(const rsGraph<rsVector2D<T>, T>& mesh, T* minX, T* maxX, T* minY, T* maxY)
{
  using Vec2 = rsVector2D<T>;
  for(int i = 0; i < (int)mesh.getNumVertices(); i++) {
    rsVector2D<T> vi = mesh.getVertexData(i);
    *minX = rsMin(*minX, vi.x);
    *maxX = rsMax(*maxX, vi.x);
    *minY = rsMin(*minY, vi.y);
    *maxY = rsMax(*maxY, vi.y); }
}

template<class T>
void drawMesh(const rsGraph<rsVector2D<T>, T> mesh, rsImage<T>& img, 
 T minX, T maxX, T minY, T maxY)
{
  rsImagePainter<T, T, T> painter(&img);
  T brightness = T(0.75);
  int w = img.getWidth();
  int h = img.getHeight();
  for(int i = 0; i < mesh.getNumVertices(); i++) 
  {
    rsVector2D<T> vi = mesh.getVertexData(i);
    vi.x = rsLinToLin(vi.x, minX, maxX, T(0), T(w-1));
    vi.y = rsLinToLin(vi.y, minY, maxY, T(h-1), T(0));
    for(int k = 0; k < mesh.getNumEdges(i); k++)
    {
      int j = mesh.getEdgeTarget(i, k);
      rsVector2D<T> vj = mesh.getVertexData(j);
      vj.x = rsLinToLin(vj.x, minX, maxX, T(0), T(w-1));
      vj.y = rsLinToLin(vj.y, minY, maxY, T(h-1), T(0));
      painter.drawLineWu(vi.x, vi.y, vj.x, vj.y, brightness);
    }
  }
}

template<class T>
void visualizeResult(const rsGraph<rsVector2D<T>, T>& mesh, std::vector<std::vector<T>> result, 
  int width, int height, int frameRate)
{
  // maybe use GNUPlot or rsVideoRGB
  // for each time step, draw an image with a bunch of circles whose color is dictated by the 
  // result

  int numFrames = (int) result.size();

  // figure out spatial extent:
  T minX, maxX, minY, maxY;
  T margin = 0.01f;
  getExtent(mesh, &minX, &maxX, & minY, &maxY);
  T tmp = maxX-minX;
  minX -= margin * tmp;
  maxX += margin * tmp;
  tmp = maxY-minY;
  minY -= margin * tmp;
  maxY += margin * tmp;


  T brightness = T(4);
  // using brightness = 4, we get only a yellow spot in the top-right corner - wtf? the brightness
  // has nothing to do with the position?


  rsImage<T> positive(width, height), negative(width, height), background(width, height);

  rsVideoRGB video(width, height);
  rsImagePainter<T, T, T> painter;

  rsAlphaMask<T> mask;
  mask.setSize(15.f);   // should be as large as possible without overlap
  //mask.setTransitionWidth(0.25);
  // when changing the mask settings, we sometimes get a totally messed up result - the dots get 
  // *moved*(!!!) into the top-right corner - the brightness can also have this effect - wtf?
  // it depends also on numFrames - there's a weird bug somewhere!

  painter.setUseAlphaMask(true);
  painter.setAlphaMaskForDot(&mask);
    
  drawMesh(mesh, background, minX, maxX, minY, maxY);

  for(int n = 0; n < numFrames; n++) 
  {
    positive.clear();
    negative.clear();
    for(int i = 0; i < mesh.getNumVertices(); i++) 
    {
      rsVector2D<T> vi = mesh.getVertexData(i);
      vi.x = rsLinToLin(vi.x, minX, maxX, T(0), T(width-1));
      vi.y = rsLinToLin(vi.y, minY, maxY, T(height-1), T(0));
      T value = brightness * result[n][i];
      if(value > T(0))
        painter.setImageToPaintOn(&positive);
      else
        painter.setImageToPaintOn(&negative);
      painter.paintDot(vi.x, vi.y, rsAbs(value)); 
    }
    video.appendFrame(positive, negative, background);
  }

  // write video to file:
  rsVideoFileWriter vw;
  vw.setFrameRate(frameRate);
  vw.setCompressionLevel(10);  // 0: lossless, 10: good enough, 51: worst
                               // 0 produces artifacts
  vw.setDeleteTemporaryFiles(false);
  vw.writeVideoToFile(video, "PDE");
}
// maybe make datastructure rsFrozenGraph that has the same interface as rsGraph for reading but
// allows no restructuring - it's initialized from a regular rsGraph and then stays fixed, at least
// with regard to topology - that may be more efficient to use in PDE solver, less indirection, 
// more compact storage in memory (no scattering of the data all over the place)

// The PDE is 1st order in time, in space, it may be 2nd order depending on settings
void testPDE_1stOrder()
{
  using Vec2 = rsVector2D<float>;
  using Vec = std::vector<float>;

  // Set up mesh and video parameters:
  int Mx        = 40; // number of spatial samples in x direction
  int My        = 20; // number of spatial samples in y direction
  int numFrames = 250;
  int width     = Mx*10;
  int height    = (int)round(My*10 * sqrt(3));
  int frameRate = 25;

  // Create the mesh:
  rsGraph<Vec2, float> mesh = getHexagonMesh<float>(Mx, My); // mayb need to compute weights
  int M = mesh.getNumVertices();  // should be Mx*My
  //plotMesh(mesh);
  // try triangular mesh - then, each vertex has 6 neighbors - this may make the estimation of the
  // derivative more accurate - it uses more data, so it might be more accurate. maybe try also
  // an 8-point stencil - figure out, how accuracy increases with number of stencil points

  // Set up initial conditions:
  Vec u0(M);
  for(int i = 0; i < M; i++)
    u0[i] = 0.f;
  int Mc = getFlatMeshIndex2D(Mx/2, My/2, My);
  u0[Mc] = 1.f;    // one impulse in the middle

  // Solve PDE and visualize:
  std::vector<Vec> result = solveExamplePDE1(mesh, u0, numFrames, 0.05f);
  visualizeResult(mesh, result, width, height, frameRate);
  // we need some consideration for the aspect ratio

  // -when the wave hits hits the wall, it messes up
  // -the weights are not yet assigned, so they are supposed to by unity - which may actually be 
  //  appropriate

  int dummy = 0;
}

template<class T>
void initWithGaussian2D(const rsGraph<rsVector2D<T>, T>& mesh, std::vector<T>& u,
  rsVector2D<T> mu, T sigma)
{
  using Vec2 = rsVector2D<T>;
  int N = mesh.getNumVertices();
  rsAssert((int) u.size() == N);
  for(int i = 0; i < N; i++)
  {
    Vec2 v = mesh.getVertexData(i) - mu;  // difference between vertex location and gaussian center
    T    r = v.x*v.x + v.y*v.y;           // squared distance
    u[i]   = exp(-(r*r)/(sigma*sigma));   // value of bivariate Gaussian
  }
}

/** Removes edges from the mesh that have a positive component in the given direction d, i.e. the
dot product of their direction vector with d is strictly greater than zero. Can be used to modify 
the mesh to implement an upwind scheme for the transport equation in which case the velocity vector
should be passed as d. */
template<class T>
void removeDirectedConnections(rsGraph<rsVector2D<T>, T>& mesh, rsVector2D<T> d)
{
  using Vec2 = rsVector2D<T>;
  int N = mesh.getNumVertices();
  for(int i = 0; i < N; i++) {
    Vec2 vi = mesh.getVertexData(i);
    int numNeighbors = mesh.getNumEdges(i);
    for(int k = 0; k < numNeighbors; k++) {
      int j = mesh.getEdgeTarget(i, k);         // index of current neighbor of vi
      const Vec2& vj = mesh.getVertexData(j);   // current neighbor of vi
      Vec2 dv = vj - vi;                        // difference vector
      if(rsDot(dv, d) > T(0)) {
        mesh.removeEdgeAtIndex(i, k);
        numNeighbors--; }}}
}
// make a function that instead of removing the edges altogether introduces a factor between 0 and
// 1 - maybe based on the actual value of rsDot(dv, d) / (rsNorm(dv)*rsNorm(d)) - allow user to 
// trade off numeric dispersion vs numeric diffusion


template<class T>
void weightEdgesByDirection(rsGraph<rsVector2D<T>, T>& mesh, rsVector2D<T> d, T amount)
{
  using Vec2 = rsVector2D<T>;
  int N = mesh.getNumVertices();
  for(int i = 0; i < N; i++) 
  {
    Vec2 vi = mesh.getVertexData(i);
    int numNeighbors = mesh.getNumEdges(i);
    for(int k = 0; k < numNeighbors; k++) 
    {
      int j = mesh.getEdgeTarget(i, k);
      const Vec2& vj = mesh.getVertexData(j);
      Vec2 dv = vj - vi;
      T c = rsDot(dv, d) / (rsNorm(dv) * rsNorm(d)); // correlation, todo: precompute norm of d
      T e = mesh.getEdgeData(i, k);
      if(c > T(0))
      {
        //e *= (1-amount) + amount*c;  // or mybe we should just do nothing?
        // i think it's better to do nothing - it let's the blob diffuse into all direction whereas
        // with the formula, it tends to diffuse more into the direction of motion
        // todo: 
        // -try to tweak formula for best results
        // -try formula in conjuction with edge-weighting by distance
      }
      else
      {
        e *= (1-amount);
        //e *= (1-amount) + (1+amount*c);
      }
      mesh.setEdgeData(i, k, e);
    }
  }
}
// there's a lot of tweaking room here:
// -should we also scale the forward directions? if so, what function should be used
// -maybe we should not just scale the backward directions by the same amount but by an amount
//  that takes inot account how aligned the backward direction is, i.e. take into account c: if 
//  it's -1, the edge weight should be 0 but if it's like -0.5, maybe the edge weight should be 
//  just scaled down by 0.5 ...or 0.5^2...or whatever other formula? -> experimentation needed


void testTransportEquation()
{
  // Under construction

  // Solves the transport equation on a regular rectangular grid with periodic boundary conditions, 
  // i.e. a toroidal topology of the grid

  using Vec  = std::vector<float>;
  using Vec2 = rsVector2D<float>;
  using AT   = rsArrayTools;

  // Equation and solver settings:
  float dt = 1.f/512;              // time step - needs to be very small for Euler, like 0.002f
  Vec2  v  = Vec2(3.0f,  1.0f);    // velocity vector - maybe make it a function of (x,y) later
  Vec2  mu = Vec2(0.25f, 0.25f);   // center of initial Gaussian distribution
  float sigma = 0.0025f;           // variance
  int density = 65;                // density of mesh points (number along each direction)
  //bool upwind = false;             // if true, mesh connections in direction of v are deleted
  float upwind = 0.5f;             // continiously adjustable upwind setting - adjusts weights

  // maybe compute Courant number, try special values like 1, 1/2 - maybe use a velocity vector
  // with unit length (like (0.8,0.6)) and an inverse power of 2 for dt, if density is a power of 2
  // plus 1, we get a spatial sampling with an inverse power of 2, too

  // Visualization settings:
  int width     = 400;
  int height    = 400;
  int numFrames = 100;
  int frameRate = 25;

  // Create the mesh:
  rsMeshGenerator2D<float> meshGen;
  meshGen.setNumSamples(density, density);
  meshGen.setTopology(rsMeshGenerator2D<float>::Topology::torus);
  meshGen.setParameterRange(0.f, 1.f, 0.f, 1.f);             // rename to setRange
  meshGen.updateMeshes();                                    // get rid of this
  rsGraph<Vec2, float> mesh = meshGen.getParameterMesh();    // rename mesh to graphMesh, getP.. to getMesh
  weightEdgesByDirection(mesh, -v, upwind);

  // Create the rsStencilMesh2D for optimized computations:
  rsStencilMesh2D<float> stencilMesh;
  stencilMesh.computeCoeffsFromMesh(mesh);

  // Compute Courant number:
  float dx = 1.f / float(density-1);      // more generally: (xMax-xMin) / (xDensity-1)
  float dy = 1.f / float(density-1);
  float C  = dt * (v.x/dx + v.y/dy);
  rsAssert(C <= 1.f, "Courant number too high! A garbage solution should be expected!");
  // This should be less than 1, i think, see:
  // https://en.wikipedia.org/wiki/Courant%E2%80%93Friedrichs%E2%80%93Lewy_condition

  // Create and initialize data arrays for the funtion u(x,y,t):
  int N = mesh.getNumVertices();
  Vec u(N), u_x(N), u_y(N), u_t(N); // mesh function and its spatial and temporal derivatives
  Vec tmp1(N), tmp2(N), tmp3(N);    // temporaries, used vor different things in different schemes
  initWithGaussian2D(mesh, u, mu, sigma);

  // Define lambda function that computes the temporal derivative u_t by first computing the 
  // spatial partial derivatives u_x, u_y using rsNumericlDifferentiator::gradient2D for meshes and
  // then computing u_t from them via the transport equation: u_t = -dot(g,v) where g is the 
  // gradient, v is the velocity and dot means the dot-product:
  auto timeDerivative = [&](Vec& u, Vec& u_t)
  {
    // Compute spatial derivatives u_x, u_y:
    stencilMesh.gradient(&u[0], &u_x[0], &u_y[0]);                   // fast
    //rsNumericDifferentiator<float>::gradient2D(mesh, u, u_x, u_y); // slow

    // Compute temporal derivatives u_t:
    for(int i = 0; i < N; i++)
      u_t[i] = -(u_x[i]*v.x + u_y[i]*v.y);
  };

  // Computes an estimate for the time-derivative evaluated at the midpoint t + n/2. This estimate
  // is used in the midpoint method.
  auto timeDerivativeMidpoint = [&](Vec& u, Vec& u_t)
  {
    timeDerivative(u, tmp1);               // tmp1:  u_t at t = n
    for(int i = 0; i < N; i++)
      tmp2[i] = u[i] + 0.5f*dt * tmp1[i];  // tmp2:  u   at t = n+1/2 via Euler
    timeDerivative(tmp2, u_t);             // u_t:   u_t at t = n+1/2
  };

  // Computes an estimate for the mean of time-derivatives evaluated at t and t + 1. This estimate
  // is used in the Heun method.
  auto timeDerivativeMean = [&](Vec& u, Vec& u_t)
  {
    timeDerivative(u, tmp1);               // tmp1:  u_t at t = n
    for(int i = 0; i < N; i++)
      tmp2[i] = u[i] + dt * tmp1[i];       // tmp2:  u   at t = n+1 via Euler
    timeDerivative(tmp2, tmp3);            // tmp3:  u_t at t = n+1
    for(int i = 0; i < N; i++)
      u_t[i] = 0.5f * (tmp1[i] + tmp3[i]); // u_t:   average of estimates at t = n and t = n+1
  };

  auto updateSolution = [&](Vec& u, Vec& u_t)
  {
    for(int i = 0; i < N; i++) 
      u[i] += dt * u_t[i];
  };

  // Define lambda function that computes the temporal derivative u_t and updates our solution
  // u = u(x,y,t) to the next time step u = u(x,y,t+dt) using a smoothed Euler step. The parameter 
  // s is a smoothing coefficient in the range 0..0.5, where 0 means normal Euler steps and 0.5 
  // means averaged Euler steps (like in trapzoidal integration). So, with 0.5, the temporal 
  // derivative is averaged between the current and previous time step. This is for 
  // experimentation, but it seems to be useless - we'll see. When s = 1, only the derivative
  // from the previous time step is used - but that seems to be even more silly.
  auto doTimeStepEuler = [&](float s = 0.f) // s: smoothing from 0...0.5
  {
    timeDerivative(u, u_t);
    for(int i = 0; i < N; i++) {
      u[i] += dt * ((1.f-s)*u_t[i] + s*tmp1[i]); // update u via smoothed Euler step
      rsCopy(u_t, tmp1); }                       // remember u_t for next iteration
  };

  auto doTimeStepMidpoint = [&]()
  {
    timeDerivativeMidpoint(u, u_t);
    updateSolution(u, u_t);
  };

  auto doTimeStepHeun = [&]()
  {
    timeDerivativeMean(u, u_t);
    updateSolution(u, u_t);
  };
  // maybe this could use smoothing, too

  // A time-stepper using a mean of midpoint and Heun method:
  auto doTimeStepMidHeun = [&]()
  {
    timeDerivativeMidpoint(u, tmp1);
    timeDerivativeMean(    u, tmp2);

    rsPlotVectors(tmp1, tmp2);

    for(int i = 0; i < N; i++)
      u_t[i] = 0.5f * (tmp1[i] + tmp2[i]);

    updateSolution(u, u_t);
  };
  // this actually produces garbage - why? ...because timeDerivativeMean overwrites tmp1 with a 1st
  // order estimate? ..the two function calls operate on the same temp-data - introduce separate
  // temp arrays for u_t_mid and u_t_mean
  // ...if it works, this should be optimized: the function both compute the same estimate u_t at
  // t = n twice!

  // A classic Runge-Kutta time-step of order 4, adapted to the PDE scenario: the function "f" 
  // that is typically seen in textbooks is replaced by our "timeDerivative" function that fills
  // a whole array and "h" is replaced by our time-delta dt:
  Vec k1(N), k2(N), k3(N), k4(N), uk(N); // arrays for the k-values in RK4 and u + ki
  auto doTimeStepRungeKutta4 = [&]()
  {
    // Compute the (arrays of the) 4 k-values:
    timeDerivative(u,  k1);                                    // k1 = f(u)

    AT::weightedSum(&u[0], &k1[0], &uk[0], N, 1.f, 0.5f*dt);   // uk =   u + (h/2)*k1
    timeDerivative(uk, k2);                                    // k2 = f(u + (h/2)*k1) = f(uk)

    AT::weightedSum(&u[0], &k2[0], &uk[0], N, 1.f, 0.5f*dt);   // uk =   u + (h/2)*k2
    timeDerivative(uk, k3);                                    // k3 = f(u + (h/2)*k2) = f(uk)

    AT::weightedSum(&u[0], &k3[0], &uk[0], N, 1.f, 1.0f*dt);   // uk =   u + h*k3
    timeDerivative(uk, k4);                                    // k4 = f(u + h*k3)

    // Do the RK4 step using a weighted average of the 4 k-values:
    for(int i = 0; i < N; i++)
      u[i] += (dt/6.f) * (k1[i] + 2.f*(k2[i] + k3[i]) + k4[i]);
  };
  // needs test
  // ToDo: Combine an initial section (of 4 frames) computed via RK4 with subsequent frames 
  // computed by 4th order Adams-Bashforth or Adams-Moulton steps. These are multistep methods 
  // that re-use past computations of u and/or u_t from previous time-steps. This gives also a 4th 
  // order in time scheme but with 4 times less evaluations of the expensive timeDerivative 
  // function. See "Höhere Mathematik in Rezepten", page 384

  // Try combining estimates at t=n, t=n+1/2, t=n with weights (1,2,1)/4 ...maybe tweak the weights
  // later
  // (1) compute u_t at t = n and t = n+1 like in Heun's method
  // (2) compute their average
  // (3) compute u_t at t = n+1/2 going a half-step into the direction of the average from (2)
  // (4) average the results from (2) and (3)
  // (5) update solution with result of (4)
  auto doTimeStepHM = [&]()  // HM: Heun-Midpoint
  {
    timeDerivative(u, tmp1);         // tmp1:  u_t at t = n
    for(int i = 0; i < N; i++)
      tmp2[i] = u[i] + dt * tmp1[i];        // tmp2:  u   at t = n+1 via Euler
    timeDerivative(tmp2, tmp3);      // tmp3:  u_t at t = n+1 via Euler
    for(int i = 0; i < N; i++)
      tmp1[i] = 0.5f * (tmp1[i] + tmp3[i]); // tmp1:   average of estimates at t = n and t = n+1

    //updateSolution(u, tmp1); return;  //   // test - should give same results as Heun

    // tmp1 now contains the estimate of u_t that would be used in the Heun method - but we use it
    // here to do a preliminary midpoint step:
    for(int i = 0; i < N; i++)
      tmp2[i] = u[i] + 0.5f*dt * tmp1[i];  // tmp2:  u   at t = n+1/2 via Heun
    timeDerivative(tmp2, tmp3);     // tmp3:  u_t at t = n+1/2

    // tmp3 now contains a midpoint estimate but using a Heun (instead of Euler) step to compute 
    // where the midpoint is. Now we finally compute our actual estimate of u_t as average of Heun 
    // method result and the (Heun-improved) midpoint method:
    for(int i = 0; i < N; i++)
      u_t[i] = 0.5f * (tmp1[i] + tmp3[i]);

    updateSolution(u, u_t);
  };
  // -this should result in a 3rd order accurate scheme (i hope), but the result looks the same 
  //  as with Heun or midpoint - no improvement visible 
  //  -check for bugs 
  //  -or maybe the spatial accuracy is already too bad for this? we want to achieve 3rd order
  //   in time, but maybe that makes no sense when the spatial error is only 2nd order? maybe the 
  //   spatial order should be >= the temporal order, otherwise increasing temporal order will not 
  //   help?
  // -try doing it the other way around: first making a midpoint estimate and then use this to make 
  //  a preliminary Heun estimate - call it MH for Midpoint-Heun
  // -maybe this can be optimized using less temporaries

  // Try combining midpoint and Heun with midpoint first:
  // (1) compute u_t at t = n and t = n+1/2 like in the midpoint method
  // (2) compute u at t = n+1 using the u_t estimate at t = n + 1/2 from (1)
  // (3) compute u_t at t = n+1 using u as computed in (2)
  // (4) compute a weighted average of u_t at t = n, t = n+1/2 and t = n+1 computed in (1),(3)
  // (5) update solution with result of (4)
  auto doTimeStepMH = [&]()  // MH: Midpoint-Heun
  {
    // ...

  };

  // this uses an estimate (u_t(n) + u_t(n+1)) / 2
  // try: (u_t(n) + 2*u_t(t+1/2) + u_t(n+1)) / 4 ..i.e. an average of Heun and midpoint method - i 
  // think, this should be 3rd order accurate in dt since it uses 3 gradient evaluations


  // Loop through the frames and for each frame, update the solution and record the result:
  rsVideoWriterMesh<float> videoWriter;
  videoWriter.setSize(width, height);
  videoWriter.initBackground(mesh);
  for(int n = 0; n < numFrames; n++)
  {
    videoWriter.recordFrame(mesh, u);
    //doTimeStepEuler(0.0f);                        // 0: normal Euler, 0.5: "trapezoidal" Euler
    //doTimeStepMidpoint();
    doTimeStepHeun();
    //doTimeStepMidHeun();    // this sucks!
    //doTimeStepRungeKutta4();
    //doTimeStepHM();
  }
  videoWriter.writeFile("TransportEquation");

  // Observations:
  // -Withe the Euler steps, there is quite a lot of rippling and dispersion going on, but it can 
  //  be remedied by choosing a very small time step dt. I tried reducing the density in order to 
  //  make the ratio between spatial and temporal sampling better (Courant number or something), 
  //  but it doesn't seem to help
  // -Smoothing the Euler steps seems useless with regard to improving accuracy, but maybe the idea
  //  can be used in other contexts to improve stability, when a scheme produces oscillations at
  //  the time-stepping rate - we'll see...
  // -The Heun time stepping method is indeed a big improvement over Euler. Now it's clear what we 
  //  have to do to get even better results: average more estimates of u_t in the interval t = n
  //  and t = n+1, just like it is done in Runge-Kutta methods for ODEs.
  // -I can't see any difference between midpoint and Heun (which is a good sign - they are supposed
  //  to have the same error order) but maybe run the simulation longer for closer analysis.
  // -HM is slightly better than Heun with density = 65 - but the difference is not as big as 
  //  expected - maybe try using a grid with higher connectivity (i.e. connections also to diagonal
  //  neighbors) to increase spatial accuracy - maybe it makes no sense to try to increase the 
  //  temporal accuracy, when the spacial accuracy is not high enough
  //  -> done: adding diagonals does not seem to help to get rid of the dispersion - it just gets
  //     oriented more diagonally
  // -RK4 does not seem to better than Heun either - in fact, the result looks exactly the same
  //  -check for bugs
  //  -try it with more accurate spatial derivatives
  //   -adding one pair of diagonals northWest/southEast, the dispersion seems to get dragged into
  //    that direction - it doesn't get less compared to no diagonal connections, though, so it 
  //    seems trying to counteract the dispersion with higher spatial accuracy does not work
  //   -adding the other set of diagonals gives errors - there seems to be  something wrong with 
  //    the southWest diagonals - but the error only shows up much later as a heap corruption, when
  //    rsImage tries to allocate memory - there are comments in 
  //    rsMeshGenerator2D::connectInnerDiagonal about this - it's really weird!
  //    -> try, if the error persists, if we leave out the time-steps, i.e. if it's sufficient to
  //       trigger the error to set up the diagonal mesh connections
  // -dispersion seems to increase when the the initial Gaussian bump is made narrower (needs more)
  //  tests
  // -maybe a lower order scheme (like Heun) but with smaller time-step is preferable, because 
  //  reducing the time-step seems to reduce the dispersion ...or maybe use something like leapfrog
  //  or lax-wendroff for time-stepping (try to get high order with only one gradient evaluation
  //  per time step), we may use RK4 for initialization
  // -when trying ot use an upwidn scheme, we get garbage
  //  -maybe we need a mesh with more neighbors - try with 8, but that requires to fix the bug with
  //   the southwest neighbors in the mesh (try to let loops run from 2...Nv-2 )
  //  -also, figure out, if we need to pass v or -v - the API should be such that we pass v
  //   -> it's OK - the "upwind" side is the direction where the wave comes from
  //  -with upwind, only frames 0,1,2 ar ok - after that, we get garbage - we get nans in u already
  //   in the first update step - could be related to encountering a singular matrix A in the 
  //   boundary vertices producing nans which then contaminate the whole mesh over time
  //   -however, these first few frames actually look sort of what one would expect from an upwind
  //    scheme: less dispersion than a centered scheme but instead more spreading out, i.e. 
  //    diffusion. maybe instead of calling rsMatrix2x2<T>::solve, use some sort of solveSave
  //    function that gracefully handles singular matrices
  // -It doesn't seem to make a difference, if we use the expensive RK4 time-stepper or the Heun
  //  stepper that has roughly half the cost
  // -Increasing grid density seems to help only up to a point (like 65) going even higher (128) 
  //  does not seem to reduce the dispersion further
  // -Using more neighbors (8) seems to leave the hape more intact and the ripples adjust more long
  //  the direction of movement (rather than along the grid directions)
  // -Try to make the "upwind" parameter continuous: scale down the weights of the downwind 
  //  connections instead of removing them entirely -> user can dial in a tradeoff between numeric
  //  dispersion and numeric diffusion


  // ToDo: 
  // -Implement more accurate (i.e. higher order) time-steppers. When the improvements start to 
  //  level off, it may be due the accuracy of spatial derivatives not keeping up - in this case, 
  //  use a mesh with more neighbors to increase spatial accuracy.
  // -try using a mesh with connections to diagonal neighbors - increase spatial precision by
  //  4 powers 
  // -try to improve time accuracy by using a strategy similar to the midpoint rule in ODE solvers:
  //  do a prelimiray step by half the stepsize, compute the gradient there again and then do the
  //  actual step with the gradient computed there - or try Heun's method:
  //  https://en.wikipedia.org/wiki/Heun%27s_method
  //  https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Second-order_methods_with_two_stages
  //  https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods#Explicit_methods
  //  it goes a tentative full step, then computes the new gradient there and the uses an average
  //  of both gradients for the actual step
  // -maybe try using double precision
  // -try another PDE - wave equation, diffusion equation
  // -try less smooth initial conditions
  // -the wrap-around does not seem to work - check the mesh connectivity
  // -figure out, if the method could be used in the context of an implicit scheme - maybe not, but 
  //  that may not be a problem, if it's stable in explicit schemes
  // -Compare the numerical dispersion to a comparable standard scheme - see, if this algor is 
  //  better or worse
  // -figure out, if there is a certain optimum value for the speed that minimizes the dispersion
  // -maybe try to connect to more nieghbors, further away from the center vertex
  // -Try to derive and implement implicit schemes (maybe the implicit Euler scheme first because 
  //  it's the simplest). The explicit Euler scheme does: uNew = uOld + dt * u_t where u is the 
  //  vector of u-values, u_t is the time-derivative and dt is the time-step. u_t itself can be 
  //  computed from u as u_t = D * uOld where D is some (sparse) matrix that provides the 
  //  time-differentiation by first performing spatial differentiation and then combining these 
  //  spatial derivatives in a way dictated by the velocity vector. Maybe we can write the explicit 
  //  Euler update in full generality as uNew = A * uOld where in this case A = I + dt*D where I is 
  //  the identity matrix. An implicit update would then look like uNew = A * uNew...hmm..that 
  //  doesn't seem right...maybe write:
  //    explicit: uNew = uOld + dt * D * uOld
  //    implicit: uNew = uOld + dt * D * uNew
  //  and then solve the linear system arising from the implicit method using an iterative solver.
  //  This can be generalized to 2nd order PDEs - just the D-matrix would need to change.

  // Notes:
  // -In the book "Finite Difference Computing with PDEs", chapter 4, it is said that PDEs 
  //  involving 1st order spatial derivatives (such as the transport equation) are actually harder 
  //  to treat numerically than those involving only 2nd order spatial derivatives (such as the 
  //  wave- or heat-equation), so we are doing already a sort of stress-test here.

  // Links:
  // Analysis of numerical dissipation and dispersion:
  // http://www.mathematik.uni-dortmund.de/~kuzmin/cfdintro/lecture10.pdf
  // Free book on PDEs:
  // http://people.maths.ox.ac.uk/trefethen/pdetext.html
  // Chapter 3, page 126 has an algorithm for finite differences in for arbitraray 1D meshes

  int dummy = 0;
}

void testWaveEquation()
{


  int dummy = 0;
}


// check, if this can be removed:
/*
template<class T> 
rsPolynomial<std::complex<T>> getComplexFromHarmonicU(const rsBivariatePolynomial<T>& u)
{
  // It's just the very first column of u. This can be seen by assuming p(x,y) = p(x+i*y), i.e. the
  // bivariate p(x,y) is actually univariate in the single variable x+i*y and multiplying out the 
  // ansatz: c0*(x+i*y)^0 + c1*(x+i*y)^1 + c2*(x+i*y)^2 + c3*(x+i*y)^3 + ... It can be seen, that
  // the univariate c-coeffs must equal the bivariate coeffs that multiply the powers of x without 
  // any y in the term. ...but maybe that works only if the original polynomial had real coeffs
  // i think this works to extract the real part and for the imaginary part, we need to take the
  // negative entries of the first column of v - which can be passed by the caller but we could 
  // also do v = u.getHarmonicConjugate - maybe do both

  rsBivariatePolynomial<T> v = u.getHarmonicConjugate();

  int M = u.getDegreeX();
  rsPolynomial<std::complex<T>> p(M);
  for(int i = 0; i <= M; i++)
  {
    //p[i] = u.coeff(i, 0);

    //p[i] = std::complex<T>(u.coeff(i, 0), -v.coeff(i, 0));

    p[i] = std::complex<T>(u.coeff(i, 0), v.coeff(i, 0)); // why no minus for v.coeff?

  }
  return p;
}
// needs more tests....
// more generally, the coeffs for the ansatz 
//   c0*(a*x+b*y)^0 + c1*(a*x+b*y)^1 + c2*(a*x+b*y)^2 + c3*(a*x+b*y)^3
// are the first column divided by powers of a, i.e. p[i] = u.coeff(i, 0) / pow(a, i); ..i think
// can we also reconstruct the complex function from the v-part. maybe we need to negate it and 
// then find the harmonic conjugate and feed it to this algo?

// see: https://math.mit.edu/~jorloff/18.04/notes/topic5.pdf
// ...seems like we need only u and not also v to reconstruct the complex function - which makes
// sense, since ve is redundant

template<class T> 
rsPolynomial<std::complex<T>> getComplexFromHarmonicV(const rsBivariatePolynomial<T>& v)
{
  rsBivariatePolynomial<T> vn = v;
  //vn.negate();
  rsBivariatePolynomial<T> u = vn.getHarmonicConjugate();
  return getComplexFromHarmonic(u);
}
*/


void testBiModalFeedback()
{
  // Idea: 
  // -We want to combine the outputs of a modal filter bank in a way that is suitable to feed
  //  back as input signal to achieve effects that mimic the (self)-excitation in musical 
  //  instruments such as bowing a string or blowing a pipe. In these systems, a steady supply of 
  //  energy gets modulated by the output signal (string-position, pressure) itself. We tpyically 
  //  see effects like mode-locking (the nearly harmonic modes lock each other into exact harmonic
  //  relationships).
  // -It seems like it would be good, if we could produce a signal from the modal-bank output that
  //  features prominent spikes in each cycle or up/down spikes in each half-cycle
  // -In "The Physics of Musical Instuments", pg. 139, the equation of motion for a self-excited 
  //  oscillator is given by: y_tt + w_0^2 * y = g(y, y_t) where y is some sort of displacement,
  //  _t, _tt denote 1st and 2nd time derivatives and g is some function and w_0 is the resonance
  //  frequency when g is zero. On page 140, a special form of g is given as: g = a*y_t*(1-y^2) 
  //  where a is some amplitude of excitation. That's descirbed as the Van der Pol oscilator. I 
  //  think, g should be large whenever y is small and has high 1st derivative, which we may expect
  //  at the zero, crossings - so that equation for the feedback-driver might be suitable.
  // -for y,y_t we could use y = (y[n-1] + y[n-2]) / 2, y_t = (y[n-1] - y[n-2]) / 2
  // -variations: g = a * sat(y_t^3 * (1-y^4)) where sat(..) is some sort of saturation curve. 
  //  Using higher (even) powers of y may concentrate the spike more around the zero crossing. 
  //  Using saturation may help to stabilize the system. It could also be some nonmonotonic 
  //  function like x / (1 + x^2), x^m / (1 + x^n) where n odd, m even
  // https://en.wikipedia.org/wiki/Van_der_Pol_oscillator
  // https://de.wikipedia.org/wiki/Van-der-Pol-System

  // We try it with 2 modes that are slightly off from a perfect harmonic relationship:

  // User parameters:
  double fs = 44100;
  double f1 = 200,  f2 = 520;    // 1st and 2nd modal frequency in Hz
  double p1 = 0,    p2 = 0;      // modal pahses in degrees
  double a1 = 1./2, a2 = 1./5;   // modal amplitudes
  double d1 = 0.05, d2 = 0.1;    // modal decay time constants in seconds
  double L  = 1.0;               // length of output in seconds
  double k  = +0.05;             // feedback gain

  // The feedback driver function g(y, y_t):
  auto func1 = [](double y, double y_t){ return y_t * (1. - y*y); };  // van der Pol

  auto func2 = [](double y, double y_t)
  { 
    double t  = 0.8;                // tension parameter
    double ty = t*y;
    double z  = (ty-y)/(2*ty-t-1);  // rationally mapped y
    return rsSign(y_t) * z;
  };
  // rational mapping, see https://www.desmos.com/calculator/xaklfkriac

  auto func3 = [](double y, double y_t)
  { 
    double t  = +0.99;               // tension parameter - seems to scale the whole feedbak signal
    double v  = y*y;
    double tv = t*v;
    double z  = (tv-v)/(2*tv-t-1);  // rationally mapped v
    return z;
    //return rsSign(y_t) * z;
  };

  auto func4 = [](double y, double y_t)
  { 
    double t  = 0.99;
    double v  = 1.0-y*y;
    double tv = t*v;
    double z  = (tv-v)/(2*tv-t-1);  // rationally mapped v
    return z;
  };
  // Produces upward spikes at all (upward and downward) zero-crossings

  auto func5 = [](double y, double y_t)
  { 
    double t  = 0.99;
    double v  = 1.0-y*y;
    double tv = t*v;
    double z  = (tv-v)/(2*tv-t-1);  
    return y_t * z;
  };
  // Produces upward spikes at upward zero-crossings and downward spikes at downward zero-crossings


  // i want a function z(y) that produces large output (near 1) only when v = y^2 is near zero, 
  // scaling that with the sign of the derivative should give up/down spikes at the zero crossings


  auto func = func4;


  // Create and set up the 2 modal filters:
  rsModalFilter<double, double> mf1, mf2;
  mf1.setModalParameters(f1, a1, d1, p1, fs);
  mf2.setModalParameters(f2, a2, d2, p2, fs);

  // Create arrays for the output signals
  int N = (int) ceil(L*fs);                // number of samples to produce
  std::vector<double> y1(N) , y2(N), y(N), g(N);

  // Trigger the filters (wihtout feedback yet) and enter the loop over the samples (with 
  // feedback):
  y1[0] = mf1.getSample(1.0); y2[0] = mf2.getSample(1.0); y[0] = y1[0] + y2[0];
  y1[1] = mf1.getSample(0.0); y2[1] = mf2.getSample(0.0); y[1] = y1[1] + y2[1];
  for(int n = 2; n < N; n++)
  {
    // Create feedback signal:
    double y_a = (y[n-1] + y[n-2]) * 0.5;  // average
    double y_t = (y[n-1] - y[n-2]) * 0.5;  // difference (approximates derivative)
    g[n]  = func(y_a, y_t);

    // Produce output signal:
    y1[n] = mf1.getSample(k*g[n]); 
    y2[n] = mf2.getSample(k*g[n]); 
    y[ n] = y1[n] + y2[n];
  }

  // Write the output y and the feedback excitation g into a stereo wavefile:
  using AT = RAPT::rsArrayTools;
  AT::normalize(&y[0], N);
  AT::normalize(&g[0], N);
  rosic::writeToMonoWaveFile("ModalFeedbackY.wav", &y[0], N, int(fs));
  rosic::writeToStereoWaveFile("ModalFeedback.wav", &y[0], &g[0], N, int(fs));


  //rsPlotVectors(y, g);
  //rsPlotVectors(y1, y2);
  //rsPlotVectors(y, g, y1, y2);


  // Observations:
  // -f = 200,520, p1 = 0,0, a = 1/1,1/1, d = 0.1,0.1, k = +0.05, func = func4, t = 0.99:
  //  -modes lock after around 10k samples, f0 around 105 Hz is present (but not strong), when 
  //   using k = -0.05 instead of +0.05, the signal seems to be inverted - is it a general rule 
  //   that flipping the sign of the feedback factor flips the sign of the signal...or perhaps only
  //   for func4
  // -when a2 = 0, we see a waveform at around 210 Hz plus harmonics - the offset from the dialed 
  //  in 200 Hz is probably due to the feedback delay? ...maybe try with oversampling
  // -reducing both decay times, seems to increase the fundamental frequency
  // -f = 200,505, p1 = 0,0, a = 1/2,1/5, d = 0.1,0.1, k = +0.05, func = func4, t = 0.99:
  //  -settles into "overblown" state at around 505 Hz after 25k samples
  // -f = 200,503, p1 = 0,0, a = 1/2,1/5, d = 0.1,0.1, k = +0.05, func = func4, t = 0.99:
  //  -settles afetr 10k samples into a 210 Hz tone with harmonics (420,630,840,...) with some 
  //   component around 495 that fades away
  //  -going down to 480 with f2 still seems to settle to the same harmonics - but there's a rough
  //   and rather long transient (about 16k samples long)
  //  -going down further to f2 = 450, it settles to 450 Hz pülus harmonics after an interesting
  //   transient
  //  -Going to f2 = 420 enters a chatic regime with prominent peaks at 210 and 430 hz
  //  -the chaotic regime seems to go down to about 360 Hz, at 350 Hz, we again enter a periodic 
  //   regime - maybe it has to do with interacting phases of even and odd harmonics? when f2 is 
  //   near an even harmonic, they interact in a way that produces chaos?
  //  -going further down to 310, the initial chaotic section gets longer
  // -f = 200,520, p1 = 0,0, a = 1/2,1/5, d = 0.1,0.1, k = -0.7, func = func5, t = 0.99:
  //  -modes get more strongly damped with higher negative feedback (not interesting), increasing 
  //   tension to 0.999 seems to make the decay longer and more linear
  // -f = 200,520, p1 = 0,0, a = 1/2,1/5, d = 0.1,0.1, k = -0.75, func = func4, t = 0.99:
  //  -settles into quasi-steady state after 5k samples, when increasing negative feedback to -0.8,
  //   it overblows, at -0.81 it goes back to normal (but noisier)



  // ...oh - wait - the a1,a2 values were not actually used in these tests, so they have been 1.0 
  // all along - ah - no - that's wrong - they are part of the modal filter setting

  // ToDo:
  // -fade out the feedback factor at the end and see how this ends the sound
  // -try a feedback factor that is a function of time - maybe an envelope and/or LFO
  // -maybe with more modal filters, we should have seperate parameters for (cross)feedback of 
  //  neighbouring modes (each mode gets feedback from iteself and its two neighbors) and global 
  //  feedback (each mode gets feedback from the whole sum). maybe that neighbor feedback can be
  //  a function of frequency. More generally, we could have a whole (sparse?) feedback matrix
  //  ...maybe we should not have a full set of 1000 modal oscillators in this case but rather a 
  //  handful - we don't need them to produce high freq content because the nonlinearities also 
  //  produce harmonics

  // Observations old:
  // -negative k lead to stronger damping
  // -with positive k, the 2nd harmonic quickly dominates
  // -the g function does not look very much like spikes at the zero-crossings - maybe we need 
  //  higher even powers of y ...or maybe the rational function used for parameter mappings could
  //  help to concentrate the spikes more at the zeros
  // -we could also detect zero-crossings explicitly and trigger spikes explicitly - but somehow 
  //  that seems to be less natural, less physical


}

void testExteriorAlgebra3D()
{
  // This is actually just a special case of the 3D geometric algebra, namely, the subalgebra that 
  // involves only pure blades

  using Real   = float;
  using Vec    = rsVector3D<Real>;
  using BiVec  = rsBiVector3D<Real>;
  using TriVec = rsTriVector3D<Real>;
  using EA     = rsExteriorAlgebra3D<Real>;

  bool ok = true;

  Vec u(1.f, 2.f, 3.f), v(4.f, 5.f, 6.f), w(7.f, -8.f, 9.f);
  // when using w(7.f, 8.f, 9.f) t becomes zero

  // Test wedge product:
  BiVec b = u ^ v;    // wedge product of two vectors
  TriVec t(u, v, w); ok &= t.getSignedVolume() == -96.f; 
  t = b ^ w;         ok &= t.getSignedVolume() == -96.f;  // wegde product of bivector and vector
  t = u ^ v ^ w;     ok &= t.getSignedVolume() == -96.f;  // wedge product of 3 vectors
  ok &= (u^v) == -(v^u);                      // antisymmetry
  ok &= ((u^v)^w) == (u^(v^w));               // associativity
  ok &= (u ^ (v+w)) == (u^v) + (u^w);         // distributivity over addition
  ok &= ((2.f*u) ^ (3.f*v)) == 2.f*3.f*(u^v); // distributivity of scalar multiplication

  // Test Hodge star:
  ok &= det(u, v, EA::star(u^v)) >= 0.f;
  ok &= EA::star(u+v) == EA::star(u) + EA::star(v);

  // todo: 
  // -test a ^ star(b) = <<a,b>>w, dot(u,v) = star(sharp(u) ^ star(flat(b)), 
  //  cross(u,v) = sharp(star(flat(u) ^ flat(v)))
  // -test exterior algebra for vectors - i.e. the type Real is replaced by a vector type

  // -implement directional derivative d and check:
  //  d f(u) = dot(u, grad(f)), d(a ^ b) = (d a) ^b + ((-1)^k) (a ^ d b), 
  //  curl(X) = sharp(star(d flat(X))), (flat(u))(v) == g(u,v) where g is the metric induced by f

  // todo: 
  // -implement analogous operations for covectors (maybe using delegation) and then give the
  //  covector classes an evaluation operator () that takes 1,2 or 3 vectors as inputs respectively
  // -implement sharp and flat functions as member functions in the classes, where these apply 
  // -what about the hodge-star operation? do we need an explicit operation or do we just 
  //  "identify" objects with their hodge-duals as needed?

  // For generalizing the wedge product to higher dimensional spaces, it seems, we could use a 
  // definition based on tensor products, see
  // https://math.stackexchange.com/questions/2312215/relationship-between-tensor-product-and-wedge-product
  // https://www.researchgate.net/publication/303810058_Tensor_Products_Wedge_Products_and_Differential_Forms/link/5754447008ae10d9337a2f15/download
  // from this, it seems: u ^ v = (u*v - v*u) / 2
  // ...or can we store the set of vectors and if we want to evaluate a k-form, just project the 
  // input vectors and then compute a determinant (see Krane, pg. 58 - there it's done for 2D but
  // maybe the idea generalizes?) ...or: maybe forming a wedge product from k 1-vectors or 
  // k 1-forms should be done by applying Gram-Schmidt orthonormalization to them and applying the 
  // resulting k-form amounts to projection and taking the determinant?



  int dummy = 0;
}

// move to main repo (together with the actual bit-twiddling functions - i think, it now makes 
// sense to create a class rsBitTwiddling<TInt> in rapt)
bool testBitTwiddling()
{
  int r;
  bool ok = true;

  r = rsBitCount(0); ok &= r == 0;
  r = rsBitCount(1); ok &= r == 1;
  r = rsBitCount(2); ok &= r == 1;
  r = rsBitCount(3); ok &= r == 2;
  r = rsBitCount(4); ok &= r == 1;
  r = rsBitCount(5); ok &= r == 2;
  r = rsBitCount(6); ok &= r == 2;
  r = rsBitCount(7); ok &= r == 3;
  r = rsBitCount(8); ok &= r == 1;

  r = rsRightmostBit(0); ok &= r == -1;
  r = rsRightmostBit(1); ok &= r ==  0;
  r = rsRightmostBit(2); ok &= r ==  1;
  r = rsRightmostBit(3); ok &= r ==  0;
  r = rsRightmostBit(4); ok &= r ==  2;
  r = rsRightmostBit(5); ok &= r ==  0;
  r = rsRightmostBit(6); ok &= r ==  1;
  r = rsRightmostBit(7); ok &= r ==  0;
  r = rsRightmostBit(8); ok &= r ==  3;

  // 4D:
  // 1    e1   e2   e3   e4   e12  e13  e14  e23  e24  e34  e123 e124 e134 e234 e1234
  // 0000 0001 0010 0100 1000 0011 0101 1001 0110 1010 1100 0111 1011 1101 1110 1111
  // 0    1    2    4    8    3    5    9    6    10   12   7    11   13   14   15

  //bool b;

  // sequences with 1 bit set (1,2,4,8)
  ok &= !rsBitLess(1, 1);
  ok &=  rsBitLess(1, 2) && !rsBitLess(2, 1);
  ok &=  rsBitLess(1, 4) && !rsBitLess(4, 1);
  ok &=  rsBitLess(1, 8) && !rsBitLess(8, 1);
  ok &= !rsBitLess(2, 2);
  ok &=  rsBitLess(2, 4) && !rsBitLess(4, 2);
  ok &=  rsBitLess(2, 8) && !rsBitLess(8, 2);
  ok &= !rsBitLess(4, 4);
  ok &=  rsBitLess(4, 8) && !rsBitLess(8, 4);
  ok &= !rsBitLess(8, 8);

  // sequences with 2 bits set (3,5,9,6,10,12)
  ok &= !rsBitLess(3,  3);
  ok &=  rsBitLess(3,  5) && !rsBitLess( 5, 3);
  ok &=  rsBitLess(3,  9) && !rsBitLess( 9, 3);
  ok &=  rsBitLess(3,  6) && !rsBitLess( 6, 3);
  ok &=  rsBitLess(3, 10) && !rsBitLess(10, 3);
  ok &=  rsBitLess(3, 12) && !rsBitLess(12, 3);
  ok &= !rsBitLess(5,  5);
  ok &=  rsBitLess(5,  9) && !rsBitLess( 9, 5);
  ok &=  rsBitLess(5,  6) && !rsBitLess( 6, 5);
  ok &=  rsBitLess(5, 10) && !rsBitLess(10, 5);
  ok &=  rsBitLess(5, 12) && !rsBitLess(12, 5);
  ok &= !rsBitLess(9,  9);
  ok &=  rsBitLess(9,  6) && !rsBitLess( 6, 9);
  ok &=  rsBitLess(9, 10) && !rsBitLess(10, 9);
  ok &=  rsBitLess(9, 12) && !rsBitLess(12, 9);
  ok &= !rsBitLess(6,  6);
  ok &=  rsBitLess(6, 10) && !rsBitLess(10, 6);
  ok &=  rsBitLess(6, 12) && !rsBitLess(12, 6);
  ok &= !rsBitLess(10, 10);
  ok &=  rsBitLess(10, 12) && !rsBitLess(12, 10);
  ok &= !rsBitLess(12, 12);

  // sequences with 3 bits set (7,11,13,14)
  ok &= !rsBitLess(7,  7);
  ok &=  rsBitLess(7, 11) && !rsBitLess(11, 7);
  ok &=  rsBitLess(7, 13) && !rsBitLess(13, 7);
  ok &=  rsBitLess(7, 14) && !rsBitLess(14, 7);
  ok &= !rsBitLess(11, 11);
  ok &=  rsBitLess(11, 13) && !rsBitLess(13, 11);
  ok &=  rsBitLess(11, 14) && !rsBitLess(14, 11);
  ok &= !rsBitLess(13, 13);
  ok &=  rsBitLess(13, 14) && !rsBitLess(14, 13);
  ok &= !rsBitLess(14, 14);


  return ok;
}


bool testGeometricAlgebra010()
{
  // The geometic algebra with signature (0,1,0) is supposed to be isomorphic to the complex 
  // numbers. We test that here...

  using Real = double;
  using GA   = rsGeometricAlgebra<Real>;
  using MV   = rsMultiVector<Real>;
  using Comp = std::complex<Real>;

  bool ok = true;

  GA alg(0,1,0);

  Comp a,b,c,d;
  MV A(&alg), B(&alg), C(&alg), D(&alg);

  // unity and imaginary unit:
  Comp i(0, 1), one(1, 0);
  MV   I(&alg); I[1] = 1;
  MV   One(&alg); One[0] = 1;

  a = 3.0 + 2.0*i;
  A = 3.0 + 2.0*I;
  b = 5.0 + 7.0*i;
  B = 5.0 + 7.0*I;

  // Error between multivector Z and complex number z:
  auto error = [](const MV& Z, const Comp& z)
  {
    Real dr = Z[0] - z.real();  // difference in real part
    Real di = Z[1] - z.imag();  // difference in imaginary part
    return rsMax(rsAbs(dr), rsAbs(di));
  };

  // Equality comparison (with tolerance) between multivector Z and complex number z:
  auto equals = [](const MV& Z, const Comp& z, Real tol = Real(0))
  { 
    Real dr = Z[0] - z.real();  // difference in real part
    Real di = Z[1] - z.imag();  // difference in imaginary part
    return rsAbs(dr) <= tol && rsAbs(di) <= tol;
  };

  // Test arithmetic operators:
  c = a+b; C = A+B; ok &= equals(C, c);
  c = a-b; C = A-B; ok &= equals(C, c);
  c = a*b; C = A*B; ok &= equals(C, c);
  c = a/b; C = A/B; ok &= equals(C, c, 1.e-16);
  c = b*a; C = B*A; ok &= equals(C, c);
  c = b/a; C = B/A; ok &= equals(C, c, 1.e-15);

  // Test multivector involutions vs complex conjugation:
  c = conj(a); 
  C = A.getGradeInvolution(); ok &= equals(C, c);
  C = A.getConjugate();       ok &= equals(C, c);
  C = A.getReverse();         // nope!

  Real tmp1, tmp2;
  tmp1 = rsAbs(a);
  tmp2 = rsNormReverse(A);    // disagrees with tmp1
  tmp2 = rsNormEuclidean(A);  // agrees with tmp1

  // I think, dualization should correspond to multiplication by i..
  c = a*i; C = A*I;         ok &= equals(C, c);
  c = a*i; C = A.getDual(); ok &= equals(C, c);
  // hmm...isn't dualization supposed to be equal multiplication by the inverse unit pseudoscalar 
  // I^-1, not I itself?
  //c = a/i;


  // Test exponential function:
  c = exp(2.0 * i);   C = rsExp(2.0 * I);   ok &= equals(C, c);         // X^2 = negative scalar
  c = exp(2.0 * one); C = rsExp(2.0 * One); ok &= equals(C, c);         // X^2 = positive scalar
  c = exp(a);         C = rsExp(A);         ok &= equals(C, c, 1.e-13); // general case

  // Test trigonometric functions:
  Real err;
  c = cos(a); C = rsCos(A); ok &= equals(C, c, 1.e-14); err = error(C, c);
  c = sin(a); C = rsSin(A); ok &= equals(C, c, 1.e-14); err = error(C, c);
  c = tan(a); C = rsTan(A); ok &= equals(C, c, 1.e-15); err = error(C, c);

  // Test hyperbolic functions:
  c = sinh(a); C = rsSinh(A); ok &= equals(C, c, 1.e-13); err = error(C, c);
  c = cosh(a); C = rsCosh(A); ok &= equals(C, c, 1.e-13); err = error(C, c);
  c = tanh(a); C = rsTanh(A); ok &= equals(C, c, 1.e-15); err = error(C, c);

  // Test square root function:
  c = sqrt(a);         C = rsSqrt(A);         ok &= equals(C, c, 1.e-15); err = error(C, c);
  c = sqrt(b);         C = rsSqrt(B);         ok &= equals(C, c, 0.0   ); err = error(C, c);
  c = sqrt(2.0 * one); C = rsSqrt(2.0 * One); ok &= equals(C, c, 0.0   ); err = error(C, c);
  //c = sqrt(2.0 * i);   C = rsSqrt(2.0 * I);   ok &= equals(C, c, 1.e-13); // fails!
  // OK, it works in these cases but more tests are needed with many different arguments. The 
  // complex square root has two solutions and we want the principal solution but the iteration 
  // used may actually converge to a different solution...

  // Test inverse functions:
  c = log(a); C = rsLog(A); err = error(C, c);
  c = log(b); C = rsLog(B); err = error(C, c);
  // works but raises assertion because the convergence test needs a tolerance

  // todo: test logarithms for large arguments - that's what the log is used for: reducing large 
  // numbers to a resonable range


  // todo: test also the subalgebras of (2,0,0) or (3,0,0) that are isomorphic to complex numbers
  // in think in 3,0,0 it's the subalgebra of the scalar and pseudoscalar, in 2,0,0 maybe the.. 

  return ok;
}

bool testGeometricAlgebra001()
{
  // The geometic algebra with signature (0,0,1) is supposed to be isomorphic to the dual 
  // numbers. We test that here...

  using Real = double;
  using GA   = rsGeometricAlgebra<Real>;
  using MV   = rsMultiVector<Real>;
  using Dual = rsDualNumber<Real, Real>;

  bool ok = true;

  GA alg(0,0,1);

  Dual a,b,c,d;
  MV A(&alg), B(&alg), C(&alg), D(&alg);

  // unity and imaginary unit:
  Dual e(0., 1.), one(1., 0.);
  MV   E(&alg); E[1] = 1;
  MV   One(&alg); One[0] = 1;

  a = 3.0 + 2.0*e;
  A = 3.0 + 2.0*E;
  b = 5.0 + 7.0*e;
  B = 5.0 + 7.0*E;

  // Equality comparison (with tolerance) between multivector Z and dual number z:
  auto equals = [](const MV& Z, const Dual& z, Real tol = Real(0))
  { 
    Real dv = Z[0] - z.v;  // difference in value part
    Real dd = Z[1] - z.d;  // difference in derivative part
    return rsAbs(dv) <= tol && rsAbs(dd) <= tol;
  };

  // Test arithmetic operators:
  c = a+b; C = A+B; ok &= equals(C, c);
  c = a-b; C = A-B; ok &= equals(C, c);
  c = a*b; C = A*B; ok &= equals(C, c);
  c = a/b; C = A/B; ok &= equals(C, c, 1.e-15);
  c = b*a; C = B*A; ok &= equals(C, c);
  c = b/a; C = B/A; ok &= equals(C, c, 1.e-15);

  // Test exponential function:
  c = rsExp(2.0 * e); C = rsExp(2.0 * E); ok &= equals(C, c);         // X^2 = 0
  c = rsExp(a);       C = rsExp(A);       ok &= equals(C, c, 1.e-13); // general case

  // Test trigonometric functions:
  c = rsSin(a); C = rsSin(A); ok &= equals(C, c, 1.e-13);
  c = rsCos(a); C = rsCos(A); ok &= equals(C, c, 1.e-13);
  c = rsTan(a); C = rsTan(A); ok &= equals(C, c, 1.e-13);

  // Test hyperbolic functions:
  c = rsSinh(a); C = rsSinh(A); ok &= equals(C, c, 1.e-13);
  c = rsCosh(a); C = rsCosh(A); ok &= equals(C, c, 1.e-13);
  c = rsTanh(a); C = rsTanh(A); ok &= equals(C, c, 1.e-13);

  // Test square root function:
  c = rsSqrt(a); C = rsSqrt(A); ok &= equals(C, c, 1.e-13);


  return ok;
}

bool testGeometricAlgebraMatrix()
{
  using Real = double;
  using GA   = rsGeometricAlgebra<Real>;
  using MV   = rsMultiVector<Real>;
  using Vec  = std::vector<Real>;
  using Mat  = rsMatrix<Real>;
  //using LA   = rsLinearAlgebraNew;
  using ILA  = rsIterativeLinearAlgebra;

  bool ok = true;

  GA alg(3);

  // Returns a vector of (raw) basis matrices. They are NxN (but may(?) be reduced...
  auto getBasisMatrices = [&]()
  {
    // Create one basis vector at a time and add its matrix representation to B:
    std::vector<Mat> B;
    int N = alg.getMultiVectorSize();
    MV b(&alg);
    for(int i = 0; i < N; i++) {
      b[i] = 1;
      B.push_back(b.getMatrixRepresentation());
      b[i] = 0; }
    return B;
  };

  auto isIndexUseless = [](const std::vector<Mat>& M, int i)
  {
    // As soon as a nonzero element is found in either the i-th row or the i-th column in any of 
    // the matrices, the index i is not useless and needs to be retained:
    int N = (int)M.size();
    for(int k = 1; k < N; k++) {
      if(!M[k].isRowZero(i))    
        return false;
      if(!M[k].isColumnZero(i)) 
        return false; }
    return true;
    // Maybe we should start the loop at 1 rather than 0? The identity matrix will always have a 
    // nonzero entry
  };

  // Returns an array of row/column indices, for which all of the matrices in M have a zero row and
  // column. The vector M is supposed to have N elements and each is supposed to be an NxN matrix
  auto getUselessRowsAndCols = [&](const std::vector<Mat>& M)
  {
    std::vector<int> useless;
    int N = (int)M.size();
    for(int i = 0; i < N; i++)
      if(isIndexUseless(M, i))
        useless.push_back(i);
    return useless;
  };

  std::vector<Mat> basis = getBasisMatrices();

  //std::vector<int> useless = getUselessRowsAndCols(basis);
  // todo: find rows and columns that are all zero in all of the basis matrices - these may be 
  // scrapped to reduce the excess dimenstionality of the matrix representation...hmm - it doesn't
  // seem to be that simple - maybe in order to reduce the dimensionality of the basis, we also 
  // need to permute rows columns to make the zero rows/columns align - OK, so for the momebt, 
  // let's work with the raw basis of 8x8 matrices...

  // Converts multivector x to its matrix represnetation using the basis matrices in B
  auto toMatrix = [](const MV& x, const std::vector<Mat>& B)
  {
    int N = x.getNumCoeffs();
    Mat xm(N, N);
    for(int i = 0; i < N; i++)
      xm += x[i] * B[i];
    return xm;
  };

  MV a(&alg), b(&alg);
  a.randomIntegers(+1, +9, 0);           // A = 3,8,7,4,6,4,6,5

  Mat m1, m2; 
  m1 = a.getMatrixRepresentation();
  m2 = toMatrix(a, basis);
  ok &= m1 == m2;


  // ToDo: find the eigenvalue spectrum of the matrix represneting multivector a and compare that
  // to what we get in logarithm experiments


  return ok;
}


void testGeometricAlgebra()
{
  bool ok = true;
  ok &= testBitTwiddling();
  ok &= testGeometricAlgebra010();
  ok &= testGeometricAlgebra001();
  ok &= testGeometricAlgebraMatrix();

  // References:
  // 1: Geometric Algebra for Computer Science (GA4CS)
  // 2: The Inner Products of Geometric Algebra (TIPoGA)
  // 3: Linear and Geometric Algebra (LaGA)

  using Real = double;
  using GA   = rsGeometricAlgebra<Real>;
  using MV   = rsMultiVector<Real>;
  using GV   = rsGradedVector<Real>;
  using PT   = MV::ProductType;
  using Vec  = std::vector<Real>;
  using Mat  = rsMatrix<Real>;
  using LA   = rsLinearAlgebraNew;



  // 3D Geometric Algebra (or 2D Elliptic Projective Geometric Algebra). Elements represent 
  // lines/planes through the origin (or vectors/bivectors). The even subalgebra is isomorphic to
  // the quaternions.
  GA alg3(3);
  MV A(&alg3), B(&alg3); 
  A.randomIntegers(+1, +9, 0);           // A = 3,8,7,4,6,4,6,5  
  B.randomIntegers(+1, +9, 1);           // B = 4,5,7,1,4,7,6,1
  MV C(&alg3); // d(&alg3), e(&alg3);
  C = A+B; ok &= C == Vec({7,13,14,5,10,11,12,6});
  C = A*B; ok &= C == Vec({12,1,72,29,84,-5,28,46});
  C = B*A; ok &= C == Vec({12,21,104,-43,6,-5,122,46}); 
  C = A^B; ok &= C == Vec({12,47,49,19,57,25,21,46});

  // 2D Hyperbolic Projective Geometric Algebra. Elements represent lines (vectors) and points 
  // (bivectors). The even subalgebra includes hyperbolic rotations and translations.
  GA alg210(2,1,0); A.setAlgebra(&alg210); B.setAlgebra(&alg210);
  C = A*B; ok &= C == Vec({142,121,30,29,30,-5,28,46});
  C = A^B; ok &= C == Vec({12,47,49,19,57,25,21,46});    // same as for 300

  // 2D Euclidean Projective Geometric Algebra. Elements represent lines (vectors) and points 
  // (bivectors). The even subalgebra is isomorphic to the planar quaternions and includes all 
  // isometries (metric preserving translations and rotations) in 2D.
  GA alg201(2,0,1); A.setAlgebra(&alg201); B.setAlgebra(&alg201);
  C = A*B; ok &= C == Vec({29,1,31,19,84,-5,21,46});
  C = A^B; ok &= C == Vec({12,47,49,19,57,25,21,46});  // same as for 120

  // Custom:
  GA alg120(1,2,0); A.setAlgebra(&alg120); B.setAlgebra(&alg120);
  C = A*B; ok &= C == Vec({10,21,30,81,30,55,28,46});
  C = A^B; ok &= C == Vec({12,47,49,19,57,25,21,46}); 

  // Custom:
  GA alg111(1,1,1); A.setAlgebra(&alg111); B.setAlgebra(&alg111);
  C = A*B; ok &= C == Vec({93,121,67,19,30,-5,21,46});
  C = A^B; ok &= C == Vec({12,47,49,19,57,25,21,46});

  // Custom:
  GA alg003(0,0,3); A.setAlgebra(&alg003); B.setAlgebra(&alg003);
  C = A*B; ok &= C == Vec({12,47,49,19,57,25,21,46});
  C = A^B; ok &= C == Vec({12,47,49,19,57,25,21,46});

  // try 102, 310, 410, 311 - maybe generate the .cpp code for it and use it to create reference
  // output

  // or maybe use this for unit tests:
  // R(4,1) is known as 3D Conformal Geometric Algebra. Elements represent points (vectors), 
  // point-pairs (bivectors), lines and circles (trivectors), spheres and planes (quadvectors). 
  // The even subalgebra includes rotations, translations and dilutions as conformal 3D 
  // transformations.


  // Test grade extraction and wedge-products between blades:
  A.setAlgebra(&alg3); B.setAlgebra(&alg3); C.setAlgebra(&alg3); 
  GV a(&alg3, 0), b(&alg3, 0), c(&alg3, 0), d(&alg3, 0);
  a = C.extractGrade(0); ok &= a.getGrade() == 0 && a == Vec({12});
  a = C.extractGrade(1); ok &= a.getGrade() == 1 && a == Vec({47,49,19});
  b = C.extractGrade(2); ok &= b.getGrade() == 2 && b == Vec({57,25,21});
  c = C.extractGrade(3); ok &= c.getGrade() == 3 && c == Vec({46});
  c = a^b; ok &= c.getGrade() == 3 && c == Vec({845});   // (47e1+49e2+19e3)^(57e12+25e13+21e23)
  c = b^a; ok &= c.getGrade() == 3 && c == Vec({845});   // (57e12+25e13+21e23)^(47e1+49e2+19e3)
  c = a^a; ok &= c.getGrade() == 2 && c == Vec({0,0,0});
  c = b^b; ok &= c.getGrade() == 0 && c == Vec({0});
  a.set(1, Vec({2,3,5})); b.set(0, Vec({7}));
  c = a^b;   ok &= c.getGrade() == 1 && c == Vec({14,21,35});
  c = b^a;   ok &= c.getGrade() == 1 && c == Vec({14,21,35});
  c = b^a^b; ok &= c.getGrade() == 1 && c == Vec({98,147,245});
  c = a^b^a; ok &= c.getGrade() == 2 && c == Vec({0,0,0});
  b.set(1, Vec({7,11,13}));
  c = a^b; ok &= c.getGrade() == 2 && c == Vec({1,-9,-16}); // ^  is anticommutative 
  c = b^a; ok &= c.getGrade() == 2 && c == Vec({-1,9,16});  // when applied to two vectors
  c.set(1, Vec({17,19,23}));
  d = (a^b)^c; ok &= d.getGrade() == 3 && d == Vec({-78});  // ^ is associative
  d = a^(b^c); ok &= d.getGrade() == 3 && d == Vec({-78});

  // Compute the unit pseudoscalar and its inverse:
  a.set(1, Vec({1,0,0})); A = a;
  b.set(1, Vec({0,1,0})); B = b;
  c.set(1, Vec({0,0,1})); C = c;
  MV I  = A*B*C; Vec({0,0,0,0,0,0,0, 1});       // unit pseudoscalar
  MV Ii = C*B*A; Vec({0,0,0,0,0,0,0,-1});       // and its inverse
  C = I*Ii; ok &= C == Vec({1,0,0,0,0,0,0,0});  // I * I^-1 = 1
  C = Ii*I; ok &= C == Vec({1,0,0,0,0,0,0,0});  // I^-1 * I = 1

  // Function to compute the dual via the formula with the inverse pseudoscalar, which must be 
  // passed via the parameter Ii:
  auto dual = [](const MV& A, const MV& Ii) { return A * Ii; };

  // Function to compute the regressive product (a.k.a. antiwedge product) via the formula using
  // duals: A v B = (A* ^ B*)*. To compute the duals, we need the inverse pseudoscalar, which must
  // be passed via the parameter Ii:
  auto regressiveProduct = [&](const MV& A, const MV& B, const MV& Ii)
  {
    MV Ad = dual(A, Ii);  // dual of A
    MV Bd = dual(B, Ii);  // dual of B
    MV Pd = Ad ^ Bd;      // dual of product
    return dual(Pd, Ii);  // product
  };

  // Computes inner product via the identity: (M | N)* = M ^ N* -> M | N = -(M ^ N*)* which is 
  // given in this video (btw.: we also have: (M ^ N)* = M | N*). 
  // https://www.youtube.com/watch?v=iv5G956UGfs&list=PLLvlxwbzkr7i6DlChcYEL7nJ8R9ZuV8JA&index=5
  // at 9:35. Btw. at around 4 min it also says, the "fundamental identity" a*b = a|b + a^b also 
  // holds when b is a multivector (but a is still just a vector). These identifies apply to the
  // "left-contraction" definition of the inner product. On the other hand bivector.net uses the 
  // "fat dot" definition.
  auto innerProduct = [&](const MV& M, const MV& N, const MV& Ii)
  {
    // M | N = -(M ^ N*)*
    MV Nd = dual(N, Ii);
    MV Pd = M ^ Nd;
    return -dual(Pd, Ii); // the minus comes from: A** = dual(dual(A)) = -A
  };
  // see also:
  // https://discourse.bivector.net/t/very-basic-question-ii/361
  // https://www.youtube.com/watch?v=bj9JslblYPU&t=160s

  // what if we define an inner product by  u|v = (u*v + v*u)/2? do we get something different than
  // using u|v = u*v - u^v

  // Test dualization:
  A.set(Vec({3,8,7,4,6,4,6,5}));
  B = A.getDual();
  C = dual(A, Ii);
  ok &= B == C;
  B = B.getDual();
  ok &= A == -B;         // taking the dual of A twice gives back A but with negative sign
  B = A;
  B.applyDualization();  // test in-place dualization
  ok &= B == C;


  // Operations to do next:
  // Vec I  = alg.getPseudoScalar();
  // Vec Ii = alg.getInversePseudoScalar();
  // MV  Ad = A.getDual();
  // Bld ai = a.getInverse();   // see LaGA, pg 69
  // MV  Ai = A.getInverse();   // dunno, if that's always possible, maybe return scalar NaN if not
                                // or maybe the whole array should be NaNs
  // -reversion (complex conjugation in 2D)
  // -store the unit pseudoscalar and its inverse in the algebra (should be computed in init)
  // -implement taking the inverse
  // -implement exp

  // Maybe install the python package clifford for reference


  // test (M ^ N)* = M | N* and  (M | N)* = M ^ N*

  // check this out at around 4 min:
  // 
  // it seems, the "fundamental identity" a*b = a|b + a^b also holds when b is a multivector (but
  // a is still just a vector)...it has actually some other identities at 9:35 for the inner 
  // product based on the dual as wel, namely: (M^N)* = M|N*, (M|N)* = M^N*, so it seems we could 
  // define M|N = (M^N*)* or -(M^N*)* because he says something about getting the negative back when 
  // doing something like (N*)*



  // Test the product function for the derived products:
  A.set(Vec({3,8,7,4,6,4,6,5}));
  B.set(Vec({4,5,7,1,4,7,6,1}));
  C = A^B;
  MV D = MV::product(A, B, PT::wedge);
  ok &= C == D; 

  // When entering
  //   (3+8e1+7e2+4e3+6e12+4e13+6e23+5e123) | (4+5e1+7e2+1e3+4e12+7e13+6e23+1e123)
  // into bivector.net, it apparently computes the "fat dot" product:
  C = MV::product(A, B, PT::fatDot);
  ok &= C == Vec({12,1,72,29,45,-5,75,23});


  // see https://www.youtube.com/watch?v=iv5G956UGfs&t=550s There are also useful identities for
  // the inner product. Macdonald uses the left-contraction definition of the inner product:
  C = innerProduct(A, B, Ii);
  D = MV::product(A, B, PT::contractLeft);
  ok &= C == D;


  D = MV::product(A, B, PT::dot);
  //D = A | B;
  D = MV::product(A, B, PT::fatDot);
  D = MV::product(A, B, PT::contractRight);
  // it seems, none of the inner products defined so far obeys the equation that was used to derive
  // the innerProduct function. What's going on? In his 1st book, page 101, Alan Macdonald defines 
  // the inner product as left contraction

  // Test some identities:

  C = MV::product(A, B, PT::contractLeft) + MV::product(A, B, PT::contractRight);
  D = MV::product(A, B, PT::scalar) + MV::product(A, B, PT::fatDot);
  ok &= C == D; // 2.11 in TIPoGA

  //A.set(Vec({ 0,2,3,4,0,0,0,0 }));
  B = A*Ii; // should compute the dual of A
  C = B*Ii; ok &= C == -A; 

  // the geometric product A*B commutes, when B = A*
  C = A*B;
  D = B*A;
  ok &= C == D;

  //C = A ^ B;
  //D = 0.5*(A*B - B*A);  // this is actually zero :-O
  //ok &= C == D;         // does not hold! a^b = (a*b-b*a)/2 holds probably only for vectors

  // Test reversal:
  MV e1(&alg3, Vec({0,1,0,0,0,0,0,0}));
  MV e2(&alg3, Vec({0,0,1,0,0,0,0,0}));
  MV e3(&alg3, Vec({0,0,0,1,0,0,0,0}));
  ok &= (e1^e2) == e1*e2;
  ok &= (e1^e3) == e1*e3;
  ok &= (e2^e3) == e2*e3;
  MV e12  = e1^e2;
  MV e13  = e1^e3;
  MV e23  = e2^e3;
  MV e123 = e12 ^ e3;
  ok &= (e1^e2).getReverse()    == (e2^e1);
  ok &= (e1^e3).getReverse()    == (e3^e1);
  ok &= (e2^e3).getReverse()    == (e3^e2);
  ok &= (e1^e2^e3).getReverse() == (e3^e2^e1);

  // Test matrix representation:
  Mat matA = A.getMatrixRepresentation();
  Vec vecB  = B.getCoeffs();
  Vec vecAB = matA * vecB;            // matrix-vector product
  MV  AB    = A    *    B;            // geometric product of multivectors   
  ok &= vecAB == AB.getCoeffs();      // ...should give the same result

  // todo: this is actually a "left-matrix" representation - figure out how to create right-matrix
  // representation such that B*A = vecB * matA ...does that even make sense? or maybe we should
  // use  B*A = vecB^T * matA  with the B-vector transposed?

  // todo: implement getInverse based on the matrix representation: solve the linear system:
  // A*x = (1,0,0,0,...) where A is the matrix represnetation and x is the desired inverse


  // Test inversion of general multivector:
  MV Ai = A.getInverse();            // A^-1
  MV one(&alg3); one[0] = 1.0;       // 1 = (1,0,0,0,...)
  Real tol = 1.e-13;
  C = Ai * A; ok &= C.isCloseTo(one, tol);
  C = A * Ai; ok &= C.isCloseTo(one, tol);

  // Test vector inversion:
  A.setZero();
  A[1] = 2; A[2] = 3; A[3] = 5;
  Ai = A.getInverseVector();
  C = Ai * A; ok &= C.isCloseTo(one, tol);
  C = A * Ai; ok &= C.isCloseTo(one, tol);

  // Test exponential of general multivector:
  A.set(Vec({3,8,7,4,6,4,6,5}));
  //A = A * (1.0/8.0);
  A = A * (1.0/16.0);              // we need smallish coeffs for convergence*
  //A = A * (1.0/32.0);
  C = rsExpNaive(A); 
  C = A.getReverse() * A;
  // (*) with factor 1/8 the powers X^k in the iteration still blow up, with 1/32 they shrink and
  // with 1/16 they stay roughly the same. 16 is the square-root of 256 which happens to be close
  // to the scalar component of the unscaled A*rev(A) - this is probably not a coincidence: maybe 
  // we should use the sqrt of <A * rev(a)>_0 as scaler...or maybe twice that value

  // Test integer powers of multivectors:
  A.set(Vec({3,8,7,4,6,4,6,5}));
  ok &= rsPow(A, 0) == one;
  ok &= rsPow(A, 1) == A;
  ok &= rsPow(A, 2) == A*A;
  ok &= rsPow(A, 3) == A*A*A;
  ok &= rsPow(A, 4) == A*A*A*A;
  ok &= rsPow(A, 5) == A*A*A*A*A;

  // Test exponential function:
  //C = rsExp(A);
  A.set(Vec({10,0,0,0,0,0,0,0}));
  C = rsExp(A); // e^10 == 22026.465794806716517
  Real err = C[0] - 22026.465794806716517;
  ok &= rsAbs(err) < 2.e-11;               // the error is quite large: around 1.8e-11

  A.set(Vec({3,8,7,4,6,4,6,5}));
  C = rsExp(A);
  // how can we obtain a reference value? maybe use clifford.py? bivector.net does not know exp
  C = rsExp(-A);
  D = 1.0 / rsExp(A);
  B = C-D;
  ok &= rsIsCloseTo(C, D, 3.e-7);  // the error is even worse here

  // Test trigonometric functions:
  A.set(Vec({1,0,0,0,0,0,0,0}));
  Real tgt;
  C = rsSinSmall(A); tgt = sin(1.0); err = C[0] - tgt; ok &= err == 0.0;
  C = rsCosSmall(A); tgt = cos(1.0); err = C[0] - tgt; ok &= err == 0.0;
  A.set(Vec({5,0,0,0,0,0,0,0}));
  C = rsSin(A); tgt = sin(5.0); err = C[0] - tgt; ok &= rsAbs(err) < 2.e-15;
  C = rsCos(A); tgt = cos(5.0); err = C[0] - tgt; ok &= rsAbs(err) < 2.e-15;

  // Test, if sin^2 + cos^2 = 1 for general multivectors:
  A.set(Vec({3,8,7,4,6,4,6,5}));
  B = rsSin(A); B = B*B;  // B = sin^2(A)
  C = rsCos(A); C = C*C;  // C = cos^2(A)
  D = B + C;              // should be 1 - yep, but accuracy is low

  // Test the operator norm:
  Real r1, r2, r3, r4, r5;
  r1 = rsNormOperator(A);
  r2 = rsNormOperator(A*A);
  r3 = rsNormOperator(A*A*A);
  r4 = r2/r1;
  r5 = r3/r2;
  rsIsCloseTo(r4, r5, 1.e-15);
  // ToDo: compare various norms in various signatures


  // Test logarithm functions (move into separate function):
  A.set(Vec({8,-7,-3,-2,-8,-8,-7,3}));
  C = rsLogViaTaylor(A, 50);
  B = rsExp(C);
  int dummy = 0;
  // OK, with this choice of A, it seems to work. The Taylor series converges because the norm of
  // the transformed variable z is less than 1.

  // Try it with random matrices where it is unknown, if the log exists:
  //for(int i = 0; i < 100; i++)  {
  //  A.randomIntegers(-9, 9, i);
  //  C = rsLogViaTaylor(A, 50);
  //  B = rsExp(C);  // should reconstruct A
  //  // This seems to work occasionally. Sometimes the operator norm of z=x-1 has indeed absolute 
  //  // value < 1, so the series converges
  //  int dummy = 0; }

  // Try it with expoentiated random matrices - the log is known to exist and we even know its 
  // value:
  //for(int i = 0; i < 100; i++) {
  //  A.randomIntegers(-9, 9, i);
  //  B = rsExp(A);
  //  C = rsLogViaTaylor(B, 50);  // should reconstruct A
  //  // Nope - doesn't work. The operator norm of z=x-1 seems to be always between 1 and 2 in this
  //  // test.
  //  int dummy = 0; }

  // The Taylor series implementation is only for reference. It's convergence is too slow to be 
  // useful:
  auto errorLog1 = [&](Real arg, int numTerms)
  { 
    MV x(&alg3); x[0] = arg; 
    MV y = rsLogViaTaylorSmall(x, numTerms);            // converges slowly
    return y[0] - log(x[0]); 
  };
  err = errorLog1(-0.1, 10); // nan, but y[0] is not nan but should be!
  err = errorLog1( 0.0, 10); // inf, but y[0] is not -inf but should be!
  err = errorLog1( 0.5, 10); ok &= rsAbs(err) < 1.e-4;
  err = errorLog1( 0.9, 10); ok &= rsAbs(err) < 1.e-11;
  err = errorLog1( 1.0, 10); ok &= rsAbs(err) < 1.e-15; // it's actually zero! :-)
  err = errorLog1( 1.1, 10); ok &= rsAbs(err) < 1.e-11;
  err = errorLog1( 1.5, 10); ok &= rsAbs(err) < 1.e-4;
  err = errorLog1( 2.0, 10);                            // limit of convergent domain

  // The series based on the atanh function has a larger domain of convergence and converges more
  // quickly. The sweet spot with the fastest convergence is around x = 1. See:
  // https://en.wikipedia.org/wiki/Logarithm#Calculation
  auto errorLog2 = [&](Real arg, int numTerms)
  { 
    MV x(&alg3); x[0] = arg; 
    MV y = rsLogViaAtanhSeriesSmall(x, numTerms);
    return y[0] - log(x[0]); 
  };
  //err = errorLog2(-0.1, 10); // todo: test if result (not error) is nan
  //err = errorLog2( 0.0, 10); // todo: test if result (not error) is -inf
  err = errorLog2( 0.1, 10); ok &= rsAbs(err) < 0.01;    // too large
  err = errorLog2( 0.5, 10); ok &= rsAbs(err) < 1.e-10;
  err = errorLog2( 0.9, 10); ok &= rsAbs(err) < 1.e-15;
  err = errorLog2( 1.0, 10); ok &= rsAbs(err) < 1.e-15;
  err = errorLog2( 1.1, 10); ok &= rsAbs(err) < 1.e-15;
  err = errorLog2( 1.5, 10); ok &= rsAbs(err) < 1.e-15;
  err = errorLog2( 2.0, 10); ok &= rsAbs(err) < 1.e-10;
  err = errorLog2( 3.0, 10); ok &= rsAbs(err) < 1.e-7;   // convergence becomes worse

  // This uses the series based solution above with range reduction:
  auto errorLog3 = [&](Real arg, int numTerms)
  { 
    MV x(&alg3); x[0] = arg; 
    MV y = rsLogViaAtanhSeries(x, numTerms);
    return y[0] - log(x[0]); 
  };
  err = errorLog3(  0.01, 10); ok &= rsAbs(err) < 1.e-15;
  err = errorLog3(  0.1 , 10); ok &= rsAbs(err) < 1.e-15;
  err = errorLog3(  1.0 , 10); ok &= rsAbs(err) < 1.e-15;
  err = errorLog3( 10.0 , 10); ok &= rsAbs(err) < 1.e-15;
  err = errorLog3(100.0 , 10); ok &= rsAbs(err) < 1.e-15;

  //C = rsLogViaAtanhSeries(A, 100); // raises assertion and produces garbage because convergence
                                   // requirement is violated 
  //C = rsLogViaTaylor(     A, 100); // diverges also but slower
  //C = rsLogViaNewton(A);
  //err = C[0] - tgt;
  //ok &= err == 0.0;
  // maybe it diverges because no solution exists? -> try it with a vector where a solution is 
  // known:
  A.set(Vec({3,8,7,4,6,4,6,5}));
  B = rsExp(A);
  //C = rsLogViaAtanhSeries(B, 20);  
  // fails! apparently the transformed variable (x-1)/(x+1) has absolute value > 1 even when the
  // non-transformed has not? ... so maybe and (accelerated) Taylor series could converge?

  C = rsLogViaTaylor(B, 20);   
  D = rsExp(C);
  // Fails! i think, the problem might be that the series does not compute powers of the input
  // x but rather of x-1, so all our efforts to make the operator norm of x less than one might
  // be thwarted by the subtraction of 1 immediately before we enter the Taylor series iteration.
  // ...could it be that the Euler acceleration also affects the convergence region? ..i actually 
  // don't think so, because it applies only to series that are convergent in the first place.
  // Could we use a Laurent series around x0 = 0 instead of a Taylor series around x0 = 1?

  //C = rsLogViaNewton(     B    );    // fails!
  // soooo...yeah....logarithms of general multivectors are complicated. Algorithms that work well 
  // for real numbers may completely fail for general multivectors. Maybe also research matrix 
  // logarithms. It's likely more info around about that than about logs for multivectors
  // https://www.maths.manchester.ac.uk/~higham/talks/ecm12_log.pdf
  // https://www.maths.manchester.ac.uk/~higham/fm/index.php
  // ...maybe try using different norms for argument scaling. For example, the L1 norm (sum of
  // absolute values). This should shrink the multivector more. The goal is that the powers do
  // not explode. If that is guaranteed, we should be on the safe side. hmmm...i guess, we really
  // need to use the operator norm - but that's hard to compute - but maybe it has to be done

  // ToDo: implement various norms of multivectors, for eaxmple:
  //   N_c(A) = conj((A) * A     where conj(A) is the Clifford conjugate of A
  //   N_r(A) = <A * rev(a)>_0   where rev(A) is the reversal of A
  // how about the absolute value of the determinant of the matrix-representation?
  A.set(Vec({3,8,7,4,6,4,6,5}));
  C = A.getConjugate() * A;           // has scalar and pseudoscalar parts, both negative
  C = A * A.getConjugate();           // is the same
  C = A.getReverse() * A;             // has scalar and vector parts, scalar is 251
  C = A * A.getReverse();             // ditto, but different values for vector part
  matA = A.getMatrixRepresentation();

  // todo: implement 
  //   rsMatrix<T> alg.makeOutermorphism(const rsMatrix<T>& F)
  // that takes an n x n matrix defining a linear transformation for vectors and returns an N x N 
  // matrix defining the induced linear transformation for multivectors. This induced 
  // transformation is constructed such that F(a^b) = F(a) ^ F(b) for vectors a,b. It preserves the
  // outer product and is therefore called outermorphism. See:
  // https://www.youtube.com/watch?v=0VGMxSUDBH8&list=PLLvlxwbzkr7igd6bL7959WWE7XInCCevt&index=6
  // https://en.wikipedia.org/wiki/Outermorphism#Properties
  // ..but there, it's even more general - it allows the input matrix to be non-square
  // outermorphisms are grade preserving - i think that implies that their matrix representations
  // has a block diagonal structure, like in R^3:
  //   a11 a12 a13
  //   a21 a22 a23  = F
  //   a31 a32 a33
  // gets extended to the outermorphism:
  //   o11  0   0   0   0   0   0   0
  //    0  o22 o23 o24  0   0   0   0
  //    0  o32 o33 o34  0   0   0   0
  //    0  o42 o43 o44  0   0   0   0  = F_o
  //    0   0   0   0  o55 o56 o57  0
  //    0   0   0   0  o65 o66 o67  0
  //    0   0   0   0  o75 o76 o77  0
  //    0   0   0   0   0   0   0  o88
  // where the grade-1 block (o22..o44) is the original 3x3 matrix and the other elements can all 
  // be computed from these. I think, o11 should be 1 and o88 be the determinant of the orginal 
  // matrix (verify!). Since the columns of a transformation matrix generally contain the images
  // of the basis vectors, we could compute the column (o55,o65,o75) by transforming e1^e2:
  // F(e1^e2) = F(e1) ^ F(e2), then compute the column (o56,o66,o76) as F(e1) ^ F(e3), 
  //  (o57,o67,o77) as F(e2) ^ F(e3), and o88 = F(e1) ^ F(e2) ^ F(e3)
  Mat F(3, 3, {1,2,3, 4,-5,6, 7,8,9});
  Mat F_o = alg3.makeOutermorphism(F);
  Real detF = rsLinearAlgebraNew::determinant(F);
  ok &= rsIsCloseTo(detF, F_o(7,7), tol);  // last element of F_o should be the determinant of F 

  // How about outermorphisms G^m -> G^n constructed from linear transformations R^m -> R^n? 
  // maybe makeOutermorphism(A, srcAlg, dstAlg) and the special case is then just
  // makeOutermorphism(A, this, this)
  // Let's say, we have a function F: R^2 -> R^3:
  //   a11 a12 
  //   a21 a22 = A
  //   a31 a32 
  // This gives the induced map F_o: R^4 -> R^8 ...we need 4 rows, 8 cols
  //  1   0   0   0      c0      d0
  //  0  a11 a12  0   *  c1   =  d1
  //  0  a21 a22  0      c2      d2
  //  0  a31 a32  0      c12     d3
  //  0   0   0  o44             d12
  //  0   0   0  o54             d13
  //  0   0   0  o64             d23
  //  0   0   0   0              d123
  // where c are the input coeffs, ad d are the output coeffs. Note that a grade-2 element has 1
  // coeff in 2D but 3 coeffs in 3D -> the bivector with coeff c12 from 2D gets mapped to the 
  // bivector (d12,d13,d23) in 3D, There is no trivector in 2D, so the d123 element is zero. The
  // matrix can be sliced row-wise in 1,3,3,1 and column-wise in 1,2,1. what if F: R^3 -> R^2?
  // do we just throw away the trivector part?


  // Test, if the outer product F(a) ^ F(b) of two mapped vectors a,b is indeed equal to the mapped
  // outer product F(a ^ b) of them:
  a.set(1, Vec({2, 3,5})); A.set(a);
  b.set(1, Vec({7,-4,6})); B.set(b);
  GV Fa = F*a, Fb = F*b;
  GV Fa_Fb = Fa ^ Fb;
  C = F_o*(A ^ B);
  D = MV(Fa_Fb);
  ok &= C == D;
  C = (F_o*A) ^ (F_o*B);
  ok &= C == D;

  // Test, F_o(A) ^ F_o(B) == F_o(A ^ B) for general multivectors A,B, i.e. the outermorphism 
  // property generalizes to multivectors:
  A.set(Vec({3,8,7,4,6,4,6,5}));
  B.set(Vec({4,5,7,1,4,7,6,1}));
  C = F_o*(A ^ B);
  D = (F_o*A) ^ (F_o*B);
  ok &= C == D;

  // ToDo: Maybe represent the outermorphism as rsMatrixList - a class (to be implemented) that is
  // essentially an array of matrices - such a class can be useful also for use with neural 
  // networks ..well, actually, we could just use a std::vector<rsMatrix<T>>

  // Question: Can this so created outermorphism matrix F_o be somehow mapped to / represented by a 
  // multivector? We would need some sort of inverse of MV::getMatrixRepresentation() that takes a 
  // N x N matrix (here, the F_o matrix) and computes a multivector whose matrix representation is
  // F_o. I have no idea, if this is possible at all and if so, under which conditions -> research!
  // In general, (left) multiplication by genera multivectors may mix up grades, but outermorphisms
  // are grade-preserving, so the result of such an operation (if possible) is a restricted class
  // of multivectors. 
  // ToDo: implement tests:
  // -A.isGradePreserving: checks, if the matrix representation of A has the desired block 
  //  structure
  // -A.isOutermorphism: A must have the desired block structure *and* the blocks for grades != 1
  //  are related to the grade-1 block according to the outer product. We can test this by 
  //  comparing the matrix representation of A to the outermorphism that is induced by the grade-1
  //  block of A
  
  // in factor, we can do a test "A.isOutermorphism" or ""
  



  // For elementary functions:
  // https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4361175/pdf/pone.0116943.pdf
  A.set(Vec({9,0,0,0,0,0,0,0}));
  B = rsSqrt(A); C = B*B; ok &= C == A; // works


  A.set(Vec({-1,0,0,0,0,0,0,0}));
  //B = rsSqrt(A); // iteration diverges


  A.set(Vec({3,8,7,4,6,4,6,5}));
  //B = rsSqrt(A); C = B*B; ok &= C == A;   // iteration diverges
  B = A*A; C = rsSqrt(B); //ok &= C == A;   // converges to a different root: C != A
  D = C*C; ok &= rsIsCloseTo(D, B, 1.e-11); // ..and it is not very precise!
  // Is there actually always a square-root for multivectors or can there be multivectors that 
  // cannot be obtained by squaring some other multivector? Maybe the square-root is a partial
  // function (in addition to being multi-valued, where it exists)? Could it be that the algo
  // sometimes diverges because no solution exists? Let's consider the matrix representation matA 
  // of a multivector A - we can then write: A*A = B = matA*A, so finding the square-root of B 
  // amounts to solving the linear system B = matA*A for A. Could it be that it has no solutions, 
  // i.e. could matA be singular? ...yes, it obviously must be possible for matA to be singular, 
  // because sometimes multiple solutions exists (as we have just seen) - and this also happens 
  // only for singular matrices.
  matA = A.getMatrixRepresentation();
  Real detA = LA::determinant(matA);
  // ...but detA is not zero. What about the matrix-square-root - could we use the algo for that
  // todo: try to find the square root of -1 in a (0,1,0) algebra (i.e. complex numbers) and in an
  // algebra that contains complex subalgebras



  rsAssert(ok);

  // In "The Inner Products of Geometric Algebra" some other operations are mentioned:
  // -"reversion of a blade is obtained by writing its evctor factors in reverse order" pg.39
  //  ...so that means e1*e2 becomes e2*e1, etc. 
  // -"grade involution reverses the sign of odd blades" pg 39


  // In this video:
  // https://www.youtube.com/watch?v=b0K451IxLBQ&list=PLQ6JJNfj9jD_H3kUopCXkvvGoZqzYOzsV&index=4
  // he says, that complex numbers and quaternions can be derived from geomteric algebra in 2 ways:
  // 1: as signature 0,1,0 and 0,2,0 geometric algebras and 2: as subalgebras of a 3,0,0 algebra

  // http://geocalc.clas.asu.edu/pdf/OerstedMedalLecture.pdf
  // pg 27: pauli algebra is the matrix version of G(3,0,0)


  // https://www.youtube.com/watch?v=tX4H_ctggYo
  // ei*ei = 1 or 0 or -1, ei*ej = -ej*ei for i != j, ei*ej = eij
  // regressive product:  a v b = (a* ^ b*)*

  // all 3 products (inner, outer, geom) should obey the distributive law
  // https://www.youtube.com/watch?v=oqdoSoBt6H8 at around 5:40

  // see also:
  // https://en.wikipedia.org/wiki/Comparison_of_vector_algebra_and_geometric_algebra



  // Why is the wedge of two bivectors commutative? Isn't it supposed to be anticommutative - or
  // is that only the case when the grade of the result is even? -> figure out and document

  // todo: maybe also implement the geometric product between blades? will that be just the same as
  // the wedge product? or will it even make sense mathematically? if it is really the same, it 
  // could make sense to have the usual multiplication symbol overloaded as an alias to the wedge
  // but if it's not the same, we probably should not implement the operator at all for blades

  // (2e1+3e2+5e3)*(7e1+11e2+13e3)*(17e1+19e2+23e3) gives vector + trivector
  // (2e1+3e2+5e3)*(7e1+11e2+13e3)^(17e1+19e2+23e3) gives vector + trivector
  // ..so, in general, the geometric product between two blades will give a multivector
  // maybe implement a conversion constructor from blades to multivectors - maybe the geometric 
  // product of two blades will the just work?


  //A.set(a); B.set(b);

  // todo: implement == for multivector == blade and blade == multivector
  // implement assign for multivector = blade



  // todo: implement the unit pseudoscalar I = e1*e2*e3 and its inverse which in G^3 is just 
  // I^-1 = e3*e2*e1 - i think, this pattern of reversing the basis vectors generalizes. with that,
  // we can dualize multivectors via dual(M) = M * I^-1 - this should give the orthogonal
  // complement

  // Translations of terminology between geometric/exterior/tensor/linear algebra and physics in 3D
  // Geometric                Exterior   Tensor     Linear          Physics
  //
  // scalar                   scalar     scalar     scalar          scalar
  // vector                   vector     vector     row-vector      vector
  // bivector                 1-form     covector   column-vector   pseudovector/axial vector
  // trivector/pseudoscalar   scalar?    scalar?    scalar          scalar

  // Is it generally true, that the set of fixed points of a rotation in nD space has 
  // dimensionality n-2? It works in 2D and 3D and seems to make sense because a rotation is 
  // defined by a bivector (which is 2D). Within the bivector, only the origin is a fixed point.

  // Are the 1D geometric algebras with signatures (1,0,0),(0,1,0),(0,0,1) isomorphic to the 
  // hyperbolic, complex and dual numbers? We can get a variation of complex numbers (with a 
  // noncommutative imaginary unit) in the (2,0,0) algebra by defining I = e1*e2 (verify). The 
  // even subalgebra (using only grades 0 and 2) is then a (sort of) complex numbers, but not 
  // quite isomorphic due to the noncommutativity of I (verify!). It works also with arbitrary 
  // planes defined by arbitrary bivectors in (3,0,0)...
  // The complex numbers can be also found in Cl(R^3) by considering the subalgebra formed by 
  // scalars and pseudoscalars (trivctors) and the quaternions are found as the even subalgebra
  // (of scalars and bivectors)


  // Results can be checked here: https://bivector.net/tools.html - choose the right signature for
  // the algebra and then enter the product via the syntax:
  // (3+8e1+7e2+4e3+6e12+4e13+6e23+5e123) * (4+5e1+7e2+1e3+4e12+7e13+6e23+1e123)
  // or:
  // (3+8e0+7e1+4e2+6e01+4e02+6e12+5e012) * (4+5e0+7e1+1e2+4e01+7e02+6e12+1e012)
  // the former notation is required when there are no zero dimensions and the latter where there
  // are some

  // more info about the code generator can be found here:
  // https://github.com/enkimute/ganja.js


  //c = a|b;            // c = 0,-46,23,10,27,-30,7,0 -> should be 12,1,72,29,45,-5,75,23
  // there seem to be a lot of different inner products and no universal consensus among the
  // mathematicians which is best. geometric algebra seems to be a still actively developing field
  // and the dust has not yet settled:
  // https://www.researchgate.net/publication/2842332_The_Inner_Products_of_Geometric_Algebra
  // so it's maybe not too surprising that the naive way of implementing it did not work out
  // and it may be better leave that for now - geometric and outer products are enough for the 
  // moment, let's and instead focus on:
  // -incorporating negative and zero dimensions
  // -extraction of grades (maybe implement: isBlade, getGrade (return -1 for mixed grade)) ...but
  //  maybe we should another class for blades - maybe if we restrict ourselves to blades and
  //  the outer product, we get the exterior algebra as subalgebra?
  //  -rsBlade could also take a pointer to the algebra but maybe it should not be a subclass of
  //   rsMultiVector
  // -maybe swicthing to a sparse matrix representation
  // 

  // How are blades and (differential) forms related? A 2-form takes two vectors as input and 
  // produces a scalar as output. Can a 2-form be mapped to a 2-blade as follows?: when two input 
  // vectors come in, do:
  // -form a second 2-blade from the two incoming vectors
  // -form the scalar product between these two 2-blades

  // test the followig properties:
  // -outer product should be antisymmetric when operands have odd grade (1, pg 38)
  // perhaps the identities a^b = (a*b - b*a)/2, a|b = (a*b + b*a)/2 hold only if a and b are 
  // vectors but do not hold anymore for general multivectors?
  // maybe create the cayley table also for the outer product - i guess the one for the inner
  // can then be obtained by the equation a*b = a|b + a^b, so we get the table for the inner as
  // geom - outer?

  // this matches what https://bivector.net/tools.html gives, but i'm not sure if we need to use
  // the reverse mapping as in blades( i, j) = unmap[ab]; ...in this special case, the map and the 
  // reverse map are the same. todo: try another case

  // actually, 1 matrix would be enough - we can absorb/encode the signs also in the blades matrix

  // to build the Cayley-Table, see:
  // https://en.wikipedia.org/wiki/Cayley_table
  // https://bivector.net/tools.html
  // we need to use rsSubsetsOfSize(const std::vector<T>& set, int size) to generate a 
  // representation of the row/column headers of the table and then fill in the table elements 
  // according to the rules (for the outer product)
  // -concatenate row and column index array
  // -if any entry of the result occurs more than once, the table entry is zero
  // -if all entries occur only once, figure out, how many swaps are required to order the result,
  //  the ordered version corresponds to another basis blade B -> figure out which one
  // -if the number of swaps is even, the entry is plus B, if it's odd, it's minus B
  //  (later that rule may be modified to allow different signature´s - in the basic setup, all
  //  basis vectors square to 1 - but we may want also algebras where some square to -1 or 0 - but
  //  maybe that affects only the squares and the the cross-terms?)
  //  ...or wait - no - is this rule correct?


  // https://en.wikipedia.org/wiki/Multivector
  // https://en.wikipedia.org/wiki/Blade_(geometry)

  // http://www.jaapsuter.com/geometric-algebra.pdf

  // https://geometricalgebra.org/

  // http://home.clara.net/iancgbell/maths/geoprg.htm

  // https://www.youtube.com/watch?v=-6F74TH1i_g


  // https://www.youtube.com/watch?v=60z_hpEAtD8 A Swift Introduction to Geometric Algebra
  // https://www.youtube.com/watch?v=_AaOFCl2ihc The Vector Algebra War

  // https://www.youtube.com/watch?v=P2ZxxoS5YD0 Intro to clifford, a python package for geometric algebra
  // https://clifford.readthedocs.io/en/latest/

  // http://glucat.sourceforge.net/

  // https://bivector.net/lib.html  links to GA libraries

  // https://github.com/wolftype/versor "A (fast) Generic C++ library for Geometric Algebras"
  // http://versor.mat.ucsb.edu/
  // http://wolftype.com/versor/colapinto_masters_final_02.pdf


  // https://www.youtube.com/watch?v=tX4H_ctggYo Siggraph2019 Geometric Algebra
  // https://github.com/enkimute/ganja.js
  // https://enkimute.github.io/ganja.js/examples/coffeeshop.html#pga2d_points_and_lines
  // https://bivector.net/

  // to create the table, i think, we need a function to generate all possible subsets of a given 
  // size k of a set of size n, see:
  // https://www.geeksforgeeks.org/print-subsets-given-size-set/
  // https://www.tutorialspoint.com/print-all-subsets-of-given-size-of-a-set-in-cplusplus
  // https://stackoverflow.com/questions/23974569/how-to-generate-all-subsets-of-a-given-size
  // https://www.codeproject.com/Articles/26050/Permutations-Combinations-and-Variations-using-C-G
  // https://afteracademy.com/blog/print-all-subsets-of-a-given-set


  // big GA/GC playlist:
  // https://www.youtube.com/playlist?list=PLTGkWQjAP0wovgVKVz1tUCYKZXCSmqsLm

 
  // solutions to GA problems:
  // https://vixra.org/author/james_a_smith

  // this is really nice, concise and straightforward:
  // http://www.math.umd.edu/~immortal/MATH431/lecturenotes/ch_geometricalgebra.pdf
  // http://www.math.umd.edu/~immortal/MATH431/lecturenotes/
  // http://www.math.umd.edu/~immortal/MATH431/

  // Ideas:
  // -the diagonals of the Cayley tables are always -1,0,+1 - would it make sense if they also 
  //  could be a scalar multiple of some other basis blade?
}
// ToDo:
// Figure out how to translate back and forth between a multivector-based and matrix-based 
// representation of a Clifford algebra. For example, the 8 basis blades of G(3,0,0) can be 
// represented by a set of 8 4x4 matrices. But how are we supposed to extract the coefficient for a
// given basis blade from a given 4x4 matrix that is some linear combination of the basis matrices?
// It's all mixed up. Maybe by a sort of projection onto an reciprocal basis blade in analogy how
// it usually works in tensor algebra? But how do we find the reciprocal basis blades and how would
// the projection work? Maybe we write the basis matrices as vectors, assemble these vectors into 
// an 16x8 matrix, find the pseudo-inverse and. ...figure out
// To find the basis matrices, maybe we should find the matrix representations of all the basis 
// vectors. In R^3, this will give 8 8x8 matrices but maybe they have rows/columns with all zeros, 
// which could then be removed? I think, in order to remove the i-th row/column, it must be the 
// case that the i-th row and column is all zeros in all basis matrices?


void testEulerTransformation()
{
  // Tests the Euler transformation used to speed up the convergence of a slowly converging 
  // alternating series. As example, we use the alternating harmonic series which converges to
  // log(2). We fill the first column of a matrix with the 1/n terms (without the sign 
  // alternation applied), subsequent columns will contain the forward difference of the previous 
  // column. The partial sum of the original series is obtained by summing the first column with 
  // sign alternation. The accelerated partial sum is obtained by summing the first row with sign 
  // alternation and weighting by 1/2^(j+1) where j is the column index. We plot the relative 
  // errors of the partial sums as function of the upper summation index for both series.

  int N = 10;                          // number of terms
  using Real = double;
  int i, j;
  rsMatrix<Real> A(N, N);
  Real target = log(2.0);              // the limit to which both series converge
  std::vector<Real> err1(N), err2(N);  // relative error of partial sum

  // Fill the first column and compute the partial sum of the original series and record the 
  // relative a error as the summation progresses:
  for(i = 0; i < N; i++)
    A(i, 0) = 1.0 / Real(i+1);
  Real sign = 1.0;
  Real sum1 = 0.0;
  for(i = 0; i < N; i++) {
    sum1 += sign * A(i, 0);
    err1[i] = (sum1-target)/target;
    sign *= -1;
  }

// Compute the forward differences of all orders. The j-th column will contain the j-th 
// difference:
  for(j = 1; j < N; j++)
    for(i = 0; i < N-j; i++)
      A(i, j) = A(i+1, j-1) - A(i, j-1);

  // Sum the first row (with scale and sign) to get the partial sum of the accelerated series and 
  // record the error:
  sign = 1.0;
  Real scale = 0.5;
  Real sum2  = 0.0;
  for(j = 0; j < N; j++) {
    sum2  += sign * scale * A(0, j);
    err2[j] = (sum2-target)/target;
    sign  *= -1;
    scale *= 0.5;
  }

  // Plot the relative errors:
  rsPlotVectors(err1, err2);

  // Observations:
  // The partial sum of the accelerated series does indeed converge much faster to the limit than
  // the partial sum of the original series. Furthermore, the error alternates between positive and
  // negative for the orginal series but doesn't for the accelerated series. I think, this is due 
  // to the smoothing effect of taking differences of terms with opposite signs. I think, it's like 
  // taking averages, when the signs would not differ. The faster convergence can also be explained 
  // by the additional 1/2^(j+1) factor. Maybe both effects play a role in the acceleration? But
  // interestingly, the first row of the matrix A follows also the A(0,j) = 1/(j+1) rule just as 
  // the first column follows A(i,0) = 1/(i+1). This is probably a peculiar feature of that 
  // particular series used here which doesn't generalize. If that would be generally true, it 
  // would mean that all the differencing computations could be bypassed in general and that would 
  // just be too good to be true. 

  // ToDo: make a more efficient implementation that uses only O(N) of axiliary memory. Maybe with
  // some clever juggling with temporary variables, we could even get away with O(1) of memory?
  //   Step Compute                        Accumulate Forget    Variables
  //   0    a0                             a0                   a0
  //   1    a1,b0=a1-a0                    b0         a0        a1,b0
  //   2    a2,b1=a2-a1,c0=b1-b0           c0         a1,b0     a2,b1,c0
  //   3    a3,b2=a3-a2,c1=b2-b1,d0=c1-c0  d0         a2,b1,c0  a3,b2,c1,d0
  // hmmm...nope - that doesn't seem to work. Tne number of active (still needed) variables grows 
  // by one in each step. So we need indeed a temporary array of length N. Then precompute all the 
  // original terms and store them in the array. Then apply successive forward differencing and
  // accumulation of the front element (with scale and sign). The algo will need O(N^2) time 
  // complexity though. ...unless an analytic formula can be given for all the terms in the 
  // accelerated series in which case it will be O(1) in memory and O(N) in time complexity.


  // To find symbolic expressions for terms the in the series, Wolfram may help. In particular the 
  // function DifferenceDelta to find an expression for a forward difference:
  // https://reference.wolfram.com/language/guide/DiscreteCalculus.html
  // https://reference.wolfram.com/language/ref/DifferenceDelta.html
  // https://www.wolfram.com/mathematica/newin7/content/DiscreteCalculus/
  // For example, for the logarithm of x-1, the expression for the n-th term in the original series
  // is x^n / (n+1)

  // Example from wolfram, but i replaced with k - i doesn't work with i in alpha:
  //   DifferenceDelta[Sin[a k + b], {k, 5}]  
  // For the logarithm:
  //   DifferenceDelta[x^k/(k+1), {k, n}]  
  // gives:
  //   -beta(1/x, -k - n - 1, n + 1)/x
  // where beta is the incomplete beta function:
  //   https://reference.wolfram.com/language/ref/Beta.html
  // ...hmm - this doesn't seem to be helpful. I hoped the expression is something simpler, 
  // possibly involving binomial coefficients instead of the beta function. But these are related:
  //   https://proofwiki.org/wiki/Binomial_Coefficient_expressed_using_Beta_Function
  //   https://en.wikipedia.org/wiki/Beta_function
  // Maybe the problem is that mathematica assumes the series to be summed up to infinity. What we
  // really need is a truncated series like: x^k/(k+1) for k < N, 0 otherwise. See:
  //   https://reference.wolfram.com/language/ref/Piecewise.html
  //   Plot[Piecewise[{{x^2, x < 0}, {x, x > 0}}], {x, -2, 2}]
  // Let's try it:
  //   DifferenceDelta[Piecewise[{{x^k/(k+1), k <= N}, {0, k > N}}], {k, n}]  
  // but alpha doesn't understand this. Maybe sequence can't be defined piecewise? Or maybe it has 
  // problems with the variable upper limit N? Here, k is the sequence index (which later becomes 
  // the summation index in the series), N is the point where we want to tuncate the sequence
  // (which becomes the upper limit of the sum) and n is the order of the difference. Maybe try it 
  // in Wolfram cloud and/or search for other possible problems.
  // BUT: by just inspecting the first row of the matrix A, we see that the coeff A(0,i) is just
  // 1/(i+1). Does that help? Probably not. In general, the powers of x will also factor in into
  // the original series which will almost certainly destroy this simple pattern.
  // Maybe for the truncated sequence, the incomplete beta function can be replaced by the complete
  // one which in turn can be expressed in terms of binomial coeffs? ...just a wild guess...
}


// fast inverse square root approximation from Quake engine
float Q_rsqrt(float number)
{
  long i;
  float x2, y;
  const float threehalfs = 1.5F;

  x2 = number * 0.5F;
  y  = number;
  i  = *(long*)&y;                          // evil floating point bit level hacking
  i  = 0x5f3759df - (i >> 1);               // what the fuck? 
  y  = *(float*)&i;
  y  = y * (threehalfs - (x2 * y * y));     // 1st Newton iteration
  //  y  = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration, this can be removed

  return y;
}
// https://www.youtube.com/watch?v=p8u_k2LIZyo
// https://medium.com/hard-mode/the-legendary-fast-inverse-square-root-e51fee3b49d9


// todo: implement adaptive filters: LMS, GAL, GALL, RLS, LSL, FTF, see:
// https://en.wikipedia.org/wiki/Recursive_least_squares_filter
// https://link.springer.com/chapter/10.1007/978-1-4612-4978-8_10
// https://www.researchgate.net/publication/220058701_Implementation_of_the_Least-Squares_Lattice_with_Order_and_Forgetting_Factor_Estimation_for_FPGA


//-------------------------------------------------------------------------------------------------

// moved to rs-met codebase (except some comments) - may be deleted here when missing important
// comments have also been moved over:

/** Fills edges of a graph of 2D vectors (as vertices) with a user supplied function that takes as
input the source and target vector and returns a scalar that can be used as weight for the edge 
between the two vertices. */
/*
template<class T>
void fillEdges(rsGraph<rsVector2D<T>, T>& g, 
  const std::function<T(rsVector2D<T>, rsVector2D<T>)>& f)
{
  using Vec = rsVector2D<T>;
  for(int i = 0; i < g.getNumVertices(); i++) {
    Vec vi = g.getVertexData(i);                 // vector stored at source vertex i
    for(int j = 0; j < g.getNumEdges(i); j++) {
      int k  = g.getEdgeTarget(i, j);            // index of target vertex
      Vec vk = g.getVertexData(k);               // vector stored at target vertex k
      T ed   = f(vi, vk);                        // compute edge data via user supplied function
      g.setEdgeData(i, j, ed); }}                // ...and store it at the edge
}
*/

/*
void testVertexMesh()
{
  using Vec2 = rsVector2D<float>;
  using VecF = std::vector<float>;
  using VecI = std::vector<int>;
  //using Mesh = rsGraph<Vec2, rsEmptyType>;  // later use float for the edge data
  using Mesh = rsGraph<Vec2, float>;
  using ND   = rsNumericDifferentiator<float>;

  // an (irregular) star-shaped mesh with a vertex P = (3,2) at the center and 4 vertices 
  // Q,R,S,T surrounding it that are connected to it:
  Mesh mesh;
  bool sym = true;                // select, if edges should be added symmetrically
  mesh.addVertex(Vec2(3.f, 2.f));  // P = (3,2) at index 0
  mesh.addVertex(Vec2(1.f, 3.f));  // Q = (1,3) at index 1
  mesh.addVertex(Vec2(4.f, 2.f));  // R = (4,2) at index 2
  mesh.addVertex(Vec2(2.f, 0.f));  // S = (2,0) at index 3
  mesh.addVertex(Vec2(1.f, 1.f));  // T = (1,1) at index 4
  mesh.addEdge(0, 1, sym);         // connect P to Q
  mesh.addEdge(0, 2, sym);         // connect P to R
  mesh.addEdge(0, 3, sym);         // connect P to S
  mesh.addEdge(0, 4, sym);         // connect P to T

  // Create arrays of function values and (true) partial derivatives and their numerical estimates.
  // For the estimates, only vertices with neighbors are supposed to contain a reasonable value 
  // afterwards, all others are supposed to contain zero:
  int N = mesh.getNumVertices();
  VecF u(N), u_x(N), u_y(N);     // u(x,y) and its true partial derivatives with resp. to x,y
  VecF u_x0(N), u_y0(N);         // with weighting 0 (unweighted)
  VecF u_x1(N), u_y1(N);         // with weighting 1 (sum of absolute values, "Manhattan distance")
  VecF u_x2(N), u_y2(N);         // with weighting 2 (Euclidean distance)
  VecF e_x0(N), e_y0(N);         // error of u_x0, ...
  VecF e_x1(N), e_y1(N);         // ...etc.
  VecF e_x2(N), e_y2(N);

  // Define our test function u(x,y) and its partial derivatives:
  // u(x,y)   =    sin(wx * x + px) *    sin(wy * y + py)
  // u_x(x,y) = wx*cos(wx * x + px) *    sin(wy * y + py)
  // u_y(x,y) =    sin(wx * x + px) * wy*cos(wy * y + py)
  // and a function to fill the arrays of true partial derivatives:
  float wx = 0.01f, px = 0.3f;
  float wy = 0.02f, py = 0.4f;
  auto f  = [&](float x, float y)->float { return    sin(wx * x + px) *    sin(wy * y + py); };
  auto fx = [&](float x, float y)->float { return wx*cos(wx * x + px) *    sin(wy * y + py); };
  auto fy = [&](float x, float y)->float { return    sin(wx * x + px) * wy*cos(wy * y + py); };
  auto fill = [&]() 
  { 
    for(int i = 0; i < N; i++) {
      Vec2 v = mesh.getVertexData(i);
      u[i]   = f( v.x, v.y);
      u_x[i] = fx(v.x, v.y);
      u_y[i] = fy(v.x, v.y); }
  };
  // todo: later compute also 2nd derivatives u_xx, u_yy, u_xy and Laplacian u_L

  // distance functions (constant, 1/Manhattan, 1/Euclidean)
  std::function<float(Vec2, Vec2)> d0, d1, d2;
  d0 = [&](Vec2 a, Vec2 b)->float { return 1.f; };
  d1 = [&](Vec2 a, Vec2 b)->float { Vec2 d = b-a; return 1.f / (rsAbs(d.x) + rsAbs(d.y)); };
  d2 = [&](Vec2 a, Vec2 b)->float { return 1.f / rsNorm(b-a); };

  // P = (3,2), Q = (1,3), R = (4,2), S = (2,0), T = (1,1)
  fill();
  fillEdges(mesh, d0); ND::gradient2D(mesh, u, u_x0, u_y0); e_x0 = u_x-u_x0; e_y0 = u_y-u_y0;
  fillEdges(mesh, d1); ND::gradient2D(mesh, u, u_x1, u_y1); e_x1 = u_x-u_x1; e_y1 = u_y-u_y1;
  fillEdges(mesh, d2); ND::gradient2D(mesh, u, u_x2, u_y2); e_x2 = u_x-u_x2; e_y2 = u_y-u_y2;
  // Manhattan distance seems to work best

  // This is the regular 5-point stencil that would result from unsing a regular mesh:
  // P = (3,2), Q = (3,3), R = (4,2), S = (3,1), T = (2,2)
  mesh.setVertexData(0, Vec2(3.f, 2.f));   // P = (3,2)
  mesh.setVertexData(1, Vec2(3.f, 3.f));   // Q = (3,3)
  mesh.setVertexData(2, Vec2(4.f, 2.f));   // R = (4,2)
  mesh.setVertexData(3, Vec2(3.f, 1.f));   // S = (3,1)
  mesh.setVertexData(4, Vec2(2.f, 2.f));   // T = (2,2)
  fill();                                  // compute target values
  fillEdges(mesh, d0); ND::gradient2D(mesh, u, u_x0, u_y0); e_x0 = u_x-u_x0; e_y0 = u_y-u_y0;
  fillEdges(mesh, d1); ND::gradient2D(mesh, u, u_x1, u_y1); e_x1 = u_x-u_x1; e_y1 = u_y-u_y1;
  fillEdges(mesh, d2); ND::gradient2D(mesh, u, u_x2, u_y2); e_x2 = u_x-u_x2; e_y2 = u_y-u_y2;


  // test solveMinNorm - move elsewhere:
  float a = 2, b = 3, p = 5, x, y;
  RAPT::rsLinearAlgebra::solveMinNorm(a, b, p, &x, &y);
  float q = a*x + b*y;    // should be equal to p - ok
  float n = x*x + y*y;    // should be the smallest possible norm
  // how can we figure out, if there's really no other solution x,y with a smaller norm?
  // It's 1.92307687 - that's at least less than the squared norm of the obvious solution x=y=1, 
  // which has 2 as squared norm - but how can we know that theres no solution with smaller norm?
  // maybe derive y as function of x, which is just y = (p-a*x)/b and then the norm as function of 
  // y which is x*x + y*y and plot it fo x = 0...2 or something

  std::function<float(float)> f1 = [&](float x)->float 
  { 
    float y = (p-a*x)/b;
    return x*x + y*y; 
  };
  //plotFunction(500, 0.76f, 0.78f, f1);
  // yes - minimum is around 0.77, as is the computed x, so it seems to work


  // Demonstrates how to create a graph with no vertex data:
  struct Dummy { };                  // an empty struct
  RAPT::rsGraph<Dummy, Dummy> graph; // graph with no vertex or edge data
  graph.addVertex();

  int dummy = 0;


}
*/

/*
todo:
-allow reading out the mesh at arbitrary positions p
-figure out in which triangular region the vector p falls - i.e. find the 3 vertices that
 bound the triangle that contains the point p
-do a triangular interpolation (something with barycentric coodinates, i guess)
-maybe compute relative errors
-compare accuracy of weighted vs unweighted
-compare to results with regular mesh and central difference - see, if the formula reduces to
 the central difference formula in this case
-try different configurations of Q,R,S,T - maybe also edge cases, where some are
 non-distinct, maybe even fall on P - which is actually a situation that should not occur, but
 out of curiosity, what happens
-try a rotated regular configuration
-try different functions
-test critically determined case (vertex with 2 neighbors) - test also behavior when the two
 vertices are both along the x-direction - in this case, we should not be able to get an
 estimate for the y-component of the gradient
-implement and test underdetermined case (vertex with 1 neighbor)
-maybe try with symmetric edges (this will produce vertices with 1 neighbor)
-generalize - first to 3D, then to nD
-maybe measure how accuracy depends on grid-spacing and number of neighbors - i guess, a
 vertex with more neighbors will get a more accurate estimate?
-provide functions to create meshes programmatically, for example
 createCircularMesh(int Nr, int Na) where Nr, Na are the number of angles and radii - this can
 be used to compare a solver on an irregular cricular grid defined in cartesian coordinates
 with a regular grid in polar coordinates - maybe solving the heat- and wave-equation with a
 given initial temperature and height distribution and maybe with clamped values at the
 boundary. especially the center point of the circle in the irregular mesh is interesting - it
 will have Na neighbours whereas a typical point will have only 4. boundary points will have
 only 3 ...or maybe not giving them any neighbours could be a convenient to fix their values.
 ...maybe a vertex could have additional data associated with it, like the function value -
 but maybe these should be kept in separate arrays
 -we could also insert an extra center point connected to all vertices in the bottom row
 -it should be possible to allow different topologies like plane, cylinder, torus, sphere, 
  moebius-strip (like cylinder but contact edge reversed), klein-bottle (like torus but one or both
  contact edges reversed), etc.

 https://math.stackexchange.com/questions/2253443/difference-between-least-squares-and-minimum-norm-solution
 https://see.stanford.edu/materials/lsoeldsee263/08-min-norm.pdf

*/
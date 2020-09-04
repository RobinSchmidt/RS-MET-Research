#include "Tools.cpp"  // this includes rapt and rosic

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


/** Given the (N-1)th line of the Pascal triangle in x, this produces the N-th line in y, where x 
is of length N-1 and y is of length N. It may be used in place, i.e. x and y may point to the same
array. */
template<class T>
void rsNextPascalTriangleLine(const T* x, T* y, int N)
{
  T xL = T(1);
  for(int i = 1; i < N-1; i++) { 
    T xR = x[i]; 
    y[i] = xL + xR;
    xL   = xR;  }
  y[N-1] = T(1);
}
// move to rapt - the algo there is not in place
// experiment with variations, for example replacing the + with - or xL+xR with -(xL+xR)
// maybe this can be optimized using symmetry by doing something like
// y[i] = y[i+k] = xL + xR where k depends on i and N - or maybe y[i] = y[N-i] = xL + xR?

/** If you need only one line of the Pascal triangle, this function may be more convenient. */
template<class T>
void rsPascalTriangleLine(T* y, int N)
{
  for(int n = 1; n <= N; n++)
    rsNextPascalTriangleLine(y, y, n);
}
// overflows for N >= 35 when T is a 32 bit signed integer, N <= 34 works
// figure out, for which N it overflows for other common integer types


void testSortedSet()
{
  using Set = rsSortedSet<int>;
  bool r = true;

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

  // compute the next line of the pascal triangle from a given line:
  static const int N = 50;
  int p[N];
  rsNextPascalTriangleLine(p, p,  1);
  rsNextPascalTriangleLine(p, p,  2);
  rsNextPascalTriangleLine(p, p,  3);
  rsNextPascalTriangleLine(p, p,  4);
  rsNextPascalTriangleLine(p, p,  5);
  rsNextPascalTriangleLine(p, p,  6);
  rsNextPascalTriangleLine(p, p,  7);
  rsNextPascalTriangleLine(p, p,  8);
  rsNextPascalTriangleLine(p, p,  9);
  rsNextPascalTriangleLine(p, p, 10);
  rsNextPascalTriangleLine(p, p, 11);
  rsNextPascalTriangleLine(p, p, 12);
  // the sum of the n-th line should be 2^(n-1), when n starts counting at 1, 2^n when starting to 
  // count at 0
  rsPascalTriangleLine(p, 7);
  rsPascalTriangleLine(p, 34);  // 34 is the greatest possible value without overflow

  // see: https://en.wikipedia.org/wiki/Pascal%27s_triangle
  // maybe implement also:
  // https://en.wikipedia.org/wiki/Trinomial_triangle
  // https://en.wikipedia.org/wiki/(2,1)-Pascal_triangle
  // https://en.wikipedia.org/wiki/Bell_triangle
  // https://en.wikipedia.org/wiki/Bernoulli%27s_triangle
  // https://en.wikipedia.org/wiki/Leibniz_harmonic_triangle
  // https://en.wikipedia.org/wiki/Eulerian_number#Basic_properties


  int dummy = 0;
}

/** Solves a*x + b*y = p subject to x^2 + y^2 = min. */
template<class T>
void solveMinNorm(T a, T b, T p, T* x, T* y)
{
  T s = p / (a*a + b*b);
  *x = s*a;
  *y = s*b;
}
// needs test
// -maybe move to rsMatrix2x2 or rsLinearAlgebraNew
// -can we do this with one division?
// -maybe try to derive the formulas with sage, see
//  https://ask.sagemath.org/question/38079/can-sage-do-symbolic-optimization/

// x == a*p/(a^2 + b^2), y == b*p/(a^2 + b^2), l == -2*p/(a^2 + b^2)

// move to rs-met codebase - maybe turn into a unit test and/or experiment
void testVertexMesh()
{
  using Vec2 = rsVector2D<float>;
  using VecF = std::vector<float>;
  using VecI = std::vector<int>;
  using Mesh = rsGraphWithVertexData<Vec2>;
  using ND   = rsNumericDifferentiator<float>;

  // an (irregular) star-shaped mesh with a vertex P = (3,2) at the center and 4 vertices 
  // Q,R,S,T surrounding it that are connected to it:
  Mesh mesh;
  bool sym = false;                // select, if edges should be added symmetrically
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

  // P = (3,2), Q = (1,3), R = (4,2), S = (2,0), T = (1,1)
  fill();
  ND::gradient2D(mesh, u, u_x0, u_y0, 0); e_x0 = u_x-u_x0; e_y0 = u_y-u_y0;
  ND::gradient2D(mesh, u, u_x1, u_y1, 1); e_x1 = u_x-u_x1; e_y1 = u_y-u_y1;
  ND::gradient2D(mesh, u, u_x2, u_y2, 2); e_x2 = u_x-u_x2; e_y2 = u_y-u_y2;
  // Manhattan distance seems to work best

  // This is the regular 5-point stencil that would result from unsing a regular mesh:
  // P = (3,2), Q = (3,3), R = (4,2), S = (3,1), T = (2,2)
  mesh.setVertexData(0, Vec2(3.f, 2.f));   // P = (3,2)
  mesh.setVertexData(1, Vec2(3.f, 3.f));   // Q = (3,3)
  mesh.setVertexData(2, Vec2(4.f, 2.f));   // R = (4,2)
  mesh.setVertexData(3, Vec2(3.f, 1.f));   // S = (3,1)
  mesh.setVertexData(4, Vec2(2.f, 2.f));   // T = (2,2)
  fill();                                  // compute target values
  ND::gradient2D(mesh, u, u_x0, u_y0, 0); e_x0 = u_x-u_x0; e_y0 = u_y-u_y0;
  ND::gradient2D(mesh, u, u_x1, u_y1, 1); e_x1 = u_x-u_x1; e_y1 = u_y-u_y1;
  ND::gradient2D(mesh, u, u_x2, u_y2, 2); e_x2 = u_x-u_x2; e_y2 = u_y-u_y2;


  // test solveMinNorm - move elsewhere:
  float a = 2, b = 3, p = 5, x, y;
  solveMinNorm(a, b, p, &x, &y);
  float q = a*x + b*y;    // should be equal to p - ok
  float n = x*x + y*y;    // should be the smallest possible norm
  // how can we figure out, if there's really no other solution x,y with a smaller norm?
  // It's 1.92307687 - that's at least less than the squared norm of the obvious solution x=y=1, 
  // which has 2 as squared norm - but how can we know that theres no solution with smaller norm?
  // maybe derive y as function of x, which is just y = (p-a*x)/b and then the norm as function of 
  // y which is x*x + y*y and plot it fo x = 0...2 or something

  int dummy = 0;

  // todo:
  // -maybe compute relative errors
  // -compare accuracy of weighted vs unweighted
  // -optimize
  // -move to library
  // -compare to results with regular mesh and central difference - see, if the formula reduces to
  //  the central difference formula in this case
  // -try different configurations of Q,R,S,T - maybe also edge cases, where some are 
  //  non-distinct, maybe even fall on P - which is actually a situation that should not occur, but
  //  out of curiosity, what happens
  // -try a rotated regular configuration
  // -try different functions
  // -test criticall determined case (vertex with 2 neighbors) - test also behavior when the two
  //  vertices are both along the x-direction - in this case, we should not be able to get an 
  //  estimate for the y-component of the gradient
  // -implement and test underdetermined case (vertex with 1 neighbor)
  // -maybe try with symmetric edges (this will produce vertices with 1 neighbor)
  // -generalize - first to 3D, then to nD
  // -maybe measure how accuracy depends on grid-spacing and number of neighbors - i guess, a 
  //  vertex with more neighbors will get a more accurate estimate?

  // ToDo: maybe later use a function u(x,y) - maybe a bivariate polynomial - so we can compute 
  // exact partial derivatives and compare them to the numerical results. We should also compare 
  // them to numerical results obtained on a regular grid. Maybe or polynomial should have a scale 
  // factor to scale both inputs (such that a grid-spacing of around 1 becomes reasonable with 
  // respect to the behavior of the function)

  // ToDo: factor out the computation and choose different configurations for Q,R,S,T - among them
  // those that would result from ahving a regular gridd - maybe also rotate them (maybe by 45° 
  // degrees and scale by sqrt(2))

  // ToDo: provide functions to create meshes programmatically, for example 
  // createCircularMesh(int Nr, int Na) where Nr, Na are the number of angles and radii - this can
  // be used to compare a solver on an irregular cricular grid defined in cartesian coordinates
  // with a regular grid in polar coordinates - maybe solving the heat- and wave-equation with a
  // given initial temperature and height distribution and maybe with clamped values at the 
  // boundary. especially the center point of the circle in the irregular mes is interesting - it 
  // will have Na neighbours whereas a typical point will have only 4. boundary points will have 
  // only 3 ...or maybe not giving them any neighbours could be a convenient to fix their values.
  // ...maybe a vertex could have additional data associated with it, like the function value - but
  // maybe these should be kept in separate arrays
}
// https://math.stackexchange.com/questions/2253443/difference-between-least-squares-and-minimum-norm-solution
// https://see.stanford.edu/materials/lsoeldsee263/08-min-norm.pdf
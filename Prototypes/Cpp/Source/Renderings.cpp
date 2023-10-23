// This file contains functions that render images, animations or audio samples. The rendered 
// results may serve multiple purposes. They can be either just be pretty artworks or they may 
// visualize some math or physics concepts and potentially go as figures into a math paper or the 
// audio signals may eventually end up in some sample library or whatever.

//=================================================================================================
// Image Rendering Tools

template<class T>
class rsArtsyBivariateFunctions
{

public:

  /** f(x,y) = tanh(tan(x^2 + y^2)) * cos(x + y) - cos(x^2 + y^2)

  This function based on the implicit curve defition:
   
    tan(x^2 + y^2) * cos(x + y) = cos(x^2 + y^2)
   
  taken from this video (at 1:29): https://www.youtube.com/watch?v=Ey-W3xwNJU8. I turned it into
  into a function f(x,y) by bringing everything to one side and replace the zero that remains on 
  the other side by the output. A contour line at height zero should reproduce the original image
  from the video. We want to create multiple contour levels instead. But there is a problem: The 
  original function has has poles due to the tan and that doesn't play well with evaluating it in 
  the whole plane so we tame it with a tanh saturator placed into various places. We have some 
  other little modifications and implement various variants. One version has also the sign of one 
  of the cosines flipped, and/or replaced cos with sin. When just rendering the function itself 
  without doing the contour filling business, it looks a bit like 3D rendered Tori, hence the 
  name. */
  static T weirdTori(T x, T y, int variant);

};

template<class T>
T rsArtsyBivariateFunctions<T>::weirdTori(T x, T y, int variant)
{
  T x2 = x*x;
  T y2 = y*y;
  T d2 = x2 + y2;
  switch(variant)
  {
  case 1: return tanh(tan(d2)) * cos(x + y) - cos(d2);  // tames only tan part by tanh
  case 2: return tanh(tan(d2)) * cos(x + y) + cos(d2);  // changed sign of last cosine
  case 3: return tanh(tan(d2)) * cos(x + y) + sin(d2);  // replaced cos with sin
  case 4: return tanh(tan(d2)) * cos(x + y) - sin(d2);  // and now negated the sin
  case 5: return tanh(tan(d2)  * cos(x + y) - cos(d2)); // this tames the end result
  }
  // ToDo:
  // -Try also to apply the tanh after tan(d2)) * cos(x + y)
}

//-------------------------------------------------------------------------------------------------

/** A class for creating contour plots from mathematical functions. The difference to
rsImageContourPlotter is that that class is a pure image-processing facility where
the input is also an image. Here, the input is a std::function. ...TBC... */

template<class TPix, class TVal>
class rsMathContourPlotter : public rsImagePlotter<TPix, TVal>
{

public:

  //-----------------------------------------------------------------------------------------------
  // \name Setup

  void setPixelSize(int width, int height) { this->width = width; this->height = height; }

  /** Decides whether we normalize the image before finding the contours. In case of normalization 
  the contour levels should be in 0...1. Otherwise they should be in the range that the function 
  produces naturally. */
  void useNormalizedLevels(bool shouldNormalize) { normalize = shouldNormalize; }


  //-----------------------------------------------------------------------------------------------
  // \name Contour Drawing


  rsImage<TPix> contourLines(const std::function<TVal(TVal, TVal)>& func, 
    const std::vector<TVal>& levels);

  rsImage<TPix> contourFills(const std::function<TVal(TVal, TVal)>& func, 
    const std::vector<TVal>& levels);


protected:

  int  width  = 480;
  int  height = 270;
  bool normalize = false;

};

template<class TPix, class TVal>
rsImage<TPix> rsMathContourPlotter<TPix, TVal>::contourLines(
  const std::function<TVal(TVal, TVal)>& func, const std::vector<TVal>& levels)
{
  // Create image with function values:
  rsImageF imgFunc(width, height);
  rsImagePlotter<TPix, TVal>::generateFunctionImage(func, imgFunc);
  if(normalize)
    rsImageProcessor<TVal>::normalize(imgFunc);

  // Create images with contours:
  rsImageContourPlotter<TPix, TVal> cp;
  rsImageF imgCont = cp.getContourLines(imgFunc, levels, { 1.0f }, true);
  return imgCont;

  // ToDo:
  // -Instead of hardcoding the colors via the { 1.0f } in the call to getContourLines, let the 
  //  user (optionally) pass in the array (of type TPix)
}

template<class TPix, class TVal>
rsImage<TPix> rsMathContourPlotter<TPix, TVal>::contourFills(
  const std::function<TVal(TVal, TVal)>& func, const std::vector<TVal>& levels)
{
  // Create image with function values:
  rsImageF imgFunc(width, height);
  rsImagePlotter<TPix, TVal>::generateFunctionImage(func, imgFunc);
  if(normalize)
    rsImageProcessor<TVal>::normalize(imgFunc);

  // Create images with bin-fills:
  rsImageContourPlotter<TPix, TVal> cp;
  int numLevels = levels.size();
  int numColors = numLevels + 1;
  std::vector<TVal> colors = rsRangeLinear(0.f, 1.f, numColors);
  rsImageF imgFills = cp.getContourFills(imgFunc, levels, colors, true);
  return imgFills;

  // ToDo:
  // -Instead of hardcoding the colors via the colors = rsRangeLinear(...) in the call to 
  //  getContourFills, let the user (optionally) pass in the array (of type TPix)
}


//=================================================================================================
// Image Rendering Scripts

void rainbowRadiation()
{
  // Renders a picture that looks like some sort of circularly outward radiating wave. Due to the
  // way we use different variations of the function and distributing these variations to the RGB
  // color channels, we get an image that features pretty rainbow-ish color gradients.

  using Real = float;
  using Func = std::function<Real(Real, Real)> ;
  using IP   = rsImageProcessor<Real>;
  using Vec  = std::vector<Real>;
  using ABF  = rsArtsyBivariateFunctions<Real>;

  // Image parameters:
  int scale  = 1;                // scaling: 1: = 480 X 270 (preview), 4: 1920 x 1080 (full)
  int width  = scale * 480;      // width in pixels
  int height = scale * 270;      // height in pixels

  // Plotting range parameters:
  Real ratio = Real(width) / Real(height);  // aspect ratio
  Real xMin  = -4 * ratio;
  Real xMax  = +4 * ratio;
  Real yMin  = -4;
  Real yMax  = +4;

 // Normalized contour levels to be used:
 Vec levels({ 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 });  // 1
 //Vec levels({ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 });  // 2
 //Vec levels({ 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95 });  // 3
 //Vec levels({ 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 1.0 }); // 4
 //Vec levels({ 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.1, 1.2 }); // 5


  // Each color channel uses a different variant of the function:
  Func fRed   = [&](Real x, Real y) { return ABF::weirdTori(x, y, 1); };
  Func fGreen = [&](Real x, Real y) { return ABF::weirdTori(x, y, 4); };
  Func fBlue  = [&](Real x, Real y) { return ABF::weirdTori(x, y, 2); };
  // It looks good 142 for the variants of the functiosn for the RGB channels


  rsMathContourPlotter<Real, Real> cim;
  cim.setRange(xMin, xMax, yMin, yMax);
  cim.setPixelSize(width, height);
  cim.useNormalizedLevels(true);
  rsImageF red   = cim.contourFills(fRed,   levels);
  rsImageF green = cim.contourFills(fGreen, levels);
  rsImageF blue  = cim.contourFills(fBlue,  levels);
  writeImageToFilePPM(red, green, blue, "RainbowRadiation.ppm");

  // Write the color channels seperately to images:
  writeImageToFilePPM(red,   "RainbowRadiationR.ppm");
  writeImageToFilePPM(green, "RainbowRadiationG.ppm");
  writeImageToFilePPM(blue,  "RainbowRadiationB.ppm");



  // ToDo:
  // -Rotate the whole picture by 45° cunterclockwise. That gives symmetry over the x and y axis 
  //  instead of the diagonals. Use rsRotationXY for this.
  // -The function is a bit to busy in the outward range and a bit too slow in the middle. Try to 
  //  modify it such that the ripples are less dense far away from the origin and the inner circle
  //  might be smaller. Maybe some nonlinear mapping like.
  //  r = (x^2 + y^2), s = pow(r, p), x *= r, y *= r  could achieve this where p is some number to 
  //  be found by trial and error (p=1 would change nothing). I think, to achieve the desired 
  //  effect, we need p < 1
  // -There are also some white-ish lines that look a bit artifacty. Try to figure out how they 
  //  arise and get rid of them. Maybe by some post-processing. Maybe soft-clipping the intensity
  //  values could help. Maybe green should be soft-clipped more (i.e. with lower threshold) than 
  //  red and blue. Or maybe green should use other levels..hmm...doesn't seem to help much
  //  Could it be that these occur where the tan changes sign abruptly? If so, maybe it could help 
  //  to use (conditional) smoothing? Using IrfanView's Blur 3 times seems to get rid of it
  //  ...but no! It's too blurry overall. We really need to restrict the blur to the artifacts.
  //  Maybe check if the pixel is white and if so, use the average of the 3x3 cell. The 
  //  artifact is always 1 pixel wide, independently from the resolution. 
  // -Maybe try to use unnormalized contour levels
}

// ToDo:
// -Maybe prefix the function names with img for images, ani for animations and snd for sounds.

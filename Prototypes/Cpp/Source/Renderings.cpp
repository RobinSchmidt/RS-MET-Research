// This file contains functions that render images, animations or audio samples. The rendered 
// results may serve multiple purposes. They can be either just be pretty artworks or they may 
// visualize some math or physics concepts and potentially go as figures into a math paper or the 
// audio signals may eventually end up in some sample library or whatever.

void rainbowRadiation()
{
  // Renders a picture that looks like some sort of circularly outward radiating wave. Due to the
  // way we use different variations of the function and distributing these variations to the RGB
  // color channels, we get an image that features pretty rainbow-ish color gradients.

  using Real = float;
  using Func = std::function<Real(Real, Real)> ;
  using IP   = rsImageProcessor<Real>;
  using Vec  = std::vector<Real>;

  // Image parameters:
  int scale  = 4;                // scaling: 1: = 480 X 270, 4: 1920 x 1080
  int width  = scale * 480;      // width in pixels
  int height = scale * 270;      // height in pixels

  // Plotting range parameters:
  Real ratio = Real(width) / Real(height);  // aspect ratio
  Real xMin  = -4 * ratio;
  Real xMax  = +4 * ratio;
  Real yMin  = -4;
  Real yMax  = +4;
  Vec levels({ 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 });  // Normalized contour levels to be used


  // https://www.youtube.com/watch?v=Ey-W3xwNJU8  at 1:29 has the implicit curve:
  //
  //   tan(x^2 + y^2) * cos(x + y) = cos(x^2 + y^2)
  //
  // We turn it into a function f(x,y) as 
  //
  //   f(x,y) = tan(x^2 + y^2) * cos(x + y) - cos(x^2 + y^2)
  //
  // A contour line at height zero should reproduce the original image from the video. We want to
  // create multiple contour levels instead. But there is a problem: The original function has has
  // poles due to the tan and that doesn't play well with evaluating it in the whole plane so we 
  // tame it with a tanh saturator placed into various places. We have some other little 
  // modifications and implement various variants. One version has also the sign of one of the 
  // cosines flipped, and/or replaced cos with sin.
  auto weirdTori = [&] (Real x, Real y, int variant) 
  { 
    Real x2 = x*x;
    Real y2 = y*y;
    Real d2 = x2 + y2;
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
    // -Move that function into a class where we collect such functions. Maybe it should go into
    //  a file RenderTools.cpp
  };


  // Helper functions to produce an rsImage with contour lines or filled contours from a 
  // std::function. ToDo: Factor these two functions out into a class - it should have the 
  // range settings as members:
  auto getContourLineImage = [&](const Func& func, const Vec& levels)
  {
    // Create image with function values:
    rsImageF imgFunc(width, height);
    rsImagePlotter<Real, Real> plt;
    plt.setRange(xMin, xMax, yMin, yMax);
    plt.generateFunctionImage(func, imgFunc);
    IP::normalize(imgFunc);

    // Create images with contours:
    rsImageContourPlotter<Real, Real> cp;
    rsImageF imgCont = cp.getContourLines(imgFunc, levels, { 1.0f }, true);
    return imgCont;
  };
  auto getContourFillImage = [&](const Func& func, const Vec& levels)
  {
    // Create image with function values:
    rsImageF imgFunc(width, height);
    rsImagePlotter<Real, Real> plt;
    plt.setRange(xMin, xMax, yMin, yMax);
    plt.generateFunctionImage(func, imgFunc);
    IP::normalize(imgFunc);

    // Create images with bin-fills:
    rsImageContourPlotter<Real, Real> cp;
    int  numLevels = levels.size();
    int  numColors = numLevels + 1;
    std::vector<Real> colors = rsRangeLinear(0.f, 1.f, numColors);
    rsImageF imgFills = cp.getContourFills(imgFunc, levels, colors, true);
    return imgFills;
  };


  // Each color channel uses a different variant of the function:
  Func fRed   = [&](Real x, Real y) { return weirdTori(x, y, 1); };
  Func fGreen = [&](Real x, Real y) { return weirdTori(x, y, 4); };
  Func fBlue  = [&](Real x, Real y) { return weirdTori(x, y, 2); };
  // It looks good 142 for the variants of the functiosn for the RGB channels

  rsImageF red   = getContourFillImage(fRed,   levels);
  rsImageF green = getContourFillImage(fGreen, levels);
  rsImageF blue  = getContourFillImage(fBlue,  levels);

  writeImageToFilePPM(red, green, blue, "RainbowRadiation.ppm");

  // ToDo:
  // -Rotate the whole picture by 45° cunterclockwise. That gives symmetry over the x and y axis 
  //  instead of the diagonals. Use rsRotationXY for this.
  // -The function is a bit to busy in the outward range and a bit too slow in the middle. Try to 
  //  modify it such that the ripples are less dense far away from the origin and the inner circle
  //  might be smaller. Maybe some nonlinear mapping like.
  //  r = (x^2 + y^2), s = pow(r, p), x *= r, y *= r  could achieve this where p is some number to 
  //  be found by trial and error (p=1 would change nothing). i think, to achiev the desired 
  //  effect, we need p < 1
  // -There are also some white-ish lines that look a bit artifacty. Try to figure out how they 
  //  arise and get rid of them. Maybe by some post-processing. Maybe soft-clipping the intensity
  //  values could help. Maybe green should be soft-clipped more (i.e. with lower threshold) than 
  //  red and blue. 
}

// ToDo:
// -getContourLineImage/getContourFillImage should not be local functions. They may be used for
//  multiple images and could be library functions. Maybe move them into some library.
// -Maybe prefix the function names with img for images, ani for animations and snd for sounds.
//
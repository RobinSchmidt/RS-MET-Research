

void artsyContours()
{
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
  Vec levels({ 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 });


  // https://www.youtube.com/watch?v=Ey-W3xwNJU8  at 1:29 has the implicit curve:
  //
  //   tan(x^2 + y^2) * cos(x + y) = cos(x^2 + y^2)
  //
  // We turn it into a function f(x,y) as 
  //
  //   f(x,y) = tan(x^2 + y^2) * cos(x + y) - cos(x^2 + y^2)
  //
  // The contour at height zero should reproduce the original image from the video. But:
  // The original function has has poles which doesn't play well with evaluating it in the 
  // whole plane so we tame it with a tanh saturator placed into various places. One version 
  // has also the sign of one of the cosines flipped. ...TBC...
  auto weirdTori = [&] (Real x, Real y, int variant) 
  { 
    Real x2 = x*x;
    Real y2 = y*y;
    Real d2 = x2 + y2;
    switch(variant)
    {
    case 1: return tanh(tan(d2)) * cos(x + y) - cos(d2);  // tames end result
    case 2: return tanh(tan(d2)) * cos(x + y) + cos(d2);  // changed sign of last cosine
    case 3: return tanh(tan(d2)) * cos(x + y) + sin(d2);  // replaced cos with sin
    case 4: return tanh(tan(d2)) * cos(x + y) - sin(d2); 
    case 5: return tanh(tan(d2)  * cos(x + y) - cos(d2)); // tames only tan part
    }
  };


  // Factor these two functions out into a class - it should have the range settings as members:
  auto getContourLineImage = [&](const Func& func, const Vec& levels)
  {
    // Create image with function values:
    rsImageF imgFunc(width, height);
    rsImagePlotter<Real, Real> plt;
    plt.setRange(xMin, xMax, yMin, yMax);
    plt.generateFunctionImage(func, imgFunc);
    IP::normalize(imgFunc);

    // Create images with contours:
    //int  numLevels = levels.size();
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

  writeImageToFilePPM(red, green, blue, "Radiation.ppm");
}

// Maybe wrap this drawing into a class
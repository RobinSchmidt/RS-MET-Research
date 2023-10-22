

void artsyContours()
{
  using Real = float;
  using Func = std::function<Real(Real, Real)> ;
  using IP   = rsImageProcessor<Real>;


  int  w = 800;               // width in pixels
  int  h = 800;               // height in pixels


  Real xMin = -8;
  Real xMax = +8;
  Real yMin = -8;
  Real yMax = +8;

  std::vector<Real> levels({-0.5, 0.0, +0.5});




  Func f;

  // Cassini curves:
  f = [&] (Real x, Real y) { return (x*x+y*y)*(x*x+y*y) - 2*(x*x-y*y) + 1; };
  // Drawing range is not yet optimal
  // would perhaps be better to have an x-range from -2..+2 and a y-range from -1..+1
  // we should make sure that 1 is among the levels in order to see the lemniskate
  // https://de.wikipedia.org/wiki/Cassinische_Kurve
  // https://en.wikipedia.org/wiki/Cassini_oval


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
  // whole plane so we tame it with a tanh saturator. ...TBC...

  f = [&] (Real x, Real y) 
  { 
    Real x2 = x*x;
    Real y2 = y*y;
    Real d2 = x2 + y2;

    //return tanh(tan(d2) * cos(x + y) - cos(d2));  // tames only tan part

    return tanh(tan(d2)) * cos(x + y) - cos(d2);  // tames end result
  };


  // Create image with function values:
  rsImageF imgFunc(w, h);
  rsImagePlotter<Real, Real> plt;
  plt.setRange(xMin, xMax, yMin, yMax);
  plt.generateFunctionImage(f, imgFunc);
  IP::normalize(imgFunc);

  // Create images with contours:
  int  numLevels = levels.size();
  rsImageContourPlotter<Real, Real> cp;
  rsImageF imgCont = cp.getContourLines(imgFunc, levels, { 1.0f }, true);

  // Create images with bin-fills:
  int  numColors = numLevels + 1;
  std::vector<Real> colors = rsRangeLinear(0.f, 1.f, numColors);
  rsImageF imgFills = cp.getContourFills(imgFunc, levels, colors, true);


  // Write images to files:
  writeScaledImageToFilePPM(imgFunc,  "ContourInput.ppm",  1);
  writeScaledImageToFilePPM(imgCont,  "ContourLines.ppm",  1);
  writeScaledImageToFilePPM(imgFills, "ContourFills.ppm",  1);




  int dummy = 0;
}

// Maybe wrap this drawing into a class


void artsyContours()
{
  using Real = float;
  using Func = std::function<Real(Real, Real)> ;
  using IP   = rsImageProcessor<Real>;
  using Vec  = std::vector<Real>;


  int  w = 1920;               // width in pixels
  int  h = 1080;               // height in pixels




  Real xMin = -4;
  Real xMax = +4;
  Real yMin = -4;
  Real yMax = +4;

  Real ratio = Real(w) / Real(h);
  xMin *= ratio;
  xMax *= ratio;

  //Vec levels({-1.0, -0.5, 0.0, +0.5, 1.0});

  //Vec levels({ -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3 });
  //Vec levels({ -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4 });

  //Vec levels({ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 });

  //Vec levels({ 0.4, 0.5, 0.6 });
  //Vec levels({ 0.3, 0.4, 0.5, 0.6, 0.7 });
  Vec levels({ 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9 });

  Func f;



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
    // Lots of black and some lightish gray, little transition areas
    // Maybe use for green.

    return tanh(tan(d2)) * cos(x + y) - cos(d2);  // tames end result
    // Looks like hollow tori with holes in the surface
    // Use for blue!

    //return tanh(tan(d2)) * cos(x + y) + cos(d2);  // test - changed sign of last cosine

    //return tanh(tan(d2) * cos(x + y))  - cos(d2);
    // Similar but holes have weirder shape
    // Use for red!


    // Both versions look intersting. 
    // -Maybe use 3 different versions for the 3 color channels
    //  -The variant with most birght parts should go to blue, the middle brightness version 
    //   to red and the low brightness version to green
    // -Try to tuen it such that it doesn't look so diagonal
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
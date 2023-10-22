

void artsyContours()
{
  using Real = float;
  using Func = std::function<Real(Real, Real)> ;


  int  w = 800;               // width in pixels
  int  h = 800;               // height in pixels

  Real xMin = -2;
  Real xMax = +2;
  Real yMin = -1;
  Real yMax = +1;

  int  numLevels = 20;
  int  numColors = numLevels + 1;


  Func f;

  // Cassini curves:
  f = [&] (Real x, Real y) { return (x*x+y*y)*(x*x+y*y) - 2*(x*x-y*y) + 1; };
  // Drawing range is not yet optimal
  // would perhaps be better to have an x-range from -2..+2 and a y-range from -1..+1
  // we should make sure that 1 is among the levels in order to see the lemniskate
  // https://de.wikipedia.org/wiki/Cassinische_Kurve
  // https://en.wikipedia.org/wiki/Cassini_oval


  rsImageContourPlotter<Real, Real> cp;

  using IP = rsImageProcessor<Real>;

  // create image with function values:
  rsImageF imgFunc(w, h);

  //generateFunctionImage(f, xMin, xMax, yMin, yMax, imgFunc);;
  // not available! move into library!

  /*
  IP::normalize(imgFunc);

  // create images with contours:
  std::vector<float> levels = rsRangeLinear(0.f, 1.f, numLevels);
  rsImageF imgCont = cp.getContourLines(imgFunc, levels, { 1.0f }, antiAlias);
  */





  // https://www.youtube.com/watch?v=Ey-W3xwNJU8
  // 1:29

  int dummy = 0;
}

// Maybe wrap this drawing into a class
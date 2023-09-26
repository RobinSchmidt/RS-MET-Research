#include "../JuceLibraryCode/JuceHeader.h"
using namespace RAPT;   // maybe get rid
using namespace rosic;  // dito

//=================================================================================================
// Convenience functions for certain types of plots. Maybe move to library, maybe into rs_testing 
// module into TestTools/Plotting.h. Maybe at some point even into GNUPlotCPP itself.

template<class T>
void generateMatrixData(std::function<T(T x, T y)> f, 
  T xMin, T xMax, T yMin, T yMax, int Nx, int Ny,
  std::vector<T>& x, std::vector<T>& y, RAPT::rsMatrix<T>& z)
{
  x = RAPT::rsRangeLinear(xMin, xMax, Nx);
  y = RAPT::rsRangeLinear(yMin, yMax, Ny);
  z.setShape(Nx, Ny);
  for(int i = 0; i < Nx; i++)
    for(int j = 0; j < Ny; j++)
      z(i, j) = f(x[i], y[j]);
  // ToDo:
  // -Maybe use rsArrayTools::rangeLinear to make sure that no re-allocations take place in the
  //  assignemnt of std::vector
}
// Maybe pass output parameter by pointer


/** Given a bivariate function f = f(x,y), ranges for x and y and numbers of samples along x and y,
this function generates the data matrix of the heights produced by f and adds the data as matrix 
data to the plotter object. */
/*
template<class T>
void addHeightData(GNUPlotter& plt, std::function<T(T x, T y)> f,
  T xMin, T xMax, T yMin, T yMax, int Nx, int Ny)
{
  plt.addDataBivariateFunction(Nx, xMin, xMax, Ny, yMin, yMax, f);
}
*/
// Maybe factor out the function that generates the P-matrix into a function getDataMatrix(...).
// Instead of just adding the data to plt, we may want tot analyze the produced data to set up plt.
// For example, we may want to to figure out the min and max values to set up the z-range and/or 
// levels for contour lines in a contour plot

/** Produces a surface plot in dark mode. */
void plotSurfaceDark(GNUPlotter& plt)
{
  //plt.setToDarkMode();

  using CP = GNUPlotter::ColorPalette;
  //plt.addCommand("set palette rgbformulae 8, 9, 7");   // burgund...white
  //plt.setColorPalette(CP::KM_moreland);
  //plt.setColorPalette(CP::F_printable);
  //plt.setColorPalette(CP::SW_magma);


  plt.addCommand("set style fill solid 1.0 noborder");
  plt.addCommand("set pm3d depthorder noborder");


  //plt.addCommand("set pm3d lighting specular 0.25");
  plt.addCommand("set pm3d lighting primary 0.6 specular 0.0");
  //plt.addCommand("set pm3d lighting primary 0.6 specular 0.25");
  https://stackoverflow.com/questions/71490416/how-to-make-the-choice-in-3d-color-palette-in-gnuplot-with-light-effect
  // "primary 0.0" is brighter than "primary 0.5". "primary 1.0" makes the underside completely 
  // black. I think this number adjusts between ambient light and sourced light where 0.0 uses 
  // ambient light only (underside is just as bright as upside) whereas 1.0 makes the underside
  // completely black. I think, "specular 0.0" is good for coarser meshes wher the quadrilaterals
  // give cues. For fine meshes, a bit more specular light may be beneficial (or it may not).
  // "primary 0.6 specular 0.0" looks good with a coarse mesh. Some specular light (like 0.25) 
  // seems to work well with unipolar darkish color-maps like CB_YlGnBu9m but not so good with 
  // bipolar maps with a whiteish color in the middle like CJ_BuYlRd11. A little bit like 0.1
  // ight be OK - but it really depends on the angle.


  plt.addCommand("splot 'C:/Temp/gnuplotData.dat' i 0 nonuniform matrix with pm3d notitle");
  plt.invokeGNUPlot();
}
// ToDo:
// -Remove the call to setDarkMode. Maybe also the set palette ... command. This should be done by the 
//  caller
// -Maybe move into class GNUPlotter. If so, adapt the splot command to use the dataPath instead of the 
//  hardcoded path.

// Under construction. Does not yet seem to work. Intentional usage:
//
//   int N = 20;
//   setLineStyles(plt, "lc rgb \"red\"",   1, N, 3);
//   setLineStyles(plt, "lc rgb \"green\"", 2, N, 3);
//   setLineStyles(plt, "lc rgb \"blue\"",  3, N, 3);
//
// All lines up to N should be alternatingly red, green and blue. It doesn't seem to work yet, 
// though. At least not in the context of contour plots. The contour lines are always black even if
// we call the function like that. Calling:
//
//   plt.addCommand("set style increment user");
//
// before also doesn't help. The contours are still black. -> Figure out!
void setLineStyles(GNUPlotter& plt, const std::string& style, int iStart, int iEnd, int iInc)
{
  std::string range = 
    std::to_string(iStart) + ":" + std::to_string(iEnd) + ":" + std::to_string(iInc);
  std::string cmd = "do for [i=" + range  + "] { set style line i " + style + " }";
  plt.addCommand(cmd);

  // See:
  // https://stackoverflow.com/questions/35818875/gnuplot-pm3d-with-contour-lines
  // http://www.gnuplotting.org/tag/for/
}
// Move into class GNUPlotter as member function. It's generally useful.


// ToDo:
// -Move that functionality into GNUPlotter
void plotContours(GNUPlotter& plt, const std::vector<float> levels, bool useConstColors = true)
{
  // Add the contour lines:
  std::string cmd;
  cmd = "set cntrparam levels discrete " + std::to_string(levels[0]);
  for(int i = 1; i < levels.size(); i++)
    cmd += "," + std::to_string(levels[i]);
  plt.addCommand(cmd);

  // Use constant color fills between the contour lines if desired:
  if(useConstColors)
  {
    cmd = "set palette maxcolors " + std::to_string(levels.size() - 1);
    plt.addCommand(cmd);
    std::string range = "[" + std::to_string(levels[0]) + ":" + std::to_string(rsLast(levels)) + "]";
    plt.addCommand("set zrange " + range);   // range for z values
    plt.addCommand("set cbrange " + range);  // color bar range
  }

  // Plot:
  plt.addCommand("set pm3d map impl");
  plt.addCommand("set contour");
  plt.addCommand("splot 'C:/Temp/gnuplotData.dat' i 0 nonuniform matrix w pm3d notitle");
  plt.invokeGNUPlot();

  // Questions:
  // -What happens, if the levels are non-equidistant? I guess, in this case, the alignment between
  //  constant color region boundaries and contour lines gets messed up.
}

// See:
// https://stackoverflow.com/questions/35818875/gnuplot-pm3d-with-contour-lines
// https://stackoverflow.com/questions/20977368/filled-contour-plot-with-constant-color-between-contour-lines
// https://subscription.packtpub.com/book/data/9781849517249/10/ch10lvl1sec101/making-a-labeled-contour-plot
// http://lowrank.net/gnuplot/plotpm3d-e.html
// -has interesting function: x^2 * y^2 * exp(-(x^2 + y^2))
//
// ToDo:
// Maybe plot also the Polya vector field. Use strength of color (opacity and/or brightness) so 
// indicate field strength. Let the arrows all have the same length.




// Maybe move them into rs_testing/TestTools/Plotting.h:

/** Baseclass for 2 field plotters */

template<class T>
class rsFieldPlotter2D
{

public:

  void setInputRange(T minX, T maxX, T minY, T maxY) 
  { xMin = minX; xMax = maxX; yMin = minY; yMax = maxY; }

  void setPixelSize(int width, int height) { pixelWidth  = width; pixelHeight = height; }

  void setDrawRectangle(double newLeft, double newRight, double newBottom, double newTop)
  {
    left   = newLeft;
    right  = newRight;
    bottom = newBottom;
    top    = newTop;
  }

  void setTitle(const std::string& newTitle) { title = newTitle; }


  void setColorPalette(GNUPlotter::ColorPalette newMap, bool reverse)
  { colorMap = newMap; reverseColors = reverse; }

  // Adds a custom command that will be passed to the plotter after the standard commands have been
  // passed. Can be used to set up special plotting options or to override the default behavior:
  void addCommand(const std::string& command) { commands.push_back(command); }

  void clearCommands() { commands.clear(); }


  void setupPlotter(GNUPlotter* plt);

protected:

  // Data setup:
  T xMin = 0;
  T xMax = 1;
  T yMin = 0;
  T yMax = 1;

  // Plotting setup:
  int pixelWidth  = 600;
  int pixelHeight = 600;
  std::string title;
  bool dark = false;

  GNUPlotter::ColorPalette colorMap = GNUPlotter::ColorPalette::EF_Viridis;
  bool reverseColors = false;

  std::vector<std::string> commands;  // Additional commands set by the user for customization
  // maybe rename to userCommands or customCommands

  double left   = 0.07;  // Left margin.
  double right  = 0.87;  // One minus right margin.
  double bottom = 0.1;   // Bottom margin
  double top    = 0.9;   // One minus top margin
  // Let's abbreviate "left" by "L", "right" by "R", etc.. I think, we need T-B = R-L to get an 
  // aspect ratio of 1? Here, we use T-B = R-L = 0.8. -> Figure this out and document it properly.
  // What setting do we need to get a 1:1 aspect ratio, given that the pixel-size is square?
};


template<class T>
void rsFieldPlotter2D<T>::setupPlotter(GNUPlotter* plt)
{
  if(dark)
    plt->setToDarkMode();
  else
    plt->setToLightMode();

  if(!title.empty())
    plt->setTitle(title);


  plt->setColorPalette(colorMap, reverseColors);
  plt->setPixelSize(pixelWidth, pixelHeight);
  plt->setRange(xMin, xMax, yMin, yMax);

  plt->addCommand("set lmargin at screen " + std::to_string(left));
  plt->addCommand("set rmargin at screen " + std::to_string(right));
  plt->addCommand("set tmargin at screen " + std::to_string(top));
  plt->addCommand("set bmargin at screen " + std::to_string(bottom));

  /*
  // Preliminary - make this customizable:
  plt->addCommand("set bmargin at screen 0.1");  // B: bottom
  plt->addCommand("set tmargin at screen 0.9");  // T: top
  plt->addCommand("set lmargin at screen 0.07"); // L: left
  plt->addCommand("set rmargin at screen 0.87"); // R: right
  // I think, we need T-B = R-L to get an aspect ratio of 1? Here, we use T-B = R-L = 0.9. 
  // ToDo: 
  // -Verify, if the aspect ratio is indeed 1.
  // -Don't hardcode these numbers! Have a function that can be called like 
  //  setDrawingArea(0.1, 0.9, 0.07, 0.87)
  */


  // Get rid of the axis labels:
  plt->addCommand("set xlabel \"\""); // use empty string as label
  plt->addCommand("set ylabel \"\"");
  // ToDo: let the user specify axis labels

  // Use the custom command list to set additional user-defined options or to override the default
  // settings:
  for(size_t i = 0; i < commands.size(); i++)
    plt->addCommand(commands[i]);
}




template<class T>
class rsContourMapPlotter : public rsFieldPlotter2D<T>
{

public:

  void setFunction(const std::function<T(T x, T y)>& newFunction) { f = newFunction; }

  void setOutputRange(T minZ, T maxZ) { zMin = minZ; zMax = maxZ; }

  void setNumContours(int newNumber) { numContours = newNumber; }

  void setSamplingResolution(int numSamplesX, int numSamplesY) 
  { Nx = numSamplesX; Ny = numSamplesY; }


  void plot();

protected:

  // Data setup:
  std::function<T(T x, T y)> f;
  T zMin = 0;
  T zMax = 0;
  bool clipData = true;

  // Plotting setup:
  int Nx          = 101;  // rename to resX of x-resolution or numSamplesX
  int Ny          = 101;
  int numContours = 21;

};

template<class T>
void rsContourMapPlotter<T>::plot()
{
  // Generate the data and figure out appropriate values for zMin/zMax if the user hasn't given a 
  // valid z-range and generate the array of the contour levels that will drawn in as contour 
  // lines:
  std::vector<T> x, y;
  RAPT::rsMatrix<T> z;
  generateMatrixData(f, xMin, xMax, yMin, yMax, Nx, Ny, x, y, z);
  if(zMin >= zMax) {
    zMin = z.getMinimum();
    zMax = z.getMaximum(); }
  std::vector<T> levels = RAPT::rsRangeLinear(zMin, zMax, numContours);  // Array of contour levels

  // Clip the matrix data:
  if(clipData == true) {
    for(int i = 0; i < z.getNumRows(); i++)
      for(int j = 0; j < z.getNumColumns(); j++)
        z(i, j) = RAPT::rsClip(z(i, j), zMin, zMax); }
  // ToDo: Maybe implement and use a function z.clipToRange(zMin, zMax);


  GNUPlotter plt;
  plt.addDataMatrixFlat(Nx, Ny, &x[0], &y[0], z.getDataPointer());
  setupPlotter(&plt);
  plotContours(plt, levels, true);   // Make this a member function! It's currently free.
}



template<class T>
class rsVectorFieldPlotter : public rsFieldPlotter2D<T>
{

public:

  void setFunction(const std::function<void(T x, T y, T* u, T* v)>& newFunction) 
  { f = newFunction; }

  void setArrowDensity(int densX, int densY) { numArrowsX = densX; numArrowsY = densY; }


  void plot();

protected:

  // Data setup:
  std::function<void(T x, T y, T* u, T* v)> f;

  // Plotting Setup:
  int numArrowsX = 21;
  int numArrowsY = 21;
};

template<class T>
void rsVectorFieldPlotter<T>::plot()
{
  // Some API adaptor business:
  std::function<T(T x, T y)> fu, fv;
  fu = [&](T x, T y) { T u, v; f(x, y, &u, &v); return u; };
  fv = [&](T x, T y) { T u, v; f(x, y, &u, &v); return v; };
  // ToDo: Let GNUPlotter accept a function like our f to define a 2D vector field directly.

  // Plotting:
  GNUPlotter plt;
  plt.addVectorField2D(fu, fv, numArrowsX, xMin, xMax, numArrowsY, yMin, yMax);

  //plt.setRange(xMin, xMax, yMin, yMax);  // Maybe move into setupPlotter()
  // Maybe we should have margins? It tends to look better with margins when some of the arrows 
  // point out of the rectangle defined by xMin, xMax, yMin, yMax. But then, it may be out of sync
  // with a corresponding contour map plot


  setupPlotter(&plt);
  plt.plot();
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
  for a rotation of 90∞ such that there is no difference between centerSumHorz and centerSumVert 
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



//=================================================================================================

/** A class to let console applications show their progress when performing a long task. It 
repeatedly writes a "percentage done" in the format " 45%" to the console. Note that the initial
whitespace is intentional to allow for the first digit when it hits 100%. It deletes and overwrites
the old percentage by programmatically writing backspaces to the console, so the indicator doesn't
fill up the console - it just stays put and counts up. */

class rsConsoleProgressIndicator
{

public:

  // todo: have a setting: setNumDecimalPlaces

  void deleteCharactersFromConsole(int numChars) const
  {    
    unsigned char back = 8;                // backspace, see http://www.asciitable.com/
    for(int i = 1; i <= numChars; i++)
      std::cout << back;                   // delete  characters

    // Can we do better? Like first creating a string of backspaces and writing it to cout in one 
    // go?
  }

  std::string getPercentageString(double precentage) const 
  {
    int percent = (int) round(precentage);
    // pre-padding (this is dirty - can we do this more elegantly?)
    std::string s;
    if(percent < 100) s += " ";
    if(percent < 10)  s += " ";
    s += std::to_string(percent) + "%";
    return s;
  }
  // always prodcues a 4 character string with a rounded percentage value
  // todo: produce a 7 character string with two decimal places after the dot, like:
  // " 56.87%" (the leading whitespace is intentional) because for longer tasks, we may need a 
  // finer indicator - but maybe let the user set up the number of decimals

  /** Prints the initial 0% in the desired format (i.e. with padding) to the console. */
  void init() const
  {
    std::cout << getPercentageString(0.0);
  }

  /** Deletes the old percentage from the console and prints the new one. Should be called 
  repeatedly by the worker function. */
  void update(int itemsDone, int lastItemIndex) const // use numItems - more intuitive to use
  {
    deleteCharactersFromConsole(4);                  // delete old percentage from console
    double percentDone = 100.0 * double(itemsDone) / double(lastItemIndex);
    std::cout << getPercentageString(percentDone);   // write the new percentage to console
  }
  // maybe implement a progress bar using ascii code 178 or 219
  // http://www.asciitable.com/
  // -maybe make the function mor general: itemsDone, lastItemIndex
  // -have different versions - one taking directly a percentage, another in terms of 
  //  itemsDone, numItems

  // have a function init that write 0% in the desired format


protected:

  //int numDecimals = 0; // 0: only the rounded integer percentage is shown

};

//=================================================================================================

/** A class for representing videos. It is mostly intended to be used to accumulate frames of an
animation into an array of images. */

class rsVideoRGB
{

public:

  rsVideoRGB(int width = 0, int height = 0/*, int frameRate*/)
  {
    setSize(width, height);
    //this->frameRate = frameRate;
  }


  //-----------------------------------------------------------------------------------------------
  /** \name Setup */

  /** Selects, whether or not the pixel values should be clipped to the valid range. If they are
  not clipped, they will wrap around to zero when they overflow the valid range. */
  void setPixelClipping(bool shouldClip) { clipPixelValues = shouldClip; }

  void setSize(int width, int height)
  {
    this->width  = width;
    this->height = height;
  }

  //-----------------------------------------------------------------------------------------------
  /** \name Inquiry */

  /** Returns the pixel width. */
  int getWidth() const { return width; }

  /** Returns the pixle height. */
  int getHeight() const { return height; }

  /** Returns the number of frames. */
  int getNumFrames() const { return (int) frames.size(); }

  /** Returns a const-reference to the i-th frame. */
  const rsImage<rsPixelRGB>& getFrame(int i) const { return frames[i]; }


  //-----------------------------------------------------------------------------------------------
  /** \name Manipulations */

  /** Appends a frame to the video. The images for the r,g,b channels must have the right width
  and height. */
  void appendFrame(const rsImage<float>& R, const rsImage<float>& G, const rsImage<float>& B)
  {
    rsAssert(R.hasShape(width, height));
    rsAssert(G.hasShape(width, height));
    rsAssert(B.hasShape(width, height));
    frames.push_back(rsConvertImage(R, G, B, clipPixelValues));
  }

  // have a function that just accepts pointers to float for r,g,b along with width,height, so we
  // can also use it with rsMatrix, etc. - it shouldn't depend on frames being represented as 
  // rsImage - maybe also allow for rsMultiArray (interpret a 3D array as video) 
  // -> appendFrames(rsMultiArray<float>& R
  // the range of the 1st index is the number of frames to be appended
  

protected:

  // video parameters:
  int width  = 0;
  int height = 0;
  //int frameRate = 0;  // can be decided when writing the file

  // audio parameters:
  // int numChannels = 0;
  // int sampleRate  = 0;

  // conversion parameters:
  bool clipPixelValues = false;  // clip to valid range (otherwise wrap-around)

  // video data:
  std::vector<rsImage<rsPixelRGB>> frames;

  // audio data:
  // todo: have support for (multichannel) audio
  // float** audioSamples = nullptr;
};

//=================================================================================================

/** A class for writing videos (of type rsVideoRGB) to files. It will first write all the frames as
separate temporary .ppm files to disk and then invoke ffmpeg to combine them into an mp4 video 
file. It requires ffmpeg to be available. On windows, this means that ffmpeg.exe must either be 
present in the current working directory (typically the project directory) or it must be installed 
somewhere on the system and the installation path must be one of the paths given the "PATH" 
environment variable. 

todo: maybe do it like in GNUPlotCpp: use an ffmpegPath member variable
*/

class rsVideoFileWriter
{

public:

  //-----------------------------------------------------------------------------------------------
  /** \name Setup */

  void setFrameRate(int newRate) { frameRate = newRate; }

  /** Sets the amount of compression where 0 means lossless and 51 gives the highest compression
  and worst quality. Corresponds to the CRF (constant rate factor) encoder setting, see:  
  https://trac.ffmpeg.org/wiki/Encode/H.264  */
  void setCompressionLevel(int newLevel) { compression = newLevel; }

  /** Selects, whether or not the temporary image files should be deleted after the final video has 
  been created. By default, they are deleted but keeping them can be useful for two reasons: 
  1: they are full quality, 2: it is possible to use the left/right keys in IrfanView to manually 
  "play" the video forward or backward - just load one of the ppm files and use left/right. */
  void setDeleteTemporaryFiles(bool shouldDelete) { cleanUp = shouldDelete; }

  // todo: setOutputFileName, setOutputDirectory, setTempDirectory


  //-----------------------------------------------------------------------------------------------
  /** \name File writing */

  /** Writes the given video into a file with given name. The name should NOT include the .mp4 
  extension - this will be added here. */
  void writeVideoToFile(const rsVideoRGB& vid, const std::string& fileName) const
  {
    writeTempFiles(vid);
    encodeTempFiles(fileName);
    if(cleanUp)
      deleteTempFiles(vid.getNumFrames());
  }

  /** Writes the given image as temporary .ppm file to disk. The caller needs to pass the frame 
  index and the total number of frames. The latter is needed fro the progress bar. */
  void writeTempFile(const rsImage<rsPixelRGB>& img, int frameIndex, int numFrames = 0) const
  {
    std::string path = getTempFileName(frameIndex);
    writeImageToFilePPM(img, path.c_str());
    if(numFrames != 0)
      progressIndicator.update(frameIndex, numFrames-1);  // maybe pass frameIndex+1 and numFrames
  }

  /** Writes the temporary .ppm files to the harddisk for the given video. */
  void writeTempFiles(const rsVideoRGB& vid) const
  {
    int numFrames = vid.getNumFrames();
    std::cout << "Writing ppm files: ";
    progressIndicator.init();
    for(int i = 0; i < numFrames; i++) 
    {
      writeTempFile(vid.getFrame(i), i, numFrames);

      // old:
      //std::string path = getTempFileName(i);
      //writeImageToFilePPM(vid.getFrame(i), path.c_str());
      //progressIndicator.update(i, numFrames-1);  // maybe pass i+1 and numFrames
    }
    std::cout << "\n\n";
  }

  /** Combines the temporary .ppm files that have been previously written to disk (via calling 
  writeTempFiles) into a single video file with given name.  */
  void encodeTempFiles(const std::string& fileName) const
  {
    // rsAssert(isFfmpegInstalled(), "This function requires ffmpeg to be installed");
    std::string cmd = getFfmpegInvocationCommand(fileName);
    std::cout << "Invoking ffmpeg.exe with command:\n";
    std::cout << cmd + "\n\n"; 
    system(cmd.c_str());
  }

  /** Deletes the temporary .ppm files that have been created during the process. */
  void deleteTempFiles(int numFrames) const 
  {
    std::cout << "Deleting temporary .ppm files\n";
    for(int i = 0; i < numFrames; i++)
      remove(getTempFileName(i).c_str());
    // maybe this should also show progress?
  }


  //-----------------------------------------------------------------------------------------------
  /** \name Internals */

  /** Creates the command string to call ffmpeg. */
  std::string getFfmpegInvocationCommand(const std::string& fileName) const
  {
    std::string cmd;
    cmd += "ffmpeg ";                                                 // invoke ffmpeg
    cmd += "-y ";                                                     // overwrite without asking
    cmd += "-r " + std::to_string(frameRate) + " ";                   // frame rate
    //cmd += "-f image2 ";                                              // ? input format ?
    //cmd += "-s " + std::to_string(w) + "x" + std::to_string(h) + " "; // pixel resolution
    cmd += "-i " + framePrefix + "%d.ppm ";                           // ? input data ?
    cmd += "-vcodec libx264 ";                                        // H.264 codec is common
    //cmd += "-vcodec libx265 ";                                        // H.265 codec is better
    cmd += "-crf " + std::to_string(compression) + " ";               // constant rate factor
    cmd += "-pix_fmt yuv420p ";                                       // yuv420p seems common
    cmd += "-preset veryslow ";                                       // best compression, slowest
    cmd += fileName + ".mp4";                                         // output file
    return cmd;

    // The command string has been adapted from here:
    // https://hamelot.io/visualization/using-ffmpeg-to-convert-a-set-of-images-into-a-video/
    // and by trial and error, i have commented out those options which turned out to be 
    // unnecessarry and added a few other options

    // Notes on the options:
    // -wihtout -pix_fmt yuv420p, it seems to use yuv444p by default and VLC player can't play it
    //  correctly (i think, the "p" stands for "predictive"?) - try it with ffprobe
  }

  /** Returns the name that should be used for the temp-file for a given frame. */
  std::string getTempFileName(int frameIndex) const
  {
    return framePrefix + std::to_string(frameIndex) + ".ppm";
  }


protected:

  // encoder settings:
  int compression = 0;   // 0: lossles, 51: worst
  int frameRate   = 25;

  std::string framePrefix = "VideoTempFrame";  
  // used for the temp files: VideoTempFrame1.ppm, VideoTempFrame2.ppm, ...

  // have strings for other options: pix_fmt, vcodec, preset:
  // std::string pix_fmt = "yuv420p";
  // std::string vcodec  = "libx264"
  // std::string preset  = "veryslow";


  bool cleanUp = true;

  rsConsoleProgressIndicator progressIndicator;

  //std::string tempDir, outDir, outFileName;
};
// todo: maybe allow to encode into different qualities using the same temporary files, i.e. create
// temp files once but run the encoder several times over the files with different quality 
// settings (lossless should be among them for reference)

// Infos about the video codecs of the H.26x family:
// H.261 (1988): https://en.wikipedia.org/wiki/H.261
// H.262 (1995): https://en.wikipedia.org/wiki/H.262/MPEG-2_Part_2
// H.263 (1996): https://en.wikipedia.org/wiki/H.263
// H.264 (2003): https://en.wikipedia.org/wiki/Advanced_Video_Coding
// H.265 (2013): https://en.wikipedia.org/wiki/High_Efficiency_Video_Coding
// H.266 (202x): https://en.wikipedia.org/wiki/Versatile_Video_Coding

// Guidelines for encoding video for youtube:
// https://support.google.com/youtube/answer/1722171?hl=en
// container format: .mp4
// video codec: H.264,
// audio codec: AAC-LC, stereo or 5.1, 48 kHz
// common frame rates: 24,25,30,48,50,60
// resolution: 16:9 aspect ratio is standard,

// about encoding losses:
// https://superuser.com/questions/649412/ffmpeg-x264-encode-settings
// https://slhck.info/video/2017/02/24/crf-guide.html


//=================================================================================================

/** A class to facilitate the creation of video files for recording videos of time-variant 
functions that are defined on a mesh of type rsGraph<rsVector2D<T>, T> and writing them to disk.
The user is supposed to create and set up the writer object and repeatedly call recordFrame in a 
loop and calling writeFile when done. It's mainly intended to record the time evolution of mesh
functions that are created, for example, by a PDE solver for irregular meshes for the purpose of
visualizing what the PDE solver does. ...tbc... */

template<class T>
class rsVideoWriterMesh
{

public:

  rsVideoWriterMesh();

  void setSize(int width, int height);

  /** Before calling recordFrame in a loop, you may call this function to initialize the 
  background onto which the mesh function is drawn - otherwise the background will be black. */
  void initBackground(const rsGraph<rsVector2D<T>, T>& mesh);

  void recordFrame(const rsGraph<rsVector2D<T>, T>& mesh, const std::vector<T>& meshFunction);

  void writeFile(const std::string& name, int frameRate = 25);

  void copyPositivePixelDataFrom(const rsImage<float>& source, rsImage<float>& target) 
  { 
    rsAssert(target.hasSameShapeAs(source));
    float* src = source.getPixelPointer(0, 0);
    float* dst = target.getPixelPointer(0, 0);
    for(int i = 0; i < source.getNumPixels(); i++)
      dst[i] = rsMax(0.f, src[i]);
  }

  void copyNegativePixelDataFrom(const rsImage<float>& source, rsImage<float>& target) 
  { 
    rsAssert(target.hasSameShapeAs(source));
    float* src = source.getPixelPointer(0, 0);
    float* dst = target.getPixelPointer(0, 0);
    for(int i = 0; i < source.getNumPixels(); i++)
      dst[i] = rsMax(0.f, -src[i]);
  }


protected:

  rsImage<float> background;
  rsImage<float> foreground;
  rsImage<float> frameR, frameG, frameB; // image objects to store r,g,b values of current frame
  rsVideoRGB video;                      // video object to acculate the frames

  rsAlphaMask<float> brush;
  rsImagePainterFFF painter;

  // range of the mesh-vertices - to be used for conversion from mesh-coordinates to pixel 
  // coordinates (todo: make user adjustable):
  T xMin = T(0), xMax = T(1);
  T yMin = T(0), yMax = T(1);

};

template<class T>
rsVideoWriterMesh<T>::rsVideoWriterMesh()
{
  painter.setImageToPaintOn(&foreground);
  brush.setSize(10.f);
  painter.setAlphaMaskForDot(&brush);
  painter.setUseAlphaMask(true);
}

template<class T>
void rsVideoWriterMesh<T>::setSize(int w, int h)
{
  foreground.setSize(w, h);
  background.setSize(w, h);
  frameR.setSize(w, h);
  frameG.setSize(w, h);
  frameB.setSize(w, h);
  video.setSize(w, h);
}

template<class T>
void rsVideoWriterMesh<T>::initBackground(const rsGraph<rsVector2D<T>, T>& mesh)
{
  painter.setImageToPaintOn(&background);
  float brightness = 0.25f;
  int w = background.getWidth();
  int h = background.getHeight();
  for(int i = 0; i < mesh.getNumVertices(); i++) 
  {
    rsVector2D<T> vi = mesh.getVertexData(i);
    vi.x = rsLinToLin(vi.x, xMin, xMax, T(0), T(w-1));
    vi.y = rsLinToLin(vi.y, yMin, yMax, T(h-1), T(0));
    for(int k = 0; k < mesh.getNumEdges(i); k++)
    {
      int j = mesh.getEdgeTarget(i, k);
      rsVector2D<T> vj = mesh.getVertexData(j);
      vj.x = rsLinToLin(vj.x, xMin, xMax, T(0), T(w-1));
      vj.y = rsLinToLin(vj.y, yMin, yMax, T(h-1), T(0));
      painter.drawLineWu(float(vi.x), float(vi.y), float(vj.x), float(vj.y), brightness);
    }
  }
  painter.setImageToPaintOn(&foreground);
}
// optimize away the rsLinToLin calls

template<class T>
void rsVideoWriterMesh<T>::recordFrame(
  const rsGraph<rsVector2D<T>, T>& mesh, const std::vector<T>& u)
{
  foreground.clear();
  int w = foreground.getWidth();
  int h = foreground.getHeight();
  float brightness = 4.f;
  for(int i = 0; i < mesh.getNumVertices(); i++) 
  {
    rsVector2D<T> vi = mesh.getVertexData(i);
    vi.x = rsLinToLin(vi.x, xMin, xMax, T(0), T(w-1));
    vi.y = rsLinToLin(vi.y, yMin, yMax, T(h-1), T(0));
    T value = brightness * float(u[i]);
    painter.paintDot(vi.x, vi.y, rsAbs(value)); 
  }

  //frameB.copyPixelDataFrom(background);

  // The blue channel is used for drawing the mesh in the background, the red channel draws 
  // positive values and the green channel draws (absolute values of) negative values
  copyPositivePixelDataFrom(foreground, frameR);
  copyNegativePixelDataFrom(foreground, frameG);
  frameB.copyPixelDataFrom(background);  // we don't need to do this for each frame
  video.appendFrame(frameR, frameG, frameB);

  int dummy = 0;
}

template<class T>
void rsVideoWriterMesh<T>::writeFile(const std::string& name, int frameRate)
{
  rsVideoFileWriter vw;
  vw.setFrameRate(frameRate);
  vw.setCompressionLevel(10);  // 0: lossless, 10: good enough, 51: worst - 0 produces artifacts
  vw.setDeleteTemporaryFiles(false);
  vw.writeVideoToFile(video, name);
}

//=================================================================================================

/** Given 3 arrays of images for the red, green and blue color channels, this function writes them
as video into an .mp4 file. It is assumed that the arrays R,G,B have the same length and each image
in all 3 arrays should have the same pixel-size. The function first writes each frame as a separate
.ppm file into the current working directory (typically the project folder) and then invokes ffmpeg
to stitch the images together to a video file. For this to work (on windows), ffmpeg.exe must be 
either present in the current working directory (where the .ppm images end up) or it must be 
installed on the system and the PATH environment variable must contain the respective installation 
path. You may get it here: https://www.ffmpeg.org/ */
void writeToVideoFileMP4(
  const std::vector<rsImage<float>>& R,
  const std::vector<rsImage<float>>& G,
  const std::vector<rsImage<float>>& B,
  const std::string& name,                    // should not include the .mp4 suffix
  int frameRate = 25,
  int qualityLoss = 0)                        // 0: lossless, 51: worst
{
  size_t numFrames = R.size();
  rsAssert(G.size() == numFrames && B.size() == numFrames);
  if(numFrames == 0)
    return;

  // write each frame into a ppm file:
  int w = R[0].getWidth();                // pixel width
  int h = R[0].getHeight();               // pixel height
  std::string prefix = name + "_Frame";   // prefix for all image file names
  std::cout << "Writing ppm files\n\n";
  for(size_t i = 0; i < numFrames; i++)
  {
    std::string path = prefix + std::to_string(i) + ".ppm";
    rsAssert(R[i].hasShape(w, h)); 
    rsAssert(G[i].hasShape(w, h));
    rsAssert(B[i].hasShape(w, h));
    writeImageToFilePPM(R[i], G[i], B[i], path.c_str());
  }

  // invoke ffmpeg to convert the frame-images into a video (maybe factor out):

  // rsAssert(isFfmpegInstalled(), "This function requires ffmpeg to be installed");
  // compose the command string to call ffmpeg:
  std::string cmd;
  cmd += "ffmpeg ";                                                 // invoke ffmpeg
  cmd += "-y ";                                                     // overwrite without asking
  cmd += "-r " + std::to_string(frameRate) + " ";                   // frame rate
  //cmd += "-f image2 ";                                              // ? input format ?
  //cmd += "-s " + std::to_string(w) + "x" + std::to_string(h) + " "; // pixel resolution
  cmd += "-i " + prefix + "%d.ppm ";                                // ? input data ?
  cmd += "-vcodec libx264 ";                                        // very common standard
  //cmd += "-vcodec libx265 ";                                      // ...newer standard
  cmd += "-crf " + std::to_string(qualityLoss) + " ";               // constant rate factor
  //cmd += "-pix_fmt yuv420p ";                                       // seems to be standard
  //cmd += "-pix_fmt rgb24 ";                                       // ..this does not work well
  cmd += "-preset veryslow ";                                       // best compression, slowest
  cmd += name + ".mp4";                                             // output file


  // crf: quality - smaller is better, 0 is best, 20 is already quite bad, 10 seems good enough
  // see: https://trac.ffmpeg.org/wiki/Encode/H.264
  // The range of the CRF scale is 0ñ51, where 0 is lossless, 23 is the default, and 51 is worst 
  // quality possible. Even the lossless setting may achieve compression of factor 30 with the SIRP
  // model


  std::cout << "Invoking ffmpeg.exe with command:\n";
  std::cout << cmd + "\n\n"; 
  system(cmd.c_str());

  // maybe optionally clean up the (temporary) .ppm files - but the can be useful - one can "run"
  // the animation manually in IrfanView by using the cursor keys
}
// this function has quite a lot parameters - maybe about time to turn it into a class - the can be
// a convenience function, though
// ...but maybe keep this as a quick-and-dirty solution

// pixel-formats:
// -with yuv420p, VLC player can play it nicely but windows media player shows garbage
// -with rgb24, VLC player shows garbage and wnidows media player gives an error message and 
//  refuses to play at all
// -for the meaning of pixel-formats, see:
//  https://github.com/FFmpeg/FFmpeg/blob/master/libavutil/pixfmt.h

// for audio encoding using flac, see
// https://www.ffmpeg.org/ffmpeg-codecs.html#flac-2

// maybe try to encode with H.265 instead of H.264
// https://trac.ffmpeg.org/wiki/Encode/H.265
// this is a newer version of the encoder that may give better compression
// it seems to make no difference - using the option -vcodec libx265 instead of -vcodec libx265,
// i get exactly the same filesize

//=================================================================================================

/** Class for representing a parametric plane (parametrized by s and t) given in terms of 3 vectors
u,v,w:

   P(s,t) = u + s*v + t*u 
   
Aside from being able to compute points on the plane, it can also compute parameter pairs s,t such
that the point on the plane at these parameters is close to some given target point...  */

template<class T>
class rsParametricPlane3D
{


public:


  void setVectors(const rsVector3D<T>& newU, const rsVector3D<T>& newV, const rsVector3D<T>& newW)
  {
    u = newU;
    v = newV;
    w = newW;
  }

  /** Returns the point (x,y,z) on the plane that corresponds to the pair of parameters (s,t). */
  rsVector3D<T> getPointOnPlane(T s, T t) const
  {
    return u + s*v + t*w;
  }

  /** Computes parameters s,t such that a point on the plane with these parameters will be have 
  matched x and y coordinates with the given target point. */
  void getMatchParametersXY(const rsVector3D<T>& target, T* s, T* t) const
  {
    rsVector3D<T> c = target-u;
    T k =   1 / (v.y*w.x - v.x*w.y);
    *s  =  (c.y*w.x - c.x*w.y) * k;
    *t  = -(c.y*v.x - c.x*v.y) * k;
  }
  // var("s t cx cy vx vy wx wy")
  // e1 = cx == s*vx + t*wx
  // e2 = cy == s*vy + t*wy
  // solve([e1,e2],[s,t])
  // -> [[s == (cy*wx - cx*wy)/(vy*wx - vx*wy), t == -(cy*vx - cx*vy)/(vy*wx - vx*wy)]]

  /** Computes parameters s,t such that a point on the plane with these parameters will be have 
  matched x and z coordinates with the given target point. */
  void getMatchParametersXZ(const rsVector3D<T>& target, T* s, T* t) const
  {
    rsVector3D<T> c = target-u;
    T k =   1 / (v.z*w.x - v.x*w.z);
    *s =  (c.z*w.x - c.x*w.z) * k;
    *t = -(c.z*v.x - c.x*v.z) * k;
  }

  /** Computes parameters s,t such that a point on the plane with these parameters will be have 
  matched y and z coordinates with the given target point. */
  void getMatchParametersYZ(const rsVector3D<T>& target, T* s, T* t) const
  {
    rsVector3D<T> c = target-u;
    T k =   1 / (v.z*w.y - v.y*w.z);
    *s =  (c.z*w.y - c.y*w.z) * k;
    *t = -(c.z*v.y - c.y*v.z) * k;
  }

  /** Computes parameters s,t such that a point on the plane with these parameters will be closest 
  to the given target point in the sense of minimizing the Euclidean distance. */
  void getMatchParameters(const rsVector3D<T>& target, T* s, T* t) const
  {
    // Idea: minimize the squared norm of a-b where a = u + s*v + t*u is a vector produced by our 
    // plane equation and b is the target vector. This leads to finding the minimum of the 
    // quadratic form: err(s,t) = A*s^2 + B*s*t + C*t^2 + D*s + E*t + F with respect to s and t 
    // with the coefficients defined as below:

    rsVector3D<T> c = u-target;  // c := u-b -> now we minimize norm of c + s*v + t*w
    T A =   dot(v,v);
    T B = 2*dot(v,w);
    T C =   dot(w,w);
    T D = 2*dot(c,v);
    T E = 2*dot(c,w);
    T F =   dot(c,c);
    T k = 1 / (B*B - 4*A*C); // can we get a division by zero here?
    *s  = (2*C*D - B*E) * k;
    *t  = (2*A*E - B*D) * k;

    // Is it possible to derive simpler formulas? maybe involving the plane normal (which can be 
    // computed via the cross-product of v and w)? There are a lot of multiplications being done 
    // here....
  }
  // sage:
  // var("s t A B C D E F")
  // f(s,t) = A*s^2 + B*s*t + C*t^2 + D*s + E*t + F
  // dfds = diff(f, s);
  // dfdt = diff(f, t);
  // e1 = dfds == 0
  // e2 = dfdt == 0
  // solve([e1,e2],[s,t])
  // -> [[s == (2*C*D - B*E)/(B^2 - 4*A*C), t == -(B*D - 2*A*E)/(B^2 - 4*A*C)]]

  // symbolic optimization with sage
  // stationary_points = lambda f : solve([gi for gi in f.gradient()], f.variables())
  // f(x,y) = -(x * log(x) + y * log(y))
  // stationary_points(f)
  // [[x == e^(-1), y == e^(-1)]]
  // doesnt work here because we have all these parameters A,B,C,... and this code finds the 
  // gradient also with respct to these parameters, if we adapt it

  // stationary_points = lambda f : solve([gi for gi in f.gradient()], f.variables())

  // https://ask.sagemath.org/question/38079/can-sage-do-symbolic-optimization/


  /** Returns a point on the plane that is closest to the given target point in the sense of having
  the minimum Euclidean distance. */
  rsVector3D<T> getClosestPointOnPlane(const rsVector3D<T>& target) const
  {
    T s, t;
    getMatchParameters(target, &s, &t);
    return getPointOnPlane(s, t);
  }

  /** Computes the distance of the given point p from the plane. */
  T getDistance(const rsVector3D<T>& p) const
  {
    rsVector3D<T> q = getClosestPointOnPlane(p);
    q -= p;
    return q.getEuclideanNorm();
  }
  // needs test
  // todo: compute the signed distance - maybe that requires translating to implicit form:
  //   A*x + B*y + C*z + D = 0


protected:

  rsVector3D<T> u, v, w;

};
// todo: clean up and move to rapt (into Math/Geometry)


/** Returns a vector that contains a chunk of the given input vector v, starting at index "start" 
with length given by "length". */
template<class T>
inline std::vector<T> rsChunk(const std::vector<T>& v, int start, int length)
{
  rsAssert(length >= 0);
  rsAssert(start + length <= (int) v.size());
  std::vector<T> r(length);
  rsArrayTools::copy(&v[start], &r[0], length);
  return r;
}

//-------------------------------------------------------------------------------------------------

/** Extends rsMultiArray by storing information whether a given index is covariant or contravariant
and a tensor weight which is zero for absolute tensors and nonzero for relative tensors. 

A tensor is:
-A geometric or physical entity that can be represented by a multidimensional array, once a 
 coordinate system has been selected. The tensor itself exists independently from the coordinate 
 systems, but its components (i.e. the array elements) are always to be understood with respect to 
 a given coordinate system.
-A multilinear map that takes some number of vectors and/or covectors as inputs and produces a 
 scalar as output. Multilinear means that it is linear in all of its arguments.
-When converting tensor components from one coordinate system to another, they transform according
 to a specific set of transformation rules.
-Scalars, vectors (column-vectors), covectors (row-vectors) and matrices are included as special 
 cases, so tensors really provide a quite general framework for geometric and physical 
 calculations.

 References:
   (1) Principles of Tensor Calculus (Taha Sochi)


*/

template<class T>
class rsTensor : public rsMultiArray<T>
{

public:

  // todo: setters

  using rsMultiArray::rsMultiArray;  // inherit constructors


  //-----------------------------------------------------------------------------------------------
  // \name Setup


  //-----------------------------------------------------------------------------------------------
  // \name Inquiry

  int getRank() const { return getNumIndices(); }
  // this is the total rank, todo: getCovariantRank, getContravariantRank

  bool isIndexCovariant(int i) const { return covariant[i]; }

  bool isIndexContravariant(int i) const { return !isIndexCovariant(i); }

  int getWeight() const { return weight; }

  /** Returns true, iff indices i and j are of opposite variance type, i.e. one covariant the other
  contravariant. This condition is required for the contraction with respect to i,j to make 
  mathematical sense, so it can be used as sanity check to catch errors in tensor computations. */
  bool areIndicesOpposite(int i, int j) const
  {
    return (covariant[i] && !covariant[j]) || (!covariant[i] && covariant[j]);
  }

  // isScalar, isVector, isCovector, 

  // compares ranks, shape, variances and weights
  bool isOfSameTypeAs(const rsTensor<T>& A) const
  { return this->shape == A.shape && this->covariant == A.covariant && this->weight == A.weight; }

  /** Compares this tensor with another tensor A for equality with a tolerance. */
  bool equals(const rsTensor<T>& A, T tol) const
  {
    bool r = true;
    r &= getShape() == A.getShape();
    r &= covariant  == A.covariant;
    r &= weight     == A.weight;
    r &= rsArrayTools::almostEqual(getDataPointerConst(), A.getDataPointerConst(), getSize(), tol);
    return r;
  }

  //-----------------------------------------------------------------------------------------------
  // \name Operations
  // see rsMultiArrayOld for possible implementations

  /** Returns a tensor that results from contracting tensor A with respect to the given pair of 
  indices.  */
  static rsTensor<T> getContraction(const rsTensor<T>& A, int i, int j)
  {
    // sanity checks:
    rsAssert(A.covariant.size() == 0 || A.areIndicesOpposite(i, j),
      "Indices i,j must have opposite (co/contra) variance type for contraction");
    rsAssert(A.shape[i] == A.shape[j], "Summation indices must have the same range");
    rsAssert(A.getRank() >= 2, "Rank must be at least 2");

    // verify this:
    rsTensor<T> B;
    B.shape.resize(A.shape.size() - 2);
    int k = 0, l = 0;
    while(k < B.getRank()) {
      if(l == i || l == j)  {
        l++; continue; }
      B.shape[k] = A.shape[l];
      k++; l++; }
    B.adjustToNewShape();

    // this needs thorough verifications - step/jump/start were mostly guessed:
    int step = A.strides[i] + A.strides[j];
    int jump = rsArrayTools::sum(&A.strides[0], (int) A.strides.size()) - step;
    int num  = A.shape[i];                    // == A.shape[j]
    for(k = 0; k < B.getSize(); k++) {
      B.data[k] = T(0);
      int start = k*jump;
      for(l = 0; l < num; l++)
        B.data[k] += A.data[start + l*step]; }
    return B;
  }


  /** Computes the outer product between tensors A and B, also known as direct product, tensor 
  product or Kronecker product. Every element of tensor A is multiplied with every element from 
  tensor B. todo: explain, how the data is arranged in the result */
  static rsTensor<T> getOuterProduct(const rsTensor<T>& A, const rsTensor<T>& B)
  {
    // Allocate and set up result:
    size_t i;
    rsTensor<T> C;
    C.shape.resize(A.shape.size() + B.shape.size());
    for(i = 0; i < A.shape.size(); i++) C.shape[i]                  = A.shape[i];
    for(i = 0; i < B.shape.size(); i++) C.shape[i + A.shape.size()] = B.shape[i];
    C.adjustToNewShape();

    // Set up the weight and the "covariant" array of the result:
    C.weight = A.weight + B.weight;  // verify
    C.covariant.resize(A.covariant.size() + B.covariant.size());
    for(i = 0; i < A.covariant.size(); i++) C.covariant[i]                      = A.covariant[i];
    for(i = 0; i < B.covariant.size(); i++) C.covariant[i + A.covariant.size()] = B.covariant[i];

    // Fill the data-array of the result and return it:
    int k = 0;
    for(int i = 0; i < A.getSize(); i++) {
      for(int j = 0; j < B.getSize(); j++) {
        C.data[k] = A.data[i] * B.data[j]; k++; }}
    return C;
  }

  static rsTensor<T> getInnerProduct(const rsTensor<T>& A, const rsTensor<T>& B, int i, int j)
  {
    return getContraction(getOuterProduct(A, B), i, j);  
    // preliminary - todo: optimize this by avoiding the blown up outer product as intermediate
    // result
  }


  static int getDivisionIndex(const rsTensor<T>& A) // maybe rename to getBestDivisorIndex
  {
    //return 0;  // preliminary
    //return 7;  // test
    //return 5; // test
    //return rsArrayTools::firstIndexWithNonZeroValue(A.getDataPointerConst(), A.getSize());
    return rsArrayTools::maxAbsIndex(A.getDataPointerConst(), A.getSize());
      // maybe later use an entry that causes the least rounding errors (i think, we should look 
      // for numbers, whose mantissa has the largest number trailing zero bits) - in case of 
      // complex numbers we should use the one for which the minimum of both re/im of the number
      // of trailing mantissa bits is largest
  }

  /** Given a tensor product C = A*B and the right factor B, this function retrieves the left
  factor A. */
  static rsTensor<T> getLeftFactor(const rsTensor<T>& C, const rsTensor<T>& B)
  {
    rsTensor<T> A;      // result
    int rankA = C.getRank() - B.getRank();
    A.setShape(rsChunk(C.shape, 0, rankA));
    if(C.covariant.size() > 0)
      A.covariant = rsChunk(C.covariant, 0, rankA);
    A.weight = C.weight - B.weight;

    int offset = getDivisionIndex(B);
    int k = B.getSize();
    for(int i = 0; i < A.getSize(); i++)
      A.data[i] = C.data[k*i + offset] / B.data[offset];
    return A;
  }
  // seems to work

  /** Given a tensor product C = A*B and the left factor A, this function retrieves the right 
  factor B. */
  static rsTensor<T> getRightFactor(const rsTensor<T>& C, const rsTensor<T>& A)
  {
    rsTensor<T> B;      // result
    int rankA = A.getRank();
    int rankB = C.getRank() - rankA;
    B.setShape(rsChunk(C.shape, rankA, rankB));
    if(C.covariant.size() > 0)
      B.covariant = rsChunk(C.covariant, rankA, rankB);
    B.weight = C.weight - A.weight;

    // needs verification:
    int offset = getDivisionIndex(A);
    int k = B.getSize();
    for(int i = 0; i < B.getSize(); i++)
      B.data[i] = C.data[i + k*offset] / A.data[offset];
    return B;
  }
  // seems to work



  //-----------------------------------------------------------------------------------------------
  // Factory functions:

  /** Creates the Kronecker delta tensor (aka unit tensor) for the given number of dimensions. This
  is a rank-2 tensor represented by the NxN identity matrix. */
  static rsTensor<T> getDeltaTensor(int numDimensions)
  {
    int N = numDimensions;
    rsTensor<T> D;
    D.setShape({N, N});
    D.setToZero();
    for(int i = 0; i < N; i++)
      D(i, i) = T(1);
    return D;
  }
  // see (1) pg. 76
  // todo: create generalized delta tensor - maybe use an optional parameter...
  // maybe rename to getUnitTensor

  /** Returns the permutation tensor for the given number of dimensions. It's also known as 
  epsilon tensor, Levi-Civita tensor, anti-symmetric and alternating tensor. It has a rank 
  equal to the number of dimensions, so if this number is N, this tensor has N^N components, which 
  means it grows really fast with the number of dimensions. It's very sparse though, i.e. many 
  components are zero, so it's probably not a good idea to use this tensor explicitly in 
  computations in production code - especially in higher dimensions. 
  ...todo: use weight...set it in the tensor and fill the "covariant" array accordingly
  */
  static rsTensor<T> getPermutationTensor(int numDimensions, int weight = 0)
  {
    int N = numDimensions;
    std::vector<int> indices(N);  
    rsFill(indices, N);             // we "abuse" the indices array here to represent the shape
    rsTensor<T> E(indices);
    E.setToZero();
    for(int i = 0; i < E.getSize(); i++)
    {
      E.structuredIndices(i, &indices[0]);
      E.data[i] = (T) rsLeviCivita(&indices[0], N);
      // this implementation may still be highly suboptimal - i just wanted something that works
      int dummy = 0;
    }

    // maybe factor out, so we may have a version that doesn't use the weight and variance-flags:
    E.weight = weight;
    E.covariant.resize(N);
    if(weight == -1)
      rsFill(E.covariant, char(1));
    else
      rsFill(E.covariant, char(0));


    return E;
  }
  // see (1) pg 77
  // it's either a contravariant relative tensor of weight +1 or a covariant tensor of weight -1
  // (see (1) pg. 81) -> set this up correctly! ...maybe optionally - pass -1 or +1 as parameter - 
  // or 0, if the weight should not be used





  static rsTensor<T> getGeneralizedDeltaTensor(int numDimensions)
  {
    int N = numDimensions;
    rsTensor<T> Eu = getPermutationTensor(N, +1); // contravariant version of permutation tensor
    rsTensor<T> El = getPermutationTensor(N, -1); // covariant version
    return Eu * El;                               // (1) Eq. 201
  }
  // it's a mixed tensor - the first N indices are upper the second N indices are lower, (1) Eq 201



  

  // todo:
  // -factory functions for special tensors: epsilon, delta (both with optional argument to produce
  //  the generalized versions
  //  -maybe verify some epsilon-delta identities in unit test
  // -(anti)symmetrizations

  //-----------------------------------------------------------------------------------------------
  // Operators:

  rsTensor<T> operator+(const rsTensor<T>& B) const
  { 
    rsAssert(isOfSameTypeAs(B), "Tensors to be added must have same shape, variances and weights");
    rsTensor<T> C(this->shape); 
    this->add(*this, B, &C); 
    return C; 
  }

  rsTensor<T> operator-(const rsTensor<T>& B) const
  { 
    rsAssert(isOfSameTypeAs(B), "Tensors to be subtracted must have same shape, variances and weights");
    rsTensor<T> C(this->shape); 
    this->subtract(*this, B, &C); 
    return C; 
  }

  rsTensor<T> operator*(const rsTensor<T>& B) const
  { return getOuterProduct(*this, B); }

  rsTensor<T> operator/(const rsTensor<T>& B) const
  { return getLeftFactor(*this, B); }

  // maybe we should override the multiplication operator? the baseclass defines it as element-wise
  // multiplication...but then, maybe we should also override the division - but should be then 
  // return the left or the right factor? is one more common than the other? what about the
  // quotient-rule (Eq. 141) - in this case, C and B are given and A is computed....but no - this
  // rule involves a contraction...i think, when starting from A*B = C, then dividing both sides by 
  // B leading to A = A*B/B = C/B is more natural than dividing both sides by a and writing
  // B = A*B/A = C/A, so we should probably compute A - the left factor - or maybe disallow 
  // division by the / operator...or see what sage or matblab or mathematica do

  // todo: multiplication by a scalar (outside the class)

  rsTensor<T> operator==(const rsTensor<T>& B) const
  {
    return this->isOfSameTypeAs(B) 
      && rsArrayTools::equal(this->getDataPointer(), B.getDataPointer(), getSize());
  }

protected:

  void adjustToNewShape()
  {
    updateStrides(); 
    updateSize();
    data.resize(getSize());
    updateDataPointer();
  }
  // maybe move to baseclass

  // using these is optional - if used, it allows for some sanity checks in the arithmetic 
  // operations such as catching (i.e. raising an assertion) when you try to add tensors of 
  // different variance types and/or weight or contracting with respect to two indices of the same
  // variance type (none of which makes sense mathematically):
  int weight = 0;


  //std::vector<bool> covariant; // true if an index is covariant, false if contravariant
  std::vector<char> covariant; // 1 if an index is covariant, 0 if contravariant
    // todo: use a more efficient datastructure - maybe rsFlags64 - this would limit us to tensors
    // up to rank 64 - but that should be crazily more than enough

  //A.covariant = rsChunk(C.covariant, 0, rankA);
  // doesn't compile with vector<bool> because:
  // https://stackoverflow.com/questions/17794569/why-is-vectorbool-not-a-stl-container
  // https://howardhinnant.github.io/onvectorbool.html
  // that's why i use vector<char> for the time being - eventually, it's perhaps best to use 
  // rsFlags64 (which may have to be written)
};

/** Multiplies a scalar and a tensor. */
template<class T>
inline rsTensor<T> operator*(const T& s, const rsTensor<T>& A)
{ rsTensor<T> B(A); B.scale(s); return B; }


//-------------------------------------------------------------------------------------------------

/** Class for doing computations with N-dimensional manifolds that are embedded in M-dimensional
Euclidean space, where M >= N. The user must provide a function that takes as input an 
N-dimensional vector of general curvilinear coordinates and produces a corresponding M-dimensional 
vector of the resulting coordinates in the embedding space. The case N = M = 3 can be used to do 
work with cylindrical or spherical coordinates in 3D space. The class can compute various geometric
entities on the manifold, such as the metric tensor...

todo: compute lengths of curves, areas, angles, geodesics, ...

References:
  (1) Principles of Tensor Calculus (Taha Sochi)
  (2) Aufstieg zu den Einsteingleichungen (Michael Ruhrl‰nder)


todo: 
-all numerical derivatives are computed using a central difference approximation and they all use
 the same stepsize h (which can be set up from client code) - maybe let the user configure the 
 object with some sort of rsNumercialDifferentiator where such things can be customized
-avoid excessive memory allocations: maybe pre-allocate some temporary vectors, matrices and 
 higher rank tensors which are used over and over again for intermediate results instead of 
 allocating them as local variables

*/

template<class T>
class rsManifold
{

public:

  rsManifold(int numManifoldDimensions, int numEmbeddingSpaceDimensions)
  {
    N = numManifoldDimensions;
    M = numEmbeddingSpaceDimensions;

    rsAssert(M >= N, "Can't embedd manifolds in lower dimensional spaces");
    // ...you may consider to recast your problem as an M-dimensional manifold in N-space, i.e. 
    // swap the roles of M and N
  }


  /** An array of N functions where each functions expects N arguments, passed as length-N vector, 
  where N is the dimensionality of the space. */
  //using FuncArray = std::vector<std::function<T(const std::vector<T>&)>>;

  using Vec  = std::vector<T>;
  using Mat  = rsMatrix<T>;
  using Tens = rsMultiArray<T>;

  using FuncSclToVec = std::function<void(T, Vec&)>;
  // function that maps scalars to vectors, i.e defines parametric curves

  using FuncVecToVec = std::function<void(const Vec&, Vec&)>;
  // 1st argument: input vector, 2nd argument: output vector

  using FuncVecToMat = std::function<void(const Vec&, Mat&)>;
  // 1st argument: input vector, 2nd argument: output Jacobian matrix

  //-----------------------------------------------------------------------------------------------
  // \name Setup


  /** Sets the approximation stepsize for computing numerical derivatives. Client code may tweak 
  this for more accurate results. */
  void setApproximationStepSize(T newSize) { h = newSize; }

  /** Sets the functions that convert from general curvilinear coordinates to the corresponding 
  canonical cartesian coordinates. */
  void setCurvilinearToCartesian(const FuncVecToVec& newCoordFunc)
  { u2x = newCoordFunc; }


  /** Sets the functions that convert from canonical cartesian coordinates to the corresponding 
  general curvilinear coordinates.. */
  void setCartesianToCurvilinear(const FuncVecToVec& newInverseCoordFunc)
  { x2u = newInverseCoordFunc; }
  // this should be optional, too


  // all stuff below is optional...

  void setCurvToCartJacobian(const FuncVecToMat& newFunc)
  { u2xJ = newFunc; }

  void setCartToCurvJacobian(const FuncVecToMat& newFunc)
  { x2uJ = newFunc; }



  //-----------------------------------------------------------------------------------------------
  // \name Computations

  /** Converts a vector of general curvilinear coordinates to the corresponding canonical cartesian
  coordinates. */
  void toCartesian(const Vec& u, Vec& x) const 
  { checkDims(u, x); u2x(u, x); }
  // x should be M-dim, u should be N-dim

  /** Converts a vector of canonical cartesian coordinates to the corresponding general curvilinear
  coordinates. */
  void toCurvilinear(const Vec& x, Vec& u) const 
  { checkDims(u, x); x2u(x, u); }
  // x should be M-dim, u should be N-dim


  /** The i-th column of the returned matrix E is the i-th covariant basis vector emanating from a 
  given (contravariant) position vector u. E(i,j) is the j-th component of the i-th basis 
  vector. */
  Mat getCovariantBasis(const Vec& u) const
  {
    rsAssert((int)u.size() == N);
    if( u2xJ ) {  // is this the right way to check, if std::function is not empty?
      Mat E(M, N); u2xJ(u, E); return E; }
    else
      return getCovariantBasisNumerically(u);
  }







  /** Returns MxN matrix, whose columns are the covariant(?) basis vectors in R^M at the 
  contravariant(?) coordinate position u in R^N. */
  Mat getCovariantBasisNumerically(const Vec& u) const
  {
    Vec x(M);

    // Compute tangents of the coordinate curves at x - these are the basis vectors - and 
    // organize them in a MxN matrix:
    Mat E(M, N);
    Vec up = u, um = u;   // u with one of the coordinates wiggled u+, u-
    Vec xp = x, xm = x;   // resulting wiggled x-vectors - why init with an uninitialized x?
    T s = 1/(2*h);
    for(int j = 0; j < N; j++)
    {
      // wiggle one of the u-coordinates up and down:
      up[j] = u[j] + h;    // u+
      um[j] = u[j] - h;    // u-

      // compute resulting x-vectors:
      toCartesian(up, xp); // x+
      toCartesian(um, xm); // x-

      // approximate tangent by central difference:
      x = s * (xp - xm); // maybe this can be get rid of - do it directly in the loop below..

      // write tangent into matrix column:
      //E.fillColumn(i, &x[0]);
      for(int i = 0; i < M; i++)
        E(i, j) = x[i];  // we could do directly E(i, j) = s * (xp[i] - xm[i])

      // restore u-coordinates:
      up[j] = u[j];
      um[j] = u[j];
    }

    return E; // E = [ e_1, e_2, ..., e_N ], the columns are the basis vectors
  }
  // maybe, if we express this stuff in terms of tensors, we can also check, if the incoming vector
  // is a (1,0)-tensor (as opposed to being a (0,1)-tensor) - the index should be a contravariant, 
  // i.e. upper index.
  // wait - isn't this matrix here the Jacobian matrix? see Eq 64
  // maybe rename to getCoordinateCurveTangents - see Eq. 45
  // wait - is it consistent to use the columns as basis vectors? column-vectors are supposed to
  // be contravariant vectors, but these basis-vectors are covariant. On the other hand, when 
  // forming a vector as linear combination from these basis vectors, we use contravariant 
  // components and these should multiply column-vectors such that the result is also a column 
  // vector

  Mat getContravariantBasis(const Vec& u) const
  {
    rsAssert((int)u.size() == N);
    if( x2uJ ) {  // is this the right way to check, if std::function is not empty?
      Mat E(N, M); Vec x(M); u2x(u, x); x2uJ(x, E); return E; }
    else
      return getContravariantBasisNumerically(u);
  }

  void fillCovariantBasisVector(const Vec& u, int i, Vec& Ei)
  {
    Vec up = u, um = u;    // u with one of the coordinates wiggled u+, u-
    Vec xp = x, xm = x;    // resulting wiggled x-vectors
    up[i] = u[i] + h;      // u+
    um[i] = u[i] - h;      // u-
    toCartesian(up, xp);   // x+
    toCartesian(um, xm);   // x-
    Ei = (1/(2*h)) * (xp - xm) ;
  }




  /*
  rsMatrix<T> getContravariantBasis(const std::vector<T>& u) const
  {
    rsAssert((int)u.size() == N);

    return getContravariantBasisNumerically(u);
  }
  */


  /** Computes the contravariant basis vectors according to their definition in (1), Eq. 45 as the
  gradients of the N u-functions as functions of the cartesian x-coordinates. Each of the N 
  u-functions can be seen as a scalar function of the M coordinates of the embedding space. The 
  gradients of the scalar fields in M-space are the contravariant basis vectors. The basis vectors 
  are the rows of the returned matrix - this is consistent with the idea that a gradient is 
  actually not a regular (column) vector but a covector (i.e. row-vector, aka 1-form). So the 
  function will return an NxM matrix whose rows are the contravariant basis vectors. The input 
  vector u is assumed to be in contravariant form as well (as usual). */
  Mat getContravariantBasisNumerically(const Vec& u) const
  {
    rsAssert((int)u.size() == N);
    Vec x(M);
    toCartesian(u, x); 
    T s = 1/(2*h);
    Vec xp = x, xm = x;   // wiggled x-vectors
    Vec up(N), um(N);
    Mat E(N, M);
    for(int j = 0; j < M; j++)
    {
      // wiggle one of the x-coordinates up and down:
      xp[j] = x[j] + h;    // x+
      xm[j] = x[j] - h;    // x-
      // this may actually produces invalid coordinates that are not inside the manifold - may this
      // be the reason, that things go wrong with gl*gu != id, i.e. the co-and contravarinat metric 
      // tensors do not multiply to the identity? ...chack with metrics computed from analytic 
      // Jacobians - but wait - i did this by hand - mayb it was wrong to throw away the z coordinate
      // in the computation of u,v

      // compute resulting u-vectors:
      toCurvilinear(xp, up); // u+
      toCurvilinear(xm, um); // u-

      // compute partial derivatives of the N u-coordinates to the j-th x-coordinate and write them
      // into the matrix:
      for(int i = 0; i < N; i++)
        E(i, j) = s * (up[i] - um[i]);

      // restore x-coordinates:
      xp[j] = x[j];
      xm[j] = x[j];
    }
    return E;
  }
  // factor out into a getJacobian - should be in rsNumericDifferentiator
  // maybe rename to getSpatialCoordinateGradients
  // i think the gradient of a scalar function should indeed be represented as a covector, i.e. the
  // matrix returned from here should be NxM and not MxN - such that the rows of the matrix are 
  // the individual gradients of the individual scalar functions u1(x1,...,xM),...,uN(x1,...,xM)
  // an alternative way to compute the contravariant basis vectors (that does not require client
  // code to provide the inverse coordinate mapping) would be to use the covariant basis vectors 
  // and raise the indices using the metric (perhaps the inverse metric?) ...maybe that could be 
  // used as fallback strategy when the inverse mapping is not provided? It would be really nice, 
  // if we could free client code from the requirement of providing the inverse functions, as they 
  // may be hard to obtain. All we really need is the ability to do a local inversion which is 
  // provided by inverting the Jacobian (right?)
  // when x2u is not assigned (empty): use as fallback: getCovariantBasis() and raise indices using
  // the metric


  /** Returns the covariant form of the metric tensor at given contravariant(?) curvilinear 
  coordinate position u. The metric tensor is a symmetric NxN matrix where N is the dimensionality
  of the manifold. */
  Mat getCovariantMetric(const Vec& u) const
  {
    Mat E = getCovariantBasis(u);
    return E.getTranspose() * E;  // (1), Eq. 213 (also Eq. 62 ? with E == J)
  }
  // maybe the computation of the metric can be done more efficiently using the symmetries (it is a
  // symmetric matrix)? in this case, we should not rely on first computing the basis vectors 
  // because the matrix of basis vectors is not necessarily symmetric
  // maybe Eq 216,218 - but no - we actually need all M*N partial derivatives - the symmetry 
  // results from the commutativity of multiplication of scalars - the entries are products of
  // two partial derivatives

  Mat getContravariantMetric(const Vec& u) const
  {
    Mat E = getContravariantBasis(u);
    return E * E.getTranspose();  // (1), Eq. 214
  }
  // when x2u is not assigned (empty): use as fallback: getCovariantMetric().getInverse() 


  /** Computes the mixed dot product of a covariant covector and contravariant vector or vice 
  versa. No metric is involved, so you don't need to give a position vector u - the mixed metric 
  tensor is always the identity matrix. */
  static T getMixedDotProduct(const Vec& a, const Vec& b)
  {
    rsAssert(a.size() == b.size());
    return rsArrayTools::sumOfProducts(&a[0], &b[0], (int) a.size()); // (1), Eq 259,260
  }

  /** Computes the dot-product between two vectors (or covectors) with a given metric (which should
  be in contravariant form in the case of two vectors and in covariant form in the case of two 
  covectors...or the other way around?...figure out!) */
  static T getDotProduct(const Vec& a, const Vec& b, const Mat& g)
  {
    int N = (int)a.size();
    rsAssert(a.size() == b.size());
    //rsAssert(g.hasShape(N, N));
    T sum = T(0);
    for(int i = 0; i < N; i++)
      for(int j = 0; j < N; j++)
        sum += g(i, j) * a[i] * b[j];   // (1), Eq 257,258
    return sum;
  }

  /** Computes the dot-product of two contravariant vectors (i.e. regular column-vectors) emanating 
  from position vector u (using the covariant metric at u). */
  T getContravariantDotProduct(const Vec& a, const Vec& b, const Vec& u) const
  {
    Mat g = getContravariantMetric(u);    // g^ij
    return getDotProduct(a, b, g);
  }

  /** Computes the dot-product of two covariant vectors (aka row-vectors, covectors, 1-forms) 
  emanating from position vector u (using the contravariant metric at u). */
  T getCovariantDotProduct(const Vec& a, const Vec& b, const Vec& u) const
  {
    rsMatrix<T> g = getCovariantMetric(u);       // g_ij
    return getDotProduct(a, b, g);
  }

  // position vectors like u are always supposed to be given by contravariant components, right?
  // (1) pg 60: gradients of scalar fields are covariant, displacement vectors are contravariant
  // ...and a displacement should have the same variance as a position, i think - because it is
  // typically being added to a position

  /** Shifts (raises or lowers) the index of a vector with the given metric tensor. */
  Vec shiftVectorIndex(const Vec& a, const Vec& g) const
  {
    Vec r(N);         // result
    for(int i = 0; i < N; i++) {
      r[i] = T(0);
      for(int j = 0; j < N; j++)
        r[i] += a[j] * g(j, i); }     // (1), Eq. 228,229
  }
  // maybe make static - infer N from inputs


  /** Given a covariant covector (with lower index), this function returns the corresponding
  contravariant vector (with upper index). */
  Vec raiseVectorIndex(const Vec& a, const Vec& u) const
  {
    rsAssert((int) a.size() == N);
    Mat g = getContravariantMetric(u); // g^ij
    return shiftVectorIndex(a, g);
  }
  // needs test

  Vec lowerVectorIndex(const Vec& a, const Vec& u) const
  {
    rsAssert((int) a.size() == N);
    Mat g = getCovariantMetric(u);  // g_ij
    return shiftVectorIndex(a, g);
  }
  // needs test


  Mat raiseFirstMatrixIndex(const Mat& A, const Vec& u) const
  {
    //rsAssert(A.hasShape(N,N));  // should we allow NxM or MxN matrices?
    Mat g = getContravariantMetric(u); // g^ij

    // factor out into shiftFirstMatrixIndex:
    Mat B(N, N);
    B.setToZero();
    for(int i = 0; i < N; i++)
      for(int j = 0; j < N; j++)
        for(int k = 0; k < N; k++)
          B(i,j) += g(i,k) * A(k,j);  // (1), Eq 231,2
    return B;
  }
  // needs test

  Mat raiseSecondMatrixIndex(const Mat& A, const Vec& u) const
  {
    rsAssert(A.hasShape(N,N));  // should we allow NxM or MxN matrices?
    Mat g = getContravariantMetric(u); // g^ij

    // factor out into shiftFirstMatrixIndex:
    Mat B(N, N);
    B.setToZero();
    for(int i = 0; i < N; i++)
      for(int j = 0; j < N; j++)
        for(int k = 0; k < N; k++)
          B(j,i) += g(i,k) * A(j,k);  // (1), Eq 231,1
    return B;
  }
  // needs test

  // todo: implement also Eq 230,231 - raise one index in turn - starting from first


  /** Computes the Christoffel-symbols of the first kind [ij,l] and returns them as NxNxN 
  rsMultiArray. The (i,j,l)th element is accessed via C(i,j,l) when C is the multiarray of the
  Christoffel symbols. ...these are mainly used as an intermediate step to calculate the more 
  important Christoffel symbols of the second kind - see below...  */
  Tens getChristoffelSymbols1stKind(const Vec& u) const
  {
    int i, j, l;

    // Create an array of the partial derivatives of the metric with respect to the coordinates:
    Vec up(u), um(u);               // wiggled u coordinate vectors
    Mat gp, gm;                     // metric at up and um
    Mat g = getCovariantMetric(u);  // metric at u
    std::vector<Mat> dg(N);         // we need N matrices
    T s = 1/(2*h);
    for(i = 0; i < N; i++)
    {
      up[i] = u[i] + h;
      um[i] = u[i] - h;

      gp = getCovariantMetric(up);  // metric at up
      gm = getCovariantMetric(um);  // metric at um

      dg[i] = s * (gp - gm);        // central difference approximation

      up[i] = u[i];
      um[i] = u[i];
    }

    // Compute the Christoffel-symbols from the partial derivatives of the metric and collect them
    // into an NxNxN 3D array:
    Tens C({N,N,N}); 
    for(i = 0; i < N; i++)
      for(j = 0; j < N; j++)
        for(l = 0; l < N; l++)
          C(i,j,l) = T(0.5) * (dg[j](i,l) + dg[i](j,l) - dg[l](i,j)); // (1), Eq. 307
    return C;
  }
  // the notation G_kij is also used - 3 lowercase indices
  // https://en.wikipedia.org/wiki/List_of_formulas_in_Riemannian_geometry

  /** Computes the Christoffel symbol of the second kind G^k_ij at given position u. G for the  
  commonly used uppercase gamma with the 1st index upstairs, 2nd and 3rd downstairs. It's not 
  actually a tensor, but it's commonly denoted in this way - so the up/down notation does not 
  actually mean contra- or covariant in this case. The element G(k,i,j) has the following 
  meaning: In an n-dimensional space with n covariant basis vectors E_1,...,E_n which span the 
  space, the partial derivative E_i,j of the i-th basis vector with respect to the j-th coordinate
  is itself a vector living the same space, so it can be represented as a linear combination of the
  original basis vectors E_1,...,E_n. The coefficients in this linear combination are the 
  Christoffel symbols of the 2nd kind. They have 3 indices: i,j to indicate which basis vector is 
  differentiated and with respect to which coordinate (denoted downstairs) and k to denote, which 
  basis-vector it multiplies in the linear combination (denoted upstairs). The utility of the 
  Christoffel symbols is that they facilitate the calculation of the covariant derivative without
  resorting to any reference to the basis vectors of the tangent space at u in the embedding space. 
  By some magic cancellation of terms, it happens that they may be calculated using derivatives of 
  the metric tensor alone, which is an intrinsic property of the manifold, as opposed to the 
  tangent-space basis vectors which are extrinsic to the manifold. They are symmetric in their 
  lower indices, i.e. G^k_ij = G^k_ji because each of the lower indices refers to one partial 
  differentiation of an input position vector with respect to one of the original coordinates 
  (the basis-vectors are first derivatives of the coordinates and here we take (some symmetric 
  combination of) derivatives of the basis-vectors, so we get second derivatives) and the order of 
  partial derivatives is irrelevant (by Schwarz's theorem). */
  Tens getChristoffelSymbols2ndKind(const Vec& u) const
  {
    int k, i, j;
    // k: derivative index
    // i: basis vector index
    // j: vector component index
    // ...verify these....

    Mat g = getContravariantMetric(u);  // contravariant metric at u
    Tens C = getChristoffelSymbols1stKind(u);
    Tens G({N,N,N});
    G.setToZero();
    for(i = 0; i < N; i++)
      for(j = 0; j < N; j++)
        for(k = 0; k < N; k++)
        {
          for(int l = 0; l < N; l++)
          {
            G(k,i,j) += g(k,l) * C(i,j,l);  // (1), Eq. 308
            // or should we swap i,k - because the k in the formula is notated upstairs - it feels
            // more natural, if it's the first index - done - here, the upper index is denoted 
            // first:
            // https://en.wikipedia.org/wiki/Christoffel_symbols#Christoffel_symbols_of_the_first_kind
          }
        }
    return G;
  }
  // this is not a good algorithm - it involves the contravariant metric and has a quadruple-loop
  // ...is there a more direct way? probably yes! at least, it seems to give correct results

  /** Computes the matrix D of partial derivatives of the given vector field v = f(u) with respect 
  to the coordinates. D(i,j) is the j-th component of the partial derivative with respect to 
  coordinate i. */
  Mat getVectorFieldDerivative(const Vec& u, const FuncVecToVec& f) const
  {
    int i, j;
    // i: index of coordinate with respect to which partial derivative is taken
    // j: index of vector component in vector field

    // compute matrix of partial derivatives of the vector field f(u):
    Vec up(u), um(u);        // wiggled u coordinate vectors
    Vec Ap(N), Am(N);        // A = f(u) at up and um
    Mat D(N, N);
    T s = 1/(2*h);
    for(i = 0; i < N; i++)
    {
      up[i] = u[i] + h;
      um[i] = u[i] - h;

      // Compute vector fields at wiggled coordinates:
      f(up, Ap);
      f(um, Am);

      // Approximate partial derivatives by central difference and store them into matrix:
      for(j = 0; j < N; j++)
        D(i, j) = s * (Ap[j] - Am[j]);

      up[i] = u[i];
      um[i] = u[i];
    }

    return D;
  }
  // could actually be a static function - oh - no - it uses our member h




  /** Given two vector fields f = f(u) and g = g(u), this function computes the Lie bracket of 
  these vector fields at some given position vector u. The Lie bracket of f and g is just 
  f(g) - g(f), i.e. the commutator of the two vector fields. */
  Vec getLieBracket(const Vec& u, const FuncVecToVec& f, const FuncVecToVec& g) const
  {
    return f(g(u)) - g(f(u));
  }
  // verify this


  // we may also need a covariant derivative along a given direction - is this analoguous to the
  // directional derivative, as in: the scalar product of the gradient with a given vector? should
  // we form the scalar product of the covariant derivative with a given vector? ...figure out!

  // getTorsionTensor(const Vec& u, const FuncVecToVec& f, const FuncVecToVec& g)
  // torsion tensor of two vector fields f and g at u, see:
  // https://www.youtube.com/watch?v=SfOiOPuS2_U&list=PLJHszsWbB6hpk5h8lSfBkVrpjsqvUGTCx&index=24
  // ..implement both formulas: original and simplified and compare - oh  - wait - the torsion 
  // tensor does not actually depend on the vector field, only on the connection (...which is the
  // way in which we define the Christoffel symbols - right?)
  // getLieBracket


  /** Computes the covariant derivative of a vector field using Christoffel symbols. 
  Todo: make the compuation based on derivatives of basis vectors - this avoids computing the 
  contravariant metric tensor and seems generally simpler...  **/
  Mat getCovariantDerivative(const Vec& u, const FuncVecToVec& f) const
  {
    rsAssert((int)u.size() == N, "Input vector u has wrong dimensionality");

    // Algorithm: 
    //  (1) compute contravariant output vector A at u
    //  (2) compute matrix dA of partial derivatives of the vector field
    //  (3) compute rank-3 tensor of Christoffel symbols of 2nd kind at u
    //  (4) combine them via Ref.(1), Eq. 362

    Vec  A(N); f(u, A);                        // (1)
    Mat  dA = getVectorFieldDerivative(u, f);  // (2)
    Tens C  = getChristoffelSymbols2ndKind(u); // (3)

    // (4) - apply Ref.(1), Eq. 362:
    Mat D(N, N);
    D.setToZero();
    int i, j, k;
    // i: index of covariant derivative (cov. der. with respect to i-th coordinate)
    // j: index of vector component (j-th component of cov. der.)
    // k: summation index, summation is over basis-vectors (right?)
    for(i = 0; i < N; i++) {              // The covariant derivative is...
      for(j = 0; j < N; j++) {
        D(i, j) = dA(i, j);               // ...the regular partial derivative...
        for(k = 0; k < N; k++) {
          D(i, j) += C(j, k, i) * A[k];   // ...plus a sum of Christoffel-symbol terms.
        }
      }
    }
    return D;
  }




  // The code below is under construction and has not yet been tested:




  /** Under construction... totally unverified

  Absolute differentiation of a vector field with respect to a parameter t, given a t-parametrized
  curve C(t). It's evaluated at a given parametr value t, which in turn determines the position u 
  where we are in our space...
  
  ...so, is this the change of the vector field when we move a tiny bit along the curve?
  */
  Vec getAbsoluteDerivative(T t, const FuncVecToVec& A, const FuncSclToVec& C) const
  {
    // Compute position vector u at given parameter t:
    Vec u(N); C(t, u); 

    // Compute tangent to curve at u:
    Vec up(N), um(N); 
    C(t+h, up); C(t-h, um);
    Vec ut = (1/(2*h)) * (up - um);  // ut is tangent vector

    // Compute covariant derivative of vector field A at u:
    Mat dA = getCovariantDerivative(u, A);

    // Compute inner product of covariant derivative dA with tangent ut - this is defined as the 
    // absolute derivative:
    Vec r(N);
    for(int i = 0; i < N; i++)
    {
      r[i] = 0;
      for(int j = 0; j < N; j++)
        r[i] += A(i,j) * ut(j);   // verify this!
    }

    return r;
  }




  /** Under construction... not yet tested */
  Tens getRiemannTensor1stKind(const Vec& u) const
  {
    int i, j, k, l, r;
    Vec up(u), um(u);         // wiggled u coordinate vectors
    Tens cp, cm;              // Christoffel symbols at up and um

    // derivatives of Christoffel symbols of 1st kind - maybe factor out - and maybe instead of 
    // using a vector of tensors, use a rank-4 tensor - maybe we need a facility to insert a 
    // lower-rank tensor at a given (multi) index in a higher rank tensor - like 
    // chrisDeriv.insert(currentChristoffelDerivative, i); ..we want to avoid memory allocations
    std::vector<Tens> dc(N);  
    T s = 1/(2*h);
    for(i = 0; i < N; i++)
    {
      up[i] = u[i] + h;
      um[i] = u[i] - h;

      cp = getChristoffelSymbols1stKind(up);
      cm = getChristoffelSymbols1stKind(um);

      dc[i] = s * (cp - cm);             // central difference approximation
      // rsMultiArray needs to define the operators (subtraction and scalar multiplication)

      up[i] = u[i];
      um[i] = u[i];
    }


    Tens c1 = getChristoffelSymbols1stKind(u);
    Tens c2 = getChristoffelSymbols2ndKind(u);
    Tens R({N,N,N,N});
    //R.setToZero();
    for(i = 0; i < N; i++)
      for(j = 0; j < N; j++)
        for(k = 0; k < N; k++)
          for(l = 0; l < N; l++)
          {
            // (1), Eq. 558:
            R(i,j,k,l) = dc[k](j,l,i) - dc[l](j,k,i);
            for(r = 0; r < N; r++)
              R(i,j,k,l) += c1(i,l,r)*c2(r,j,k) - c1(i,k,r)*c2(r,j,l);
          }
    return R;
  }

  // 2nd kind: use Eq 560....

  Tens getRiemannTensor2ndKind(const Vec& u) const
  {
    int i, j, k, l, r;

    // compute Christoffel symbols of 2nd kind:
    Tens c = getChristoffelSymbols2ndKind(u);

    // compute derivatives of Christoffel symbols of 2nd kind
    std::vector<Tens> dc(N);  
    Vec up(u), um(u);         // wiggled u coordinate vectors
    Tens cp, cm;              // Christoffel symbols at up and um
    T s = 1/(2*h);
    for(i = 0; i < N; i++)
    {
      up[i] = u[i] + h;
      um[i] = u[i] - h;

      cp = getChristoffelSymbols2ndKind(up);
      cm = getChristoffelSymbols2ndKind(um);
      // = assignment operator is not yet implemented...

      dc[i] = s * (cp - cm);             // central difference approximation
      // rsMultiArray needs to define the operators (subtraction and scalar multiplication)

      up[i] = u[i];
      um[i] = u[i];
    }

    //double test = dc[1](0,0,0);

    // compute Riemann-Christoffel curvature tensor via (1) Eq. 560:
    Tens R({N,N,N,N});
    //R.setToZero();
    for(i = 0; i < N; i++)
      for(j = 0; j < N; j++)
        for(k = 0; k < N; k++)
          for(l = 0; l < N; l++)
          {
            // (1), Eq. 560:
            //double test1 = dc[k](i,j,l);
            //double test2 = dc[l](i,j,k);

            R(i,j,k,l) = dc[k](i,j,l) - dc[l](i,j,k);
            for(r = 0; r < N; r++)
              R(i,j,k,l) += c(r,j,l)*c(i,r,k) - c(r,j,k)*c(i,r,l);
            //int dummy = 0;
          }
    return R;
  }

  // todo: verify Bianchi identities in test code

  /** The Ricci Curvature tensor of the first kind is obtained by contracting over the first and 
  last index of the Riemann-Christoffel curvature tensor. */
  Mat getRicciTensor1stKind(const Vec& u) const
  {
    Mat  r(N, N); r.setToZero();                     // Ricci tensor (rank 2)
    Tens R = getRiemannTensor2ndKind(u);  // has rank 4, type (1,3)
    for(int i = 0; i < N; i++)
      for(int j = 0; j < N; j++)
        for(int a = 0; a < N; a++)
          r(i,j) += R(a,i,j,a);  // (1), Eq. 589
          // todo: implement a general contraction function in rsMultiArray or rsTensor..maybe
    return r;
  }

  /** The Ricci curvature tensor of the second kind is obtained by raising the first index of the
  Ricci curvature tensor of the first kind. */
  Mat getRicciTensor2ndKind(const Vec& u) const
  {
    Mat R = getRicciTensor1stKind(u);
    return raiseFirstMatrixIndex(R, u);  // (1), Eq 593
    //rsError("Not yet complete");
    //return R;
  }
  // needs test

  /** The Ricci scalar is obtained by contracting over the two indices of the Ricci tensor of the 
  second kind, i.e. by taking the trace of the matrix. */
  T getRicciScalar(const Vec& u) const
  {
    Mat R = getRicciTensor2ndKind(u);
    return R.getTrace();  // todo
  }

  // the Ricci tensor can also be computed from the Christoffel-symbols (aufstieg zu den 
  // einstein-gleichungen, eq 20.11)...but there are also derivatives of the Christoffel symbols
  // involved, so the advantage is questionable


  //Mat getEinsteinTensor(const Vec& u) const { }
  //Tens getWeylTensor(const Vec& u) const { }

  // Einstein tensor, Weyl tensor (and more):
  // https://en.wikipedia.org/wiki/List_of_formulas_in_Riemannian_geometry

  // https://en.wikipedia.org/wiki/Riemann_curvature_tensor
  // https://en.wikipedia.org/wiki/Ricci_decomposition


  // todo: scalar invariants (pg 157), infinitesimal strain (Eq. 597) - formula seems equivalent to
  // rate of strain (Eq 609) - it's the symmetric component of the gradient of a vector field, 
  // stress (Eq. 599), vorticity (Eq. 612 - anti-symmetric part of gradient)...is the "gradient" 
  // here to be understood as covariant derivative? ...probably...




  /** For two vectors given as 1xN or Nx1 matrices, this function returns whether the two vectors
  are of different variance type, i.e one covariant and the other contravariant (the variance type
  of a vector is identified by it being a row-vector (covariant) or column-vector (contravariant) ) */
  /*
  bool haveVectorsSameVariance(const rsMatrix<T>& a, const rsMatrix<T>& b)
  {
    rsAssert(a.isVector() && b.isVector());
    return (a.isRowVector() && b.isColumnVector) 
      ||   (b.isRowVector() && a.isColumnVector);
  }
  */


  /*
  T dotProduct(const rsMatrix<T>& a, const rsMatrix<T>& b)
  {
    // if one of a,b is covariant and the other is contravariant, we can do without the metric 
    // tensor

    if(haveVectorsSameVariance)
    {
      // We need the metric tensor in one or the other form
      if(a.isRowVector())
      {
        // both are row-vectors (covectors), so we need the contravariant version of the metric


      }
      else  // b is the row-vector
      {
        // both are regular column-vectors, so we need the covariant version of the metric

      }

    }
    else
    {
      // We don't need the metric tensor and can directly form the sum of the products

    }


  }
  // todo: provide implementation that takes the metric as parameter so we don't have to recompute
  // it each time
  */

  // todo: lowerVectorIndex, raiseCovectorIndex (Eq 228,229)
  // raiseIndex, lowerIndex (should apply to arbitrary tensors, Eq 230,231)

  // todo: getContravariantBasis - maybe this should be done using the covariant basis and the 
  // metric? or directly using the definition (Eq. 45 or 85) as gradients of the coordinate functions? or both and
  // then be compared? also, should the contravariant basis vectors be stored as columns of the 
  // matrix? According to Eq. 45, i think, we need to define two scalar functions u1(x,y,z) and
  // u2(x,y,z) and the first contravariant basis vector is the gradient of u1 and the 2nd the 
  // gradient of u2

  // getScaleFactors Eq. 53 69

  // getJacobian

  /*
  void computeMetric(const std::vector<T>& u, rsMatrix<T>& g)
  {
    rsAssert((int)u.size() == N);
    //rsAssert(g.hasSameShapeAs(N, N));

    // todo: convert u to basis vectors - for spherical coordinates (longitude, latitute) we would 
    // get 2 basis-vectors in 3D space - the metric tensor would be the matrix product of the 
    // matrix of these two basis vectors with the same matrix transposed
  }
  // computes covariant metric tensor at given coordinate position u = (u1, u2,..., uN)
  //
  */



  //-----------------------------------------------------------------------------------------------
  // \name Tests. After configuring the object with the various conversion functions, you can run 
  // a couple of sanity checks in order to catch errors in the client code, like passing wrong
  // or numerically too imprecise formulas.

  /** Computes the error in the roundtrip converting from curvilinear coordinates to cartesian 
  and back for the given vector u. Your functions passed to setCurvilinearToCartesian and 
  setCartesianToCurvilinear should be inverses of each other - if they in fact are not, you may 
  catch your mistake with the help of this function. */
  Vec getRoundTripError(const Vec& u) const
  {
    rsAssert((int)u.size() == N);
    Vec x(M), u2(N);
    toCartesian(  u, x);
    toCurvilinear(x, u2);
    return u2 - u;
  }

  /** Computes the matrix that results from matrix-multiplying the matrix of contravariant basis 
  vectors with the matrix of covariant basis vectors. If all is well, this should result in an 
  NxN identity matrix (aka Kronecker delta tensor) - if it doesn't, something is wrong. */
  Mat getNumericalBasisProduct(const Vec& u)
  {
    Mat cov = getCovariantBasisNumerically(u);
    Mat con = getContravariantBasisNumerically(u);
    return con * cov;
  }

  /** Same as getNumericalBasisProduct but using the analytical formulas instead (supposing that
  the user has passed some - otherwise this test doesn't make sense). */
  Mat getAnalyticalBasisProduct(const Vec& u)
  {
    rsAssert(u2xJ != nullptr, "Test makes only sense when Jacbian was assigned via setCurvToCartJacobian");
    rsAssert(x2uJ != nullptr, "Test makes only sense when Jacbian was assigned via setCartToCurvJacobian");
    Mat cov = getCovariantBasis(u);
    Mat con = getContravariantBasis(u);
    return con * cov;
  }

  /** Returns the difference between computing the Jacobian matrix (which gives the covariant basis 
  vectors) analytically and numerically, where "analytically" refers to using the function that was
  passed to setCurvToCartJacobian (if this user-defined function actually uses analytical formulas 
  or does itself something numerical internally is of course unknown to this object - but using 
  analytical formulas is the main intention of this facility).  */
  Mat getForwardJacobianError(const Vec& u)
  {
    rsAssert(u2xJ != nullptr, "Test makes only sense when Jacbian was assigned via setCurvToCartJacobian");
    Mat ana = getCovariantBasis(u);            // (supposedly) analytic formula
    Mat num = getCovariantBasisNumerically(u); // numerical approximation
    return ana - num;
  }

  /** @see getForwardJacobianError - this is the same for the inverse transformation, i.e. u(x). */
  Mat getInverseJacobianError(const Vec& u)
  {
    rsAssert(x2uJ != nullptr, "Test makes only sense when Jacbian was assigned via setCartToCurvJacobian");
    Mat ana = getContravariantBasis(u); 
    Mat num = getContravariantBasisNumerically(u);
    return ana - num;
  }



  /** Runs all tests that make sense for the given configuration using the given tolerance and 
  input vector ... */
  bool runTests(const Vec& u, T tol)
  {
    bool result = true;

    T errMax;    // maximum error
    Vec errVec;  // error vector
    Mat errMat;  // error matrix

    // roundtrip of coordinates:
    if( x2u ) { // might be not assigned
      errVec  = getRoundTripError(u);
      errMax  = rsArrayTools::maxAbs(&errVec[0], (int) errVec.size());
      result &= errMax <= tol; }

    // product of numerically computed co- and contravariant bases:
    Mat id = Mat::identity(N);
    Mat M  = getNumericalBasisProduct(u);
    errMat = M - id;
    result &= errMat.getAbsoluteMaximum() <= tol;

    // product of analytically computed co- and contravariant bases:
    if( x2uJ && u2xJ ) {
      M = getAnalyticalBasisProduct(u);
      errMat = M - id;
      result &= errMat.getAbsoluteMaximum() <= tol; }

    // difference between numerically and analytically computed covariant bases:
    if(u2xJ) {
      M = getForwardJacobianError(u);
      result &= errMat.getAbsoluteMaximum() <= tol; }

    // difference between numerically and analytically computed contravariant bases:
    if(x2uJ) {
      M = getInverseJacobianError(u);
      result &= errMat.getAbsoluteMaximum() <= tol; }

    // ...what else? ...later maybe Christoffel symbols

    return result;
  }
  // should check, if the provided formulas make sense - like x2u being inverse of u2x, the 
  // Jacobians are inverses of each other, etc. - maybe compare analytic Jacobians to numercial 
  // ones etc. - helps to catch user error - it's easy to pass an inconsistent set of equations
  // provide
  



protected:


  /** Checks, if u and x have the right dimensions. */
  void checkDims(const Vec& u, const Vec& x) const
  {
    rsAssert((int)u.size() == N);
    rsAssert((int)x.size() == M);
  }


  int N;   // dimensionality of the manifold           (see (1), pg 46 for the conventions)
  int M;   // dimensionality of the embedding space
 

  T h = 1.e-8;  
  // approximation stepsize for numerical derivatives (maybe find better default value or justify 
  // this one - maybe the default value should be the best choice, when typical coordinate values 
  // are of the order of 1)

  /** Conversions from general (contravariant) coordinates u^i to Cartesian coordinates x_i and 
  vice versa. The x_i are denoted with subscript because the embedding space is supposed to be 
  Euclidean, so we don't need a distinction between contravariant and covariant coordinates and we 
  use subscript by default. */
  FuncVecToVec u2x, x2u;
  // u2x must produce M extrinsic coordinates from N intrinsic coordinates:
  // u2x: x1(u1, u2, ..., uN)
  //      x2(u1, u2, ..., uN)
  //      ...
  //      xM(u1, u2, ..., uN)
  //
  // x2u: u1(x1, x2, ..., xM)
  //      u2(x1, x2, ..., xM)
  //      ...
  //      uN(x1, x2, ..., xM)
  // 
  // with x2u, we must be careful to only insert valid coordinates, i.e. coordinates that actually 
  // live on the manifold, otherwise the outputs may be meaningless.

  // functions for analytic Jacobians:
  FuncVecToMat u2xJ, x2uJ;

};
// todo: make it possible that the input and output dimensionalities are different - for example, 
// to specify points on the surface of a sphere, we would have two curvilinear input coordinates 
// and 3 cartesian output coordinates
// maybe have N and M as dimensionalities of the space/manifold under cosideration (e.g. the 
// sphere) and the embedding space (e.g. R^3, the 3D Euclidean space)


// formluas from:
// https://en.wikipedia.org/wiki/Spherical_coordinate_system#Coordinate_system_conversions

template<class T>
void cartesianToSpherical(T x, T y, T z, T* r, T* theta, T* phi)
{
  *r     = sqrt(x*x + y*y + z*z);
  *phi   = atan2(y, x);
  *theta = acos(z / *r);
  //*phi   = acos(x / sqrt(x*x + y*y));        // ?= atan2(y,x)
  //*theta = acos(z / sqrt(x*x + y*y + z*z));  // == acos(z / *r)
}
// see Mathematical Physics Eq. 3.12
// https://en.wikipedia.org/wiki/Spherical_coordinate_system

template<class T>
void sphericalToCartesian(T r, T theta, T phi, T* x, T* y, T* z)
{
  *x = r * sin(theta) * cos(phi);
  *y = r * sin(theta) * sin(phi);
  *z = r * cos(theta);
}
// see Mathematical Physics Eq. 3.11
// roundtrip doesn't work

// see also:
// https://en.wikipedia.org/wiki/Hyperbolic_coordinates
// https://en.wikipedia.org/wiki/Elliptic_coordinate_system
// https://en.wikipedia.org/wiki/Toroidal_coordinates
// https://en.wikipedia.org/wiki/Curvilinear_coordinates

// maybe try triangular coordinates (in 3D):
// -select 3 points in 3-space: p1 = (x1,y1,z1), p2 = ..., p3
// -the general (u1,u2,u3) coordinates of a point p = (x,y,z) with respect to these points are 
//  defined as the distances of p to these 3 points:
//    u1^2 = (x-x1)^2 + (y-y1)^2 + (z-z1)^2
//    u2^2 = (x-x2)^2 + (y-y2)^2 + (z-z2)^2
//    u3^2 = (x-x3)^2 + (y-y3)^2 + (z-z3)^2
// -we may encode on which side of the plane spanned by p1,p2,p3 a point p is by applying a sign to
//  all the u-coordinates (uniformly, to treat all u values on same footing)
// -for different choices of p1,p2,p3, we obtain an infinite familiy of coordinate systems to play 
//  with
// -there are some conditions for which combinations of u-values are valid - it can't be that a 
//  point is very close to p1 and at the same time very far from p2 when p1 and p2 are somewhat 
//  close together (-> figure out details) - the bottom line is that we have to take care assigning
//  u-coordinates to make sure, the specify a valid point

// Set operations on std::vector (not yet tested):

/** Returns a set C consisting of elements that are in A and in B. */
template<class T>
std::vector<T> rsSetIntersection(const std::vector<T>& A, const std::vector<T>& B)
{
  std::vector<T> C;
  for(size_t i = 0; i < A.size(); i++)
    if(rsArrayTools::contains(&B[0], (int) B.size(), A[i]))
      C.push_back(A[i]);
  return C;
}

/** Returns a set C consisting of elements that are in A but not in B. */
template<class T>
std::vector<T> rsSetDifference(const std::vector<T>& A, const std::vector<T>& B)
{
  std::vector<T> C;
  for(size_t i = 0; i < A.size(); i++)
    if(!rsArrayTools::contains(&B[0], (int) B.size(), A[i]))
      C.push_back(A[i]);
  return C;
}

/** Returns a set C consisting of elements that are in A or in B. */
template<class T>
std::vector<T> rsSetUnion(const std::vector<T>& A, const std::vector<T>& B)
{
  std::vector<T> C = A;
  for(size_t i = 0; i < B.size(); i++)
    if(!rsArrayTools::contains(&C[0], (int) C.size(), B[i]))
      C.push_back(B[i]);
  return C;
}

// -maybe make a class rsSet with operators + for union, - for difference, * for intersection.
// -maybe keep the elements sorted - that reduces the complexity of "contains" from N to log(N)
//  -union would be some sort of merge of sorted arrays (similar as in merge sort?..but avoiding
//   duplicates)
//  -but that requires an element type that defines an order...maybe we should have both: sorted 
//   and unsorted sets

/** Removes duplicate elements from a vector A and returns the result - that's useful for turning 
arbitrary vectors into sets (which contain each element just once). */
template<class T>
std::vector<T> rsRemoveDuplicates(const std::vector<T>& A)
{
  std::vector<T> B;
  for(size_t i = 0; i < A.size(); i++)
    if(!rsArrayTools::contains(&B[0], (int) B.size(), A[i]))
      B.push_back(A[i]);
  return B;
}



// code adapted from: https://www.geeksforgeeks.org/print-subsets-given-size-set/

template<class T> // todo: rename
void rsSubsetUtil(const std::vector<T>& set, int r, int index, std::vector<T>& tmp, int i, 
  std::vector<T>& result)
{
  // Current subset is ready, append it to the result and return:
  if (index == r) { rsAppend(result, tmp); return; }

  // Check, if no more elements are there to put in tmp - if so, return:
  int n = (int) set.size();
  if(i >= n) return;

  // Current is included, put next at next location:
  tmp[index] = set[i];
  rsSubsetUtil(set, r, index + 1, tmp, i + 1, result);

  // current is excluded, replace it with next (Note that i+1 is passed, but index is not
  // changed):
  rsSubsetUtil(set, r, index, tmp, i + 1, result);

  int dummy = 0;
}
template<class T>
std::vector<T> rsSubsetsOfSize(const std::vector<T>& set, int size)
{
  std::vector<T> result;     // todo: reserve n-choose-k, n = set.size(), k = size
  std::vector<T> tmp(size);
  rsSubsetUtil(set, size, 0, tmp, 0, result);
  return result;
}





/** A class for representing sets. To optimize operations, they are kept sorted internally which 
means the data type T must support "less-than" comparisons. */

template<class T>
class rsSortedSet
{

public:


  rsSortedSet() {}

  rsSortedSet(const std::vector<T>& setData) : data(setData)
  { rsAssert(isValid(data)); }

  /** Returns true, iff the given vector is a valid representation of a sorted set. For this, it 
  must be ascendingly sorted and each element may occur only once. */
  static bool isValid(const std::vector<T>& A)
  {
    if(A.empty()) return true;  // the empty set is a valid set
    using AT = rsArrayTools;
    bool sorted = AT::isSortedAscending(&A[0], (int) A.size()); // use isSortedStrictlyAscending to disallow duplicates
    return sorted;
  }

  /** An element is in the intersection set of A and B if it is in A or in B. */
  static std::vector<T> unionSet(const std::vector<T>& A, const std::vector<T>& B)
  {
    size_t Na = A.size(), Nb = B.size();
    size_t ia = 0, ib = 0; // indices into A and B, maybe use i,j
    std::vector<T> C;
    C.reserve(Na+Nb);
    while(ia < Na && ib < Nb) {
      if(     B[ib] < A[ia]) { C.push_back(B[ib]); ib++;       }   // A[ia] >  B[ib]
      else if(A[ia] < B[ib]) { C.push_back(A[ia]); ia++;       }   // A[ia] <  B[ib]
      else                   { C.push_back(A[ia]); ia++; ib++; }}  // A[ia] == B[ib]
    while(ia < A.size()) { C.push_back(A[ia]); ia++; }
    while(ib < B.size()) { C.push_back(B[ib]); ib++; }
    return C;
  }
  // maybe rename to unionSet, etc. to avoid confusion with setters

  /** An element is in the intersection set of A and B if it is in A and in B. */
  static std::vector<T> intersectionSet(const std::vector<T>& A, const std::vector<T>& B)
  {
    size_t Na = A.size(), Nb = B.size();
    size_t ia = 0, ib = 0;
    std::vector<T> C;
    C.reserve(rsMin(Na, Nb));
    while(ia < Na && ib < Nb) {
      while(ia < Na && ib < Nb && A[ia] <  B[ib])   ia++;   // is ib < B.size() needed?
      while(ia < Na && ib < Nb && B[ib] <  A[ia])   ib++;   // is ia < A.size() needed?
      while(ia < Na && ib < Nb && B[ib] == A[ia]) { C.push_back(A[ia]); ia++; ib++; }}
    return C;
  }

  /** An element is in the difference set A "without" B if it is in A but not in B. */
  static std::vector<T> differenceSet(const std::vector<T>& A, const std::vector<T>& B)
  {
    size_t Na = A.size(), Nb = B.size();
    size_t ia = 0, ib = 0;
    std::vector<T> C;
    C.reserve(Na);
    while(ia < Na && ib < B.size()) {
      while(ia < Na && ib < Nb && A[ia] <  B[ib]) { C.push_back(A[ia]); ia++; }  // is ib < Nb needed?
      while(ia < Na && ib < Nb && A[ia] == B[ib]) { ia++; ib++;               }
      while(ia < Na && ib < Nb && B[ib] <  A[ia]) { ib++;                     }} // is ia < Na needed?
    while(ia < Na) { C.push_back(A[ia]); ia++; } 
    return C;
  }
  // while(ia < Na && ib < Na)
  //   add all elements from A that are less than our current element in B
  //   skip all elements in A and B that are equal
  //   skip all elements in B that are less than our current element in A
  // endwhile

  /** An element is in the intersection set of A and B if it is in A or in B but not in both, so 
  it's like the union but with the exclusive instead of the inclusive or. The symmetric difference
  is the union minus the intersection. */
  static std::vector<T> symmetricDifferenceSet(const std::vector<T>& A, const std::vector<T>& B)
  {
    size_t Na = A.size(), Nb = B.size();
    size_t ia = 0, ib = 0;
    std::vector<T> C;
    C.reserve(Na+Nb);
    while(ia < Na && ib < Nb) {
      while(ia < Na && ib < Nb && A[ia] <  B[ib]) { C.push_back(A[ia]); ia++; }   // is ib < B.size() needed?
      while(ia < Na && ib < Nb && B[ib] <  A[ia]) { C.push_back(B[ib]); ib++; }   // is ia < A.size() needed?
      while(ia < Na && ib < Nb && B[ib] == A[ia]) { ia++; ib++;               }}
    while(ia < A.size()) { C.push_back(A[ia]); ia++; }
    while(ib < B.size()) { C.push_back(B[ib]); ib++; }
    return C;
  }
  // needs more tests

  static std::vector<std::pair<T,T>> cartesianProduct(
    const std::vector<T>& A, const std::vector<T>& B)
  {
    size_t Na = A.size(), Nb = B.size();
    std::vector<std::pair<T,T>> C(Na * Nb);
    for(size_t ia = 0; ia < Na; ia++)
      for(size_t ib = 0; ib < Nb; ib++)
        C[ia*Nb+ib] = std::pair(A[ia], B[ib]);
    return C;
  }
  // -maybe allow different types for the elements of A and B
  // -maybe it should return a set

  // todo: implement int find(const T& x) - searches for element x via binary search and returns 
  // index, bool contains(const T& x), bool insert(const T& x), bool remove(const T& x) - return
  // value informs, if something was done

  // implement symmetric difference:
  // https://en.wikipedia.org/wiki/Set_(mathematics)#Basic_operations
  // https://en.wikipedia.org/wiki/Symmetric_difference
  // i think, it can be done with the same algo as for the intersection, just that we push in 
  // branches 1,2 and skip in branch 3, see also
  // https://www.geeksforgeeks.org/set-operations/

  // -implement function areDisjoint(A, B) ..or maybe as A.isDisjointTo(B),
  //  also: A.isSubsetOf(B), A.isSupersetOf(B) 
  // -maybe use < and <= operators where < should denote a strict subset. when A is a subset of B 
  //  and B has larger size than A, then A is a strict subset of B
  // -maybe also implement power set - but that scales with 2^N and so it gets impractical very 
  //  quickly. 
  // -maybe that could use a getAllSubsetsOfSize(int) function which may also be useful by itself
  // -maybe < should not be used for subsets because we may need some other notion of < that 
  //  satisfies the tritochomy condition, because if we want to make sets of sets with the 
  //  *sorted* set data structure, we need the comparison operator to work in a way that lest us 
  //  uniquely order the sets https://de.wikipedia.org/wiki/Trichotomie - on the other hand, we 
  //  could use a user-provided less-function for the comparisons, similar to rsBinaryHeap - that 
  //  would free the < operator again...but no - i think, it would be confusing to have a < 
  //  operator that doesn't satisfy trichotomy
  // -maybe the operators should be implemented as functions A.intersectWith(B), etc.

  // maybe implement relations as subsets of the cartesian product - then we may inquire if a 
  // particular tuple of elements is in a given relation - this can also be determined quickly by 
  // two successive binary searches...maybe we can do interesting computaions with relations, too


  const std::vector<T>& getData() const { return data; }

  /** Addition operator implements set union. */
  rsSortedSet<T> operator+(const rsSortedSet<T>& B) const
  { return rsSortedSet<T>(unionSet(this->data, B.data)); }

  /** Subtraction operator implements set difference. */
  rsSortedSet<T> operator-(const rsSortedSet<T>& B) const
  { return rsSortedSet<T>(differenceSet(this->data, B.data)); }

  /** Multiplication operator implements set intersection. */
  rsSortedSet<T> operator*(const rsSortedSet<T>& B) const
  { return rsSortedSet<T>(intersectionSet(this->data, B.data)); }

  /** Division operator implements set symmetric difference. */
  rsSortedSet<T> operator/(const rsSortedSet<T>& B) const
  { return rsSortedSet<T>(symmetricDifferenceSet(this->data, B.data)); }


  bool operator==(const rsSortedSet<T>& B) const
  { return this->data == B.data; }


protected:

  std::vector<T> data;

};
// goal: all basic set operations like union, intersection, difference, etc. should be O(N), 
// finding an element O(log(N))
// how would we deal with sets of complex numbers? they do not have a < operation. should we 
// implement a subclass that provides such an operator, like rsComplexOrderedReIm

// see also:
// https://www.geeksforgeeks.org/print-all-possible-combinations-of-r-elements-in-a-given-array-of-size-n/
// https://www.geeksforgeeks.org/heaps-algorithm-for-generating-permutations/
// https://en.cppreference.com/w/cpp/algorithm/next_permutation#Possible_implementation
// https://en.wikipedia.org/wiki/Heap%27s_algorithm#:~:text=Heap's%20algorithm
// https://en.wikipedia.org/wiki/Steinhaus%E2%80%93Johnson%E2%80%93Trotter_algorithm

// https://www.jmlr.org/papers/volume18/17-468/17-468.pdf
// Automatic Differentiation in Machine Learning: a Survey

// https://github.com/coin-or/ADOL-C

//=================================================================================================

/** Implements a datatype suitable for automatic differentiation or AD for short. In AD, a number 
is seen as the result of a function evaluation and in addition to the actual output value of the 
function, the value of the derivative is also computed and carried along through all subsequent 
operations and function applications. Derivative values can be useful in algorithms for numerical 
optimization (e.g. gradient descent), iterative nonlinear equation solvers (e.g. Newton iteration),
ordinary and partial differential equations solvers, differential geometry etc. In general, 
derivatives can be calculated by various means: 

  (1) Analytically, by directly implementing a symbolically derived formula. This is tedious, 
      error-prone and needs human effort for each individual case (unless a symbolic math engine 
      is available, but the expressions tend to swell quickly rendering the approach inefficient).
  (2) Numerically, by using finite difference approximations. This can be computationally 
      expensive and inaccurate.
  (3) Automatically, by overloading operators and functions for a number type that is augmented by
      a derivative field. The resulting algebra of such augmented numbers makes use of the well 
      known differentiation rules and derivatives of elementary functions.

The so called dual numbers are a way to implement the 3rd option. Each dual number has one field 
"v" representing the actual (value of the) number itself and an additional field "d" for 
representing the derivative. In all arithmetic operations that we do with a pair of operands, the 
first value component "v" is computed as usual and the derivative component "d" is computed using
the well known differentiation rules: sum-rule, difference-rule, product-rule and quotient-rule. 
Likewise, in univariate function evaluations, we apply the chain-rule. 

Mathematically, we can think of the dual numbers as being constructed from the real numbers in a 
way similar to the complex numbers. For these, we postulate the existence of a number i which has
the property i^2 = -1. No real number has this property, so i can't be a real number. For the dual
numbers, we postulate a nonzero number epsilon (denoted here as E) with the property E^2 = 0. No 
real number (except zero, which was excluded) has this property, so E can't be a real number. The 
dual numbers are then numbers of the form a + b*E, just like the complex numbers are of the form 
a + b*i, where a and b are real numbers. ..i think, this has relations to nonstandard analysis with
hyperreal numbers and E can be thought of as being the infinitesimal....figure that out and 
explain more...maybe mention also also hyperbolic numbers...they do not form a field, only a ring,
because the quotient rule demands that the v-part of divisor must be nonzero - so any dual number
with zero v-part is a number that we cant divide by - and these are not the neutral element(s)
of addition (there can be only one anyway)


To implement multidimensional derivatives (gradients, Jacobians, etc.), we can use rsMatrix as 
template type T. Then, the input is an Mx1 matrix, the output is an Nx1 matrix and the Jacobian is 
an NxM matrix. ...hmm...this is not how i did it in the examples...
...what about higher derivative via nesting?

...At some point, the derivative value must be seeded...i think, this is called 
"forward mode"...We seed the derivative field with 1 by default-constructing a dual number from a 
real number...explain also what reverse mode is..tbc...

...under construction... */

template<class TVal, class TDer>
class rsDualNumber 
{

public:



  TVal v;  // function value, "real part" or "standard part"
  TDer d;  // derivative, "infinitesimal part" (my word)

  rsDualNumber(TVal value = TVal(0), TDer derivative = TDer(1)) : v(value), d(derivative) {}
  // maybe the derivative should default to 1? what is most convenient? to seed or not to seed?


  using DN = rsDualNumber<TVal, TDer>;   // shorthand for convenience



  //-----------------------------------------------------------------------------------------------
  // \name Inquiry

  TVal getValue()      const { return v; }
  TDer getDerivative() const { return d; }

  //-----------------------------------------------------------------------------------------------
  // \name Arithmetic operators

  /** Unary minus just applies minus sign to both parts. */
  DN operator-() const { return DN(-v, -d); }

  /** Implements sum rule: (f+g)' = f' + g'. */
  DN operator+(const DN& y) const { return DN(v + y.v, d + y.d); }

  /** Implements difference rule: (f-g)' = f' - g'. */
  DN operator-(const DN& y) const { return DN(v - y.v, d - y.d); }

  /** Implements product rule: (f*g)' = f' * g + g' * f. */
  DN operator*(const DN& y) const { return DN(v * y.v, d*y.v + v*y.d ); }

  /** Implements quotient rule: (f/g)' = (f' * g - g' * f) / g^2. */
  DN operator/(const DN& y) const { return DN(v / y.v, (d*y.v - v*y.d)/(y.v*y.v) ); }
    // requires that y.v is nonzero - what does this mean for the algebraic structure of dual 
    // numbers? do they form a ring? it's not a field because *any* number y for which v is zero 
    // can't be a divisor - but if the d-field of y is nonzero, then y is not 0 (the neutral 
    // element of addition)...right? ..so we have nonzero elements that we can't divide by, so we
    // have no field
    // maybe we could do something else when y.v == 0. in this case, the infinitesimal part gets 
    // blown up to infinity, so maybe it could make sense to add the numerator 
    // (d*y.v - v*y.d) = -v*y.d to the real part? ...highly speculative - maybe try it and see 
    // what happens...but actually, the 0 in the denominator is squared, so it gets blown up to
    // "infinity-squared" ...so maybe it should overblow the real part...oh - that's what it 
    // already does anyway...maybe it should be v - v*y.d = v*(1-y.d)...maybe consider the limits
    // when y.v goes to 0

  template<class Ty> DN operator+(const Ty& y) const { return DN(v + TVal(y), d                ); }
  template<class Ty> DN operator-(const Ty& y) const { return DN(v - TVal(y), d                ); }
  template<class Ty> DN operator*(const Ty& y) const { return DN(v * TVal(y), d * TDer(TVal(y))); }
  template<class Ty> DN operator/(const Ty& y) const { return DN(v / TVal(y), d / TDer(TVal(y))); }

  // maybe rename operands from x,y to a,b - x,y should be used for function inputs and ouputs in
  // expressions like y = f(x)

  // todo: add boilerplate for +=,-=,*=,...



  //-----------------------------------------------------------------------------------------------
  // \name Comparison operators

  bool operator==(const DN& y) const { return v == y.v && d == y.d; } // maybe we should only compare v
  bool operator!=(const DN& y) const { return !(*this == y); }
  bool operator< (const DN& y) const { return v <  y.v; }
  bool operator<=(const DN& y) const { return v <= y.v; }
  bool operator> (const DN& y) const { return v >  y.v; }
  bool operator>=(const DN& y) const { return v >= y.v; }

  // maybe we should take the d-part into account in the case of equal v-parts. this video says 
  // something about the hyperreal numbers being an *ordered* field:
  // https://www.youtube.com/watch?v=ArAjEq8uFvA
  // when we do it like this, the set of relations {<,==,>} satisfy the trichotomy law: any pair 
  // of numbers is in one and only one of those 3 relations

};

//-------------------------------------------------------------------------------------------------
// Operators for left argument of type Tx that can be converted into TVal:

template<class TVal, class TDer, class Tx>
rsDualNumber<TVal, TDer> operator+(const Tx& x, const rsDualNumber<TVal, TDer>& y)
{ return rsDualNumber<TVal, TDer>(TVal(x) + y.v, y.d); } // ok

template<class TVal, class TDer, class Tx>
rsDualNumber<TVal, TDer> operator-(const Tx& x, const rsDualNumber<TVal, TDer>& y)
{ return rsDualNumber<TVal, TDer>(TVal(x) - y.v, -y.d) ; } // ok

template<class TVal, class TDer, class Tx>
rsDualNumber<TVal, TDer> operator*(const Tx& x, const rsDualNumber<TVal, TDer>& y)
{ return rsDualNumber<TVal, TDer>(TVal(x) * y.v, TDer(TVal(x)) * y.d); } // ok

template<class TVal, class TDer, class Tx>
rsDualNumber<TVal, TDer> operator/(const Tx& x, const rsDualNumber<TVal, TDer>& y)
{ return rsDualNumber<TVal, TDer>(TVal(x) / y.v, -TDer(x)*y.d/(y.v*y.v) ); } // ok

// maybe reduce the noise via #defines here, too

//-------------------------------------------------------------------------------------------------
// Elementary functions of dual numbers. The v-parts are computed as usual and the d-parts are 
// computed via the chain rule: (f(g(x)))' = g'(x) * f'(g(x)). We use some preprocessor 
// #definitions to reduce the verbosity of the boilerplate code. They are #undef'd when we are done
// with them:

#define RS_CTD template<class TVal, class TDer>  // class template declarations
#define RS_DN  rsDualNumber<TVal, TDer>          // dual number
#define RS_PFX RS_CTD RS_DN                      // prefix for the function definitions

RS_PFX rsExp(RS_DN x) { return RS_DN(rsExp(x.v),  x.d*rsExp(x.v)); }
RS_PFX rsSin(RS_DN x) { return RS_DN(rsSin(x.v),  x.d*rsCos(x.v)); }
RS_PFX rsCos(RS_DN x) { return RS_DN(rsCos(x.v), -x.d*rsSin(x.v)); }
RS_PFX rsTan(RS_DN x) { TVal t = rsTan(x.v); return RS_DN(t, x.d*TDer(TVal(1)+t*t)); }
RS_PFX rsSinh(RS_DN x) { return RS_DN(rsSinh(x.v), x.d*rsCosh(x.v)); }
RS_PFX rsCosh(RS_DN x) { return RS_DN(rsCosh(x.v), x.d*rsSinh(x.v)); }
RS_PFX rsTanh(RS_DN x) { TVal t = rsTanh(x.v); return RS_DN(t, x.d*TDer(TVal(1)-t*t)); }

// not tested:
RS_PFX rsLog( RS_DN x) { return RS_DN(rsLog(x.v),  x.d/x.v); } // requires x.v > 0
RS_PFX rsSqrt(RS_DN x) { return RS_DN(rsSqrt(x.v), x.d*TVal(0.5)/sqrt(x.v)); } 
// requires x.v > 0 - todo: make it work for x.v >= 0 - the derivative part at 0 should be computed
// by using a limit


RS_PFX rsPow(RS_DN x, int  n) { return RS_DN(rsPow(x.v, n),  x.d*n*rsPow(x.v, n-1)); }
RS_PFX rsPow(RS_DN x, TVal n) { return RS_DN(rsPow(x.v, n),  x.d*n*rsPow(x.v, n-1)); }
// Not sure, if we should have these two of just always use the more general implementation below
// which takes two dual numbers as inputs.

RS_PFX rsPow(RS_DN x, RS_DN y)
{
  return rsExp(y * rsLog(x)); // x^y = exp(y * log(x))
}
// see also:
// https://math.stackexchange.com/questions/1914591/dual-number-ab-varepsilon-raised-to-a-dual-power-e-g-ab-varepsilon
//
// Try also an implementation based on:
// https://en.wikipedia.org/wiki/Logarithmic_differentiation#Composite_exponent
// see also Hans Cycon's math book, page 99 "Logarithmische Ableitung":
//   (f^g)' = (f^g) * (g' * log(f) + g * f' / f)   for f > 0
// implementing this formula directly for the d-part along with calling pow for the v-part should
// probably give the same result. -> try it!

// todo: cbrt, pow, abs, asin, acos, atan, atan2, etc.
// what about floor and ceil? should their derivatives be a delta-comb? well - maybe i should not
// care about them - they are not differentiable anyway


#undef RS_CTD
#undef RS_DN
#undef RS_PFX




//-------------------------------------------------------------------------------------------------
// Functions for nested dual numbers - can be used to compute 2nd derivatives:
// ...works only with simply nested dual numbers - nesting twice doesn't work - maybe nesting 
// should be postponed - it's a mess

#define RS_CTD template<class T1, class T2, class T3>  // class template declarations
#define RS_IDN rsDualNumber<T2, T3>                    // inner dual number
#define RS_ODN rsDualNumber<T1, RS_IDN>                // outer dual number

// types: x.v: T1, x.d.v: T2, x.d.d: T3

RS_CTD RS_ODN rsSin(RS_ODN x) { return RS_ODN(rsSin(x.v),  x.d.v*rsCos(RS_IDN(x.v))); }
RS_CTD RS_ODN rsCos(RS_ODN x) { return RS_ODN(rsCos(x.v), -x.d.v*rsSin(RS_IDN(x.v))); }


//RS_CTD RS_ODN rsExp(RS_ODN x) { return RS_ODN(rsExp(x.v),  x.d.v*rsExp(RS_IDN(x.v))); }
//RS_CTD RS_ODN rsExp(RS_ODN x) { return RS_ODN(rsExp(x.v),  RS_IDN(x.d.v)*rsExp(RS_IDN(x.v))); }
//RS_CTD RS_ODN rsExp(RS_ODN x) { return RS_ODN(rsExp(x.v),  RS_IDN(x.d.v,0)*rsExp(RS_IDN(x.v))); }  // 3rd wrong
//RS_CTD RS_ODN rsExp(RS_ODN x) { return RS_ODN(rsExp(x.v),  RS_IDN(x.d.v,1)*rsExp(RS_IDN(x.v))); }    // 2nd,3rd wrong
//RS_CTD RS_ODN rsExp(RS_ODN x) { return RS_ODN(rsExp(x.v),  T2(x.d.v)*rsExp(RS_IDN(x.v))); } // 3rd wrong
//RS_CTD RS_ODN rsExp(RS_ODN x) { return RS_ODN(rsExp(x.v),  RS_IDN(T2(x.d.v))*rsExp(RS_IDN(x.v))); }
//RS_CTD RS_ODN rsExp(RS_ODN x) { return RS_ODN(rsExp(x.v),  RS_IDN(T2(x.d.v),0)*rsExp(RS_IDN(x.v))); }
//RS_CTD RS_ODN rsExp(RS_ODN x) { return RS_ODN(rsExp(x.v), RS_IDN(T3(x.d.v))*rsExp(RS_IDN(x.v))); } 
//RS_CTD RS_ODN rsExp(RS_ODN x) { return RS_ODN(rsExp(x.v), T3(x.d.v)*rsExp(RS_IDN(x.v))); }
// 3rd derivative is wrong - seems to be multiplied by factor 2

RS_CTD RS_ODN rsExp(RS_ODN x) 
{ 
  T1     v = rsExp(x.v);                      // T1
  //RS_IDN d = T2(x.d.v) * rsExp(RS_IDN(x.v));  // T2 * Dual<T2,T3> -> Dual(x.d.v,1) * exp(Dual(x.v,1))

  // if T2 is itself a dual number, this does not work - both factors get constructed with a 1 for the
  // d part and in the product rule f'*g + g'*f, we end up producing twice the desired value due to 
  // adding two equal terms

  RS_IDN d = RS_IDN(x.d.v,0) * rsExp(RS_IDN(x.v));
  // hmm...this also doesn't work - we still get the extra factor of 2

  //RS_IDN d = T2(x.d.v,0) * rsExp(RS_IDN(x.v)); // no compile with T2=float

  //RS_IDN d = RS_IDN(x.d.v * rsExp(RS_IDN(x.v)));

  //RS_IDN d = x.d.v * rsExp(RS_IDN(T3(x.v)));
  //RS_IDN d = T2(x.d.v) * rsExp(RS_IDN(T2(x.v)));
  //RS_IDN d = RS_IDN(T3(x.d.v)) * rsExp(RS_IDN(T2(x.v)));
  // somewhere, there's a constructor call missing and one d-element is not seeded, i think

  return RS_ODN(v,  d); 
}

// Why does it actually work when using x.d.v as inner derivative? It probably should not be 
// surprising, but i'm not sure about why. Maybe try nesting twice - doesn't work...maybe we need
// to wrap something else into a constructor call? i tried -RS_IDN(x.d.v), RS_IDN(-x.d.v) for 
// d-part of rsCos - but that doesn't work...maybe we need to use 0 as 2nd argument?

// todo: log, tan, sqrt, pow, abs, sinh, cosh, tanh, asin, acos, atan, atan2, etc.

#undef RS_CTD
#undef RS_IDN
#undef RS_NDN

// maybe for doubly nested dual numbers, we will need yet another specialzation? ...if so, will it 
// end there or will we need another for triply nested ones and so on? that would be bad!





/*
// not yet tested:
template<class TVal, class TDer>
rsDualNumber<TVal, TDer> rsSqrt(rsDualNumber<TVal, TDer> x) 
{ return rsDualNumber<TVal, TDer>(sqrt(x.v), x.d*T(0.5)/sqrt(x.v)); }  // verify

template<class TVal, class TDer>
rsDualNumber<TVal, TDer> rsLog(rsDualNumber<TVal, TDer> x) 
{ return rsDualNumber<TVal, TDer>(log(x.v), x.d/x.v); }  // requires x.v > 0

template<class TVal, class TDer>
rsDualNumber<TVal, TDer> rsPow(rsDualNumber<TVal, TDer> x, TVal p)
{ return rsDualNumber<TVal, TDer>(pow(x.v, p), x.d*TDer(p)*pow(x.v, p-1)); }  // requires x.v != 0
// what, if p is also an dual number? ...this requires more thought....

template<class TVal, class TDer>
rsDualNumber<TVal, TDer> rsAbs(rsDualNumber<TVal, TDer> x) 
{ return rsDualNumber<TVal, TDer>(rsAbs(x.v), x.d*rsSign(x.v)); }  // requires x.v != 0..really?
*/



/** A number type for automatic differentiation in reverse mode. The operators and functions are 
implemented in a way so keep a record of the whole computation. At the end of the computation, a 
call to getDerivative() triggers the computation of the derivative by running through the whole 
computation backwards

....i don't know yet, if the implementation is on the right track - i just started without much
of a plan...

*/

template<class TVal, class TDer>
class rsAutoDiffNumber
{

//protected:

public:



  enum class OperationType
  {
    neg, add, sub, mul, div, sqrt, sin, cos, exp  // more to come
  };
  // maybe sort by arity: unary first, the binary, the ternary, etc. - we may want to be able 
  // quickly distingusih arity for dispatching computations (1st dispatch is on arity, 2nd on 
  // actual func/op within a given arity)

  inline static const TVal NaN = RS_NAN(TVal);
  using ADN = rsAutoDiffNumber<TVal, TDer>;   // shorthand for convenience


  /** Structure to store the operations */
  struct Operation
  {
    OperationType type; // type of operation
    ADN op1;            // first operand
    ADN op2;            // second operand
    TVal res;           // result

    // experimental:
    TVal adj = NaN;     // adjoint
    // we could also use an ADN as res and res.d as adjoint...for what reason do we actually store
    // the result's value anyway? it's not used in the backprop 


    // -maybe we should store the partial derivatives with respect to both inputs?
    // -maybe instead of the values of the operands, we need to store the indices of the nodes 
    //  whose result is used, so we can backprop into them? but what about the initial inputs? they
    //  don't have any operation associated with them


    Operation(OperationType _type, ADN _op1, ADN _op2, TVal _res)
      : type(_type), op1(_op1), op2(_op2), res(_res) {}
  };


  std::vector<Operation>& ops;  // "tape" of recorded operations



  using OT  = OperationType;
  using OP  = Operation;



  /** Pushes an operation record consisting of a type, two operands and a result onto the operation 
  tape ops and returns the result of the operation as rsAutoDiffNumber. */
  ADN push(OperationType type, ADN op1, ADN op2, TVal res)
  {
    ops.push_back(OP(type, op1, op2, res)); 
    return ADN(res, ops);
  }

public:

  TVal v;        // value
  TDer d = NaN;  // (partial) derivative (aka adjoint?)

  rsAutoDiffNumber(TVal value, std::vector<Operation>& tape) : v(value), /*a(NaN),*/ ops(tape) {}




  void initLoc() { loc = this; }
  // inits the location pointer - this is a bad API and temporary
  // get rid of this - or at least avoid having to call it from client code - client created 
  // numbers with memory locations should always automatically assign the location and temporaries
  // shouldn't ...maybe we need to pass a flag to the constructor to indicate a memory variable
  // if this is not possible, find a better name - like enable
  // i think, maybe it's possible by making sure that push() sets loc to nullptr - then the 
  // standard constructor may automatically set loc = this. ...but is it really safe to assume,
  // that all variable locations are still valid - no: the user may define their own functions 
  // using temporary variables - these will be invalid, when the function returns - maybe we 
  // indeed need manual enable calls
  // maybe rename to setDerivativeNeeded - it should be used by client code to indicate that it 
  // wants to compute derivatives with respect to the variable

  //-----------------------------------------------------------------------------------------------
  // \name Inquiry

  TVal getValue()      const { return v; }


  void computeDerivatives()
  {
    TDer d(1);
    // i think, this accumulator is wrong - we whould use only the v,d fields of the operand(s) and
    // the result - so the result may also have to be an ADN...or maybe not?

    // I think, we need to init all ajoints in the ops array to 0, then, instead of using d, we 
    // should use the adjoint of the result?

    int numOps = (int)ops.size();
    if(numOps == 0)
      return;

    int i;

    // Init adjoints and local gradients at all nodes:
    for(i = 0; i < numOps; i++)
    {
      ops[i].adj   = 0;
      ops[i].op1.d = 0;
      ops[i].op2.d = 0;
    }

    // Backpropagate the adjoints and local gradients:
    ops[numOps-1].adj = 1; // seed for backprop
    i = (int)numOps-1;
    while(i >= 0)  // use for-loop
    {

      //ops[i].op1d = getOpDerivative(ops[i]);

      // todo:
      //TDer d = ops[i].adj;

      if(ops[i].type == OT::add)
      {
        ops[i].op1.d = d;
        ops[i].op2.d = d;
      }
      else if(ops[i].type == OT::sub)
      {
        ops[i].op1.d =  d;
        ops[i].op2.d = -d;
      }
      else if(ops[i].type == OT::mul)
      {
        ops[i].op1.d = d * ops[i].op2.v;  // (x*y)_x = y
        ops[i].op2.d = d * ops[i].op1.v;  // (x*y)_y = x
      }
      else if(ops[i].type == OT::div)
      {
        ops[i].op1.d =  d                /  ops[i].op2.v;                  // (x/y)_x =  1/y
        ops[i].op2.d = -d * ops[i].op1.v / (ops[i].op2.v * ops[i].op2.v);  // (x/y)_y = -x/y^2
      }
      else
      {
        // operation is a unary function
        d *= getOpDerivative(ops[i]);  // ??? apparently, this comes from the chain rule but i 
        // think, it doesn't generalize to arbitrary graphs
        ops[i].op1.d = d;
        ops[i].op2.d = NaN;
      }

      

      // assign derivative fields in memory variables, if applicable:
      if(ops[i].op1.loc != nullptr) ops[i].op1.loc->d = ops[i].op1.d;
      if(ops[i].op2.loc != nullptr) ops[i].op2.loc->d = ops[i].op2.d;






      i--;

    }

  }

  // rename to getFunctionDerivative and use it only for unary functions, maybe rename to 
  // getUnaryDerivative and maybe compute the partial derivative of binary functions/operators and
  // ternary functions by similar functions: getBinaryPartialDerivative, 
  // getTernaryPartialDerivative
  TDer getOpDerivative(const Operation& op) const
  {
    switch(op.type)
    {
    //case OT::add: return   op.op1.v + op.op2.v;  // is this correct?

      // for mul..do we have to call getOpDerivative twice with both operands
      // or maybe reverse mode is only good for univariate operations? they always only talk about
      // multiplying the Jacobians left-to-right or right-to-left - but here, we need to add two
      // Jacobians...maybe we have to use recursion in getDerivative
      // or maybe the Operation struct needs derivative fields for both operands? that are filled
      // in the backward pass?


    case OT::sqrt: return  TVal(0.5)/rsSqrt(op.op1.v); // maybe we could also use TVal(0.5)/op.res
    case OT::sin:  return  rsCos(op.op1.v);
    case OT::cos:  return -rsSin(op.op1.v);
    case OT::exp:  return  rsExp(op.op1.v);

      // but what about the binary operators? they are currently handled directly in 
      // computeDerivatives

    default: return TDer(0);
    }
  }

  //-----------------------------------------------------------------------------------------------
  // \name Arithmetic operators

  ADN operator-() { return push(neg, v, NaN, -v); }

  ADN operator+(const ADN& y) { return push(OT::add, *this, y, v + y.v); }
  ADN operator-(const ADN& y) { return push(OT::sub, *this, y, v - y.v); }
  ADN operator*(const ADN& y) { return push(OT::mul, *this, y, v * y.v); }
  ADN operator/(const ADN& y) { return push(OT::div, *this, y, v / y.v); }


  ADN operator=(const ADN& y) 
  { 
    v = y.v;
    ops = y.ops;
    return *this;
  }

  // void setOperationTape(std::vector<Operation>& newTape) { ops = newTape; }


  
//protected:  // try to make protected - but that may need a lot of boilerplat friend declarations

  rsAutoDiffNumber* loc = nullptr;
  // why do we need this? try to get rid!

};




#define RS_CTD template<class TVal, class TDer>  // class template declarations
#define RS_ADN rsAutoDiffNumber<TVal, TDer>      // 
#define RS_OP RS_ADN::OperationType
#define RS_PFX RS_CTD RS_ADN                      // prefix for the function definitions

RS_PFX rsSqrt(RS_ADN x) { return x.push(RS_OP::sqrt, x, RS_ADN(0.f, x.ops), rsSqrt(x.v)); }
RS_PFX rsSin( RS_ADN x) { return x.push(RS_OP::sin,  x, RS_ADN(0.f, x.ops), rsSin( x.v)); }
RS_PFX rsCos( RS_ADN x) { return x.push(RS_OP::cos,  x, RS_ADN(0.f, x.ops), rsCos( x.v)); }
RS_PFX rsExp( RS_ADN x) { return x.push(RS_OP::exp,  x, RS_ADN(0.f, x.ops), rsExp( x.v)); }
// the 2nd operand is a dummy - maybe define a macro for it


//RS_PFX rsSqrt(RS_ADN x) { return x.push(RS_OP::sqrt, x, RS_ADN(RS_ADN::NaN), rsSqrt(x.v)); }
//RS_PFX rsSin( RS_ADN x) { return x.push(RS_OP::sin,  x, RS_ADN(RS_ADN::NaN), rsSin( x.v)); }
//RS_PFX rsCos( RS_ADN x) { return x.push(RS_OP::cos,  x, RS_ADN(RS_ADN::NaN), rsCos( x.v)); }
//RS_PFX rsExp( RS_ADN x) { return x.push(RS_OP::exp,  x, RS_ADN(RS_ADN::NaN), rsExp( x.v)); }

#undef RS_CTD
#undef RS_DN
#undef RS_OP
#undef RS_PFX




// https://en.wikipedia.org/wiki/Automatic_differentiation
// https://en.wikipedia.org/wiki/Automatic_differentiation#Automatic_differentiation_using_dual_numbers
// https://www.neidinger.net/SIAMRev74362.pdf
// https://en.wikipedia.org/wiki/Dual_number

// http://www.autodiff.org/?module=Introduction&submenu=Surveys

// https://www.youtube.com/watch?v=xtZ0_0DP_GI Automatic differentiation using ForwardDiff.jl and ReverseDiff.jl (Jarrett Revels, MIT)
// http://www.juliadiff.org/ForwardDiff.jl/stable/
// https://github.com/JuliaDiff/ForwardDiff.jl
// https://github.com/JuliaDiff

// https://en.wikipedia.org/wiki/Grassmann_number
// https://en.wikipedia.org/wiki/Split-complex_number

// what about multivariate functions and partial derivatives?

template<class T>
class rsDualComplexNumber  // "duco"
{

public:

  T a, b, c, d;

  using DCN = rsDualComplexNumber<T>;


  rsDualComplexNumber(T real = T(0), T imag = T(0), T dual = T(0), T imagDual = T(0)) 
    : a(real), b(imag), c(dual), d(imagDual) {}




  DCN operator+(const DCN& z) const { return DCN(a + z.a, b + z.b, c + z.c, d + z.d); }
  DCN operator-(const DCN& z) const { return DCN(a - z.a, b - z.b, c - z.c, d - z.d); }
  DCN operator*(const DCN& z) const
  {
    return DCN(a*z.a - b*z.b, 
               a*z.b + z.a*b, 
               a*z.c + z.a*c - b*z.d - z.b*d, 
               a*z.d + z.a*d + b*z.c + z.b*c);
  }

  DCN operator/(const DCN& z) const
  {
    T A = z.a*z.a + z.b*z.b;
    T D = T(2)*(z.a*z.d - z.b*z.c);

    DCN tmp = *this * DCN(z.a, -z.b, -z.c, z.d);  // product after first augmentation
    tmp = tmp * DCN(A, T(0), T(0), -D);        // product after second augmentation
    T s = T(1) / (A*A);
    tmp.a *= s;
    tmp.b *= s;
    tmp.c *= s;
    tmp.d *= s;

    return tmp;
  }
  // todo: simplify, optimize - the second multiplication contains lots of zeros, the scaling
  // should use an operator / that takes a 2nd parameter of type T

  /*
  DCN operator/(const T& z) const { }
  */


};

#define RS_CTD template<class T> 
#define RS_DCN rsDualComplexNumber<T> 
#define RS_CMP std::complex<T> 
#define RS_PFX RS_CTD RS_DCN 

RS_PFX rsSin(RS_DCN x)
{
  // extract complex value and derivative of inputs:
  RS_CMP v(x.a, x.b);
  RS_CMP d(x.c, x.d);

  // compute derivative and value of output:
  d *= cos(v);    // chain rule
  v  = sin(v);    // regular function application

  // construct DCN from the 2 complex numbers:
  return RS_DCN(v.real(), v.imag(), d.real(), d.imag());
}
// needs verification

#undef RS_CTD
#undef RS_DCN
#undef RS_CMP 
#undef RS_PFX

// is this a restricted case of this?: https://en.wikipedia.org/wiki/Dual_quaternion
// oh - there are already dual complex numbers - but they work differently:
// https://en.wikipedia.org/wiki/Dual-complex_number
// ...so we should use another name - how about DiffComplex or DiComplex - when they are useful for
// automatic differentiation in the complex domain - we'll see
// todo: define elementary functions exp, sin, cos, sqrt
// implement operators that allow mixed operations with std::complex


//=================================================================================================

/** Defines a line in the 2D plane using 3 numbers a,b,c representing the line equation:

  a*y + b*x + c = 0



*/

template<class T>
class rsLine
{


public:


//protected:

  T a, b, c;  // coefficients of line equation: a*y + b*x + c = 0

};

template<class T>  // T should be the type for the vertices such as rsVector2D, rsVector3D
class rsTriangle
{

public:

  rsTriangle(T _A, T _B, T _C) : A(_A), B(_B), C(_C) {}


//protected:

  T A, B, C;  // coordinates of the vertices

};




//=================================================================================================

/** Class to represent complex numbers in polar form: z = r * exp(i * a) with radius r and angle 
(or argument) a. A somwhat peculiar feature of this implementation is that the angle is not 
restricted to the domain (-pi, pi] or [0, 2*pi) or whatever other interval of length 2*pi but 
instead can assume any real value and differences in the argument by multiples of 2*pi actually 
carry meaning: they distinguish between several sheets of a Riemann surface...tbc...  */

template<class T>
class rsComplexPolar
{

public:

  // ToDo: 
  // -define +,-,*,/ operators
  // -define functions: sqrt, exp, pow, log

protected:

  T r = T(0), a = T(0);

};

//=================================================================================================



//=================================================================================================

/** Class for treating pre-existing data owned elsewhere as tableau. @see rsTableau. */

template<class T>
class rsTableauView
{

public:

  //-----------------------------------------------------------------------------------------------
  /** \name Construction/Destruction */

  /** Default constructor. */
  rsTableauView() {}

  rsTableauView(int numRows, T* data, int* starts, int* lengths)
  {
    rsAssert(numRows >= 1 && data != nullptr);    
    // todo: maybe also assert that none of lengths[i] is <= 0

    this->numRows  = numRows;
    this->pDataPtr = data;
    this->pStarts  = starts;
    this->pLengths = lengths;
  }



  //-----------------------------------------------------------------------------------------------
  /** \name Element access */

  T& operator()(const int i, const int j) 
  { 
    rsAssert(i >= 0 && i < numRows,     "Invalid row index");
    rsAssert(j >= 0 && j < pLengths[i], "Invalid column index");
    return pData(starts[i] + j);
  }

  // todo: implement operators or functions returning const references like in rsMatrixView

protected:

  T*   pData    = nullptr;
  int* pLengths = nullptr;
  int* pStarts  = nullptr;
  int  numRows  = 0;


};

/** A class for efficiently storing data that has the form of an array of arrays, but unlike a 
matrix, each subarray (i.e. each row) can have a different length. 

Example - tableau with 4 rows:

  1 2 3
  4
  5 6 7 8
  9 0

The actual data is stored in a contiguous block of memory and we also store the start indices and 
lengths of each row...tbc...  */

template<class T>
class rsTableau : public rsTableauView<T>
{

public:

  rsTableau() {}

  /** Creates a new Tableau with the given number of rows where each row has a length given by the
  repective entry in the rowLengths array. We copy the content of the rowLengths array into a 
  member so after initializing, the original lengths array will not be referenced anymore */
  rsTableau(int numOfRows, int* rowLengths)
  {
    rsAssert(numOfRows > 0);  // ToDo: maybe allow to init with size zero

    numRows = numOfRows;

    lengths.resize(numRows);
    pLengths = &lengths[0];
    rsArrayTools::copy(rowLengths, pLengths, numRows);

    starts.resize(numRows);
    pStarts = &starts[0];
    pStarts[0] = 0;
    for(int i = 1; i < numRows; i++)
      pStarts[i] = pStarts[i-1] + pLengths[i-1];

    int size = rsArrayTools::sum(pLengths, numRows);
    data.resize(size);
    pData = &data[0];
  }
  // needs test, maybe factor out into init function -> useful if client needs to re-init
  // What if one or more of the rowLengths is zero? Should this be allowed? If so, check that it
  // works correctly!

protected:

  std::vector<T>   data;
  std::vector<int> lengths;
  std::vector<int> starts;
  // optimize for less allocations: use one vector ls for lengths and starts and use 
  // pLengths = &ls[0] and pStarts = &ls[numRows]...or something


};


//=================================================================================================



//=================================================================================================

/** A data structure for optimizing the computation of partial derivatives on irregular meshes. 
The mesh is given as a graph and the data stored in this graph is used to derive coefficients with
which the partial derivatives u_x, u_y at some vertex i can be computed as weighted sum of the 
vertex value itself and the values at its neighbors. These weights are stored internally in the
rsStencilMesh2D object when you call the member function computeCoeffsFromMesh. This can be done 
once and for all for a given mesh and after that, subsequent gradient computations can be computed 
by the gradient method which is (supposedly) cheaper than calling the corresponding function
rsNumericDifferentiator<T>::gradient2D(mesh, ...). The result should be the same up to roundoff 
errors. */

template<class T>
class rsStencilMesh2D
{

public:

  /** Computes the coefficients (or weights) for forming the weighted sum of the vertex value and
  its neighbors for the estimate of the partial derivatives. */
  void computeCoeffsFromMesh(const rsGraph<rsVector2D<T>, T>& mesh);

  /** Computes estimates of the partial derivatives u_x, u_y given function values u. */
  void gradient(const T* u, T* u_x, T* u_y);

  /** Given the function values u, this computes the partial derivatives u_x and u_y at the 
  grid-point with index i. Note that u should point to the start of the u-array whereas u_x and u_y
  should point to the i-th elements of the respective arrays. This makes sense because... */
  void gradient(const T* u, int i, T* u_x, T* u_y);

  // todo: hessian, laplacian, etc.


  int getNumNodes() const { return numNodes; }
  int getNumEdges() const { return numEdges; }


  int getNumNeighbors(int i) const { return numNeighbors[i]; }
  T   getSelfWeightX( int i) const { return selfWeightsX[i]; }
  T   getSelfWeightY( int i) const { return selfWeightsY[i]; }

  int getNeighborIndex(  int i, int k) const { return neighborIndices[ starts[i] + k]; }
  T   getNeighborWeightX(int i, int k) const { return neighborWeightsX[starts[i] + k]; }
  T   getNeighborWeightY(int i, int k) const { return neighborWeightsY[starts[i] + k]; }



protected:


  // internal:




  // These arrays are all numNodes long:
  int numNodes = 0;
  std::vector<int> numNeighbors; // entry is the number of neighbors of node i
  std::vector<T>   selfWeightsX, selfWeightsY; //
  std::vector<int> starts;  
  // entries are start-indices of the sections with values belonging to node i 
  // in the neighborIndices, etc arrays - maybe rename to offsets

  // These arrays are all numEdges long:
  int numEdges = 0;              // sum of entries of numNeighbors array
  std::vector<int> neighborIndices;
  std::vector<T>   neighborWeightsX, neighborWeightsY;
};

template<class T>
void rsStencilMesh2D<T>::computeCoeffsFromMesh(const rsGraph<rsVector2D<T>, T>& mesh)
{
  using Vec2 = rsVector2D<T>;

  // Allocate memory and set offsets:
  numNodes = mesh.getNumVertices();
  numNeighbors.resize(numNodes);
  selfWeightsX.resize(numNodes);
  selfWeightsY.resize(numNodes);
  starts.resize(numNodes);

  for(int i = 0; i < numNodes; i++)
    numNeighbors[i] = mesh.getNumEdges(i);
  starts[0] = 0;
  for(int i = 1; i < numNodes; i++)
    starts[i] = starts[i-1] + numNeighbors[i-1];
  numEdges = rsSum(numNeighbors);

  neighborIndices.resize(numEdges);
  neighborWeightsX.resize(numEdges);
  neighborWeightsY.resize(numEdges);
 
  // Compute the coefficients:
  for(int i = 0; i < numNodes; i++)
  {
    const Vec2& vi = mesh.getVertexData(i);

    // Compute least squares matrix A and its inverse M:
    T a11 = T(0), a12 = T(0), a22 = T(0);
    T sx = T(0), sy = T(0);
    for(int k = 0; k < numNeighbors[i]; k++)
    {
      int j = mesh.getEdgeTarget(i, k);         // index of current neighbor of vi
      const Vec2& vj = mesh.getVertexData(j);   // current neighbor of vi
      Vec2 dv = vj - vi;                        // difference vector
      T    w  = mesh.getEdgeData(i, k);         // edge weight
      a11 += w * dv.x * dv.x;
      a12 += w * dv.x * dv.y;
      a22 += w * dv.y * dv.y;
      sx  += w * dv.x;
      sy  += w * dv.y;
    }
    rsMatrix2x2 A(a11, a12, a12, a22);
    rsMatrix2x2 M = A.getInverse();
    T m11 = M.a, m12 = M.b, m22 = M.d;
    // try to optimize: get rid of using rsMatrix2x2


    selfWeightsX[i] = -(m11 * sx + m12 * sy);
    selfWeightsY[i] = -(m12 * sx + m22 * sy);   // verify!
    for(int k = 0; k < numNeighbors[i]; k++)
    {
      int j = mesh.getEdgeTarget(i, k);         // index of current neighbor of vi
      const Vec2& vj = mesh.getVertexData(j);   // current neighbor of vi
      Vec2 dv = vj - vi;                        // difference vector
      T    w  = mesh.getEdgeData(i, k);         // edge weight
      neighborWeightsX[starts[i] + k] = w * (m11 * dv.x + m12 * dv.y);
      neighborWeightsY[starts[i] + k] = w * (m12 * dv.x + m22 * dv.y);  // verify!
      neighborIndices [starts[i] + k] = j;
    }
    // maybe the sum of the neighbor-weights plus the self-weight should sum up to zero? ...they
    // seem to indeed do...so the self-weight should always be minus the sum of the neighbor 
    // weights - maybe it's numerically more precise to use that sum, such that the cancellation
    // is perfect?
  }
}

template<class T>
void rsStencilMesh2D<T>::gradient(const T* u, int i, T* u_x, T* u_y)
{
  *u_x = getSelfWeightX(i) * u[i];
  *u_y = getSelfWeightY(i) * u[i];
  for(int k = 0; k < getNumNeighbors(i); k++) {
    int j = getNeighborIndex(i, k);
    *u_x += getNeighborWeightX(i, k) * u[j];
    *u_y += getNeighborWeightY(i, k) * u[j]; }
}

template<class T>
void rsStencilMesh2D<T>::gradient(const T* u, T* u_x, T* u_y)
{
  for(int i = 0; i < numNodes; i++)
    gradient(u, i, &u_x[i], &u_y[i]);
}

//=================================================================================================

/** Computes the matrix-vector product y = A*x between a matrix A and a vector x where the types
of the elements may be different for all 3 objects. Typically, the output type TOut will be either
the same as the matrix type TMat or the input type TIn. ...tbc... */
template<class TMat, class TIn, class TOut>
void rsProduct(const rsSparseMatrix<TMat>& A, const std::vector<TIn>& x, std::vector<TOut>& y)
{
  rsAssert((int)x.size() == A.getNumColumns());
  rsAssert((int)y.size() == A.getNumRows());
  A.product(&x[0], &y[0]);
}
// maybe move to library as convenience function

/** Takes a mesh as input and produces as output a sparse matrix by which a vector u can be 
multiplied to obtain the vector of gradients: u' = A*u. The input vector u is supposed to have 
scalar elements (we assume to work with a scalar field), whereas the vector u' has elements of type
rsVector2D representing the gradient of the scalar field u(x,y). This is achieved by letting the 
matrix have elements of type rsVector2D. Using such a matrix should optimize the computations and 
also make it possible to apply implicit solver schemes. */
template<class T>
rsSparseMatrix<rsVector2D<T>> rsGradientMatrix(const rsGraph<rsVector2D<T>, T>& mesh)
{
  rsStencilMesh2D<T> stencilMesh;
  stencilMesh.computeCoeffsFromMesh(mesh);
  int numNodes = stencilMesh.getNumNodes();
  rsSparseMatrix<rsVector2D<T>> A(numNodes, numNodes);
  A.reserve(stencilMesh.getNumEdges() + numNodes);             // + numNodes for the self-weights
  for(int i = 0; i < numNodes; i++) {
    T vx = stencilMesh.getSelfWeightX(i);
    T vy = stencilMesh.getSelfWeightY(i);
    A.appendFastAndUnsafe(i, i, rsVector2D(vx, vy));           // add diagonal element
    for(int k = 0; k < stencilMesh.getNumNeighbors(i); k++) {
      vx = stencilMesh.getNeighborWeightX(i, k);
      vy = stencilMesh.getNeighborWeightY(i, k);
      int j = stencilMesh.getNeighborIndex(i, k);
      A.appendFastAndUnsafe(i, j, rsVector2D(vx, vy)); }}      // add elements for neighbors
  A.sortElements();                                            // ensure correct order
  return A;

  // ToDo:
  // -Maybe try to permute the matrix elements in a way that optimizes the access pattern into the
  //  u vector when computing the gradient. Maybe the u vector needs to be permuted, too. Maybe a 
  //  Hilbert curve could be useful for that?
  // -Try to compute a matrix that can be used to compute the Laplacian. This should have scalar
  //  elements. ...but i have not yet figured out a good formula for the Laplacian on irregular 
  //  meshes
}

//=================================================================================================

// Classes for representing the objects that are encountered in the extrerior algebra for 3D space 
// (in addition to the regular vectors, i.e. rsVector3D)

/** Bivectors are... */

template<class T>
class rsBiVector3D
{

public:

  rsBiVector3D(const rsVector3D<T>& surfaceNormal) : normal(surfaceNormal) {}

  rsBiVector3D(const rsVector3D<T>& u, const rsVector3D<T>& v) { normal = cross(u, v); }
  // maybe this constructor shouldn't exist -> enforce constructing bivectors via wedge-products

  rsVector3D<T> getSurfaceNormal() const { return normal; }

  rsBiVector3D<T> operator-() const { return rsBiVector3D<T>(-normal); }

  /** Compares two bivectors for equality. */
  bool operator==(const rsBiVector3D<T>& v) const { return normal == v.normal; }

protected:

  rsVector3D<T> normal;
  // The normal vector to the area element given by the parallelogram spanned by the two vectors 
  // from which this bivector was constructed


  template<class U> friend class rsExteriorAlgebra3D;
};



/** Trivectors are... */

template<class T>
class rsTriVector3D
{

public:

  rsTriVector3D(T signedVolume) : volume(signedVolume) {}

  rsTriVector3D(const rsVector3D<T>& u, const rsVector3D<T>& v, const rsVector3D<T>& w)
  {
    volume = rsVector3D<T>::tripleProduct(u, v, w);
  }

  T getSignedVolume() const { return volume; }


  /** Compares two trivectors for equality. */
  bool operator==(const rsTriVector3D<T>& v) const { return volume == v.volume; }

protected:

  T volume;
  // The signed volume of the parallelepiped spanned by the three vectors from which this trivector
  // was constructed

};


/** a.k.a. 1-form

*/

template<class T>
class rsCoVector3D
{

public:





protected:

  rsVector3D<T> vec;
  // The space of covectors is isomorphic to the space of regular vectors, but the interpretation 
  // and the typical operations that we do with covectors are different, so we use object 
  // composition together with delegation, where that makes sense.

  template<class U> friend class rsExteriorAlgebra3D;
};




/** Co-bivectors are ...a.k.a. 2-forms */

template<class T>
class rsCoBiVector3D
{

public:


protected:

  rsBiVector3D<T> biVec;

};

/** Co-trivectors are ...a.k.a. 3-forms */

template<class T>
class rsCoTriVector3D
{

public:


protected:

  rsTriVector3D<T> triVec;

};


/** Implements the operations of the exterior algebra of 3-dimensional Euclidean space. The objects
that this algebra deals with are of different types: 

primal: scalars (0-vectors), vectors (1-vectors), bivectors (2-vectors), trivectors (3-vectors)
dual: 0-forms (coscalars?), 1-forms (covectors), 2-forms (cobivectors?), 3-forms (cotrivectors?)

The names with the question mark are, as far as i know, not in common use. I made them up to fit 
the apparent pattern. The operations of the exterior algebra actually change the types of the 
objects. For example, the wegde-product of a bivector with a vector yields a trivector, the 
Hodge-star of a trivector yields a scalar, the sharp of a covector gives a vector, etc.

https://en.wikipedia.org/wiki/Exterior_algebra


https://en.wikipedia.org/wiki/Bialgebra
https://en.wikipedia.org/wiki/Graded_algebra

*/

template<class T>
class rsExteriorAlgebra3D
{

public:

  /** Wedge product between two vectors yielding a bivector. */
  static rsBiVector3D<T> wedge(const rsVector3D<T>& u, const rsVector3D<T>& v)
  { return rsBiVector3D<T>(u, v); }

  /** Wedge product between a vector and a bivector vectors yielding a trivector. */
  static rsTriVector3D<T> wedge(const rsVector3D<T>& v, const rsBiVector3D<T>& u)
  { T vol = dot(v, u.getSurfaceNormal()); return rsTriVector3D<T>(vol); }

  /** Wedge product between a bivector and a vector vectors yielding a trivector. */
  static rsTriVector3D<T> wedge(const rsBiVector3D<T>& u, const rsVector3D<T>& v)
  { T vol = dot(u.getSurfaceNormal(), v); return rsTriVector3D<T>(vol); }



  // what about vector and bivector, i.e. different argument order? i think, the result is the same 
  // because wedge is associative - so we may not need separate functions

  /** Converts a regular vector into its Hodge dual, which is a bivector in 3D. This operation is 
  called the "Hodge star" or just "start" and is basically just a re-interpretation or 
  type-conversion. No actual computation takes place. */
  static rsBiVector3D<T> star(const rsVector3D<T>& u) { return rsBiVector3D<T>(u); }

  /** Converts a bivector into its Hodge dual, which is a regular vector in 3D. */
  static rsVector3D<T> star(const rsBiVector3D<T>& bv) { return bv.normal; }

  /** Converts a scalar into its Hodge dual, which is a trivector in 3D. */
  static rsTriVector3D<T> star(const T& a) { rsTriVector3D<T> tv; tv.volume = a; return tv; }

  /** Converts a trivector into its Hodge dual, which is a scalar in 3D. */
  static T star(const rsTriVector3D<T>& tv) { return tv.volume = a; }



  // todo: implement the wegde- and star operations also for CoVector, CoBiVector, CoTriVectors, 
  // CoScalars - it's all very boilerplatsih and production code would probably do it differently,
  // but for keeping things clean and separated for learning the concepts, it may make sense to do 
  // it (maybe sort of the distinction between points and vectors, which is usually not made in 
  // production code)


  /** The musical isomorphism "flat" that turns a vector into a covector (aka 1-form), i.e. it 
  lowers the index, i.e. it transposes a (contravariant) column-vector into a (covariant) 
  row-vector. */
  static rsCoVector3D<T> flat(const rsVector3D<T>& vec)
  { rsCoVector3D<T> coVec; coVec.vec = vec; return coVec; }

  /** The musical isomorphism "sharp" that turns a covector (aka 1-form) into a vector, i.e. it 
  raises the index, i.e. it transposes a (covariant) row-vector into a (contravariant) 
  column-vector. */
  static rsVector3D<T> sharp(const rsCoVector3D<T>& coVec) { return coVec.vec}



  // todo: implement the additional operations of exterior calculus (maybe in another class or 
  // maybe rename this class): d (directional derivative), Lie derivative, interior product
  // https://en.wikipedia.org/wiki/Interior_product

};

// The wegde product as operator to support bivector and trivector construction via the syntax:
//   w = u ^ v, t = u ^ v ^ w:

template<class T>
rsBiVector3D<T> operator^(const rsVector3D<T>& u, const rsVector3D<T>& v)
{ return rsExteriorAlgebra3D<T>::wedge(u, v); }

template<class T>
rsTriVector3D<T> operator^(const rsVector3D<T>& u, const rsBiVector3D<T>& v)
{ return rsExteriorAlgebra3D<T>::wedge(u, v); }

template<class T>
rsTriVector3D<T> operator^(const rsBiVector3D<T>& u, const rsVector3D<T>& v)
{ return rsExteriorAlgebra3D<T>::wedge(u, v); }

/** Adds two bivectors and returns result as new bivector. */
template<class T>
rsBiVector3D<T> operator+(const rsBiVector3D<T>& v, const rsBiVector3D<T>& w)
{
  rsVector3D<T> nv = v.getSurfaceNormal();
  rsVector3D<T> nw = w.getSurfaceNormal();
  return rsBiVector3D<T>(nv + nw);
}

/** Multiplies a scalar and a bivectors and returns result as new bivector. */
template<class T>
rsBiVector3D<T> operator*(const T& s, const rsBiVector3D<T>& v)
{
  return rsBiVector3D<T>(s * v.getSurfaceNormal());
}

//#################################################################################################
// code for Geometric algebra

// taken from
// https://www.geeksforgeeks.org/count-set-bits-in-an-integer/
/** Counts the number of bits with value 1 in the given number n. */
int rsBitCount(int n)
{
  int count = 0;
  while(n) { n &= n-1; count++; }
  return count;
}

/** Return true, iff the bit at the given bitIndex (counting from right) is 1. */
bool rsIsBit1(int a, int bitIndex)
{
  return (a >> bitIndex) & 1;
}

/** Returns the index of the rightmost 1 bit, counting from right to left. If all bits are zero, it
returns -1 as code for "none". */
int rsRightmostBit(int n)
{
  int mask = 1;
  int numBits = 8 * sizeof(int);
  for(int i = 0; i < numBits; i++) {
    if(n & mask)
      return i;
    mask = (mask << 1) + 1; }  // shift one bit to left and fill set righmost bit to 1
  return -1;
}
// these 3 functions may be moved into rapt, into a rsBitTwiddling class (together with the 
// existing bit-masking and twiddling stuff that currently exists as free functions)

// -1 if a < b, +1 if a > b, 0 if a == b
int rsCompareByRightmostBit(int a, int b)
{
  int numBits = 8 * sizeof(int);
  for(int i = 0; i < numBits; i++)
  {
    bool aiIs1 = rsIsBit1(a, i);
    bool biIs1 = rsIsBit1(b, i);
    if(aiIs1 && !biIs1) return -1;
    if(biIs1 && !aiIs1) return +1;
  }
  return 0;
}
// maybe make internal to rsBitLess

/** A function used as comparison function for sorting the blades into their more natural 
order. This is needed to establish the mapping between the natural index of a basis blade and its 
bitmap representation. A given basis blade's bitmap has a 1 bit for every basis vector that is 
present in the basis blade and a 0 bit for every basis vector that is absent from the basis blade.
todo: document, how this mapping works  */
bool rsBitLess(const int& a, const int& b)  // rename
{
  int na = rsBitCount(a);
  int nb = rsBitCount(b);
  if(na < nb)
    return true;
  else if(nb < na)
    return false;
  else
  {
    int c = rsCompareByRightmostBit(a, b);
    if(c == -1)
      return true;
    else
      return false;
  }
}
// todo: test this, maybe move into a static member function of rsGeometricAlgebra - i don't think
// this will be useful in any other context


/** Class to represent a geometric algebra of an n-dimensional vector space. The main purpose of 
this class is to define, how the various products (geometric, outer, contraction, dot, ...) of two 
multivectors are calculated. Different algebras may have different rules and these rules are 
embodied in the Cayley tables that are member variables of this class and are built in the 
constructor according to the desired signature of the algebra (see below). rsGradedVector and 
rsMultiVector objects need to store a pointer to such an rsGeometricAlgebra object in order to 
delegate the actual computations in their multiplication operators to it. To the constructor of the 
algebra object, you need to pass 3 numbers that determine the number of basis vectors that square 
to +1, -1 and 0 respectively. This triple of numbers is called the signature of the algebra. For 
example, the usual 3D Euclidean space has signature (3,0,0), 4D Minkowski spacetime has signature 
(3,1,0) or (1,3,0) depending on the convention used. I prefer the (3,1,0) convention because it 
just takes the regular 3D space as is and extends it by a time dimension whereas the (1,3,0) 
convention additionally flips all spatial axes (or does it? it actually flips the squares of those
axes, i think - making them all imaginary? ...like in quaternions?), which seems to me an unnatural 
thing to do (todo: figure out the rationale behind the (1,3,0) convention). ....tbc... */

template<class T>
class rsGeometricAlgebra
{

public:

  using Vec = std::vector<T>; // for conevenience

  /** Creates an algebra object with given signature, i.e. given number of basis vectors that 
  square to +1, -1 and 0 respectively. Note that even though the zero-dimensions are specified 
  last, their basis-vectors will appear first in the canoncial ordering, adhering to the convention
  used in the algebra creation tool at bivector.net. */
  rsGeometricAlgebra(int numPositiveDimensions, int numNegativeDimensions = 0, 
    int numZeroDimensions = 0);
  // see https://en.wikipedia.org/wiki/Metric_signature

  //-----------------------------------------------------------------------------------------------
  /** \name Inquiry */

  /** Returns the dimensionality of the vector space. */
  int getNumDimensions() const { return n; }

  /** Returns the size of the multivectors, i.e. the dimensionality of the vector space. If the
  dimensionality of the underlying vectors space is n, then this is 2^n. */
  int getMultiVectorSize() const { return N; }

  /** Returns the size (i.e. number of elements) of a blade of given grade. This is the n-choose-k
  binomial coefficient where k is the grade and n the dimensionality of the underlying vector 
  space. */
  int getBladeSize(int grade) const
  { rsAssert(grade >= 0 && grade <= n); return bladeSizes[grade]; }

  /** Returns the array index at which the coefficients for a given grade begin in a multivector 
  that is represented by an array of scaling coefficients for each of the basis blades. */
  int getBladeStart(int grade) const
  { rsAssert(grade >= 0 && grade <= n); return bladeStarts[grade]; }


  const rsMatrix<int>& getCayleyTableIndices() const { return bladeIndices; } 

  const rsMatrix<T>& getCayleyTableWeightsGeom() const { return weightsGeom; } 

  const rsMatrix<T>& getCayleyTableWeightsOuter() const { return weightsOuter; } 


  //-----------------------------------------------------------------------------------------------
  /** \name Computation of products */

  /** Given multivectors a and b, this computes their geometric product and stores it in p. */
  void geometricProduct(const Vec& a, const Vec& b, Vec& p) const { product(a,b,p, weightsGeom); }
  // rename to geometricProductMM

  /** Given multivectors a and b, this computes their outer product and stores it in p. */
  void outerProduct(const Vec& a, const Vec& b, Vec& p) const { product(a,b,p, weightsOuter); }
  // rename to outerProductMM

  void contractLeftMM(const Vec& a, const Vec& b, Vec& p) const 
  { product(a,b,p, weightsContractLeft); }

  void contractRightMM(const Vec& a, const Vec& b, Vec& p) const 
  { product(a,b,p, weightsContractRight); }

  void scalarProductMM(const Vec& a, const Vec& b, Vec& p) const { product(a,b,p, weightsScalar); }

  void dotProductMM(const Vec& a, const Vec& b, Vec& p) const { product(a,b,p, weightsDot); }

  void fatDotProductMM(const Vec& a, const Vec& b, Vec& p) const { product(a,b,p, weightsFatDot); }

  void commutatorProductMM(const Vec& a, const Vec& b, Vec& p) const 
  { product(a,b,p, weightsCommutator); }



  /** Given multivectors a and b, this computes their inner product and stores it in p. */
  void innerProduct(const Vec& a, const Vec& b, Vec& p) const { product(a,b,p, weightsInner); }
  // maybe remove



  /** Given blades a and b, this computes their outer product and stores it in p. */
  void outerProduct_bld_bld(const Vec& a, int ga, const Vec& b, int gb, Vec& p) const 
  { product_bld_bld(a, ga, b, gb, p, weightsOuter); }
  // rename to outerProductGG



  /** Given an n x n matrix A, this function returns the unique N x N matrix (N=2^n) that extends
  the linear transformation of vectors that is defined by A to the induced linear transformation
  of multivectors for which holds: A(x ^ y) = A(x) ^ A(y), i.e. a transformation that preserves the
  outer product...tbc...  */
  rsMatrix<T> makeOutermorphism(const rsMatrix<T>& A);


  /** Under construction....
  A more general function to extend non-square mappings of vectors: R^m -> R^n to multivectors. */
  static rsMatrix<T> makeOutermorphism(const rsMatrix<T>& F, 
    const rsGeometricAlgebra<T>* srcAlg, const rsGeometricAlgebra<T>* dstAlg);




protected:

  //-----------------------------------------------------------------------------------------------
  /** \name Internals */

  /** Counts the number of basis vector swaps required to get (the concatenation of?) "a" and "b" 
  into canonical order and returns -1, if this number is odd and +1 if it's even. Arguments "a" and
  "b" are both bitmaps representing basis blades. This is used to build the Cayley table for the 
  gerometric products (and it's derived outer and inner products).
  adapted from "Geometric Algebra for Computer Science", page 514   */ 
  static int reorderSign(int a, int b);
  // rename to reorderSign

  /** Computes the (geometric or outer) product of the two basis blades represented by the integers
  a and b and their associated weights wa, wb and stores the result in ab/wab. 
  adapted from "Geometric Algebra for Computer Science", page 515  */
  static void basisBladeProduct(int a, T wa, int b, T wb, int& ab, T& wab, bool outer = false);

  static void basisBladeProduct(int a, int b, const std::vector<T>& M, int& ab, T& w);


  /** Creates the forward and backward mapping between basis blade indices and their bitmap 
  representations. map[i] expects an basis blade index i in their natural order and returns the 
  corresponding bitmap whereas unmap[i] expects the bitmap and returns the corresponding index. N
  ist the number of basis blades which is 2^n for an nD space. */
  static void reorderMap(std::vector<int>& map, std::vector<int>& unmap, int N);
  // find better name

  /** Builds the 2^n x 2^n matrices that define the multiplication tables for the basis blades for 
  the geometric algebra nD Euclidean space. */
  void buildCayleyTables(std::vector<T>& M, rsMatrix<int>& blades, 
    rsMatrix<T>& weightsGeom, rsMatrix<T>& weightsOuter, rsMatrix<T>& weightsInner);

  /** Computes the (i,j)th elements of the Cayley tables (i.e. multiplication tables) for the 
  geometric, outer and inner products. The metric is a vector containing the diagonal elements of 
  the desired metric tensor (todo: support non-diagonal matrics). 
  todo: document map/unmap parameters   */
  static void cayleyTableEntries(int i, int j, 
    const std::vector<T>& metric, const std::vector<int>& map, const std::vector<int>& unmap, 
    int& blade, T& wGeom, T& wOuter, T& wInner);

  /** General product of two multivectors a,b with weights passed as matrix. */
  void product(const Vec& a, const Vec& b, Vec& p, const rsMatrix<T>& weights) const;
  // rename to product_mv_mv, standing for multivector * multivector and implement also 
  // product_bld_bld for the product between two blades...and then also product_mv_bld and
  // product_bld_mv

  /** General product of two blades a,b of grades ga,gb with weights passed as matrix. */
  void product_bld_bld(const Vec& a, int ga, const Vec& b, int gb, Vec& p, 
    const rsMatrix<T>& weights) const;

  /** Initializes the object. The main task is to fill the various Cayley tables for the various
  products between two multivectors. */
  void init();

  /** Returns the bitmap corresponding to th i-th basis blade where i = 0..N-1. */
  int getBladeBitmap(int i) const { return map[i]; }

  /** Returns the bitmap of the j-th blade of grade k where k = 0..n, j = 0..n-choose-k */
  int getBladeBitmap(int k, int j) const 
  { 
    rsAssert(k >= 0 && k <= n,             "Invalid grade");
    rsAssert(j >= 0 && j <  bladeSizes[k], "Invalid blade (sub)index");
    int i = bladeStarts[k] + j;  // verify this!
    return map[i]; 
  }



  //-----------------------------------------------------------------------------------------------
  /** \name Data */

  int np = 0;  // number of basis vectors that square to +1 ("positive dimensions", space-like)
  int nn = 0;  // number of basis vectors that square to -1 ("negative dimensions", time-like)
  int nz = 0;  // number of basis vectors that square to  0 ("zero dimensions", light-like? projective? degenerate?)
  int n  = 0;  // n = np + nn + nz, the dimensionality of the vector space
  int N  = 1;  // N = 2^n, the dimensionality of the multivector space

  // The Cayley tables for the various products:
  rsMatrix<int> bladeIndices;
  rsMatrix<T> weightsGeom, weightsOuter, weightsInner; // maybe remove/rename weightsInner
  rsMatrix<T> weightsContractLeft, weightsContractRight, weightsScalar, weightsDot, weightsFatDot,
    weightsCommutator, weightsRegressive;

  // Sizes and start-indices of the coeffs of the blades of grades 1..n (or 0..n?)
  std::vector<int> bladeSizes, bladeStarts;

  // Bitmaps for the basis blades ...tbc...
  std::vector<int> map, unmap; // rename to bladeBitmaps, bladeBitmapsInv

  // Diagonal matrices for the involutions:
  std::vector<T> involutionGrade, involutionReverse, involutionConjugate;

  // Permutation and sign tables for dualization:
  //std::vector<int> dualPerm;  // the permutation is just a reversal
  std::vector<T>   dualSigns;




  // ToDo: maybe have temp arrays of size N for performing certain computations that would 
  // otherwise need to allocate memory

  // They need access to our internals:
  template<class U> friend class rsGradedVector;
  template<class U> friend class rsMultiVector;

};

//=================================================================================================

/** A class to represent k-vectors in n-dimensional space where the number k is called the grade of
the object. For example, in 3D space, we have scalars (0-vectors), vectors (1-vectors), bivectors 
(2-vectors) and trivectors (3-vectors). A k-vector is represented by a set of coefficients that 
scale the contributions of the relevant k-dimensional basis blades (a k-blade is a special kind of 
k-vector, namely one that can be factored into an outer product of k 1-vectors). A 3D scalar is 
just one coefficient that can be seen as scaling the unit scalar 1, a 3D vector has 3 coefficients 
for the 3 basis vectors (e1,e2,e3), a 3D bivector has coefficients for the 3 basis bivectors 
(e12,e13,e23) and a 3D trivector has 1 coefficient that scales the single basis trivector (e123) 
which represents a unit volume in 3D and is also known as the unit pseudoscalar or sometimes unit 
antiscalar. The highest possible grade is always equal to the dimension n of the vector space, the 
number of coefficients for a particular grade k is always the binomial coefficient "n-choose-k" and
the highest grade basis blade is always the unit pseudoscalar. In 2D space, the unit pseudoscalar 
behaves similar to the imaginary unit i in complex numbers...but i think, not exactly like it, 
because it behaves anticommutative rather than commutative? -> figure out

k-vectors of the same grade can be added or subtracted element-wise, but when an i-vector should
be added to a j-vector with i != j, the result will be of mixed grade, which means in practice that
the grade i and j components are just kept separate pretty much like real and imaginary parts in 
complex numbers z = a + b*i. These (more general) mixed-grade objects are called multivectors and 
to represent them, there's another class rsMultiVector defined below. Besides addition and 
subtraction, there are two main multiplication operations defined between k-vectors: the geometric
product and the outer product (a.k.a. wedge- or exterior product). When we multiply an i-vector and
a j-vector via the wedge product, the result will be a k-vector with k = i+j, i.e. an object of the 
same kind, albeit of different (higher) grade in general. When combining an i-vector and j-vector 
via the geometric product, the result is actually in general not a k-vector anymore but a more 
general multivector, i.e. an object of mixed grade. That means, geometric algebra with multivectors
is the more general structure but when we restrict the computations to using only k-vectors (rather
than general multivectors) and using only the outer "wedge" product (and not also the geometric 
product), we actually get a subalgebra of the geometric algebra which is known as exterior algebra. 

Note:
Graded vectors are actually a special case of multivectors, so at first glance, it may seem 
appropriate to let rsGradedVector be a subclass of rsMultiVector. However, i opted not to do this 
because multivectors can do many more things, which a k-vector should really not inherit. They are 
really a restricted special case, i.e. more like a subset rather than a subclass. But where there 
functionality overlaps, both classes provide the same API. 

References:
  https://en.wikipedia.org/wiki/Blade_(geometry)
  https://en.wikipedia.org/wiki/Exterior_algebra
  https://en.wikipedia.org/wiki/Multivector */

template<class T>
class rsGradedVector
{

public:

  //-----------------------------------------------------------------------------------------------
  /** \name Lifetime */

  /** Default constructor. */
  rsGradedVector() {}

  /** Constructor. */
  rsGradedVector(const rsGeometricAlgebra<T>* algebraToUse, int grade)
  { this->grade = grade; setAlgebra(algebraToUse); }


  rsGradedVector(const rsGeometricAlgebra<T>* algebraToUse, int grade,
    const std::vector<T>& coeffs)
  { this->grade = grade; setAlgebra(algebraToUse); set(grade, coeffs); }

  /** Factory function. Constructs the basis blade with given index which ranges from 0 to N-1 
  where N = 2^n. The grade of the blade is inferred automatically from the index. */
  //static rsGradedVector<T> basisBlade(
  //  const rsGeometricAlgebra<T>* algebraToUse, int index);

  //-----------------------------------------------------------------------------------------------
  /** \name Setup */

  /** Sets the geometric algebra to use for operations involving this multivector. */
  void setAlgebra(const rsGeometricAlgebra<T>* algebraToUse)
  { alg = algebraToUse; coeffs.resize(alg->getBladeSize(grade)); }

  void setCoefficients(const T* newCoeffs)
  { for(size_t i = 0; i < coeffs.size(); i++) coeffs[i] = newCoeffs[i]; }

  /** Sets the blade's grade and the coefficients for the projections onto the basis blades of the
  given grade. There are n-choose-k of them (n: dimensionality, k: grade), so the newCoeffs vector
  is supposed to match that size. */
  void set(int newGrade, const std::vector<T>& newCoeffs);

  //-----------------------------------------------------------------------------------------------
  /** \name Inquiry */

  /** Returns a const pointer to the std::vector of coefficients. */
  const std::vector<T>& getCoeffs() const { return coeffs; }

  int getGrade() const { return grade; }

  T getCoefficient(int i) const { return coeffs[i]; }

  /** Returns a const pointer to the algebra object that is used. */
  const rsGeometricAlgebra<T>* getAlgebra() const { return alg; }


  //-----------------------------------------------------------------------------------------------
  /** \name Operators */

  // todo: +,-,/

  /** Outer (aka wedge, exterior) product between this k-vector and k-vector b (possibly with a 
  different k). Beware of the operator precendence rules of C++: the caret operator has a very low
  precendence, so you will probably need to use parentheses a lot. */
  rsGradedVector<T> operator^(const rsGradedVector<T>& b) const;


  bool operator==(const std::vector<T>& b) const { return coeffs == b; }


  /** Read and write access to i-th coefficient. */
  T& operator[](int i) { rsAssert(i >= 0 && i < (int)coeffs.size()); return coeffs[i]; }

  /** Read access to i-th coefficient. */
  const T& operator[](int i) const { rsAssert(i >= 0 && i < (int)coeffs.size()); return coeffs[i]; }




  static bool areCompatible(const rsGradedVector<T>& a, const rsGradedVector<T>& b)
  { return a.alg == b.alg; }


protected:

  std::vector<T> coeffs;      // n-choose-k coeffs for the projections on the basis blades
  const rsGeometricAlgebra<T>* alg = nullptr;
  int grade = 0;


  //template<class U> friend class rsGeometricAlgebra;
  //template<class U> friend class rsMultiVector;

};

//=================================================================================================

/** Class to represent multivectors. A multivector can be seen as a formal sum of k-vectors of 
various grades. For example, in 3D, a general multivector M is a sum: M = S + V + B + T where
where S,V,B,T are the scalar, vector, bivector and trivector components of the multivector. The 
addition is not actually "evaluated" because you can't really add a scalar to a vector and so on. 
It is rather meant in the sense of putting things together but keeping them as seperate parts of a
single entity, very much like the real part a and the imaginary part b in z = a + b*i in the 
representation of complex numbers. You never actually evaluate any additions in expressions like 
this but just keep the coefficients a,b separate. It's the same thing here, just the V and B parts
have more than one component themselves (their components are coefficients that scale the basis 
vectors and basis bivectors respectively). Internally, the multivector is just stored as a flat 
array of coefficients for all the basis blades of various grades. In an n-dimensional vector space,
there are 2^n such basis blades in total and a blade of grade k gets n-choose-k of them. A k-vector 
can be seen as a special kind of multivector that has nonzero values only for coefficients that 
belong to the same grade, i.e. it doesn't mix up  basis blades of different grades.

The main operation between multivectors is the geometric product which is implemented via the usual
multiplication operator *. The outer or wedge product that was defined for k-vectors as ^ operator 
does also still exist for general multivectors. The geometric product is defined via a sum of the
inner and outer product of vectors: a*b = a|b + a^b, where the | operator is used here to denote 
the inner product, i.e. the scalar product. Note that a and b are supposed to be 1-vectors here and 
not general k-vectors or even more general multivectors. This equation may not hold anymore for 
these more general objects. In fact, there doesn't even seem to be a general consensus for how the 
inner product should be defined for those more general objects, which is why the operator is not 
yet implemented here (Q: could it perhaps be *defined* by this equation - would that be any 
useful?). Of course, the basic operations of addition and subtraction, which just operate 
element-wise, also still exist.

...todo: inversion, division, join, meet, rotor, projection, rejection, reflection, 
exp, sin, cos, log, pow, sqrt  */

template<class T>
class rsMultiVector
{

public:

  rsMultiVector(const rsGeometricAlgebra<T>* algebraToUse)
  { setAlgebra(algebraToUse); }

  rsMultiVector(const rsGradedVector<T>& b) { set(b); }

  rsMultiVector(const rsGeometricAlgebra<T>* algebraToUse, const std::vector<T>& coeffs)
  { setAlgebra(algebraToUse); set(coeffs); }



  //-----------------------------------------------------------------------------------------------
  /** \name Setup */

  /** Sets the geometric algebra to use for operations involving this multivector. */
  void setAlgebra(const rsGeometricAlgebra<T>* algebraToUse)
  { alg = algebraToUse; coeffs.resize(alg->getMultiVectorSize()); }

  /** Sets the coefficients to random integers in the given range. */
  void randomIntegers(int min, int max, int seed)
  { RAPT::rsArrayTools::fillWithRandomIntegers(&coeffs[0], (int)coeffs.size(), min, max, seed); }
  // maybe rename to randomize and let the rounding to integers be optional, controlled by a bool
  // parameter

  void set(const rsGradedVector<T>& b);

  void set(const std::vector<T>& newCoeffs)
  { rsAssert(newCoeffs.size() == coeffs.size()); coeffs = newCoeffs; }

  /** Scales this multivector by a scalar scaler. */
  void scale(const T& scaler) { rsScale(coeffs, scaler); }

  /** Sets this multivector to zero. */
  void setZero() { rsFill(coeffs, T(0)); }

  /** Sets this multivector to the scalar s. */
  void setToScalar(const T& s) { rsFill(coeffs, T(0)); coeffs[0] = s; }

  //-----------------------------------------------------------------------------------------------
  /** \name Inquiry */

  /** Returns a const pointer to the std::vector of coefficients. */
  const std::vector<T>& getCoeffs() const { return coeffs; }

  /** Extracts the part of a given grade that is present in this multivector. For example, 
  M.extractGrade(2) would extract the bivector part of a given multivector M. */
  rsGradedVector<T> extractGrade(int grade) const;

  /** Returns a const pointer to the algebra object that is used. */
  const rsGeometricAlgebra<T>* getAlgebra() const { return alg; }

  /** Returns the number of dimensions n of the underlying vector space in which this multivector 
  lives. Not to be confused with the number of dimensions of the multivector space, which is 2^n.*/
  int getNumDimensions() const { return alg->getNumDimensions(); }

  /** Returns the number of coefficients, i.e. the number of dimensions of the multivector space 
  which is 2^n where n is the dimensionality of the underlying vector space. */
  int getNumCoeffs() const { return alg->getMultiVectorSize(); }

  /** Returns true, iff this multivector has at least one nozero coefficient (with some tolerance) 
  for the given grade k */
  bool containsGrade(int k, T tol = T(0)) const;

  /** Returns the lowest grade that is present in this multivector. */
  int getLowestGrade(T tol = T(0)) const;
  // todo: what if the multivector is zero? should that count as zero grade (with scalar coeff 
  // zero) or be a special case (encoded by -1)? ...we'll see what's more convenient...

  /** Returns the highest grade that is present in this multivector. */
  int getHighestGrade(T tol = T(0)) const;

  /** Returns true, iff this multivector has nonzero coefficients only for a single grade. Such 
  multivectors are also called homogeneous. */
  bool isSingleGraded(T tol = T(0)) const { return getLowestGrade(tol) == getHighestGrade(tol); }

  /** Returns true, iff this multivector has nonzero coefficients only for the given grade k. */
  bool hasOnlyGrade(int k, T tol = T(0)) const 
  { return getLowestGrade(tol) == k && getHighestGrade(tol) == k; }

  /** Returns true, iff this multivector has all zero coefficients. */
  bool isZero(T tol = T(0)) const { return rsMaxAbs(coeffs) > tol; }

  /** Returns true, iff this multivector has nonzero coefficients only for the scalar (grade 0) 
  part.  ...i'm not yet sure, if 0 should be considered as the scalar 0 (of grade 0) or as special 
  case (of grade -1) - currently zero is treated as scalar, too...  */
  bool isScalar(T tol = T(0)) const { return hasOnlyGrade(0, tol); }

  /** Returns true, iff this multivector has nonzero coefficients only for the vector (grade 1) 
  part. */
  bool isVector(T tol = T(0)) const { return hasOnlyGrade(1, tol); }

  /** Returns true, iff this multivector has nonzero coefficients only for the bivector (grade 2) 
  part. */
  bool isBiVector(T tol = T(0)) const { return hasOnlyGrade(2, tol); }

  /** Returns true, iff this multivector has nonzero coefficients only for the trivector (grade 3) 
  part. */
  bool isTriVector(T tol = T(0)) const { return hasOnlyGrade(3, tol); }

  /** Returns true, iff this multivector has nonzero coefficients only for the pseudovector 
  (grade n-1) part. */
  bool isPseudoVector(T tol = T(0)) const { return hasOnlyGrade(getNumDimensions()-1, tol); }

  /** Returns true, iff this multivector has nonzero coefficients only for the pseudoscalar
  (grade n) part. */
  bool isPseudoScalar(T tol = T(0)) const { return hasOnlyGrade(getNumDimensions(), tol); }

  // todo: isBlade, isVersor

  /** Returns true, iff this multivector is numerically close the given multivector rhs within a 
  given (absolute) tolerance tol. */
  bool isCloseTo(const rsMultiVector<T>& rhs, T tol = T(0)) const
  { return rsIsCloseTo(coeffs, rhs.coeffs, tol); }

  // ToDo:
  // isBlade, isVersor. Maybe there should be algorithms to figure this out (GAfCS pg 46 says that 
  // an algorithm for this is "not elementary") as well as flags as 
  // members that keep track of the property during the construction of the multivector. For 
  // example, scalars vectors, pseudo-vectors and pseudo-scalars are blades. Forming an outer 
  // product of two blades gives again a blade. So, the outer product could assign the isBlade flag
  // of the result as boolean and of the two inputs. All other operators (add, subtract, geometric 
  // product, etc.) produce, in general, non-blades when getting blades as inputs. k-blades are 
  // sometimes called simple k-vectors. They are factorizable as an outer product of vectors.
  // Maybe wee need factory functions like makeScalar, makeVector, makePseudoVector, 
  // makePseudoScalar which would init the isBlade flag as true. Blades and versors. GAfCS, pg 529
  // says that almost all elements constructed in geometric algebra actually are blades or versors
  // and for those, there's a simpler algorithm for inversion. pg 191 expalins what versors are:
  // a k-versor is formed as a geometric product of k invertible vectors. They are typically used 
  // in a sandwich product, called the verso product, like Y = +- V X /V where V is the versor and 
  // /V means "divide by V" i.e. multiply by Vs inverse. Versors represent orthogonal 
  // transformation.
  // maybe have also: isRotor, isSpinor - see GAfCS pg 195 ff., pg 531 says versors and blades have
  // scalar squares, so that test might go into an algo that figures out bladeness and versorness.
  // Ch 21.5 also talks about this.
  // Maybe we could have functions like isBladeByConstruction/isVersorByConstruction that just 
  // check the flags. The general isBlade, isVersor, etc. functions would run the classification
  // algorithm



  T getSquaredVectorNorm() const;


  /** Returns the reverse of this multivector. */
  rsMultiVector<T> getReverse() const { rsMultiVector<T> Y(*this); Y.applyReversal(); return Y; }
  // ...i think, this may be applicable only to blades? figure out -> if so, maybe use 
  // rsAssert(isBlade())

  rsMultiVector<T> getGradeInvolution() const 
  { rsMultiVector<T> Y(*this); Y.applyGradeInvolution(); return Y; }

  /** Returns the Clifford conjugate of this multivector. */
  rsMultiVector<T> getConjugate() const 
  { rsMultiVector<T> Y(*this); Y.applyConjugation(); return Y; }

  /** Returns the dual of this multivector. If V is a homogeneous graded vector of grade k, then 
  its dual V* is of grade n-k and represents the orthogonal complement subspace of the space 
  spanned by V. The dual can be computed via V* = V / I where I is the unit pseudoscalar. */
  rsMultiVector<T> getDual() const;

  /** Returns the matrix representation of this multivector. The matrix A that corresponds to a 
  general multivector a, is the matrix for which the matrix-(multi)vector product A*b is the same
  as the geometric product a*b. */
  rsMatrix<T> getMatrixRepresentation() const;
  // todo: let the user select which of the products should be realized 

  /** Returns the inverse of this multivector. This is the general implementation based on solving
  a linear NxN system (N=2^n), so it's quite expensive. If possible, use the more efficient 
  implementations for special cases such as getInverseScalar, getInverseVector, etc. Note that an 
  inverse may not exist, in which case the linear solver will encounter a singular matrix and 
  return garbage without warning. */
  rsMultiVector<T> getInverse() const;

  rsMultiVector<T> getInverseScalar() const;

  rsMultiVector<T> getInverseVector() const;

  // todo:
  //rsMultiVector<T> getInverseVersor() const; // V^-1 = rev(V) / (V * rev(V))  GA4CS, pg 530
  //rsMultiVector<T> getInverseBlade() const; // pg 79
  //rsMultiVector<T> getInversePseudoScalar() const;
  //rsMultiVector<T> getInversePseudoVector() const;


  // todo: getInverse() - implement special formulas where available, like for scalars, vectors,
  // blades, etc. ...do we need to distinguish between left and right inverses? i think so. maybe
  // in the general case, we need to express the C = M*B as C: vector, M: matrix, B: vector and 
  // solve a linear system? the matrix M should be obtained from a multivector A by taking into 
  // account A itself and the Cayley table?
  // https://math.stackexchange.com/questions/443555/calculating-the-inverse-of-a-multivector/2985578
  // https://arxiv.org/abs/1712.05204
  // maybe write getScalarInverse, getVectorInverse, getBladeInverse, getGeneralLeftInverse,
  // getGeneralRightInverse

  //-----------------------------------------------------------------------------------------------
  /** \name Manipulations. ...that modify the multivector in place */

  /** Applies reversal of this multivector in place. */
  void applyReversal() { applyInvolution(alg->involutionReverse); }

  void applyGradeInvolution() { applyInvolution(alg->involutionGrade); }

  void applyConjugation() { applyInvolution(alg->involutionConjugate); }

  // these 3 involutions together with the identity constitute the Klein 4 group

  void applyDualization();

  // todo: applyScalarInversion, applyVectorInversion, applyBladeInversion

  //-----------------------------------------------------------------------------------------------
  /** \name Operators */

  /** Adds two multivectors. */
  rsMultiVector<T> operator+(const rsMultiVector<T>& b) const;

  /** Subtracts two multivectors. */
  rsMultiVector<T> operator-(const rsMultiVector<T>& b) const;

  /** Geometric product between this multivector and multivector b. This is the main operation that
  makes geometric algebra tick. */
  rsMultiVector<T> operator*(const rsMultiVector<T>& b) const;

  /** Divides this multivectors by multivector b. */
  rsMultiVector<T> operator/(const rsMultiVector<T>& b) const { return *this * b.getInverse(); }

  /** Adds multivector b to this multivector. */
  rsMultiVector<T>& operator+=(const rsMultiVector<T>& b);
  // todo: create function that adds a blade into a multivector

  /** Product between this multivector and scalar s. */
  rsMultiVector<T> operator*(const T& s) const;

  /** Outer (aka wedge, exterior) product between this multivector and multivector b. For vectors 
  a and b, this is the antisymmetric part of the geometric product: a^b = (a*b-b*a)/2, but this 
  identity does not generalize to multivectors. */
  rsMultiVector<T> operator^(const rsMultiVector<T>& b) const;

  /** Inner product between this multivector and multivector b. For vectors a and b, this is the 
  symmetric part of the geometric product: a|b = (a*b+b*a)/2, but this identity does not generalize
  to multivectors.  
  Maybe don't implement that operator just yet - there seems to be no general consensus yet, how
  the inner product should be defined. For example, bivector.net uses the "fat dot" definition, 
  Alan Macdonald uses the "left contraction" definition. We implement both of these and more via 
  the product function defined below - users should be explicit, which kind of "inner product" they
  want.  */
  //rsMultiVector<T> operator|(const rsMultiVector<T>& b) const;

  /** Adds a multivector and a scalar. */
  rsMultiVector<T> operator+(const T& s) const { rsMultiVector<T> R(*this); R[0] += s; return R; }

  /** Adds scalar s to this multivector. */
  rsMultiVector<T>& operator+=(const T& s) { coeffs[0] += s; }

  /** Subtracts the scalar s from a multivector. */
  rsMultiVector<T> operator-(const T& s) const { rsMultiVector<T> R(*this); R[0] -= s; return R; }

  /** Subtracts the scalar s from this multivector. */
  rsMultiVector<T>& operator-=(const T& s) { coeffs[0] -= s; }


  rsMultiVector<T>& operator*=(const rsMultiVector<T>& b) { *this = *this * b; return *this; }
  // optimize! this can be implemented without memory allocation by having a temp-array member in 
  // the algebra object, or maybe have a multiplication function that takes a workspace pointer


  rsMultiVector<T>& operator*=(const T& s) 
  { rsArrayTools::scale(&coeffs[0], getNumCoeffs(), s); return *this; }


  rsMultiVector<T>& operator/=(const T& s) 
  { rsArrayTools::scale(&coeffs[0], getNumCoeffs(), T(1)/s); return *this; }


  rsMultiVector<T> operator-()
  { rsMultiVector<T> r = *this; RAPT::rsNegate(r.coeffs); return r; }

  bool operator==(const rsMultiVector<T>& b) const { return coeffs == b.coeffs; }


  bool operator==(const std::vector<T>& b) const { return coeffs == b; }


  void operator=(const rsGradedVector<T>& b) { set(b); }


  /** Read and write access to i-th coefficient. */
  T& operator[](int i) { rsAssert(i >= 0 && i < (int)coeffs.size()); return coeffs[i]; }

  /** Read access to i-th coefficient. */
  const T& operator[](int i) const { rsAssert(i >= 0 && i < (int)coeffs.size()); return coeffs[i]; }


  static bool areCompatible(const rsMultiVector<T>& a, const rsMultiVector<T>& b)
  { bool ok = a.coeffs.size() == b.coeffs.size(); ok &= a.alg == b.alg; return ok; }
  // todo: try to relax that later: a 3D geometric algebra contains a 2D one as subalgebra
  // i think, it's enough to test a.alg == b.alg because the size of the coeffs is determined by
  // the algebra object ...at least it should be


  //-----------------------------------------------------------------------------------------------
  /** \name Functions */

  enum class ProductType
  {
    wedge,
    contractLeft,  // "contraction onto"
    contractRight, // "contraction by"
    scalar,        // for blades, nonzero only when the two blades have same grade
    dot,
    fatDot,
    commutator,    // is a "derivation", obeys product rule
    regressive
  };
  // Maybe this should be defined in class rsGeometricAlgebra - then we could have a function
  // rsMatrix<T> getCalyeyTable(ProductType type) there. This could be interesting for client code

  /** Computes one of the several products that can be derived from the geometric product. Which 
  one it is is selected by the product type parameter. See:
  https://en.wikipedia.org/wiki/Geometric_algebra#Extensions_of_the_inner_and_exterior_products */
  static rsMultiVector<T> product(const rsMultiVector<T>& A, const rsMultiVector<T>& B,
    ProductType type);


protected:

  /** This function is used in rsGeometricAlgebra::init to build the Calyey tables for the derived 
  products. */
  static rsMultiVector<T> productSlow(const rsMultiVector<T>& A, const rsMultiVector<T>& B, 
    ProductType type);

  /** Applies the involution defined by the vector inv to this multivector. inv can be something 
  like alg->involutionReverse */
  void applyInvolution(const std::vector<T>& inv);


  std::vector<T> coeffs;                      // 2^n coeffs for the projections on the basis blades
  const rsGeometricAlgebra<T>* alg = nullptr; // pointer to the algebra to use


  template<class U> friend class rsGeometricAlgebra;
  //template<class U> friend class rsGradedVector;

  // ToDo:
  // -maybe make a class rsSparseMultiVector that contains coeffs for a particular grade only if
  //  at least one of them is nonzero - for example, the product of two vectors contains only 
  //  grades 0 and 2. maybe use a std::vector<rsGradedVector<T>> grades;

};

//=================================================================================================
// Implementations of member functions and operators of classes rsGeometricAlgebra, rsGradedVector,
// rsMultiVector. The classes are so intertwined that some implementations need to know the full
// declaration of another class, so we put the implementations all together:

//-------------------------------------------------------------------------------------------------
// rsGeometricAlgebra:

template<class T>
rsGeometricAlgebra<T>::rsGeometricAlgebra(int numPos, int numNeg, int numZero)
{
  np = numPos;
  nn = numNeg;
  nz = numZero;
  n  = np + nn + nz;
  N  = RAPT::rsPowInt(2, n);
  init();
}

template<class T>
int rsGeometricAlgebra<T>::reorderSign(int a, int b)
{
  a = a >> 1;
  int sum = 0;
  while (a != 0) {
    sum = sum + rsBitCount(a & b);
    a = a >> 1; }
  return ((sum & 1) == 0) ? 1 : -1; //  even number of swaps: 1, else -1
}

template<class T>
void rsGeometricAlgebra<T>::basisBladeProduct(
  int a, T wa, int b, T wb, int& ab, T& wab, bool outer) 
{
  ab = a ^ b;                     // compute the product bitmap
  if(outer && ((a & b) != 0)) {   // if a and b are linearly independent...
    wab = T(0); return; }         // ...their outer product is zero
  int sign = reorderSign(a, b);   // compute the sign change due to reordering
  wab = T(sign) * wa * wb;        // compute weight of product blade
}

template<class T>
void rsGeometricAlgebra<T>::basisBladeProduct(int a, int b, const std::vector<T>& M, int& ab, T& w)
{
  basisBladeProduct(a, T(1), b, T(1), ab, w, false);
  int tmp = a & b;       // compute the meet (bitmap of annihilated vectors)
  int i = 0;
  while(tmp != 0) {
    if((tmp & 1) != 0) 
      w *= M[i];         // change the scale according to the metric
    i++;
    tmp = tmp >> 1; }
}
// see:
// https://bivector.net/PGA4CS.html
// pg 516 of GA4CS, or BasisBlade.java in referecne implementation, line 206

template<class T>
void rsGeometricAlgebra<T>::reorderMap(std::vector<int>& map, std::vector<int>& unmap, int N)
{
  map.resize(N);
  unmap.resize(N);
  for(int i = 0; i < N; i++)
    map[i] = i;
  rsHeapSort(&map[0], N, &rsBitLess);
  for(int i = 0; i < N; i++)
    unmap[map[i]] = i;

  // Creates a map of the basis blades that converts from their bit-based to the more natural 
  // ordering. In the bit based ordering, a bit with value 1 in the integer indicates the presence
  // of a particular basis-vector in the given basis blade. In the natural ordering, we want the 
  // scalar first, then the bivectors and then the trivectors, etc.
  //
  // 3D:
  // 0   1   2   3   4   5   6   7       array position/index
  // 1   e1  e2  e3  e12 e13 e23 e123    blade name
  // 000 001 010 100 011 101 110 111     binary code (has a 1 for each blade present)
  // 0   1   2   4   3   5   6   7       binary code translated to decimal number
  //
  // 4D:
  // 0    1    2    3    4    5    6    7    8    9    10   11   12   13   14   15
  // 1    e1   e2   e3   e4   e12  e13  e14  e23  e24  e34  e123 e124 e134 e234 e1234
  // 0000 0001 0010 0100 1000 0011 0101 1001 0110 1010 1100 0111 1011 1101 1110 1111
  // 0    1    2    4    8    3    5    9    6    10   12   7    11   13   14   15
}

template<class T>
void rsGeometricAlgebra<T>::cayleyTableEntries(int i, int j, 
  const std::vector<T>& metric, const std::vector<int>& map, const std::vector<int>& unmap,
  int& blade, T& wGeom, T& wOuter, T& wInner)
{
  int a = map[i];                             // bitmap representing 1st factor blade
  int b = map[j];                             // bitmap representing 2nd factor blade
  int ab;                                     // product blade bitmap
  basisBladeProduct(a, b, metric, ab, wGeom); // compute bitmap and weight for geometric product
  blade = unmap[ab];                          // blade index represented by product bitmap ab
  if(a & b) wOuter = T(0);                    // if (a & b) != 0, blades i,j are linearly dependent
  else      wOuter = wGeom;
  wInner = wGeom - wOuter;                    // naive ad-hoc definition, maybe useless...
  // ...there are many different possible definitions for the inner product of two multivectors. 
  // This here is based on the idea that the identity a*b = a|b + a^b (which holds for vectors), 
  // should continue to hold for general multivectors. I have no idea, if that leads to a useful 
  // concept, but some authors (Alan Macdonald) call that identity "fundamental", so it might be a 
  // good idea...we'll see...will this definition also obey the grade raising/lowering when applied
  // to blades? ...try it! ...but i can already say that it produces different results than what
  // we get on bivector.net, so it might be a "wrong" definition (definitions can't actually be 
  // "wrong" - only useful or useless)
}

template<class T>
void rsGeometricAlgebra<T>::buildCayleyTables(std::vector<T>& M, rsMatrix<int>& blades, 
  rsMatrix<T>& weightsGeom, rsMatrix<T>& weightsOuter, rsMatrix<T>& weightsInner)
{
  int n = (int) M.size();
  int N = rsPowInt(2, n);  // size of the matrices
  blades.setShape(N, N);
  weightsGeom.setShape(N, N);
  weightsOuter.setShape(N, N);
  weightsInner.setShape(N, N);

  //std::vector<int> map, unmap;
  reorderMap(map, unmap, N);    // to map back and forth between blade index and its bitmap
  // make them members bladeBitmaps, bladeBitmapsInv
  // they may be needed in some algorithms later

  for(int i = 0; i < N; i++) {
    for(int j = 0; j < N; j++) {
      int b;
      T wg, wo, wi;
      cayleyTableEntries(i, j, M, map, unmap, b, wg, wo, wi);
      blades(      i, j) = b;
      weightsGeom( i, j) = wg;
      weightsOuter(i, j) = wo;
      weightsInner(i, j) = wi; }}
}

template<class T>
void rsGeometricAlgebra<T>::init()
{
  // Create the metric for the given signature (maybe factor out into getMetric(p, n, z))
  std::vector<T> M(n);
  int i;
  for(i = 0; i < nz;       i++) M[i] = T( 0);
  for(i = i; i < nz+np;    i++) M[i] = T(+1);
  for(i = i; i < nz+np+nn; i++) M[i] = T(-1);
  // todo: maybe make the metric a member, allow for non-diagonal metrics. that requires a 
  // diagonalization of the metric before building the Cayley tables and some sort of 
  // "de-diagonalization" of the results... i'm not yet sure, how exactly that is supposed to work 
  // -> consult the GA4CS book and the author's reference implementation

  // Fill the basic Cayley (i.e. multiplication) tables for geometric, outer and inner product. 
  // These tables really define what the algebra does - they are the heart and soul of everything:
  buildCayleyTables(M, bladeIndices, weightsGeom, weightsOuter, weightsInner);

  // Create the information that is needed for extracting the coeffs for a particular grade:
  bladeSizes.resize(n+1);
  bladeStarts.resize(n+1);
  rsGetLineOfPascalTriangle(&bladeSizes[0], n);
  bladeStarts[0] = 0;
  for(i = 1; i <= n; i++)
    bladeStarts[i] = bladeStarts[i-1] + bladeSizes[i-1];

  // Build the additional Cayley tables for the derived products:
  using MV = rsMultiVector<T>;
  using PT = MV::ProductType;
  auto checkProduct = [&](const MV& P, int k)
  {
    // We assume here, that in every type of derived product, the result P has only one nonzero 
    // entry which occurs at index k (or none). This seems to be the case from observations but i 
    // have no proof for this -> figure out. But if it doesn't hold, the assertions here will 
    // catch it. In this case, we may need different "bladeIndices" tables for different products:
    int nnz = rsNumNonZeros(P.coeffs);          // number of nonzero entries in P
    int kt  = rsIndexOfFirstNonZero(P.coeffs);  // true index of the nonzero entry
    rsAssert(nnz == 1 || nnz ==  0);
    rsAssert(kt  == k || kt  == -1);
  };
  auto buildTable = [&](PT productType, RAPT::rsMatrix<T>& weights)
  {
    MV A(this), B(this), P(this);
    weights.setShape(N, N);
    for(int i = 0; i < N; i++) {
      for(int j = 0; j < N; j++) {
        // Set the i-th and j-th alement in A and B "hot", respectively and form the product:
        A.setZero(); A[i] = 1;
        B.setZero(); B[j] = 1;
        int k = bladeIndices(i, j);                 // assumed index of nozero element in product
        P = MV::productSlow(A, B, productType);
        checkProduct(P, k);                         // sanity check
        weights(i, j) = P[k];   }}  
  };
  buildTable(PT::contractLeft,  weightsContractLeft);
  buildTable(PT::contractRight, weightsContractRight);
  buildTable(PT::scalar,        weightsScalar);
  buildTable(PT::dot,           weightsDot);
  buildTable(PT::fatDot,        weightsFatDot);


  // todo: 
  // -create tables for commutator- and regressive product
  // -create permuatation and sign arrays for dualization
  MV A(this), B(this), P(this);

  // Create inverse of the unit pseudoscalar, given by the product en*...*e3*e2*e1*1:
  MV Ii(this);
  Ii[0] = T(1);                   // initialize product with unity
  for(int i = 1; i <= n; i++) {
    A.setZero(); A[i] = 1;
    Ii = A * Ii;  }
  // maybe Ii should be a member - but we can store it only as std::vector, not as rsMultiVector
  // because rsGeometricAlgebra doesn't really know about that class

  // Function to compute the dual via the formula with the inverse pseudoscalar, which must be 
  // passed via the parameter Ii:
  auto dual = [](const MV& A, const MV& Ii) { return A * Ii; };

  // Create sign table for the dualization:
  dualSigns.resize(N);
  for(int i = 0; i < N; i++)
  {
    A.setZero(); A[i] = 1;
    B = dual(A, Ii);
    int k = N-i-1;
    checkProduct(B, k);
    dualSigns[k] = B.coeffs[k];
  }


  // Function to compute the regressive product (a.k.a. antiwedge product) via the formula using
  // duals: A v B = (A* ^ B*)*:
  auto regressiveProduct = [&](const MV& A, const MV& B)
  {
    MV Ad = dual(A, Ii);  // dual of A
    MV Bd = dual(B, Ii);  // dual of B
    MV Pd = Ad ^ Bd;      // dual of product
    return dual(Pd, Ii);  // product
  };
  
  weightsCommutator.setShape(N, N);
  //weightsRegressive.setShape(N, N);
  for(int i = 0; i < N; i++) 
  {
    for(int j = 0; j < N; j++) 
    {
      A.setZero(); A[i] = 1;
      B.setZero(); B[j] = 1;
      int k = bladeIndices(i, j);
      P = T(0.5) * (A*B - B*A);        // commutator product
      checkProduct(P, k);              // sanity check
      weightsCommutator(i, j) = P[k];
      //P = regressiveProduct(A, B);
      //checkProduct(P, k);              // sanity check
      //weightsRegressive(i, j) = P[k];
    }
  }
  // For the regressive product, it seems we cannot use the same bladeIndices table as for the 
  // other products - maybe compute the regressive product without a specific Cayley table by
  // using its definition: A v B = (A* ^ B*)*...that's how it's currently done - but maybe at some
  // point, just do it with an additional bladeIndicesRegressive table - depends on how often the
  // regressive product is used - if it's used often, that optimization may be worthwhile

  // Create the diagonal matrices for the involutions. They all just change the sign of some grades
  // according to pow(-1, f(k)) where the function f(k) is different for each kind of involution
  // see "Geometric Multiplication of Vectors", pg 22,198:
  auto fGI = [&](int k) { return T(k); };
  auto fRI = [&](int k) { return T(k*(k-1)) / T(2); };
  auto fCI = [&](int k) { return T(k*(k+1)) / T(2); };
  involutionGrade.resize(n+1);
  involutionReverse.resize(n+1);
  involutionConjugate.resize(n+1);
  for(int k = 0; k <= n; k++) {
    involutionGrade[k]     = pow(T(-1), fGI(k));
    involutionReverse[k]   = pow(T(-1), fRI(k));
    involutionConjugate[k] = pow(T(-1), fCI(k)); }
  // Wait - is this formula correct in general or does it apply only signatures n,0,0? ...and in 
  // general, we need to take the signature into account? hmm...GA4CS gives the same formula for
  // reversion on page 522 without any mention of signature dependency. yeah - why would the 
  // signature come into play anyway - there are no squared basis vectors involved

  int dummy = 0;
}

template<class T>
void rsGeometricAlgebra<T>::product(const std::vector<T>& a, const std::vector<T>& b, 
  std::vector<T>& p, const RAPT::rsMatrix<T>& weights) const
{
  rsFill(p, T(0));
  for(int i = 0; i < N; i++) {
    for(int j = 0; j < N; j++) {
      int k = bladeIndices(i, j);
      p[k] += weights(i, j) * a[i] * b[j]; }}
}
// optimize! let j start at i -> this requires symmetrizing the weight matrices of the Cayley 
// tables. This may actually create even more zeros due to cancellations (at least, in the outer 
// product) which will in turn open more optimization opportunities when a sparse matrix 
// implementation is used (...later...) ...but maybe that makes the implementation of 
// product_bld_bld inconvenient? ..or maybe that's not possible at all?
// maybe in rsMatrixView we should have a function getDensity where the density of a matrix is 
// defined as the number of nonzero entries divided by the total number of entries
// ...implement a function int rsNumNonZeroEntries(const std::vector<T>& v)

template<class T>
void rsGeometricAlgebra<T>::product_bld_bld(const std::vector<T>& a, int ga, 
  const std::vector<T>& b, int gb, std::vector<T>& p, const rsMatrix<T>& weights) const
{
  int gp = ga + gb;             // grade of product
  rsAssert(gp <= n);            // will result in zero, should be handled in the operator
  int na = getBladeSize(ga);
  int nb = getBladeSize(gb);
  int np = getBladeSize(gp);
  int sa = getBladeStart(ga);
  int sb = getBladeStart(gb);
  int sp = getBladeStart(gp);
  rsFill(p, T(0));
  for(int i = sa; i < sa+na; i++) {
    for(int j = sb; j < sb+nb; j++) {
      int k = bladeIndices(i, j);
      if(k >= sp && k < sp + np)
        p[k-sp] += weights(i, j) * a[i-sa] * b[j-sb]; }}
}
// needs more tests

template<class T>
rsMatrix<T> rsGeometricAlgebra<T>::makeOutermorphism(const rsMatrix<T>& F)
{
  rsAssert(F.getNumRows() == n && F.getNumColumns() == n);

  // We make use of two facts: 
  // (1) in order to be grade-preserving, the matrix must have block-diagonal structure: there are 
  //     n+1 blocks and the size of the k-th block (k = 0..n) is n-choose-k
  // (2) in a general transformation matrix, the j-th column of the matrix is equal to the image
  //     of the j-th basis vector (at least, that's the case for the canonical basis -> figure out
  //     if it's still true for a general basis)
  // This gives rise to the following algorithm: 
  // -precompute the images of all basis vectors F(b_1), F(b_2), F(b_3), ...
  // -loop over the grades, for each grade k, do:
  //  -figure out the number m of basis blades for given grade k (this determines size of the block
  //   that we currently need fill)
  //  -loop over the m basis blades - for each blade, do:
  //   -figure out, from which basis vectors (p,q,r,...) this blade is formed
  //   -compute the outer product of the mapped basis vectors b_p, b_q, b_r, ...:
  //      P = F(b_p) ^ F(b_q) ^ F(b_r) ^ ...
  //   -copy the result P into the appropriate column in the k-th block of the matrix B

  // Compute the images of the basis vectors. We currently use the canonical basis vectors 
  // (1,0,0,..), (0,1,0,..) etc., so it's sufficient to set one coeff in b to 1 in each iteration.
  // However, for a general basis, we may indeed need a call like the commented 
  //   Vec b = getBasisVector(i); instead of b[i] = T(1); ... b[i] = T(0);
  // where we would need to implement such a getBasisVector() member function:
  using GV  = rsGradedVector<T>;
  using Vec = std::vector<T>;
  std::vector<GV> bi(n);
  Vec b(n);
  for(int i = 0; i < n; i++)
  {
    //Vec b  = getBasisVector(i); // current basis vector...
    b[i]   = T(1);
    Vec Fb = F * b;             // ...and its image under F
    bi[i]  = GV(this, 1, Fb);
    b[i]   = T(0);
  }

  // Compute the matrix elements:
  rsMatrix<T> B(N, N);
  B(0, 0) = T(1);                      // top-left entry (grade-0 trafo) is always 1
  for(int k = 1; k <= n; k++) {        // loop over the grades starting at 1
    int n0 = bladeStarts[k];           // (n0,n0) == start location of current block
    int m  = bladeSizes[k];            // number of grade-k basis blades
    // Fill a block of shape m x m:
    for(int j = 0; j < m; j++) {       // loop over the m grade-k basis blades
      int bmp = getBladeBitmap(k, j);  // bitmap of j-th grade-k basis blade
      GV P(this, 0); P[0] = T(1);      // outer product of the images, initially 1
      for(int i = 0; i < n; i++) {
        if(bmp & 1)
          P = (P ^ bi[i]);             // accumulate factor A*b if basis blade e_i is present
        bmp >>= 1; }
      rsAssert(P.getGrade() == k);     // sanity check
      for(int i = 0; i < m; i++)       // copy P into a column of the current block
        B(n0+i, n0+j) = P[i];      }}

  return B;
}

template<class T> rsMatrix<T> rsGeometricAlgebra<T>::makeOutermorphism(const rsMatrix<T>& F,
  const rsGeometricAlgebra<T>* srcAlg, const rsGeometricAlgebra<T>* dstAlg)
{
  // Input matrix F is m x n and maps vectors from R^n to R^m. Do some sanity checks:
  int m = F.getNumRows();    // dimensionality of destination vector space (codomain)
  int n = F.getNumColumns(); // dimensionality of source vector space (domain)
  rsAssert(dstAlg->getNumDimensions() == m);
  rsAssert(srcAlg->getNumDimensions() == n);
  int M = dstAlg->getMultiVectorSize();
  int N = srcAlg->getMultiVectorSize();
  rsAssert(M == rsPow(2, m));
  rsAssert(N == rsPow(2, n));

  // Compute the images of the basis vectors:
  using GV  = rsGradedVector<T>;
  using Vec = std::vector<T>;
  std::vector<GV> bi(m);
  Vec b(n);
  for(int i = 0; i < n; i++)
  {
    //Vec b  = srcAlg->getBasisVector(i); // current basis vector...
    b[i]   = T(1);
    Vec Fb = F * b;             // ...and its image under F
    bi[i]  = GV(this, 1, Fb);
    b[i]   = T(0);
  }

  // Compute the matrix elements:
  rsMatrix<T> B(M, N);
  B(0, 0) = T(1);
  for(int k = 1; k <= rsMin(m,n); k++)   // todo: verify the rsMin
  {
    int m0 = dstAlg->bladeStarts[k];     // (m0,n0) == start location of current block
    int n0 = srcAlg->bladeStarts[k];
    int mm = dstAlg->bladeSizes[k];
    int nn = srcAlg->bladeSizes[k];
    // Fill a block of shape mm x nn starting at m0,n0:



  }



  rsError("not yet complete");
  return B;
}


//-------------------------------------------------------------------------------------------------
// rsGradedVector:

template<class T>
void rsGradedVector<T>::set(int newGrade, const std::vector<T>& newCoeffs)
{
  int m = alg->getBladeSize(newGrade);
  rsAssert((int) newCoeffs.size() == m);
  grade = newGrade;
  coeffs.resize(m);
  rsCopy(newCoeffs, coeffs);
}

template<class T>
rsGradedVector<T> rsGradedVector<T>::operator^(const rsGradedVector<T>& b) const
{
  rsAssert(areCompatible(*this, b));
  int g = getGrade() + b.getGrade();
  if(g > alg->getNumDimensions())
    return rsGradedVector(alg, 0);          // return 0 (of grade 0)
  rsGradedVector<T> p(alg, g);
  alg->outerProduct_bld_bld(this->coeffs, grade, b.coeffs, b.grade, p.coeffs);
  return p;
}

/** Product between matrix A and vector b. It is assumed that A is a square matrix whose shape 
matches the size of b. */
template<class T>
rsGradedVector<T> operator*(const rsMatrix<T>& A, const rsGradedVector<T>& b)
{
  int m = (int) b.getCoeffs().size();
  rsAssert(b.getGrade() == 1, "b is assumed to be a vector" );
  rsAssert(A.hasShape(m, m),  "Matrix A has wrong shape");
  std::vector<T> v = A * b.getCoeffs();
  return rsGradedVector<T>(b.getAlgebra(), b.getGrade(), v);
}

//-------------------------------------------------------------------------------------------------
// rsMultiVector:

template<class T>
void rsMultiVector<T>::set(const rsGradedVector<T>& b)
{
  setAlgebra(b.getAlgebra());
  int n0 = alg->getBladeStart(b.getGrade());
  int n  = alg->getBladeSize( b.getGrade());  // maybe use m
  rsFill(coeffs, T(0));
  for(int i = 0; i < n; i++)
    coeffs[n0+i] = b.getCoefficient(i);
}

template<class T>
rsGradedVector<T> rsMultiVector<T>::extractGrade(int grade) const
{
  if(grade < 0 || grade > alg->getNumDimensions())
    return rsGradedVector<T>(alg, 0); // return the zero blade
  rsGradedVector<T> B(alg, grade);
  int n0 = alg->getBladeStart(grade);
  B.setCoefficients(&coeffs[n0]);
  return B;
}

template<class T>
bool rsMultiVector<T>::containsGrade(int k, T tol) const
{
  if(k < 0 || k > alg->getNumDimensions())
    return false;
  int n0 = alg->getBladeStart(k);
  int m  = alg->getBladeSize(k);
  for(int i = 0; i < m; i++)
    if(rsAbs(coeffs[n0+i]) > tol)
      return true;
  return false;
}

template<class T>
int rsMultiVector<T>::getLowestGrade(T tol) const
{
  int n = alg->getNumDimensions();
  for(int k = 0; k <= n; k++)
    if(containsGrade(k, tol))
      return k;
  return 0;  // or should we return -1?
}

template<class T>
int rsMultiVector<T>::getHighestGrade(T tol) const
{
  int n = alg->getNumDimensions();
  for(int k = n; k >= 0; k--)
    if(containsGrade(k, tol))
      return k;
  return 0;  // or should we return -1?
}

template<class T>
T rsMultiVector<T>::getSquaredVectorNorm() const
{
  rsAssert(isVector(), "applicable only to vectors"); // may need tolerance
  int n0 = alg->bladeStarts[1];
  int m  = alg->bladeSizes[1];
  T sum = T(0);
  for(int i = 0; i < m; i++)
    sum += coeffs[n0+i] * coeffs[n0+i];
  return sum;
}

template<class T>
void rsMultiVector<T>::applyInvolution(const std::vector<T>& inv)
{
  int n = alg->getNumDimensions();
  rsAssert(inv.size() == n+1);
  for(int k = 0; k <= n; k++) {
    int n0 = alg->bladeStarts[k];
    int m  = alg->bladeSizes[k];
    T   s  = inv[k];
    for(int i = 0; i < m; i++)
      coeffs[n0+i] *= s; }
}

template<class T>
void rsMultiVector<T>::applyDualization()
{
  int N = alg->N;
  rsArrayTools::reverse(&coeffs[0], N);
  for(int i = 0; i < N; i++)
    coeffs[i] *= alg->dualSigns[i];
}

template<class T>
rsMultiVector<T> rsMultiVector<T>::getDual() const
{
  int N = alg->N;
  rsMultiVector<T> Y(alg);
  for(int i = 0; i < N; i++)
    Y.coeffs[i] = coeffs[N-i-1] * alg->dualSigns[i];
  return Y;
}

template<class T>
rsMatrix<T> rsMultiVector<T>::getMatrixRepresentation() const
{
  // Each row of the matrix is a permutation of the elements of the coeffs array, with some 
  // sign-factors applied:
  int N = alg->N;
  rsMatrix<T> M(N, N);
  for(int i = 0; i < N; i++) {
    for(int j = 0; j < N; j++) {
      int k = alg->bladeIndices(i, j);
      T   s = alg->weightsGeom( i, j);
      M(k, j) = s * coeffs[i]; }}
  return M;
}
// needs more tests

template<class T>
rsMultiVector<T> rsMultiVector<T>::getInverse() const
{
  rsMatrix<T> A = getMatrixRepresentation();
  std::vector<T> b(alg->N);
  b[0] = T(1);
  std::vector<T> x = rsLinearAlgebraNew::solve(A, b);
  return rsMultiVector<T>(alg, x);
}
// try to optimze away the extra memory allocation in the return statement

template<class T>
rsMultiVector<T> rsMultiVector<T>::getInverseScalar() const
{
  rsAssert(isScalar(), "works only for scalars"); // may need tolerance
  rsMultiVector<T> I(alg);
  I[0] = T(1) / coeffs[0];
  return I;
}

template<class T>
rsMultiVector<T> rsMultiVector<T>::getInverseVector() const
{
  rsAssert(isVector(), "works only for vectors"); // may need tolerance
  T s = T(1) / getSquaredVectorNorm();
  int n0 = alg->bladeStarts[1];
  int m  = alg->bladeSizes[1];
  rsMultiVector<T> I(alg);
  for(int i = 0; i < m; i++)
    I.coeffs[n0+i] = s * coeffs[n0+i];
  return I;
}

template<class T>
rsMultiVector<T> rsMultiVector<T>::operator+(const rsMultiVector<T>& b) const
{
  rsAssert(areCompatible(*this, b));
  rsMultiVector<T> p(alg);
  RAPT::rsArrayTools::add(&coeffs[0], &b.coeffs[0], &p.coeffs[0], (int)coeffs.size());
  return p;
}

template<class T>
rsMultiVector<T> rsMultiVector<T>::operator-(const rsMultiVector<T>& b) const
{
  rsAssert(areCompatible(*this, b));
  rsMultiVector<T> p(alg);
  RAPT::rsArrayTools::subtract(&coeffs[0], &b.coeffs[0], &p.coeffs[0], (int)coeffs.size());
  return p;
}

template<class T>
rsMultiVector<T> rsMultiVector<T>::operator*(const rsMultiVector<T>& b) const
{
  rsAssert(areCompatible(*this, b));
  rsMultiVector<T> p(alg);
  alg->geometricProduct(this->coeffs, b.coeffs, p.coeffs);
  return p;
}

template<class T>
rsMultiVector<T>& rsMultiVector<T>::operator+=(const rsMultiVector<T>& b) 
{ 
  rsAssert(areCompatible(*this, b));
  for(size_t i = 0; i < coeffs.size(); i++)
    coeffs[i] += b.coeffs[i];
  return *this; 
}

template<class T>
rsMultiVector<T> rsMultiVector<T>::operator*(const T& s) const
{
  rsMultiVector<T> p(alg);
  RAPT::rsArrayTools::scale(&coeffs[0], &p.coeffs[0], (int)coeffs.size(), s);
  return p;
}

template<class T>
rsMultiVector<T> rsMultiVector<T>::operator^(const rsMultiVector<T>& b) const
{
  rsAssert(areCompatible(*this, b));
  rsMultiVector<T> p(alg);
  alg->outerProduct(this->coeffs, b.coeffs, p.coeffs);
  return p;
}

/*
template<class T>
rsMultiVector<T> rsMultiVector<T>::operator|(const rsMultiVector<T>& b) const
{
  rsAssert(areCompatible(*this, b));
  rsMultiVector<T> p(alg);
  alg->innerProduct(this->coeffs, b.coeffs, p.coeffs);
  return p;
}
*/

template<class T>
rsMultiVector<T> rsMultiVector<T>::product(
  const rsMultiVector<T>& A, const rsMultiVector<T>& B, ProductType type)
{
  using MV = rsMultiVector<T>;
  using PT = ProductType;
  const rsGeometricAlgebra<T>* alg = A.alg;
  MV P(alg);
  if(type == PT::regressive)
  {
    MV Ad = A.getDual();
    MV Bd = B.getDual();
    alg->outerProduct(Ad.coeffs, Bd.coeffs, P.coeffs);
    P.applyDualization();
    return P;
  }
  switch(type)
  {
  case PT::wedge:         alg->outerProduct(       A.coeffs, B.coeffs, P.coeffs); break;
  case PT::contractLeft:  alg->contractLeftMM(     A.coeffs, B.coeffs, P.coeffs); break;
  case PT::contractRight: alg->contractRightMM(    A.coeffs, B.coeffs, P.coeffs); break;
  case PT::scalar:        alg->scalarProductMM(    A.coeffs, B.coeffs, P.coeffs); break;
  case PT::dot:           alg->dotProductMM(       A.coeffs, B.coeffs, P.coeffs); break; 
  case PT::fatDot:        alg->fatDotProductMM(    A.coeffs, B.coeffs, P.coeffs); break;
  case PT::commutator:    alg->commutatorProductMM(A.coeffs, B.coeffs, P.coeffs); break;
  default:                rsError("Unknown product type");
  }
  return P;
}

template<class T>
rsMultiVector<T> rsMultiVector<T>::productSlow(
  const rsMultiVector<T>& C, const rsMultiVector<T>& D, ProductType type)
{
  using MV = rsMultiVector<T>;
  using GV = rsGradedVector<T>;
  using PT = ProductType;
  MV CD(C.getAlgebra());
  int n = C.getAlgebra()->getNumDimensions();
  for(int r = 0; r <= n; r++) {
    for(int s = 0; s <= n; s++) {
      if(type == PT::dot && (r == 0 || s == 0))
        continue;
      GV Cr   = C.extractGrade(r);
      GV Ds   = D.extractGrade(s);
      MV CrDs = Cr * Ds;                 // geometric product of grades Cr,Ds
      int rs;
      switch(type)
      {
      case PT::wedge:         rs = r+s;        break;
      case PT::contractLeft:  rs = s-r;        break;
      case PT::contractRight: rs = r-s;        break;
      case PT::scalar:        rs = 0;          break;
      case PT::dot:           rs = rsAbs(s-r); break; 
      case PT::fatDot:        rs = rsAbs(s-r); break;
      default:                rs = r+s;
      }
      GV Prj  = CrDs.extractGrade(rs);   // grade projection
      CD     += Prj;  }}                 // accumulation
  return CD;
}

/** Sum of a scalar and a multivector. */
template<class T>
rsMultiVector<T> operator+(const T& s, const rsMultiVector<T>& A)
{
  rsMultiVector<T> P = A;
  P[0] += s;
  return P;
}

/** Product of a scalar and a multivector. */
template<class T>
rsMultiVector<T> operator*(const T& s, const rsMultiVector<T>& A)
{
  rsMultiVector<T> P = A;
  P.scale(s);
  return P;
}

/** Quotient of a scalar and a multivector. */
template<class T>
rsMultiVector<T> operator/(const T& s, const rsMultiVector<T>& A)
{
  rsMultiVector<T> Q = A.getInverse();
  Q.scale(s);
  return Q;
}


/** Geometric product of two k-vectors. This results in general in a multivector. If the grades of 
the inputs are i and j the result will have grades i-j and i+j (verify!).  */
template<class T>
rsMultiVector<T> operator*(const rsGradedVector<T>& a, const rsGradedVector<T>& b)
{
  rsMultiVector<T> A = a;
  rsMultiVector<T> B = b;
  return A*B;
}
// may be optimized



/** Product between matrix A and multivector b. It is assumed that A is a square matrix whose shape 
matches the size of b. */
template<class T>
rsMultiVector<T> operator*(const rsMatrix<T>& A, const rsMultiVector<T>& b)
{
  int m = (int) b.getCoeffs().size();
  rsAssert(A.hasShape(m, m),  "Matrix A has wrong shape");
  std::vector<T> v = A * b.getCoeffs();
  return rsMultiVector<T>(b.getAlgebra(), v);
}

/** Naive prototype implementation of the exponential function of a general multivector. */
template<class T>
rsMultiVector<T> rsExpNaive(const rsMultiVector<T>& X)
{
  int maxIts = 32;  // maybe make parameter
  //T tol = T(1000) * RS_EPS(T);            // ad hoc - figure out something better

  rsMultiVector<T> Y(X.getAlgebra()), Xk(X.getAlgebra());  // output Y and Xk = X^k
  Xk[0] = T(1);                          // Xk = X^0 = 1
  for(int k = 0; k < maxIts; k++)
  {
    Y  += Xk * (T(1) / T(rsFactorial(k)));
    Xk  = Xk * X;                            // implement and use Xk *= X
  }
  // todo: add convergence test

  return Y;
}

/*
template<class T>
T rsNextPowerOfTwo(T x)
{
  if(x < T(1)) 
    return T(1) / rsNextPowerOfTwo(T(1)/x);
  T tmp = rsLog2(x);
  tmp = ceil(tmp);
  tmp = rsExp2(tmp);
  tmp = round(tmp);
  return tmp;
}
*/


template <class T>
T rsSquaredNormReverse(const rsMultiVector<T>& V)
{
  rsMultiVector<T> Vr_V = V * V.getReverse();
  return Vr_V[0];
  // Vr * V has same scalar part as V * Vr but in general a different vector part
  // try to optimize this: 
  // -only scalar part is needed, so we may compute only that 
  // -also try to avoid creating the temporary objects V.getReverse(), Vr_v to avoid the memory 
  //  allocation -> instead just compute Vr_v[0] maybe like
  //  for(int i = 0; i < N; i++) {
  //    for(int j = 0; j < N; j++) {
  //      k = bladeIndices(i, j);
  //      sum += V[i] * reverseSigns[k] * V[reversIndices[k]] * weightsGeom[i, j]; }}
}
template <class T>
T rsNormReverse(const rsMultiVector<T>& V)
{
  return rsSqrt(rsSquaredNormReverse(V));
}
// Refernces:
// GMoV pg 48,
// https://math.stackexchange.com/questions/3733141/what-does-the-norm-of-a-multivector-describes-geometrically
// https://math.stackexchange.com/questions/958559/norm-on-a-geometric-algebra/2840383
// todo: 
// -implement the other norm based on Clifford conjugation: rsNormConjugate
// -optimze: if only the scalar component is extracted, compute only that - maybe make a function
//  to compute the scalar product efficiently

template <class T>
T rsSumOfSquares(const rsMultiVector<T>& V)
{
  return rsArrayTools::sumOfSquares(&V.getCoeffs()[0], (int) V.getCoeffs().size());
}
template <class T>
T rsNormEuclidean(const rsMultiVector<T>& V)
{
  return rsSqrt(rsSumOfSquares(V));
}
// rename to rsNormEuclidean
// This norm may be more suitable for argument scaling in the elementary function evaluation 
// algorithms. For G(3,0,0) (R^3 vector space) is apparently coincides with the reversal-based norm
// above, in G(0,1,0) (complex numbers), it coincides with the complex magnitude, ...tbc...
// maybe rename to rsAbs
// see pg 17 here:
// http://www.math.umd.edu/~immortal/MATH431/lecturenotes/ch_geometricalgebra.pdf
// http://www.math.umd.edu/~immortal/MATH431/lecturenotes/

template <class T>
T rsNormSumAbsolute(const rsMultiVector<T>& V)
{
  return rsArrayTools::sumOfAbsoluteValues(&V.getCoeffs()[0], (int) V.getCoeffs().size());
}
// rename to rsNormManhattan

// add: rsNormClifford, rsNormOperator, rsNormForPowerSeries - the last one delegates to one
// of the other ones (probably the operator norm) - or maybe call it rsNormPower to indicate
// that we use the multivector itself as seed - to find the true operator norm, we would 
// potentially have to try with many different start vectors

/** Experimental. This is supposed to figure out the operator norm of the multivector x, i.e. the 
maximum stretch-factor that a multiplication by x can apply to another multivector (here 
interpreted as regular 2^n dimensional vector). We use a sort of power iteration to find the 
direction of maximum stretch. When this is converged, we figure out the actual stretch factor by 
comparing norms of original and and stretched vector. This works similar to algorithms for 
computing the largest absolute eigenvalue/-vector of a matrix. The difference is that here, we use
multivector multiplication instead of the matrix-vector product and we use repeated squaring 
instead of computing powers to increase the convergence speed. The so computed norm can be seen as
operator norm, when multiplication by multivector x is seen as an operator that can act on other
multivectors. ...or can it? Or maybe this holds only if x itself contains a nonzero component in 
the direction of its own largest eigenvector? But maybe that's guaranteed to be the case? See:
https://en.wikipedia.org/wiki/Power_iteration  
https://en.wikipedia.org/wiki/Operator_norm    */
template <class T>
T rsNormOperator(const rsMultiVector<T>& x)
{
  using MV = rsMultiVector<T>;
  auto norm = [&](const rsMultiVector<T>& x) { return rsNormEuclidean(x); };
  //auto norm = [&](const rsMultiVector<T>& x) { return rsNormSumAbsolute(x); }; // no convergence!
  MV y = x;
  //y.randomIntegers(-9, +9, 42);  // test
  T ny = norm(y);
  if(ny == T(0))
    return ny;
  y   /= ny;
  int its = 0;                 // iteration counter
  T tol = T(2) * RS_EPS(T);    // tolerance (rather strict, may need tweaks)
  while(true)                 
  {
    MV yNew = y*y;             // repeated squaring
    //MV yNew = x*y;           // power iteration (same result, slower convergence)
    T nyNew = norm(yNew);
    yNew   /= nyNew;
    T ratio = nyNew / ny;      // should converge to 1
    T delta = ratio - T(1);    // should go to zero
    y  = yNew;
    if(rsAbs(delta) < tol)     // convergence check
      break;
    ny = nyNew;
    its++;
  }

  // y is now supposed to be a normalized vector pointing into the direction of the largest 
  // absolute eigenvector of x. To figure out the eigenvalue, we multiply it by x and see, how
  // much this changes the length:
  //ny = norm(y);      // test - should be 1
  T ev = norm(x*y);    // eigenvalue corresponding to largest eigenvector y
  return ev;
}
// Test results with multivector x = (3,8,7,4,6,4,6,5) in G(3,0,0):
// The iterates of y hop around wildly. They don't seem to converge to a particular direction. 
// Maybe the largest absolute eigenvalue is complex and there is some sort of 8D rotation going on?
// Power iteration is used for diagonalizable matrices and maybe this one isn't? Also, what if
// x has multiple eigenvalues of the same magnitude? Maybe the iterates converge to some vector in 
// the subspace spanned by the corresponding eigenvectors and that is good enough for the purpose
// here?
// Also, the norm computed this way is suspicously close to the Euclidean norm, so it may not 
// really improve our argument reduction scheme for log, etc. Using the sum-of-abs norm rather than
// Euclidean as underlying norm, the algo doesn't even converge.
// ToDo: 
// -compare result to largest eigenvalue of matrix representation of x - it should be the same
// -figure out the actual complex spectrum of the matrix representation of x using sage or octave 
//  or whatever - or maybe my own eigenvalue code could already be up to the task? not sure...
// -maybe figure out the singular values of the matrix representation of x
// -try to use different random start vectors for y and see if it always converges to the same 
//  result - it should! -> yes, it does, but only if power iteration is used, not repeated 
//  squaring. This is expected.



//template rsMultiVector<double> RAPT::rsPow(const rsMultiVector<double>& base, int exponent);
//extern template rsMultiVector<double> RAPT::rsPow(const rsMultiVector<double>& base, int exponent);
// instantiating the template doesn't seem to work, so for the time being, we use a specialized
// implementation of rsPow:
template <class T>
rsMultiVector<T> rsPow(const rsMultiVector<T>& base, int exponent)
{
  rsMultiVector<T> square(base);
  rsMultiVector<T> result(base.getAlgebra());
  result[0] = T(1);
  while(true)
  {
    if(exponent & 1)
      result = result * square;
    exponent /= 2;
    if(exponent == 0)
      break;
    square = square * square;
  }
  return result;
}

/** Returns true, iff adding any of the dx-values (scaled by s) to the corresponding x-value 
actually changes the x-value. Useful for multidimensional convergence tests. */
template<class T>
bool rsMakesDifference(const T* x, const T* dx, int N, T s = T(1))
{
  for(int n = 0; n < N; n++) {
    T y = x[n] + s*dx[n];
    if(y != x[n])
      return true; }
  return false;
}
// move to rsArrayTools, try to find a better name

template<class T>
bool rsMakesDifference(const rsMultiVector<T>& X, const rsMultiVector<T>& dX, T weight = T(1))
{
  int N = X.getAlgebra()->getMultiVectorSize();
  rsAssert(dX.getAlgebra()->getMultiVectorSize() == N);
  return rsMakesDifference(&X.getCoeffs()[0], &dX.getCoeffs()[0], N, weight);
}

/** Exponential function of multivectors A that square to zero. */
template<class T>
rsMultiVector<T> rsExpSqrZero(const rsMultiVector<T>& A)
{
  return A + T(1);
}

/** Exponential function of multivectors A that square to a positive scalar a2. */
template<class T>
rsMultiVector<T> rsExpSqrPosScalar(const rsMultiVector<T>& A, const T& a2)
{
  rsAssert(a2 > T(0));
  T a = sqrt(a2);
  return (sinh(a)/a)*A + cosh(a); // maybe implement and use rsSinch(a)
}

/** Exponential function of multivectors A that square to a negative scalar a2. */
template<class T>
rsMultiVector<T> rsExpSqrNegScalar(const rsMultiVector<T>& A, const T& a2)
{
  rsAssert(a2 < T(0));
  T a = sqrt(-a2);
  return (sin(a)/a)*A + cos(a);  // maybe use rsSinc(a)
}


/** Exponential function of general multivectors X, evaluated by accelerated Taylor expansion. */
template<class T>
rsMultiVector<T> rsExpViaTaylor(const rsMultiVector<T>& X)
{
  // The algorithm uses the fact that exp(X) = (exp(X/s))^s to use a scaled version of X in the 
  // Taylor expansion that ensures that the powers X^k of X don't explode. In fact, they decay 
  // exponentially and additionally get divided by k!, so the terms in the series get small very 
  // fast.

  using MV = rsMultiVector<T>;
  //T s = rsNorm(X);
  T s = rsNormEuclidean(X);
  s = rsNextPowerOfTwo(s) * 2;  // factor 2 gives best numeric accuracy for the scalar exp(10)
  MV Z = (T(1)/s) * X;          // scaled X: Z = X/s
  MV Y( X.getAlgebra());        // output Y 
  MV Zk(X.getAlgebra());        // Zk = Z^k in the iteration
  Zk[0] = T(1);                 // Zk = Z^0 = 1 initially
  int maxIts = 32;              // 32 is the length of rsInverseFactorials
  int k;                        // iteration number == current power of Z
  for(k = 0; k < maxIts; k++) {
    if(!rsMakesDifference(Y, Zk, rsInverseFactorials[k]))  // convergence test
      break;  
    Y  += Zk * T(rsInverseFactorials[k]);
    Zk *= Z; }
  rsAssert(k <= maxIts, "rsExp for rsMultiVector did not converge");
  return rsPow(Y, (int)s);     // Y^s undoes the scaling
}
// -needs more tests
// -i'm not sure, if this norm makes the most sense - maybe we should use the largest absolute
//  eigenvalue of the matrix representation of X instead because this is the maximum blow-up factor
//  that this multivector can cause
// Idea: can we use a variant of Newton iteration to evaluate a function? Newton iteration itself 
// uses function and derivative evaluations, so it can't be used as is. But maybe we can use the
// differential equation f(x) - f'(x) = 0 or one of the functional equations, like 
// f(2*x) - (f(x))^2 = 0 and approximate the f,f' values via Taylor series? Or maybe somehow use
// the inverse slope: f^-1(y) = log(y) = x to somehow find a crossing of the tangent with the 
// y-axis? -> research needed -> if so, maybe we could refine the exp with a single such step at
// the very end. And/or maybe fixed point iteration may also be adapted?
// for function evaluation, see:
// https://arxiv.org/abs/1004.3412
// https://link.springer.com/chapter/10.1007/978-1-4757-2736-4_47
// https://link.springer.com/chapter/10.1007/978-1-4757-2736-4_56
// https://cr.yp.to/bib/1976/brent-elementary.pdf
// maybe implement also expm1 = exp(x)-1, see:
// https://en.wikipedia.org/wiki/Exponential_function#Computation

/** Exponential function of multivectors. Dispatches between various implementations, depending on 
the square of A. */
template<class T>
rsMultiVector<T> rsExp(const rsMultiVector<T>& X)
{
  rsMultiVector<T> X2 = X*X;
  T tol = T(0);             // preliminary
  if(X2.isScalar(tol))
  {
    T x2 = X2[0];                                          // x2 = x^2
    if(     x2 > T(0)) return rsExpSqrPosScalar(X, x2);    // exp(X) = (sinh(x)/x)*X + cosh(x)
    else if(x2 < T(0)) return rsExpSqrNegScalar(X, x2);    // exp(X) = (sin(x) /x)*X + cos(x)
    else               return rsExpSqrZero(X);             // exp(X) =             X + 1
  }
  else
    return rsExpViaTaylor(X);                              // fallback for the general case
}
// needs tests for the special cases
// see GA4CS, pg 531

// todo: implement: sin, cos, log, asin, acos, asinh, acosh, pow, agm, meet, join

// maybe rename to rsSinhViaExp etc.:
template<class T>
rsMultiVector<T> rsSinh(const rsMultiVector<T>& X)
{
  rsMultiVector<T> eX = rsExp(X);
  return T(0.5) * (eX - eX.getInverse());   // sinh(X) = (exp(X) - exp(-X)) / 2
}
template<class T>
rsMultiVector<T> rsCosh(const rsMultiVector<T>& X)
{
  rsMultiVector<T> eX = rsExp(X);
  return T(0.5) * (eX + eX.getInverse());   // cosh(X) = (exp(X) + exp(-X)) / 2
}
template<class T>
rsMultiVector<T> rsTanh(const rsMultiVector<T>& X)
{
  rsMultiVector<T> eX  = rsExp(X);
  rsMultiVector<T> eXi = eX.getInverse();
  return (eX - eXi) / (eX + eXi);           // tanh(X) = sinh(X) / cosh(X)
}





template<class T>
rsMultiVector<T> rsInvSqrt(const rsMultiVector<T>& X)
{
  using MV = rsMultiVector<T>;
  MV one(X.getAlgebra());
  one[0] = T(1);        // todo: use a scalar, i.e. type T
  MV Y = T(1) / X;
  int maxIts = 32;
  int k;
  for(k = 0; k < maxIts; k++)
  {
    MV E  = X*Y*Y - one;
    MV dY = -T(0.5)*Y*(E - T(0.75)*E*E);
    if(!rsMakesDifference(Y, dY))             // convergence test
      break;
    Y += dY;
  }
  // OK - asymptotic convergence works and is fast but we need a good initial guess - how about
  // 1/X? ..looks good for scalars - more tests needed
  // ..the convergence test does not work - we cannot assume that the addition makes no difference
  // after convegence - it may make a difference of the order of epsilon...what exactly should we 
  // use? 2*eps? 4*eps?

  //rsAssert(k < maxIts, "rsInvSqrt did not converge");
  return Y;
}

template<class T>
rsMultiVector<T> rsSqrt(const rsMultiVector<T>& X)
{
  return X * rsInvSqrt(X);
}

/** Sine function for small arguments with |X| <= 1. */
template<class T>
rsMultiVector<T> rsSinSmall(const rsMultiVector<T>& X)
{
  using MV = rsMultiVector<T>;
  MV X2 = X*X;                          // X^2
  MV Xk = X;                            // X^(2*k+1)
  MV Y(X.getAlgebra());                 // output Y 
  T s = T(1);                           // sign factor
  int kLim = 16;                        // 32 = 2*16 is the length of rsInverseFactorials
  int k;                                // iteration number
  for(k = 0; k < kLim; k++) {
    int k2p1 = 2*k+1;
    T w = s*rsInverseFactorials[k2p1];  // weight
    if(!rsMakesDifference(Y, Xk, w))    // convergence test
      break;  
    Y  += Xk * w;
    Xk *= X2;
    s  *= T(-1); }
  rsAssert(k <= kLim, "rsSin for rsMultiVector did not converge");
  return Y;
}

template<class T>
rsMultiVector<T> rsCosSmall(const rsMultiVector<T>& X)
{
  using MV = rsMultiVector<T>;
  MV X2 = X*X;                        // X^2
  MV Xk(X.getAlgebra());              // X^(2*k)
  Xk[0] = T(1);
  MV Y( X.getAlgebra());              // output Y 
  T s = T(1);                         // sign factor
  int kLim = 16;                      // 32 = 2*16 is the length of rsInverseFactorials
  int k;                              // iteration number
  for(k = 0; k < kLim; k++) {
    int k2 = 2*k;
    T w = s*rsInverseFactorials[k2];  // weight
    if(!rsMakesDifference(Y, Xk, w))  // convergence test
      break;  
    Y  += Xk * w;
    Xk *= X2;
    s  *= T(-1); }
  rsAssert(k <= kLim, "rsCos for rsMultiVector did not converge");
  return Y;
}

template<class T>
rsMultiVector<T> rsSin(const rsMultiVector<T>& X)
{
  // Uses sin(n*x) = 2*cos(x) * sin((n-1)*x) - sin((n-2)*x) to apply a similar argument reduction 
  // trick as for the exponential function.
  using MV = rsMultiVector<T>;
  static const T scl = T(1);
  //T s = rsNorm(X);
  T s = rsNormEuclidean(X);
  s = ceil(s) * scl;
  MV Z = (T(1)/s) * X;
  MV S = rsSinSmall(Z);
  MV C = rsCosSmall(Z);
  MV S2(X.getAlgebra());
  MV S1 = S;
  int n = (int)s;
  for(int i = 2; i <= n; i++) {
    S  = T(2)*C*S1 - S2;
    S2 = S1;
    S1 = S;   }
  return S;
  // The factor scl may be tweaked emprically to give a good compromise between fast convergence
  // (high scl) and accuracy (low or mid scl). ..well..actually it seems a factor of gives best 
  // accuracy and 4 seems to give good tradeoff. Currently I just use 1 for best accuracy. More 
  // tests needed, wiht various arguments
}
// needs more tests - especially with small arguments

template<class T>
rsMultiVector<T> rsCos(const rsMultiVector<T>& X)
{
  // Uses cos(n*x) = 2*cos(x) * cos((n-1)*x) - cos((n-2)*x) to apply a similar argument reduction 
  // trick as for the exponential function.
  using MV = rsMultiVector<T>;
  static const T scl = T(1);
  //T s = rsNorm(X);
  T s = rsNormEuclidean(X);
  s = ceil(s) * scl;       // todo: multiply by a factor - figure out which works best with respect to
  MV Z = (T(1)/s) * X;     // fast convergence and high accuracy of the result
  MV C = rsCosSmall(Z);
  MV C2(X.getAlgebra());
  C2[0] = T(1);
  MV C1 = C;
  MV Cn(X.getAlgebra());
  int n = (int)s;
  for(int i = 2; i <= n; i++) {
    Cn = T(2)*C*C1 - C2;
    C2 = C1;
    C1 = Cn;   }
  return Cn;
}
// For sin/cos, maybe a similar trick for accelerating the convergence can be used as for the
// exponential, based on this formula:
// https://en.wikipedia.org/wiki/De_Moivre%27s_formula#Formulae_for_cosine_and_sine_individually
// https://en.wikipedia.org/wiki/List_of_trigonometric_identities#Sine,_cosine,_and_tangent_of_multiple_angles
// or maybe with the Chebychev method:
// https://en.wikipedia.org/wiki/List_of_trigonometric_identities#Chebyshev_method
// initialized with cos(0) = 1 and cos(x/n). maybe first implement a simple version based on Taylor
// series and then try the trick on top of that implementation. in some cases, we may also use the 
// exp-function to compute sin/cos - i think, i works whenever we have a commuting pseudoscalar 
// that squares to -1

template<class T>
rsMultiVector<T> rsTan(const rsMultiVector<T>& X)
{
  return rsSin(X) / rsCos(X);
  // When the denominator becomes zero, the result is +-infinity, when the numerator is nonzero. 
  // This is the correct, desired behavior. However - can it happen that the numerator is also 
  // zero at these points? Not along the real axis and neither in the complex plane (i think). But
  // what can be said about general multivectors?
}
// ToDo: maybe have an rsSinCos function - might be more efficient

// Ideas for sin/cos:
// -use a coupled Newton iteration:
//   s[i+1] = x - s[i]/c[i];  // f[i+1] = x - f[i]/f'[i]
//   c[i+1] = x + c[i]/s[i];  // may also use c[i]/s[i+1]
//  ‰‰‰hh...wait...not - we need to reformulate it as root finding problem, like sin(x) - s = 0
//  but if we replace sin(x) by our current estimate, that becomes a useless trivial identity
// -maybe enforce sin^2(x) + cos^2(x) = 1 by renormalization: compute sin^2(x) + cos^2(x), take the
//  norm and divide both components by it (maybe not in every iteration)
// -How about using std::complex<rsMultiVector<T>> and then compute the complex exponential? This 
//  way of computing sin/cos seems to be common practice is multiprecision arithmetic libraries.
// -Maybe we may use 2D Newton iteration on the pair s,c with the two equations:
//  s^2 + c^2 = 1, s^4 - c^4 - s^2 + c^2 = 0, see: https://www.desmos.com/calculator/5xpr3ugtsa

// Ideas for logarithm:
// -Newton iteration with log'(x) = 1/x
// -Taylor series with acceleration via: log(n*x) = log(x) + log(n)
// -maybe for square root, we can also use a Taylor series with acceleration via 
//  sqrt(n*x) = sqrt(n) * sqrt(x)...or at least use the first few terms of such a series f to find
//  a good initial guess for Newton iteration
// see also:
// https://en.wikipedia.org/wiki/Series_acceleration
// https://en.wikipedia.org/wiki/Sequence_transformation
// https://mathworld.wolfram.com/ConvergenceImprovement.html
// https://arxiv.org/abs/1702.01129
// https://math.stackexchange.com/questions/585154/taylor-series-for-logx
// http://www.math.com/tables/expansion/log.htm
// https://math.stackexchange.com/questions/61209/what-algorithm-is-used-by-computers-to-calculate-logarithms
// https://en.wikipedia.org/wiki/Taylor_series#Natural_logarithm
// https://www.efunda.com/math/taylor_series/logarithmic.cfm
// http://www.math.com/tables/expansion/log.htm

/** Computes the logarithm of x using a Taylor series of given order centered around x0 = 1. 
Converges for 0 < x < 2 and converges most quickly for x = 1. */
template<class T>
rsMultiVector<T> rsLogViaTaylorSmall(const rsMultiVector<T>& x, int order)
{
  using MV = rsMultiVector<T>;
  MV z  = x; z[0] -= T(1);           // z = x-1

  // test:
  T nx = rsNormOperator(x);
  T nz = rsNormOperator(z);
  //MV t = (1.0/x) - 1.0;
  MV t = (x*x) - 1.0;
  T nt = rsNormOperator(t);
  //rsAssert(nz <= T(1));
  // OK, yes - the norm of x is indeed 1 but the norm of z is mostly > 1, so the powers of z blow 
  // up exponentially, making the series divergent. What we actually need to do is to scale x in 
  // such a way that the norm of z becomes less than 1. Using powers like t = (x*x) - 1.0; or
  // (1.0*x) - 1.0; does indeed sometimes yield a norm < 1 when the norm of z is > 1, so it seems
  // like there *may* be a suitable tansformation of the form z = (x^p / s) - 1 that makes the 
  // series convergent. We "just" need to find the right exponent and scaler. But how? And what 
  // about using negative scalers s? Maybe the rsNormOperator can also figure out whether the 
  // eigenvector get flipped or not, i.e. if the eigenvalue is positive or negative - but maybe 
  // it's complex? I think, the subtraction of 1 shifts all eigenvalues by -1, so if x has a 
  // spectrum of (1,-0.9,...), z will have a spectrum (0,-1.9,...). What we need for convergence is
  // that the *shifted* spectrum has largest absolute eigenvalue of < 1. Maybe if we can't arrange 
  // for that, we should pick a different expansion point like x0 = 2, i.e. z = x-2 - wolfram:
  //   Series[Log[x], {x, x0, 10}] 
  // gives:
  //   sum_(n>=1) ((-1)^(1 + n) (x - x0)^n x0^(-n))/n + log(x0)
  //   converges when abs(x - x0) < abs(x0)
  // so that could lead to a viable approach: we just take a greater expansion point and thereby
  // increase the region of convergence. Maybe the ideal expansion point is nz itself? No - nz is
  // computed *after* the expansion point has been selected. Oh - wait, that means we can't really
  // change the expansion point without also shifting the eigenvalues along with it

  MV zk = z;                         // z^k, initially z^1 = z
  MV y(x.getAlgebra());              // result
  T s = T(1);                        // sign
  for(int k = 1; k <= order; k++) {
    T w  = s / T(k);                 // weight
    y   += w * zk;
    zk  *= z;
    s   *= T(-1);   }                // sign alternation
  return y;
}
// Converges rather slowly because the weights do not fall off as rapidly as in the case of exp, 
// sin, etc. Can we find a faster converging series for a function that is (simply) related to
// log? Or maybe speed up the convergence in other ways? maybe this could be applicable:
// https://www.youtube.com/watch?v=wqMQRwX4Zn0
// We need to apply the same technique to the series of weights which is actually the harmonic 
// series. So it seems reasonable to start with trying to make a quickly converging harmonic 
// series....
// See also:
//   https://en.wikipedia.org/wiki/Logarithm#Calculation
// See also the book Functions of Matrices: Theory and Computation:
//   https://archive.siam.org/books/ot104/
//   http://www.ma.man.ac.uk/~higham/mftoolbox/
// it also covers logarithms of matrices, maybe that's useful

template<class T>
rsMultiVector<T> rsLogViaTaylor(const rsMultiVector<T>& x, int order)
{
  // Uses log(s*x) = log(x) + log(s) for argument reduction.
  using MV = rsMultiVector<T>;
  static const T scl = T(1);
  T s;
  //s = rsNormEuclidean(x) * scl;
  s = rsNormOperator(x)  * scl;
  MV z = (T(1)/s) * x;
  MV y = rsLogViaTaylorSmall(z, order);
  return y + log(s);
}
// We cannot only scale x but also raise it to a (possibly negative) power: z = x^p / s  
// return p*y + log(s)

template<class T>
rsMultiVector<T> rsAtanhViaSeriesSmall(const rsMultiVector<T>& x, int numTerms)
{
  // test:
  T xa = rsNormEuclidean(x);   // absolute value
  rsAssert(xa < T(1)); // convergence requirement
  // maybe return nan, if requirement is violated - returning nan is better than returning garbage
  // in this case

  using MV = rsMultiVector<T>;
  MV xk = x;                           // x^k, initially x^1 = x
  MV x2 = x*x;                         // x^2
  MV y(x.getAlgebra());                // result
  int k;
  for(k = 0; k < numTerms; k++) {
    T w = T(1)/T(2*k+1);               // weight
    if(!rsMakesDifference(y, xk, w))   // convergence test
      break;  
    y  += w * xk;
    xk *= x2;  }
  return y;
}
// References:
// This says that the absolute value of the input should be less than 1:
// https://en.wikipedia.org/wiki/Inverse_hyperbolic_functions#Principal_values_of_the_inverse_hyperbolic_tangent_and_cotangent

template<class T>
rsMultiVector<T> rsLogViaAtanhSeriesSmall(const rsMultiVector<T>& x, int numTerms)
{
  return T(2) * rsAtanhViaSeriesSmall((x-T(1))/(x+T(1)), numTerms);
}
template<class T>
rsMultiVector<T> rsLogViaAtanhSeries(const rsMultiVector<T>& x, int order)
{
  // Uses log(n*x) = log(x) + log(n) for argument reduction.
  using MV = rsMultiVector<T>;
  T scl = T(1.0);
  T s;
  s = rsNormEuclidean(x) * scl;
  //s = rsNormOperator(x)  * scl;
  MV z = (T(1)/s) * x;
  MV y = rsLogViaAtanhSeriesSmall(z, order);
  return y + log(s);
}
// -may diverge
// -maybe instead of the norm, we need to use the squared norm, because we multiply by x^2 in each 
//  iteration step in rsAtanhViaSeriesSmall? wait - no: rsAtanhViaSeriesSmall gets the transformed
//  variable (x-1)/(x+1) as input - for positive scalars x, this has indeed always an absolute 
//  value less than 1

template<class T>
rsMultiVector<T> rsLogViaNewton(const rsMultiVector<T>& x)
{
  using MV = rsMultiVector<T>;
  //MV y = rsLogViaTaylor(x, 3);       // initial guess, todo: tweak order
  //MV y = rsLogViaAtanhSeries(x, 3);
  //MV y = rsLogViaAtanhSeries(x, 10);
  MV y(x.getAlgebra());
  int maxIts = 32;
  int i;
  for(i = 1; i <= maxIts; i++) {
    MV ey = rsExp(y);
    MV dy = (x-ey) / ey;
    if(!rsMakesDifference(y, dy))    // convergence test - we need a tolerance!
      break;
    y += dy; }
  rsAssert(i < maxIts, "rsLog for rsMultiVector did not converge");
  return y;
}
// It's interesting that we seem to get more accurate results for log than for exp even though exp
// is used (iteratively!) in the computation of log. It probably has to do with the self-correcting
// behavior of Newton iteration. More experiments needed...

template<class T>
rsMultiVector<T> rsLog(const rsMultiVector<T>& x)
{
  return rsLogViaAtanhSeries(x, 50);  // todo: check, if 20 is enough as upper limit
  //return rsLogViaNewton(x);
}
// ToDo: maybe instead of using Newton, use the fast converging series with an internal 
// convergence test. 

template<class T>
bool rsIsCloseTo(const rsMultiVector<T>& X, const rsMultiVector<T>& Y, T tol)
{
  int N = X.getAlgebra()->getMultiVectorSize();
  if(Y.getAlgebra()->getMultiVectorSize() != N)
    return false;
  return rsArrayTools::almostEqual(&X.getCoeffs()[0], &Y.getCoeffs()[0], N, tol);
}

// implement power sum: sum_{k=0}^n M^k = (1 - M^(n+1)) / (1-M) ...yeah, that formula works for
// multivectors, too (Eq. 12 in Functions for Multivector Variables (Chappell et al))

// template instantiation


// At 00:30:00, there's an interesting formula from geometric calculus:
//   https://www.youtube.com/watch?v=PGZNYGwsXTw
//   grad(F) = div(F) + curl(F)
// in the sense of 
//   geometricProduct(Nabla, F) = innerProduct(Nabla, F) + outerProduct(Nabla, F)
// for any vector field F

//=================================================================================================

/** A class that tabulates the prime factorizations of all numbers up to some given upper limit. */

template<class T>  // T should be an integer type (may be unsigned)
class rsPrimeFactorTable
{

public:

  rsPrimeFactorTable(T maxNumber) { buildTable(maxNumber); }


  /** Returns the number of prime factors of the given n. By convention, it will return 1 when n is
  zero or one. */
  T getNumFactors(T n) const { return (T) factors[n].size(); }

  T getNthPrime(T n) const { return primes[n]; }  // verify!


  bool isPrime(T n) const { return getNumFactors(n) == 1; }


  const std::vector<T>& getFactors(T n) const { return factors[n]; }




protected:

  void buildTable(T maxNum);

  std::vector<T> primes;

  std::vector<std::vector<T>> factors; 

  // ToDo:
  // -Maybe split buildTable into initTable() and increaseTableSize(T) or growTable(T). The idea is
  //  that client code may later discover that it needs a table larger than it originally thought 
  //  and we want to be able to increase the size dynamically, without re-computing all previously 
  //  calculated values.
  // -To optimize memory access, maybe later use class rsTableau (which is not yet finished) for 
  //  storing the factors. It should use contiguous memory and pre-allocate enough. -> figure out, 
  //  how much is enough, i.e. find a formula for an approximation (upper bound) for the cumulative
  //  count of prime factors. A crude upper bound could perhaps be the integral of the base-2 
  //  logarithm because the base-2 log of a number n is an upper bound for the number of prime 
  //  factors in n. The integral of log(x) is x*(log(x)-1), so the required size grows slightly 
  //  superlinearly, i.e. as x*log(x) - kind of similar to how the number of primes grows slightly
  //  sublinearly as x/log(x)

};

template<class T> 
void rsPrimeFactorTable<T>::buildTable(T N)
{
  using Vec = std::vector<T>;

  // By convention, the numbers 0 and 1 have themselves as their only factors:
  factors.reserve(N+1);
  factors.push_back(Vec({0}));
  factors.push_back(Vec({1}));
  factors.push_back(Vec({2}));

  // The table of primes up to N is built along the way as well:
  // https://en.wikipedia.org/wiki/Prime-counting_function#Inequalities
  //static const double c = 30.0 * log(113.0) / 113.0;  // < 1.25506
  T maxPrimes = (T) ceil(1.25506 * double(N) / log(double(N)));
  primes.reserve(maxPrimes);
  primes.push_back(2);

  // Returns the smallest prime factor in the given n:
  auto leastFactor = [this](T n)
  {
    T m = rsIntSqrt(n);
    size_t i = 0;
    while(primes[i] <= m) {
      if(n % primes[i] == 0)
        return primes[i];
      i++;  }
    return n;  // n is itself prime and therefore its own smallest factor
  };

  // Loop to fill the table:
  for(T n = 3; n <= N; n++)
  {
    T s = leastFactor(n);
    if(s == n) {                       // n is prime
      primes.push_back(n);
      factors.push_back(Vec({n}));  }
    else
    {
      T g = n/s;  

      // suboptimal:
      //factors.push_back(factors[g]); // new element is at index n 
      //rsPrepend(factors[n], s);      // suboptimal!!!

      // optimized:
      factors.push_back(Vec());                      // new empty vector at n-th position
      factors[n].resize(factors[g].size()+1);        // it needs space for the factors of g...
      factors[n][0] = s;                             // ...plus the additional factor s
      for(size_t i = 0; i < factors[g].size(); i++)  // the factors of g get copied into the 
        factors[n][i+1] = factors[g][i];             // factors of n with offset 1
    }
  }

  // The algorithm is based on the idea of doing trial divisions of each number n by primes, 
  // starting at the smallest prime 2 moving upwards. As soon as a number is discovered to be 
  // divisible by some prime s we know that n is not a prime and s is its smallest prime factor. We
  // therefore compute its greatest (possibly still composite) factor g = n/s. The prime 
  // factorization of n is then given by the prime factorization of g but with an additional factor
  // of s. If the smallest divisor s of n is n itself, then n is a prime and therefore its own only
  // factor. As side effect of building the table of factorizations of all numbers up to N, we also
  // build a table of primes up to N.
  //
  // I think, the complexity of the algorithm should be of the order O(N*sqrt(N)) = O(N^1.5). The 
  // outer loop runs up to N, the inner loop (buried in the call to leastFactor) up to the sqrt
  // of the current outer loop index n. Actually, it doesn't run over all numbers up to sqrt(n) but
  // only over the primes <= n, so we may get O(N^1.5 / log(N)) due to the prime counting function. 
  // However, that may be eaten up by the requirement to copy arrays in the "else" branch and the 
  // size of these arrays also grows logarithmically. All in all, O(N^1.5) doesn't seem too bad for
  // an algo that computes a complete table of the prime factorizations of all numbers up to N.
  // Hmm...actually, the loop in the else branch is a 2nd loop independent from the loop in 
  // leastFactor, so the overall complexity of the inner loops should be given by the maximum of 
  // these 2 complexities, so we may actually still get away with O(N^1.5 / log(N)). The first 
  // inner loop has complexity O(sqrt(n)/log(sqrt(n))) and the second has O(log(n)) - only the 
  // first one really counts. Maybe make some more thorough analysis taking into account the 
  // average time for the inner loop - it rarely loops up to sqrt(n) - maybe that contributes
  // only a constant factor but maybe it actually changes the order. The fact that the inner loop
  // only runs up to n and not up to N perhaps contributes a constant factor of 1/2 which is the 
  // average of n taken over all N (roughly). We could do away with the trial division by 2 for
  // all even n - but the increased complexity of the implementation may not be justified. We could
  // perhaps handle that in leastFactor by bitmasking without increasing complexity too much, i.e.
  // do if(n & 1) return 2; as very first line in leastFactor. That early return would kick in half
  // of the time and even save the rsIntSqrt. Maybe do all that together with using rsTableau when
  // a production version of this code is needed and keep the implementation here as prototype for
  // comparisons in unit tests.

  int dummy = 0;
}

//=================================================================================================

/** just a stub at the moment

In mathematics, a field is a set in which certain operations are defined and these operations
behave in the same way as addition and multiplication in rational or real numbers. That means, 
besides other things that, the multiplicative inverses must exist for each element except zero.
Rational, real or complex numbers are all infinite fields but finite fields also exist. The 
simplest finite fields are the modular integers when the modulus is a prime number. The only other
finite fields that exist are (isomorphic to) those, whose number of elements is an integer power
of a prime. But for these, simple modular arithmetic doesn't produce the field. Addition and 
multiplication require some more elaborate algorithms. These are implemented in a naive way for
learning purposes by the classes rsFiniteFieldNaive and rsFiniteFieldElementNaive which work 
together in tandem in the same way as rsGeometricAlgebra and rsMultiVector: every element holds a
pointer to the algebra object which is consulted to perform the arithmetic operation.

...tbc...

See:
https://en.wikipedia.org/wiki/Finite_field


*/

template<class T>  // T should be an integer type (may be unsigned)
class rsFiniteFieldNaive
{

public:

  rsFiniteFieldNaive(T base, T exponent) : p(base), k(exponent)
  {
    //RAPT::rsAssert(rsIsPrime(p));  // todo
    //generateTables();
  }

protected:

  T p;  // Base in p^m, should be prime
  T k;  // Exponent in p^k, a positive integer


  // We need to form the field of polynomials of degree <= k over the modular integers with 
  // modulus p. Then we take that field modulo a specific polynomial M(x) that plays the role
  // of a modulus ...tbc...
  //using ModInt = RAPT::rsModularInteger<T>; // 
  //rsPolynomial<ModInt> M;  // M(x) is an degree k polynomial that is irreducible in Z_p = Z/pZ
  // ...so we have two levels of modular arithmetic at play here? The lower level being the usage 
  // of modular integers and the higher level being the use doing all polynomial operations modulo
  // the given M(x), requiring polynomial division with remainder?


};

// In a non-naive implementation, we should build tables for addition and multiplication in the 
// constructor. Each polynomial, i.e. each array of polynomial coeffs, maps to a unique integer
// in the range 0...p^k-1. For each pair of such integers (mapped polynomials) we need to specify
// what the result of their addition and multiplication should be - coming from the same set of 
// 0...p^k-1. The multiplication table can be turned into a 1D array rather than a full blown 2D
// matrix by a trick explained in Weitz pg. 744. I hope, a similar trick is possible for addition
// too. Weitz says nothing about that because he's only covering the case for p=2 in which addition
// reduces to xor such that no table is needed. The method there uses a primitive k-th (?) root of 
// unity, i.e. a number that, when multiplied by itself k times gives one. Maybe an analog for
// addition could be a number that when added to itself k times gives zero? And that number would 
// be just the number 1, regardless of the modulus? ...  figure this out!


//=================================================================================================

template<class T> 
class rsParticleSystem2D
{

public:

  // Lifetime:
  rsParticleSystem2D(int numParticles);


  // Setup:
  void setNumParticles(int newNumber);
  void setMasses(std::vector<T>& newMasses);
  void setPositions(std::vector<rsVector2D<T>>& newPositions);





  // Processing:
  void computeForcesNaive(std::vector<rsVector2D<T>>& forces); // for referencce in unit tests, inefficient
  void computeForcesFast(std::vector<rsVector2D<T>>& forces); 
  // Maybe the forces vector doesn't need to be stored as member. it can be passed in as
  // parameter. That makes it also convenient to test the force computation





protected:

  std::vector<rsVector2D<T>> positions, velocities;
  std::vector<T> masses;

};

template<class T> 
rsParticleSystem2D<T>::rsParticleSystem2D(int numParticles)
{
  setNumParticles(numParticles);
}

template<class T> 
void rsParticleSystem2D<T>::setNumParticles(int newNumber)
{
  positions.resize(newNumber);
  velocities.resize(newNumber);
  //forces.resize(newNumber);
  masses.resize(newNumber);
}

template<class T> 
void rsParticleSystem2D<T>::setMasses(std::vector<T>& newMasses)
{
  rsAssert(newMasses.size() == masses.size());
  rsCopy(newMasses, masses);
}

template<class T> 
void rsParticleSystem2D<T>::setPositions(std::vector<rsVector2D<T>>& newPositions)
{
  rsAssert(newPositions.size() == positions.size());
  rsCopy(newPositions, positions);
}


template<class T> 
void rsParticleSystem2D<T>::computeForcesNaive(std::vector<rsVector2D<T>>& forces)
{
  int N = positions.size(); 
  rsAssert((int)velocities.size() == N);
  rsAssert((int)forces.size() == N);

  for(int i = 0; i < N; i++)
  {
    forces[i] = 0;
    rsVector2D<T> p = positions[i];

    // Loop over all other particles and sum up their contributions to the total force that acts
    // on the current particle i:
    for(int j = 0; j < N; j++)
    {
      if(j != i)
      {
        rsVector2D<T> q = positions[j] - p; // delta-vector

        // ToDo: factor out, use modified gravity laws:
        T d = rsNorm(q);
        forces[i] += masses[j] * masses[i] * q / (d*d*d);
        // we use d^3 and not d^2 because q in the numerator also still contains a factor of d, 
        // i.e. q is not normalized to unit length

        int dummy = 0;

      }
    }
  }
}
// maybe it should take the positions as parameter, too?

template<class T> 
void rsParticleSystem2D<T>::computeForcesFast(std::vector<rsVector2D<T>>& forces)
{
  int N = positions.size(); 
  rsAssert((int)velocities.size() == N);
  rsAssert((int)forces.size() == N);

  T M = 0;                     // total mass of all particles (maybe precompute - it's a constant!)
  rsVector2D<T> S(0, 0);       // sum of all position vectors weighted by the mass of the particle
  for(int i = 0; i < N; i++)
  {
    M += masses[i];
    S += masses[i] * positions[i];
  }
  // By the way: the center of mass of all particles would now be given by S/M. But we don't need 
  // that in further computations. We will only need the weighted sum itself.

  for(int i = 0; i < N; i++)
  {
    T M_i = M - masses[i];
    // sum of all masses except the i-th

    rsVector2D<T> S_i = S - masses[i] * positions[i];  
    // Weighted sum of all positions expcept the i-th

    rsVector2D<T> C_i = S_i / M_i;
    // Center of mass of all particles except the i-th

    rsVector2D<T> Q = C_i - positions[i];
    // Difference vector between current particle i and the center of mass of all other particles

    T D = rsNorm(Q);
    // Distance from i-th particle to center of mass of all others

    forces[i] = M_i * masses[i] * Q / (D*D*D);
    // Force excerted on i-th particle by all the others
  }


  // Idea:
  // -We first compute the center of mass of all particles. This is an O(N) operation.
  // -For each particle, we subtract out its contribution to the center of mass which gives the
  //  center of all other masses. We treat this center of all other masses as a single replacement
  //  mass that acts as a stand-in for all others. I'm not sure, if that is supposed to work. Maybe
  //  only for very specific force laws like Hooke's but not for the (nonlinear) gravitational law?
  //  But maybe the law can be taken into account when we compute S?
}

//=================================================================================================

/** A class for representing polynomials that are (pre-) multiplied by some power of x, say x^m. 
This shifts all exponents by m. The coefficient array a[0]...a[N] of an N-th degree polynomial will
then represent:

  p(x) = x^m * (a0 + a1*x + a2*x^2 + a3*x^3 + ... + aN*x^N)
       = a0*x^m + a1*x^(m+1) + a2*x^(m+2) + a3*x^(m+3) + ... + aN*x^(m+N)

where for m = 0,we just get our regular old polynomial. The power m can also be negative. This 
makes this class helpful when we want to represent polynomials that are meant to be in inverse 
powers of x: we simply choose m = -N and reverse the coefficient array. Polynomials in inverse 
powers of z are a very common thing in DSP. Normally, when working with polynomials, it doesn't 
really matter, whether these represent polynomials in x or in 1/x as long as the usage is 
consistent among them all. But as soon as we want to mix both types, like multiply a polynomial p 
in x by another polynomial q which is in 1/x, this class can be used to make them compatible. Such
a situation occurs in the definition of paraunitary filters.

See here, section 1.3.3:
https://www.snnu.uni-saarland.de/wp-content/uploads/2015/05/BMT1822-Introduction-to-Paraunitary-Filter-Banks-and-Orthogonal-Expansions-in-l2.pdf

...tbc...  */

template<class T> 
class rsLiftedPolynomial : private RAPT::rsPolynomial<T> 
{

public:

  using Base = RAPT::rsPolynomial<T>;

  //-----------------------------------------------------------------------------------------------
  /** \name Lifetime */

  
  rsLiftedPolynomial(int degree = 0, int power = 0, bool initWithZeros = true)
    : RAPT::rsPolynomial<T>::rsPolynomial(degree, initWithZeros) { m = power; }
  
  rsLiftedPolynomial(std::initializer_list<T> l, int power = 0)
    : RAPT::rsPolynomial<T>::rsPolynomial(l) { m = power; }

  /*
  rsLiftedPolynomial(const std::vector<T>& coefficients)
    : RAPT::rsPolynomial<T>::rsPolynomial(coefficients) {}


  rsLiftedPolynomial(const T& number) 
    : RAPT::rsPolynomial<T>::rsPolynomial(number) {}
    */


  //-----------------------------------------------------------------------------------------------
  /** \name Setup */

  void setPower(int newPower) { m = newPower; }


  //-----------------------------------------------------------------------------------------------
  /** \name Evaluation (High Level) */

  /** Evaluates the polynomial at the given input x. */
  T evaluate(T x) const { return evaluate(x, &coeffs[0], getDegree(), m); }


  //-----------------------------------------------------------------------------------------------
  /** \name Misc



  /** If the function p(x) is currently a (generalized) polynomial in x, calling this will turn it
  into one in x^(-1) = 1/x instead. This is achieved by reversing the coefficient array and making
  appropriate adjustments to the power m of the pre-factor x^m. */
  void invert();


  static T evaluate(const T& x, const T *a, int degree, int offset);

  static int multiply(const T *a, int aDegree, int aOffset,
                      const T *b, int bDegree, int bOffset, T *result);
  // todo: 
  // -document, how long the result array needs to be...actually, in this respec, nothing really
  //  changes - the same rules as for normal polynomials apply

  // todo:
  // -implement add, subtract, divide ...maybe divide is like regular divide and the result-offset
  //  is obtained by subtracting the input offsets? what about the remainder?


protected:

  int m = 0;  // The offset, i.e. the power of x by which the whole thing is multiplied

};

template <class T>
void rsLiftedPolynomial<T>::invert()
{
  rsReverse(Base::coeffs);    // maybe we need this->coeffs syntax for clang, gcc?
  int N = Base::getDegree();  // dito?
  m = -N - m;                 // verify this!...ok - seems to work
}

template <class T>
T rsLiftedPolynomial<T>::evaluate(const T& x, const T *a, int N, int m)
{
  return rsPow(x, m) * Base::evaluate(x, a, N);
}

template <class T>
int rsLiftedPolynomial<T>::multiply(const T *a, int aDegree, int aOffset, const T *b, int bDegree,
  int bOffset, T *result)
{
  Base::multiply(a, aDegree, b, bDegree, result);
  return aOffset + bOffset;  // verify this!
}

// ToDo:
// -Implement evaluation, addition, subtraction, multiplication and (maybe) division.
// -What about integration and differentiation? I gues the x^(-1) = 1/x term may be problematic 
//  because the integral of 1/x is log(x) which is not a function in terms of powers of x. However
//  differntiation may still be unproblematic. And maybe at some stage, we can also extend the
//  class to contain logarithmic terms? But via repeated integration, these will then also 
//  proliferate into terms like x * (log(x) - 1) which is the antiderivative of log(x). Maybe we 
//  could have a regular polynomial and a 2nd polynomial that is multiplied by log(x) and get a set
//  of functions f(x) = p(x) + log(x) * q(x) wher p,q are polynomials that is closed under 
//  integration and differentiation? But then what about composition?
// -Can M be a real or even complex number?
// -Find a better name...maybe PoweredPolynomial, ElevatedPolynomial (!), BoostedPolynomial, 
//  LiftedPolynomial (!!), HoistedPolynomial, RaisedPolynomial
// -Maybe make also classes for (p(x))^M, p(x^M), p(x+a), p(1/x) - the latter could be done in 
//  terms of this class by reversing the coeff array and choosing M = -N but we probably should
//  not implement it this way. Maybe evaluation should just compute the reciprocal (left inverse in
//  general?) and then call the regular evaluation routine. Possible names:
//    p(x+a): rsShiftedPolynomial
//    p(1/x): rsInvertedPolynomial
//    p(x^m): rsPowerPolynomial
//    p^m(x): rsRaisedPolynomial
// -Define whether we mean to pre- or post-multiply by x^M. This may make a difference if 
//  the type T has a noncommutative multiplication such as matrices. Polynomials of matrices are
//  actually not an uncommon thing so that may actually be of practical relevance. Perhaps 
//  pre-multiply is the better choice because in DSP texts, we sometimes see things like
//  z^(-M) * H(z) (see Introduction to Digital Filters by JOS, page 133) but rarely, if ever 
//  H(z) * z^(-M). But maybe then it should be called PreLiftedPoly or PreMultipliedPoly.
// -But perhaps we need to require T to have a commutative multiplication anyway because otherwise 
//  products like r(x) = (x^m * p(x)) * (x^n * q(x)) cannot be generally computed as
//  r(x) = x^(m+n) * (p(x) * q(x)) which is what our multiplier does. But maybe we may not strictly
//  require that *all* multiplications are commutative but in this case, just the one between
//  p(x) and x^n, i.e. the polynomial factor of the 1st argument and the pre-multiplier of the 2nd
//  argument. These need to commute and this will of course be ensured, if all multiplications 
//  commute anyway but maybe we don't need to require that *all* multiplications need to commute? 
//  Figure this out and document it.
// -Make also a class LiftedRationalFunction because eventually, we want to apply it also to IIR
//  transfer functions


//=================================================================================================

/** A class for implementing digital filters in state space form. It implements the (vector/MIMO)
difference equation:

  y[n]   = C * x[n] + D * u[n]           output generation for sample index n
  x[n+1] = A * x[n] + B * u[n]           state update to prepare for the next sample

where x is a length N state vector, u is a length p input vector, y is a length q output vector, A
is an N-by-N state transition matrix, B is an N-by-p injection matrix, C is an q-by-N output matrix 
and D is a q-by-p feedthrough matrix. (verify sizes and terminology - I made some of them up. Edit:
on page 357, in a code comment JOS calls B,C,D the input, output, feed-around matrices 
respectively)

The system has the q-by-p MIMO transfer function matrix:

  H(z) = D + C*(z*I - A)^(-1) * B     (1) Eq G.5

 ...tbc...

References:
  (1) Introduction to Digital Filters with Audio Application (Julius O. Smith)

*/


template<class T> 
class rsStateSpaceFilter
{

public:

  /** Sets up the shapes of our matrices to allow for the desired number of ins/outs/states but 
  doesn't initialize the contents of those matrices. If you expect to change these sizes later 
  during realtime processing, this function can also be used once with the maximum epxected sizes
  to reserve enough memory to avoid need ing re-allocations later when the size changes dynamically
  during processing. */
  void setDimensions(int numIns, int numOuts, int numStates);

  /** Sets up our A,B,C,D matrices. This may reshape our member matrices and will copy the data 
  from the given argument matrices into them. Reshaping may re-allocate memory unless our member 
  matrices already have (more than) enough space, which you can ensure using setDimensions(). */
  void setup(const rsMatrixView<T>& A, const rsMatrixView<T>& B, const rsMatrixView<T>& C,
    const rsMatrixView<T>& D);
  // ToDo:
  // -Maybe we should keep only references or pointers to these matrices here? The way we do it now
  //  requires redundant existence of these matrices in memory which is bad. But maybe the state 
  //  vector x and its temporary storage t should nevertheless stay non-reference members?
  //  ...hmmm...

  /** Processes a single MIMO output frame at a time. */
  void processFrame(T* ins, T* outs)
  {
    // Wrap some matrix-view objects around inputs and output to view them as p-by-1 and q-by-1 
    // column-vectors:
    using MV = rsMatrixView<T>;
    MV u(p, 1, ins);                                               // u = input vector
    MV y(q, 1, outs);                                              // y = output vector

    // Compute output y = C*x + D*u:
    MV::matrixMultiply(          C, x, &y);                        // y = C*x
    MV::matrixMultiplyAccumulate(D, u, &y);                        // y = C*x + D*u

    // Update the state x = A*x + B*u:
    MV::matrixMultiply(          A, x, &t);                        // t = A*x
    MV::matrixMultiplyAccumulate(B, u, &t);                        // t = A*x + B*u
    rsArrayTools::copy(t.getDataPointer(), x.getDataPointer(), N); // x = t
  }
  // Notes:
  // -Needs more tests, if it does the right thing.
  // -We currently have no safeguards against the client passing too short arrays - should we? We 
  //  already know, that ins should have length q and outs have length p and expect the caller to
  //  know and respect that, too. After all, the client code must have set us up that way at some 
  //  previous point via calling e.g. setup(). Safeguarding here would require otherwise useless 
  //  and redundant function parameters like numIns, numOuts. Not sure, if that's a good idea.
  //  ...but maybe...we'll see....

  /** Resets out internal state vector to all zeros. */
  void reset() { x.setToZero(); }
  // ToDo: 
  // -Maybe have a function setState(T* newState) that lets the client explicitly set up any 
  //  desired initial state/condition.


protected:


  int N = 0;  // number of internal states
  int p = 0;  // number of inputs
  int q = 0;  // number of outputs
  // These are actually redundant but convenient. They could be inferred from certain row- and 
  // column settings in our matrices below if saving that little amount of extra space seems 
  // worthwhile. Maybe rename them into numStates, numIns, numOuts.


  rsMatrix<T> x, t, A, B, C, D;
  // Meaning of those matrices
  //   x: state vector, N-by-1 column vector
  //   t: temporary storage for x during state update
  //   A: state transition matrix, N-by-N matrix
  //   B: injection matrix, N-by-p matrix (verify!)
  //   C: output matrix, q-by-N matrix (verify!)
  //   D: passthrough matrix, q-by-p (verify!)
  // We also use the notation:
  //   u: input vector, p-by-1
  //   y: output vector, q-by-1
  // but we don't need any class members for these I/O variables. See (1) pg 345, Appendix G.

  // ToDo: 
  // -Check terminology. I made some of it up myself (feedthrough, injection). The book (1) 
  //  doesn't give them special names.
  // -Perhaps production code should use sparse matrices? I think, the state update matrices are
  //  typically sparse, right? But what about the other matrices? Are they also typically sparse?
  //  Maybe only A should be sparse but B,C,D dense? ...figure out!
  // -Implement a getTransferFunction() function that returns an rsMatrix of type
  //  rsRationalFunction and a getTransferFunctionAt(Complex z, int i, int j) that evaluates
  //  the (i,j)th point-to-point transfer function from input j to output i at the given z. Can we
  //  do this without memory allocations? I mean conveniently, without introducing new members or
  //  a workspace? I think, we nee to implement: H(z) = D + C*(z*I - A)^(-1) * B. There appears an
  //  inverse matrix

};

template<class T> 
void rsStateSpaceFilter<T>::setDimensions(int numIns, int numOuts, int numStates)
{
  p = numIns;
  q = numOuts;
  N = numStates;
  x.setShape(N, 1);
  t.setShape(N, 1);
  A.setShape(N, N);
  B.setShape(N, p);
  C.setShape(q, N);
  D.setShape(q, p);
  reset();
}


template<class T> 
void rsStateSpaceFilter<T>::setup(const rsMatrixView<T>& newA, const rsMatrixView<T>& newB,
  const rsMatrixView<T>& newC, const rsMatrixView<T>& newD)
{
  // Retrieve and set up desired dimensions:
  N = newA.getNumRows();     // number of states
  p = newB.getNumColumns();  // number of inputs
  q = newC.getNumRows();     // number of outputs
  setDimensions(p, q, N);
  // ToDo: Verify and document that no allocations take place here, when alreade enough memory was
  // allcoated previously. See rsMatrix::setShape - it calls resize on a std::vector which should
  // reallocate only in case of growth.

  // Perform som sanity checks on the input matrices:
  rsAssert(newA.isSquare(), "State transition matrices must be square");
  // ...more to come: make sure, that all the desired relations between the shapes of the given 
  // matrices are satisfied.

  // Copy the new matrix data into our members:
  A.copyDataFrom(newA);
  B.copyDataFrom(newB);
  C.copyDataFrom(newC);
  D.copyDataFrom(newD);
  // hmm...copyDataFrom also calls setShape. These calls are redundant with those in setDimensions.
  // Maybe it doesn't matter but perhaps it would be nicer to avoid it...we'll see...
  // Maybe we should just keep references to some A,B,C,D matrices owned by cleint code anyway.
  // That avoids redundancies and makes it easier to implement time-variant operation. Client code
  // could just vary the matrices. 

  //rsError("Not yet implemented");
}

//=================================================================================================

/** A class for representing Quaternions. Quaternions are a kind of 4-dimensional numbers similar 
to how complex numbers are 2-dimensional numbers. In fact, the complex numbers are found as a 2D 
subset within the quaternions. A quaternion q can be written as:

  q = q0 + i*q1 + j*q2 + k*q3

where i,j,k are three different imaginary units which all satisfy the squaring to one property, 
i.e. i^2 = j^2 = k^2 = 1. They also satisfy the cyclic product relations: i*j = -j*i = k, 
j*k = -k*j = i, k*i = -i*k = j. In some contexts, it is useful to view a quaternion as being 
composed of a scalar part q0 and a 3D vector part (q1,q2,q3). This is also the way, it is 
internally represented here in this class. It's - of course - also possible to just represent it 
by 4 raw numbers without structuring them further but I think, the formal scalar + vector 
representation reveals some more insights. One could also represent quaternions a pair of complex
numbers z,w as q = z + w*j...tbc... 

References
  (1) Mathematik mit 2x2 Matrizen (Hans J¸rgen Korsch)   */

template<class T> 
class rsQuaternion2
{
  // we need to append a "2" because there is already some other definition of rsQuaternion 
  // in Relativity.h ...this old one should probably be superseded by this...or maybe implement
  // various representations - maybe they all have theri strengths and weaknesses? Maybe if one
  // turns out to be the most efficient computationally (maybe the raw one?), the others may 
  // nevertheless provide certain other insights?

public:

  rsQuaternion2(const T& a, const T& b, const T& c, const T& d)
    : s(a), v(b, c, d) {}

  rsQuaternion2(const T& scalarPart, const rsVector3D<T>& vectorPart)
    : s(scalarPart), v(vectorPart) {}


  rsQuaternion2 operator*(const rsQuaternion2& p) const 
  { 
    return rsQuaternion2(s*p.s - rsDot(v, p.v), s*p.v + p.s*v + rsCross(v, p.v));
  }
  // see (1) pg 128

  /** [q,p] = q*p - p*q */
  static rsQuaternion2 commutator(const rsQuaternion2& q, const rsQuaternion2& p)
  {
    return rsQuaternion2(T(0), T(2)*rsCross(q.v, p.v));
  }

  /** {q,p} = q*p + p*q */
  static rsQuaternion2 anticommutator(const rsQuaternion2& q, const rsQuaternion2& p)
  {
    rsQuaternion2(T(-2)*rsDot(q.v*p.v), rsVector3D<T>(0,0,0));
  }


protected:

  T s;              // scalar part q0
  rsVector3D<T> v;  // vector part (q1,q2,q3)

};

//-------------------------------------------------------------------------------------------------

/** Another representation of quaternions as a pair of complex numbers. To distinguish these two
complex numbers, we actually use a nested complex type whose real and imaginary part are used our
two (level 1) complex numbers. */

template<class T> 
class rsQuaternion3
{

public:

  rsQuaternion3(const T& a, const T& b, const T& c, const T& d)
    : q(rsComplex<T>(a, b), rsComplex<T>(c, d)) {}

  rsQuaternion3(const rsComplex<T>& ab, const rsComplex<T>& cd)
    : q(rsComplex<T>(ab), rsComplex<T>(cd)) {}

  rsQuaternion3 operator*(const rsQuaternion3& p) const 
  { 
    rsComplex<T> re = q.re * p.q.re  -  q.im * rsConj(p.q.im);
    rsComplex<T> im = q.re * p.q.im  +  q.im * rsConj(p.q.re);
    return rsQuaternion3(re, im);
  }
  // This needs thorough verifications..I'm not even sure, if this "nested complex" is the right
  // way to think about it. Maybe instead, we need indeed just a pair of complex numbers. see
  // (1) pg 129


protected:

  rsComplex<rsComplex<T>> q;

};
// We need also the other operators and more tests


//-------------------------------------------------------------------------------------------------

/** Yet another representation of quaternions as a complex 2x2 matrix of the form:

  q = (a,b,c,d) = [ a+b*i  c+d*i] = [ A  B ]
                  [-c+d*i  a-b*i]   [-B* A*]

where A* denotes the complex cojugate of A. 

*/

template<class T> 
class rsQuaternion4
{

public:

  rsQuaternion4(){}

  rsQuaternion4(const T& a, const T& b, const T& c, const T& d)
  {
    using C = rsComplex<T>;
    Q.setValues(C(a,b), C(c,d), C(-c,d), C(a,-b));
  }

  rsQuaternion4 operator*(const rsQuaternion4& p) const
  {
    rsQuaternion4 r;
    r.Q = Q * p.Q;
    return r;
  }
 
protected:


  rsMatrix2x2<rsComplex<T>> Q;

};

// implement rsConj for rsQuaternion4. We need to take the Hermitian conjugate of the matrix

//-------------------------------------------------------------------------------------------------

/** Another representation of quaternions as a complex 2x2 matrix of the form:

q = (a,b,c,d) = [a-d*i -c-b*i] = [A  -B ]
                [c-b*i  a+d*i]   [B*  A*] */

template<class T> 
class rsQuaternion5
{

public:

  rsQuaternion5(){}

  rsQuaternion5(const T& a, const T& b, const T& c, const T& d)
  {
    using C = rsComplex<T>;
    Q.setValues(C(a,-d), C(-c,-b), C(c,-b), C(a,d));
  }

  rsQuaternion5 operator*(const rsQuaternion5& p) const
  {
    rsQuaternion5 r;
    r.Q = Q * p.Q;
    return r;
  }

protected:


  rsMatrix2x2<rsComplex<T>> Q;

};

// It is also possible to translate these two different representations as complex 2x2 matrices 
// into corresponding representations of real 4x4 block matrices by replacing 
//
//   1 = I = [1 0],   J = [ 0 1]
//           [0 1]        [-1 0]
//
// But we don't do a implementation of that here, though because that would be boring.


//=================================================================================================

// This code below may go elsewhere eventually. It's currently here only because we need such a 
// formula/code as subroutine to find the potential for the zeta function via its Laurent series
// representation. That's the context in which it was developed, but it may be useful in 
// different contexts as well. Maybe eventually these (mini-)algorithms should go into class 
// rsBivariatePolynomial. Maybe have a function getRealCoeffsComplexPower,
// getImagCoeffsComplexPower there.
//
// When forming a power of a complex variable z, i.e. w = z^n, we want to find explicit 
// expressions for real and imaginary part of w in terms of real and imaginary parts of 
// z = x + i*y. Set w = u + i*v. We get:
// 
//   re(w) = u(x,y) = \sum_{k=0}^{n/2}     (-1)^k * B(n,2*k)   * x^(n-2*k)     * y^(2*k)
//   im(w) = v(x,y) = \sum_{k=0}^{(n-1)/2} (-1)^k * B(n,2*k+1) * x^(n-(2*k+1)) * y^(2*k+1)
//
// where a integer-division is used in the upper summation limit, if necessarry and B(n,k) is the
// binomial coefficient n-choose-k.

template<class T>
int rsRealCoeffsComplexPower(int n, T* coeffs, int* xPowers, int* yPowers)
{
  int k;
  for(k = 0; k <= n/2; k++) 
  {
    coeffs[k]  = pow(-1, k) * rsBinomialCoefficient(n, 2*k);
    xPowers[k] = n-2*k;
    yPowers[k] = 2*k;
  }
  return k;   // We return the number of nonzero terms
}
// Maybe rename xPowers to xExponents etc.

template<class T>
int rsImagCoeffsComplexPower(int n, T* coeffs, int* xPowers, int* yPowers)
{
  // The edge case must be treated separately:
  if(n == 0)
  {
    //coeffs[0]  = 0;
    //xPowers[0] = 0;
    //yPowers[0] = 0;
    return 0;
  }
  // Check, if we really need to assign zeros to coeffs[0], etc. That might actually be 
  // superfluous because if we return 0, client code is not supposed to read the arrays anyway.

  int k;
  for(k = 0; k <= (n-1)/2; k++) 
  {
    coeffs[k]  = pow(-1, k) * rsBinomialCoefficient(n, 2*k+1);
    xPowers[k] = n-(2*k+1);
    yPowers[k] = 2*k+1;   
  }
  return k;
}

template<class T>
int rsPotentialCoeffsComplexPower(int n, T* coeffs, int* xPowers, int* yPowers)
{
  int m = rsImagCoeffsComplexPower(n, coeffs, xPowers, yPowers);
  for(int k = 0; k < m; k++) {             // Integrate wrt y by incrementing y exponent and
    yPowers[k] += 1;                       // dividing the coeff by the new y-exponent.
    coeffs[k] = -coeffs[k] / yPowers[k]; } // The negation is because we use the imag part.
  coeffs[m]  = T(1) / (n+1);               // The integration constant is a function of x
  xPowers[m] = n+1;                        // given by x^(n+1) / (n+1)
  yPowers[m] = 0;                          // ...times y^0
  return m+1;                              // Output arrays are longer by 1 now
}
// The function is based on integrating rsImagCoeffsComplexPower with respect to y, then adding as
// "integration constant" a term of the form x^(n+1) / (n+1). It's constant with respect to y but
// a function of x. Alternatively, one could also integrate rsRealCoeffsComplexPower with respect 
// to x and then add a y^(n+1) / (n+1) term when n is odd. See the pattern of coeffs for re/im 
// parts in the unit tests or the zeta paper. Maybe implement both and test both - just for proof 
// of concept. They should lead to coeff arrays that represent the same polynomial but the coeffs 
// may come out in different order when doing it the other way. The negation is because we have 
// based our function on integrating the imag part (with respect to y). And that imag part needs to
// be negated for the Polya vector field. 

template<class T>
T rsEvaluateBivariatePolynomial(T x, T y, int m, T* coeffs, int* xPowers, int* yPowers)
{
  T z = T(0);
  for(int k = 0; k < m; k++)
    z += coeffs[k] * pow(x, xPowers[k]) * pow(y, yPowers[k]);
  return z;
}
// Maybe not the most efficient way to do it, but this is just for experimentation. When doing 
// something like this in production, we may write a class for sparse bivariate polynomials that
// stores an array of triples: (coeff, xPower, yPower). Maybe the struct for that triple could be 
// named rsSparseBivariatePolynomial::Term. The general rsBivariatePolynomial class stores the 
// coeffs as matrix which is overkill here because most entries would be zero.


//=================================================================================================

/** A class that implements various algorithms to evaluate the Riemann zeta function for complex 
arguments and some related functions. It's meant for exploring the properties of the function by
facilitating to plot it but also to numercially verify certain indentities etc. */

class rsRiemannZetaFunction
{

public:

   //using Complex = std::complex<double>;

   //----------------------------------------------------------------------------------------------
   /** \name Evaluation of zeta itself */

   /** Evaluates the zeta function via z(s) = \sum_{n=1}^N n^(-s) where N = numTerms. This sum 
   converges only for real(s) > 1. */
   static std::complex<double> evalViaOriginalSum(std::complex<double> s, int numTerms);
   // todo: say something about the speed of convergence. I think, it will converge faster when
   // real(s) is larger - see experiment testRiemannZeta()

   /** Evaluates the Riemann zeta function via the related Dirichlet eta function like so:
   z(s) = eta(s) / (1-2^(1-s). This converges for real(s) > 0. */
   static std::complex<double> evalViaAlternatingSum(std::complex<double> s, int numTerms);

   static std::complex<double> evalViaBinomialSum(std::complex<double> s, int numTerms);

   static std::complex<double> evalViaEulerProduct(std::complex<double> s, int numTerms, 
     const int* primeTable);
   // requires a table of prime numbers at least as large as numTerms

   static std::complex<double> evalViaLaurentSeries(std::complex<double> s, int numTerms);


   static std::complex<double> evalViaBoostSum(std::complex<double> s, int n);
   // implements this formula:
   // https://www.boost.org/doc/libs/1_65_0/libs/math/doc/html/math_toolkit/zetas/zeta.html
   //
   // Function returns totally wrong result for s = 2.0 + 1.0*i; There's some code in the test
   // function that tests this but is currently commented out... -> figure this out!

   // ToDo: add evaluation function based on the code posted here:

   /**
   
   Based on:
   https://stackoverflow.com/questions/41549533/riemann-zeta-function-with-complex-argument */
   static std::complex<double> evalViaStackOverflowAlgo(std::complex<double> s, int n);
   // ToDo: figure out and document, what formula the algo is based on



   //----------------------------------------------------------------------------------------------
   /** \name Evaluation of the Polya vector field of zeta */

   static void vectorFieldViaOriginalSum(double x, double y, double* u, double* v, 
     int numTerms);

   static void vectorFieldViaLaurentSeries(double x, double y, double* u, double* v, 
     int numTerms);


   //----------------------------------------------------------------------------------------------
   /** \name Evaluation of the Polya potential of zeta */

   /** Evaluates the Polya potential using the formula that was derived from the original sum.
   It converges only for x > 1 (I think - verify!). */
   static double potentialViaOriginalSum(double x, double y, int numTerms);


   static double potentialViaLaurentSeries(double x, double y, int numTerms);

   //----------------------------------------------------------------------------------------------
   /** \name Evaluation of functions related to zeta */

   /** The Dirichlet eta function is a variant of the Riemann zeta function which alternating 
   signs in the summation. Its given by  eta(s) = \sum_{n=1}^N (-1)^(n-1) * n^(-s). The sum 
   converges for real(s) > 0. */
   static std::complex<double> evalDirchletEta(std::complex<double> s, int numTerms);


   //----------------------------------------------------------------------------------------------
   /** \name Some subroutines, e.g. functions to evaluate individual terms in series expansions 
   etc. */

   /** Implements the formula to compute real and imaginary part of the n-th term in the Dirichlet 
   function, i.e. n^(-s) / (1 - 2^(1-s)) but without resorting to complex numbers. This is just for
   verifying the derived formulas for re and im numerically and probably has no practical 
   application in production. These formulas may serve as starting point for finding a Polya 
   potential for such a term. I did not yet succeed cracking the integrals, though. */
   static std::complex<double> dirichletTermViaReIm(std::complex<double> s, int n);

   /** Returns the n-th coefficient in the Laurent series expansion of zeta. The first nonzero 
   coeff has index -1 and is equal to 1. For n >= 0 coeffs, a table involving the Stieltjes 
   constants gamma[n] is used. */
   static double laurentSeriesCoeff(int n);


protected:

  static const int numGammas = 32;
  static const double gamma[numGammas]; 
  // Precomputed table of the Stieltjes constants, see:
  // https://en.wikipedia.org/wiki/Stieltjes_constants
  // https://de.wikipedia.org/wiki/Stieltjes-Konstanten

};

const double rsRiemannZetaFunction::gamma[32] = 
{ 
  +.5772156649015328606065121,
  -.7281584548367672486058638e-1,
  -.9690363192872318484530386e-2,
  +.2053834420303345866160047e-2,
  +.2325370065467300057468170e-2,
  +.7933238173010627017533349e-3,
  -.2387693454301996098724218e-3,
  -.5272895670577510460740975e-3,
  -.3521233538030395096020527e-3,
  -.3439477441808804817791462e-4,
  +.2053328149090647946837223e-3,
  +.2701844395439035266729021e-3,
  +.1672729121051401933535015e-3,
  -.2746380660376015886000760e-4,
  -.2092092620592999458371397e-3,
  -.2834686553202414466429345e-3,
  -.1996968583089697747077846e-3,
  +.2627703710991833669946660e-4,
  +.3073684081492528265927548e-3,
  +.5036054530473556290555964e-3,
  +.4663435615115594494005948e-3,
  +.1044377697560001158107957e-3,
  -.5415995822039977016551962e-3,
  -.1243962090408245779299742e-2,
  -.1588511278903561561906197e-2,
  -.1074591952738488824724292e-2,
  +.6568035186371544315047730e-3,
  +.3477836913618538209007360e-2,
  +.6400068531700629458107228e-2,
  +.7371151770472239134412402e-2,
  +.3557728855573160947913538e-2,
  -.7513325997815228933135160e-2
};
// These numbers have been taken from here: 
//   http://www.plouffe.fr/simon/constants/stieltjesgamma.txt
// This textfile is also here in the repo in the Misc folder. It has many more digits. I have used
// the first 25 digits and rounded up the digit(s) before the first removed digit whenever the 
// first removed digit was >= 5. The table there has more gammas, but I have used only the first 32
// so far because that nicely matches with the table-length of RAPT::rsInverseFactorials which we 
// need in conjunction with the Stieltjes constants to produce the Laurent series coeffs for zeta.
// Maybe instead of storing the gammas, we could store the final coeffs (-1)^n * g[n] / n!. The 
// gammas will eventually grow large as n gets larger, so it may make sense to divide by the n! to 
// keep them in check. On the other hand, the unadorned Stieltjes constants themselves might be 
// useful in other contexts as well, so maybe it's better to keep them in pure form. 


// Old - obsolete:
/*
const double rsRiemannZetaFunction::gamma[11] = 
{ 
  +0.5772156649015328606065120900824024310421593359,
  -0.0728158454836767248605863758749013191377363383,
  -0.0096903631928723184845303860352125293590658061,
  +0.0020538344203033458661600465427533842857158044,
  +0.0023253700654673000574681701775260680009044694,
  +0.0007933238173010627017533348774444448307315394,
  -0.0002387693454301996098724218419080042777837151,
  -0.0005272895670577510460740975054788582819962534,
  -0.0003521233538030395096020521650012087417291805,
  -0.0000343947744180880481779146237982273906207895,
  +0.0002053328149090647946837222892370653029598537 
};
*/
// This table is preliminary. ToDo: 
// Remove leading zeros and let all coeffs have the same number of significant digits (~20). Use
// g[1] = -7.281e-2 notation. 


std::complex<double> rsRiemannZetaFunction::evalViaOriginalSum(
  std::complex<double> s, int numTerms)
{
  std::complex<double> sum = 0;
  for(int n = 1; n <= numTerms; n++)
    sum += pow(n, -s);
  return sum;
}

std::complex<double> rsRiemannZetaFunction::evalViaAlternatingSum(
  std::complex<double> s, int numTerms)
{
  std::complex<double> eta = evalDirchletEta(s, numTerms);
  return eta / (1.0 - pow(2.0, 1.0-s));
}

std::complex<double> rsRiemannZetaFunction::evalViaBinomialSum(
  std::complex<double> s, int numTerms)
{
  RAPT::rsAssert(numTerms <= 29);
  // In some first tests, the error decreased until numTerms reached 29. With 30, it went up 
  // again. I guess, we have overflow issues in the computation of the binomial coeffs. 
  // -> Figure out. For production, the computation of binomial coeffs should be replaced by an 
  // algo based on Pascal's triangle anyway. It's still a very early prototype

  std::complex<double> sum = 0;
  for(int n = 0; n <= numTerms; n++)
  {
    std::complex<double> subsum = 0;
    for(int k = 0; k <= n; k++)
    {
      int bnk = RAPT::rsBinomialCoefficient(n, k);
      int sign = pow(-1, k);  // optimize!
      subsum += double(sign * bnk) * pow(k+1.0, -s);
    }
    sum += subsum / pow(2, n+1);
  }
  return sum / (1.0 - pow(2.0, 1.0-s));
}
// template rsUint64 RAPT::rsBinomialCoefficient(rsUint64, rsUint64); // doesn't work!
// ..try to move into rs_testing and see if it works from there


std::complex<double> rsRiemannZetaFunction::evalViaEulerProduct(
  std::complex<double> s, int numTerms, const int* primes)
{
  //RAPT::rsAssert(numTerms <= 10000);
  // 10000 is the size of the array returned by rosic::PrimeNumbers::_getPrimeArray() which
  // is passed in the test...but ensuring that primes has at least numTerms entries is actually
  // the business of the caller

  std::complex<double> prod = 1;
  for(int n = 0; n < numTerms; n++)
    prod *= 1.0 / (1.0 - pow(double(primes[n]), -s));
  return prod;
}

std::complex<double> rsRiemannZetaFunction::evalViaLaurentSeries(
  std::complex<double> s, int numTerms)
{
  RAPT::rsAssert(numTerms <= numGammas);

  std::complex<double> z   = s - 1.0;  // we expand in power of (s-1)^n
  std::complex<double> sum = 1.0 / z;  // first term for n = -1
  for(int n = 0; n < numTerms; n++) {
    double c = laurentSeriesCoeff(n);
    sum += c * pow(z, n); }
  return sum;
}

std::complex<double> rsRiemannZetaFunction::evalViaBoostSum(std::complex<double> s, int n)
{
  auto coeff = [](int j, int n)
  {
    int sum = 0;
    for(int k = 0; k <= j-n; k++)
      sum += rsBinomialCoefficient(n, k);
    return pow(-1, j) * (sum - pow(2, n));  // optimize away the pow(-1, j)
  };

  std::complex<double> sum = 0;
  for(int j = 0; j <= 2*n-1; j++)
    sum += coeff(j, n) / pow(j+1, s);

  return -sum / (pow(s, n) * (1.0-pow(2.0, 1.0-s)));
}

std::complex<double> rsRiemannZetaFunction::evalViaStackOverflowAlgo(
  std::complex<double> s, int maxNumTerms)
{
  const double lowerThresh = 1.0e-6;                      // Precision, error tolerance (?)
  const double upperBound  = 1.0e+4;                      // For divergence check
  using Complex = std::complex<double>;
  std::vector<Complex> a(maxNumTerms+1);                  // Array of...what?
  Complex half(0.5), one(1), two(2), neg(-1);             // Some constants in complex format
  Complex sum(0);                                         // Main accumulator for result
  Complex sumOld(1.0e+20);                                // Convergence check facilitator
  a[0] = half / (one - pow(two, (one - s)));              // Init a[0] = 0.5 / (1 - 2^(1-s))
  sum += a[0];
  for(int n = 1; n <= maxNumTerms; n++) {
    Complex nC(n);                                        // Index n converted to complex
    for (int k = 0; k < n; k++) {
      Complex kC(k);                                      // Index k converted to complex
      a[k] *= half * (nC / (nC - kC));
      sum += a[k];  }
    a[n] = (neg * a[n-1] * pow((nC/(nC+one)), s) / nC);
    sum += a[n];
    if(abs(sumOld - sum) < lowerThresh) 
      break;                                              // Algorithm has converged
    if(abs(sum)          > upperBound)  
      break;                                              // Algorithm has diverged
    sumOld = sum; }                                       // Facilitate convergence check
  return sum;
  // ToDo: 
  // -Figure out why this array is needed. Can we reformulate the algo in a way that avoids it? 
  //  It causes a heap allocation which is undesirable.
  // -Maybe indicate divergence somehow to the caller. Maybe return sign(sum) * inf or nan
  // -Make tests to figure out at which values of s divergence happens and document that.
  // -Maybe make lowerThresh a user parameter, maybe call it precision or errorBound or maxError 
  //  or something. I think, that is what it is.
  // -maybe ise rsAbs, rsPow etc instead of abs, pow

  // Implementation is based on the code posted here:
  // https://stackoverflow.com/questions/41549533/riemann-zeta-function-with-complex-argument
  // The original code is also stored in Libraries/Snippets/Misc/RiemannZeta.cpp
}

void rsRiemannZetaFunction::vectorFieldViaOriginalSum(
  double x, double y, double* u, double* v, int numTerms)
{
  *u = 0;
  *v = 0;
  for(int n = 1; n <= numTerms; n++)
  {
    double w = log(double(n));
    *u += exp(-w*x) * cos(w*y);
    *v += exp(-w*x) * sin(w*y);
  }
}

void rsRiemannZetaFunction::vectorFieldViaLaurentSeries(
  double x, double y, double* u, double* v, int numTerms)
{
  RAPT::rsAssert(numTerms <= numGammas);

  // Internal subroutines to compute the u_n, v_n terms, i.e. the real and imaginary parts of
  // (x + i*y)^n:
  double a[numGammas]; 
  int px[numGammas], py[numGammas];  // Temp arrays for coeffs and powers
  auto u_n = [&](double x, double y, int n)
  {
    int m = rsRealCoeffsComplexPower(n, a, px, py);
    double sum = 0.0;
    for(int k = 0; k < m; k++)
      sum += a[k] * pow(x, px[k]) * pow(y, py[k]);
    return sum;
  };
  auto v_n = [&](double x, double y, int n)
  {
    int m = rsImagCoeffsComplexPower(n, a, px, py);
    double sum = 0.0;
    for(int k = 0; k < m; k++)
      sum += a[k] * pow(x, px[k]) * pow(y, py[k]);
    return sum;
  };

  // Compute the Laurent series up to the desired number of terms:
  x  -= 1.0;              // Because of the shift: z = s-1, see paper for explanation
  *u  = x / (x*x + y*y);  // That's the n = -1 term for u_n, i.e. u_{-1}
  *v  = y / (x*x + y*y);  // ...same for v_n
  for(int n = 0; n < numTerms; n++)
  {
    *u += laurentSeriesCoeff(n) * u_n(x, y, n);
    *v -= laurentSeriesCoeff(n) * v_n(x, y, n); // Minus due to negation in Polya vector field.
  }
}

double rsRiemannZetaFunction::potentialViaOriginalSum(double x, double y, int numTerms)
{
  static const double K = -0.24323834874662564;
  // Constant offset to make output zero at (x,y) = (0,0). Maybe use that in the initialization
  // of the sum. Maybe do
  // sum = x - 0.24... 


  double sum = 0;
  for(int n = 2; n <= numTerms; n++)
  {
    double w = log(double(n));
    //sum += cos(w*y) / (w *pow(n, x));  // this formula looks nicer in a textbook
    sum += exp(-w*x) * cos(w*y) / w;     // this formula is more efficient
  }
  return K + x - sum;
}

double rsRiemannZetaFunction::potentialViaLaurentSeries(double x, double y, int numTerms)
{
  RAPT::rsAssert(numTerms <= numGammas);

  double a[numGammas]; 
  int px[numGammas], py[numGammas]; 
  auto P_n = [&](double x, double y, int n)
  {
    int m = rsPotentialCoeffsComplexPower(n, a, px, py);
    double sum = 0.0;
    for(int k = 0; k < m; k++)
      sum += a[k] * pow(x, px[k]) * pow(y, py[k]);
    return sum;
  };
  // Is numGammas actually the correct size for these arrays? I think not. They are not necessarily
  // the same size as numGammas, I think. They might be smaller. Maybe more in the numGammas/2 
  // ballpark. Figure out and change it also in vectorFieldViaLaurentSeries. Look at the sage 
  // output for the polynomials in the test code. We need arrays long enough to hold all coeffs, 
  // i.e. the length should be given by the maximum number of terms in the bivariate polynomial to
  // be expected. I think, it's around n/2 because half of the coeffs go into u and half into v.
  // P has one coeff more that v.

  static const double K = 0.53929867655706210;
  // The constant that ensures that the potential is 0 at s = 0. When we have more coeffs to be 
  // able to make more accurate computations, that should be recomputed by setting it temporarily 
  // to 0, computing the potential at s=0, and then subtracting it, i.e. use its negation as K. 
  // When doing this, we should ensure in the degbugger that the series actually converges, i.e. 
  // later iterations do not change the value of P anymore. With at at most 11 terms, it was not 
  // yet fully converged to double precision so the value here is preliminary. After it has been 
  // recomputed more accurately, the constant in potentialViaOriginalSum must also be recomputed by
  // taking this functionas reference and make both functions match for example at s=2 (we can't 
  // use the other sum at s=0 because it diverges there).

  x -= 1.0;
  double P = K + log(x*x + y*y) / 2;
  for(int n = 0; n < numTerms; n++)
  {
    double c  = laurentSeriesCoeff(n);  // the n-th coeffs
    double Pn = P_n(x, y, n);           // the P_n contribution
    double d  = c*Pn;                   // delta to be added
    P += d;

    // During development, it may make sense to split it up into baby-steps like that to see 
    // what's going on. Later, we'll use just one line like so:

    //P += laurentSeriesCoeff(n) * P_n(x, y, n);
  }
  return P;

  // The code here closely parallels the one in vectorFieldViaLaurentSeries. It's just simpler
  // because we need to compute only one funtion rather than two. See comments there for more 
  // details about what's going on
}

std::complex<double> rsRiemannZetaFunction::evalDirchletEta(std::complex<double> s, int numTerms)
{
  double sign = +1.0;
  std::complex<double> sum = 0;
  for(int n = 1; n <= numTerms; n++) {
    sum  += sign * pow(n, -s);
    sign *= -1.0; }
  return sum;
}

std::complex<double> rsRiemannZetaFunction::dirichletTermViaReIm(std::complex<double> s, int n)
{
  // n^(-s) / (1 - 2^(1-s))
  std::complex<double> tn = pow(n, -s) / (1.0 - pow(2.0, 1.0-s)); // preliminary, for test
  // Maybe move into a function dirichletTermDirect and use in unit tests

  double x = real(s);
  double y = imag(s);
  double w = log(n);
  double p = log(2);

  double sp = sin(p*y);
  double cp = cos(p*y);
  double sw = sin(w*y);
  double cw = cos(w*y);
  double ep = exp(p*x);
  double ew = exp(w*x);
  double d  = 4*ep*cp*ew - (ep*ep + 4)*ew;
  double a  = 2*ep*cp - ep*ep;
  double re = (2*ep*sp*sw + a*cw) / d;
  double im = (2*ep*sp*cw - a*sw) / d;
  // If all is coorect, then (re,im) should match (tn.re,tn.im). Yep - looks good.

  return std::complex(re, im);
  //return tn;

  // ToDo: 
  // -Optimize:
  //  -get rid of second division by using s = 1/d
  //  -use rsSinCos
  //  -p is a constant, define as static const
  //  -maybe some exp calls can be merged with some log calls and cancel? 
  //   exp(w*x) = exp(log(n)*x) = n^x...yeah...no...that would be a pessimization.
  //   Likewise exp(p*x) = exp(log(2)*x) = 2^x. That could perhaps be worthwhile when a fast
  //   routine for 2^x is available (faster than exp).
  //  -Maybe a couple of products that are used twice can be computed once: ep*cp, ep*ep, 2*ep*sp.
  //   But that will probably be done by the compiler anyway.
  //  -Overall, it seems that we need one log, 2 exp, 2 sin/cos pairs, 1 div and a bunch of 
  //   add, sub, mul
  //  -On the other hand, this function is not supposed to be relevant for production code anyway
  //   so maybe optimizing it is pointless.
}


double rsRiemannZetaFunction::laurentSeriesCoeff(int n)
{
  RAPT::rsAssert(n < 32, "We have not yet tabulated coeffs higher than n=31");

  if(n <  -1) return 0.0;  // Laurent series coeffs for (s-1)^(-n) are 0 for n > 1.
  if(n == -1) return 1.0;  // This is the residue at the pole at s = 1.
  return gamma[n] * pow(-1.0, n) * RAPT::rsInverseFactorials[n]; 
  // The coeffs for nonnegative powers of s can be computed using the Stieltjes constants.
}

//=================================================================================================

/** Given a data matrix P containing values P(i,j) of a scalar (or potential) field P(x,y) with 
equidistantly sampled data with stepsize dx in the x-direction, this function computes a numerical 
estimate of the partial derivative of P with respect to x. It uses a second order central 
difference formula for the inner points and a first order forward or backward difference formula 
for the boundary points. */
template<class T>
rsMatrix<T> rsNumericDerivativeX(const rsMatrix<T>& P, T dx)
{
  int I = P.getNumRows();     // Number of rows in data matrix
  int J = P.getNumColumns();  // Number of columns in data matrix
  rsMatrix<T> P_x(I, J);      // Our result: P_x = dP/dx
  T s = T(1) / dx;
  for(int j = 0; j < J; j++) 
    P_x(0, j) = (P(1, j) - P(0, j)) * s;          // forward diff at left boundary / top row
  for(int j = 0; j < J; j++) 
    P_x(I-1, j) = (P(I-1, j) - P(I-2, j)) * s;    // backward diff at right boundary / bottom row
  s *= T(0.5);
  for(int i = 1; i < I-1; i++)
    for(int j = 0; j < J; j++) 
      P_x(i, j) = (P(i+1, j) - P(i-1, j)) * s;    // central diff for inner point
  return P_x;
}

/** Like rsNumericDerivativeX but for the partial derivative with respect to y. */
template<class T>
rsMatrix<T> rsNumericDerivativeY(const rsMatrix<T>& P, T dy)
{
  int I = P.getNumRows();     // Number of rows in data matrix
  int J = P.getNumColumns();  // Number of columns in data matrix
  rsMatrix<T> P_y(I, J);      // Our result: P_y = dP/dy
  T s = T(1) / dy;
  for(int i = 0; i < I; i++)
    P_y(i, 0) = (P(i, 1) - P(i, 0)) * s;          // forward diff at bottom boundary / left column
  for(int i = 0; i < I; i++)
    P_y(i, J-1) = (P(i, J-1) - P(i, J-2)) * s;    // backward diff at top boundary / right column
  s *= T(0.5);
  for(int i = 0; i < I; i++)
    for(int j = 1; j < J-1; j++) 
      P_y(i, j) = (P(i, j+1) - P(i, j-1)) * s;    // central diff for inner point
  return P_y;
}
// Move these 2 into RAPT::rsNumericDifferentiator
// Maybe we could use a second order formula for the boundary points, too? Then we may also change
// the ansatz equations for the numeric potential routine accordingly (which is supposed to be the
// inverse of these two functions). Maybe have both versions. If we use 3 term formulas for the 
// boundary points as well, the data matrices must be at least of size 3x3. Now we can also allow
// 2x2 data matrices.

/** Computes a potential for a vector field given in the matrices P_x(i,j), P_y(i,j) numerically. 
The notation P_x, P_y is meant to suggest that these data matrices represent the partial 
derivatives of some potential P with respect to x and y. The data is assumed to be equally spaced 
with stepsizes dx, dy in the x- and y-directions repsectively. A potential is unique only up to a 
constant shift. That's why the caller can specify the desired value "Konstant" of the potential at 
some index pair "iKonstant", "jKonstant". By default, the potential will be zero at i=0, j=0 but 
the caller can change that via these parameters.

If P_x and P_y were obtained by numerically differentiating some scalar field (or potential) P with 
respect to x and y via the routines rsNumericDerivativeX/Y, then this function should reconstruct 
the potential P up to roundoff error because the formulas we used in our ansatz here match the 
numerical differentiation formulas used in these routines. Note that the roundoff error may 
actually be quite substantial though because the systems to be solved tend to be quite large. We 
may need better numeric linear algebra routines someday. Eventually, this should be done using a 
sparse system solver anyway. This implementation here is more for proof of concept. It can be used
for data matrices of sizes of maybe 20x20 (i.e. 400 unknowns for P) or maybe a bit more. But not 
like 100x100 or 1000x1000 as would typically be needed for numerical simulation and plotting 
purposes. For details about the idea behind the algorithm, see the file 
Notes/PotentialNumerical.txt here in this repo.   */
template<class T>
rsMatrix<T> rsNumericPotential(const rsMatrixView<T>& P_x, const rsMatrixView<T>& P_y, 
  T dx, T dy, T Konstant = T(0), int iKonstant = 0, int jKonstant = 0)
{
  int I = P_x.getNumRows();     // Number of rows in data matrices
  int J = P_x.getNumColumns();  // Number of columns in data matrices
  int N = I*J;                  // Number of unknowns = number of columns of coeff matrix
  rsAssert(P_y.hasShape(I, J), "P_x and P_y must have the same shape");

  // Now we assemble the coefficient matrix:
  using Mat = rsMatrix<T>;
  Mat M(2*N+1, N);

  // Compute the coeffs that appear in the matrix:
  T a = 1/(2*dx); 
  T A = 1/dx; 
  T b = -a; 
  T B = -A;
  T c = 1/(2*dy); 
  T C = 1/dy; 
  T d = -c; 
  T D = -C;

  // Add the b,a and B,A coeffs to the matrix:
  for(int k = 0; k < N-2*J; k++) {  
    M(k+J, k)     = b;                // b,a coeffs for inner points
    M(k+J, k+2*J) = a; }
  for(int k = 0; k < J; k++) {    
    M(k,     k)       = B;            // B,A coeffs for boundary points
    M(k,     k+J)     = A;
    M(N-1-k, N-1-k)   = A;
    M(N-1-k, N-1-k-J) = B; }

  // Add the d,c, and D,C coeffs to the matrix:
  for(int i = 0; i < I; i++) {        // loop over the blocks
    int s = i*J;                      // start of i-th block
    for(int k = 1; k < J-1; k++) {    // d,c coeffs for inner points
      M(N+s+k, s+k-1) = d;
      M(N+s+k, s+k+1) = c; }}
  for(int i = 0; i < I; i++) { 
    int s = i*J;
    M(N+s,     s    ) = D;
    M(N+s,     s+1  ) = C;
    M(N+s+J-1, s+J-1) = C;
    M(N+s+J-1, s+J-2) = D; }



  // Add the last row for the additional condition to let the potential have some given value at
  // some given position:
  int i = iKonstant;     // Row index in data matrix Q or P.
  int j = jKonstant;     // Column index in data matrix Q or P.
  int k = i*J + j;       // Column index coefficient matrix R.
  M(2*N, k) = 1;         // Add a coeff on 1 at position k in the last line
  //plotMatrix(M, true); // Uncomment to inspect the coefficient matrix

  // Assemble the right hand side vector w as concatentation of vectorized P_x, P_y and the 
  // additional constant K (== Konstant) as last element:
  Mat w(2*N+1, 1);
  const T* ptr = P_x.getDataPointerConst();
  for(int n = 0; n < N; n++)
    w(n, 0) = ptr[n];
  ptr = P_y.getDataPointerConst();
  for(int n = 0; n < N; n++)
    w(N+n, 0) = ptr[n];
  w(2*N, 0) = Konstant;         // Desired value K = P(i, j) must be added to RHS as last element

  // Maybe write this as:
  //   rsMatrixView p_x(N, 1, p_x.getDataPointerConst()); // vectorized view of P_x
  //   rsMatrixView p_x(N, 1, p_y.getDataPointerConst()); // vectorized view of P_y
  //   rsMatrixView one(1, 1, { 1 });
  //   w = rsMatrix::concatenateVertically(p_x, p_y);
  //   w = rsMatrix::concatenateVertically(w, one);
  // should behave like Matlab's vertcat:
  //   https://www.mathworks.com/help/matlab/ref/double.vertcat.html
  // Although, doing it like this is more efficient because it uses only one allocation and the two
  // w = ...; assignments would each do an allocation. But maybe we can have a function that takes a 
  // variable number of arguments. 

  // Solve the system via a least squares approach:
  Mat MT  = M.getTranspose();   // M^T
  Mat MTM = MT * M;             // M^T * M. Least squares coeff matrix for the solver.
  Mat wp  = MT * w;             // w' = M^T * w. Right hand side for the solver.
  Mat P(I, J);                  // Our result. The estimate of the potential for U and V.
  rsMatrixView p(N, 1, P.getDataPointer());        // The solver wants p, a vectorized view of P.
  int its = rsLinearAlgebraNew::solve(MTM, p, wp); // Invoke the linear system solver.
  return P;
}
// ToDo: 
// -To make this idea useful in practice, we need an implementation based on sparse matrices. The 
//  dense matrix based implementation here can only serve as proof of concept. Maybe to handle 
//  larger data, we could compute the potential for (overlapping) patches and stitch them together.
//  The overlap ensures that for the final result, we use only inner points. We should maybe match 
//  up the first inner point via the constant offsets.
// -see rsSparseMatrix in Prototypes.h in the main RS-MET repo. There's also an experiment
//  iterativeLinearSolvers() that uses it already and there are also some unit test already.
// -factor out the loops involing the B,A,D,C coeffs into a function "addBoundaryCoeffs"
//  the idea is that we may later give the user the option to select between using different
//  ansatz formulas for the boundary points (and maybe also for the inner points). That's why it
//  will be convenient to have function to add these. Maybe move the whole code into a class
//  rsNumericalPotentialFinder2D. have functions like setInnerPointFormula, 
//  setBoundaryPointFormula or setFormulaForInterior, setFormulaForBoundary

// sparse-matrix based implementation of rsNumericPotentialSparse. This should eventually become the
// regular one (without the qualification "Sparse") and the other should be named "...Proto" to 
// indicate that it's a prototype (based on dense matrices which is impractical for all but very 
// small data matrices.
template<class T>
rsMatrix<T> rsNumericPotentialSparse(const rsMatrixView<T>& P_x, const rsMatrixView<T>& P_y, 
  T dx, T dy, T Konstant = T(0), int iKonstant = 0, int jKonstant = 0)
{
  int I = P_x.getNumRows();     // Number of rows in data matrices
  int J = P_x.getNumColumns();  // Number of columns in data matrices
  int N = I*J;                  // Number of unknowns = number of columns of coeff matrix
  rsAssert(P_y.hasShape(I, J), "P_x and P_y must have the same shape");
  using Vec  = std::vector<T>;
  using Mat  = rsMatrix<T>;
  using MatS = rsSparseMatrix<T>;

  // Compute the coeffs that appear in the matrix:
  T a = 1/(2*dx); 
  T A = 1/dx; 
  T b = -a; 
  T B = -A;
  T c = 1/(2*dy); 
  T C = 1/dy; 
  T d = -c; 
  T D = -C;

  // Assemble the coefficient matrix:
  MatS M(2*N+1, N);
  M.reserve(4*N+1);  // Verify formula!
  auto setCoeff = [&](int i, int j, T c) { M.appendFastAndUnsafe(i, j, c); };
  for(int k = 0; k < N-2*J; k++) {
    setCoeff(k+J, k,     b);        // M(k+J,   k)       = b;
    setCoeff(k+J, k+2*J, a); }      // M(k+J,   k+2*J)   = a;
  for(int k = 0; k < J; k++) {
    setCoeff(k,     k,       B);    // M(k,     k)       = B;
    setCoeff(k,     k+J,     A);    // M(k,     k+J)     = A;
    setCoeff(N-1-k, N-1-k,   A);    // M(N-1-k, N-1-k)   = A;
    setCoeff(N-1-k, N-1-k-J, B); }  // M(N-1-k, N-1-k-J) = B;
  for(int i = 0; i < I; i++) {
    int s = i*J; 
    for(int k = 1; k < J-1; k++) {
      setCoeff(N+s+k, s+k-1, d);
      setCoeff(N+s+k, s+k+1, c); }}
  for(int i = 0; i < I; i++) { 
    int s = i*J;
    setCoeff(N+s,     s,     D);
    setCoeff(N+s,     s+1,   C);
    setCoeff(N+s+J-1, s+J-1, C);
    setCoeff(N+s+J-1, s+J-2, D); }
  int i = iKonstant;                     // Row index in data matrix Q or P.
  int j = jKonstant;                     // Column index in data matrix Q or P.
  int k = i*J + j;                       // Column index coefficient matrix R.
  setCoeff(2*N, k, 1);                   // Add a coeff on 1 at position k in the last line
  //plotMatrix(MatS::toDense(M), true);

  // Establish the right hand side vector w as concatentation of vectorized P_x, P_y and the 
  // additional contant K (== Konstant) as last element:
  Vec w(2*N+1);
  const T* ptr = P_x.getDataPointerConst();
  for(int n = 0; n < N; n++)
    w[n] = ptr[n];
  ptr = P_y.getDataPointerConst();
  for(int n = 0; n < N; n++)
    w[N+n] = ptr[n];
  w[2*N] = Konstant;
  // Maybe use: 
  //   rsArrayTools:copy::(P_x.getDataPointerConst(), &w[0], N);
  //   rsArrayTools:copy::(P_y.getDataPointerConst(), &w[N], N);
  //   w(2*N) = Konstant;


  // Solve the system via a least squares approach and Gauss-Seidel iteration:
  Mat  P(I, J);                // Our result. The potential P.
  MatS MT  = M.getTranspose(); // M^T
  MatS MTM = MT * M;           // M^T * M. Least squares coeff matrix for the solver.
  Vec  wp  = MT * w;           // w' = M^T * w. Right hand side for the solver.
  T    tol = 1.e-6;            // Error tolerance for iterative solver, ToDo: make parameter

  //int  its = MatS::solveGaussSeidel(MTM, P.getDataPointer(), &wp[0], tol); 

  T   sor = 1.9;              // SOR parameter. Must be < 2. 0: Jacobi, 1: Gauss-Seidel
  std::vector<T> wrk(P.getSize());
  int its = MatS::solveSOR(MTM, P.getDataPointer(), &wp[0], tol, &wrk[0], sor); 

  return P;
}
// ToDo: 
// -Maybe instead of handpicking the sor parameter, try to use the optimal one for the problem at 
//  hand. That depends on the spectral radius of the iteration matrix. Maybe it makes sense to 
//  estimate that before starting the iteration. Maybe a few vector iterations to estimate the 
//  largest eigenvalue can be used for this. If we only do a few such iterations (say 20), it 
//  could be worthwhile as a precompuation. Or maybe we could do some empirical tests with 
//  matrices of different sizes and shapes and use the data obtained to derive an empirical 
//  formula for the optimal SOR coeff as function of I and J. Maybe it's just a 1D function of 
//  N. Maybe it's just a constant. But the important thing to note is that this value depends 
//  only on the coefficient matrix and not on the input data P_x, P_y (because that goes only 
//  into the right hand side). The matrices are always the same and depend onyl on the size 
//  (maybe shape) of the input matrices.
// -Can we assemble MTM directly without resorting to the (expensive) matrix multiplication step?
// -Maybe try to improve the convergence by implementing a multigrid method. Let's for example 
//  assume the data to be originally on a 100x30 grid. First, downsample to a 64x16 grid (take 
//  half-sizes and round up to the next power of two). Then downsample these grids further to 
//  32x8, 16x4, 8x2. This downsampling should use averaging of the 4 involved datapoints. Then 
//  solve the 8x2 problem. Interpolate the solution to 16x4 and use that as initial guess to solve
//  the 16x4 problem. Then use that (interpolated) solution as initial guess for the 32x8 problem,
//  then for 64x16 problem. Finally interpolate the 64x16 problem to the original 30x100 grid
//  and solve the problem there. The first decimation and last interpolation steps may be a bit 
//  more complicated but as soon as we have grids with powers of 2, decimation and interpolation
//  are easy. For development, use power-of-two grids first and as soon as that works, implement
//  the decimation and interpolation to arbitrary grids on top of that. In the downsampled steps,
//  we may use a rather high error tolerance because they are supposed to just produce initial 
//  guesses so letting them converge to full precision may be pointless.



/** Under construction. Not yet tested */

template<class T>
void rsHelmholtzDecomposition(
  const rsMatrix<T>& F1, const rsMatrix<T>& F2, T dx, T dy,
  rsMatrix<T>& G1, rsMatrix<T>& G2, rsMatrix<T>& R1, rsMatrix<T>& R2)
{
  rsMatrix<T> P = rsNumericPotential(F1, F2, dx, dy); // todo: use sparse implementation
  G1 = rsNumericDerivativeX(P, dx);
  G2 = rsNumericDerivativeY(P, dy);
  R1 = F1 - G1;
  R2 = F2 - G2;

  // The idea is to enforce G to be (numerically) curl-free by letting it be the numeric gradient
  // of a sort of pseudo-potential for F that we obtain via our rsNumericPotential routine. If F 
  // isn't (numerically) curl-free, then no potential exists for F and the rsNumericPotential 
  // routine should produce the best approximation to an actual potential in a least squares sense
  // (I think). Then, G will be the best curl-free approximation to F. And R is then just the 
  // residual F - G. See:
  // https://en.wikipedia.org/wiki/Helmholtz_decomposition
}
// Not yet tested, just an idea, so far

// This video, at around 7:30
// https://www.youtube.com/watch?v=NtoIXhUgqSk
// says that the divergence of a Polya vector field of an analytic functions is also zero. What
// remains of a vector field if both divergence and curl are zero?

// see:
// https://www.youtube.com/watch?v=xa5xornH2ok
// Are all vector fields the gradient of a potential? ... and the Helmholtz Decomposition

// https://oliver-richters.de/helmholtz/
// https://arxiv.org/pdf/2102.09556.pdf 
// Helmholtz decomposition and potential functions for n-dimensional analytic vector fields
// -defines a generalization of curl as antisymmetric matrix -> could be relevant for 
//  rsNumericDifferentiator

// -In the Helmholtz decomposition, a vector field is split into a gradient field and the rest. 
//  The gradient field will be guaranteed to be curl free, but it may still feature divergence. 
//  Can we split it further into curl and div by making an appropriate ansatz using numerical 
//  differentiation formuals?

// -Maybe wa can split an arbitrary vector field into 3 components: curl, div, rest. I'm not yet 
//  sure how to do that, though, I think, via the potential, we can split off a component that is
//  both curl-free and divergence-free. The residual would be curl + divergence. But could we 
//  split that further? 


// Another idea would be to use the same idea to reconstruct a harmonic conjugate of a given U or V
// numerically (I think, that's what it is called what we are doing here. Verify that!): Assume, we 
// have given only U = dP/dx = P_x and want to reconstruct V = dP/dy = P_y from it. We could use 
// the same approach just with a shorter matrix M. We would use only the upper half of it (plus the 
// one line for the extra condition to make the system nonsingular). So, instead of 2*N+1 equations 
// for N unknowns, we'd get N+1 equations. It would still be a least-squares problem due to the 
// extra + 1 line (unless we implement the idea of not adding the condition as extra line but using 
// it to replace 2 of our original equations. Here then, we would use it to replace only 1 
// equation, hence leading to a square matrix M and thus avoiding the M^T * M step, see 
// PotentialNumerical.txt for more details). Having the potential P reconstructed from U alone, we 
// could numercally differentiate P with respect to y to obtain V. Likewise, if we would only have 
// V, we could reconstruct U by using only the lower half of the matrix and differentiating the 
// resulting P with respect to X. (Q: What happens, if we do it the wrong way, i.e. feed in V when 
// the algo expects U or the other way around?)
//
// To facilitate this, maybe we should factor out the loops to add the the coeffs to the matrix, 
// i.e. the code that follows the comments:
//
//    // Add the b,a and B,A coeffs to the matrix:
//    // Add the d,c, and D,C coeffs to the matrix:
// 
// If we only want to use the second, lower half of the matrix, only the lower loop would have to 
// be used but without the "N+" in the row index, i.e. all assigments like:
//
//   M(N+s,     s    ) = D;
//
// would be need to replaced by
//
//   M(0+s,     s    ) = D;
//
// which we could generalize to a
//
//   M(S+s,     s    ) = D;
//
// where S is a shift/offset that can be passed to the factored out function. It would be N or 0
// depending on whether we want to put that part of the matrix on top or halfway down. Maybe for
// consistency, we could add such a shift parameter also to the routine to assemble the upper part.
// Maybe the functions could be named assemblePotentialCoeffsDiffX, assemblePotentialCoeffsDiffY.
// When refactoring like this, make sure that testNumericPotential() passes at all stages.
//
// Under construction. Does not yet work. It is supposed to construct the potential from P_x 
// alone. We don't give the caller the opportunity to fix the potential bcs that would make the 
// code here messier. It can be adjusted afterwards, if desired.
template<class T>
rsMatrix<T> _rsNumericPotential(const rsMatrix<T>& P_x, T dx)
{
  int I = P_x.getNumRows();     // Number of rows in data matrices
  int J = P_x.getNumColumns();  // Number of columns in data matrices
  int N = I*J;                  // Number of unknowns = number of columns of coeff matrix

                                // Now we assemble the coefficient matrix:
  using Mat = rsMatrix<T>;
  Mat M(N, N);

  // Compute the coeffs that appear in the matrix:
  T a = 1/(2*dx); 
  T A = 1/dx; 
  T b = -a; 
  T B = -A;

  // Add the b,a and B,A coeffs to the matrix:
  for(int k = 0; k < J; k++) {     
    M(k, k)           = B;
    M(k, k+J)         = A;
    M(N-1-k, N-1-k)   = A;
    M(N-1-k, N-1-k-J) = B;  }
  for(int k = 0; k < N-2*J; k++) {
    M(k+J, k)     = b;
    M(k+J, k+2*J) = a;  }
  M(0, 0) = 1;             // 0th row is for constant term. We overwrite the values that the
  M(0, J) = 0;             // loop wrote here (I didn't want to mess with the loop itself)
  plotMatrix(M, true);
  //plotMatrix(M, false);
  // Actually the M(0,0) = 1; M(0,J) = 0 is supposed to make the matrix nonsingular. But it 
  // doesn't! Oh! When we don't make demands about the y-derivative and only prescribe an 
  // x-derivative, it actually means that we could add an arbitrary function of y, i.e. we would 
  // have to prescribe a whole matrix row of P-values to make the problem nonsingular. We are 
  // actually just trying to solve mutliple decoupled 1D problems simultaneously, but each
  // solution to such a 1D problem could have its own shift which would have to be determined
  // by a constant. I think, the whole idea of computing the potential from a single partial 
  // derivative alone may not be workable. Could there be other ways to find the harmonic 
  // conjugate numerically? Maybe something based on Laplace's equation:
  //   https://en.wikipedia.org/wiki/Harmonic_function
  // Maybe: differentiate with respect to x, then integrate result with respect to y, then 
  // negate? We would use P_xx + P_yy = 0  ->  P_yy = -P_xx  ->  P_y = integrate -P_xx dy
  // ...but that would also require an integration "constant" that is a function of x. Maybe that
  // could be provided by the user as a "boundary condition"? But if the user provides such a
  // boundary condition, then this could be also used in an idea like this one here.

  // Assemble the right hand side vector w:
  Mat w(N, 1);
  w(0, 0) = 1.0;           // fix constant
  const T* ptr = P_x.getDataPointerConst();
  for(int n = 1; n < N; n++)
    w(n, 0) = ptr[n];

  // Solve the system via a least squares approach:
  Mat P(I, J);                              // Our result.
  rsMatrixView p(N, 1, P.getDataPointer()); // The solver wants p, the vectorized view of P.
  rsLinearAlgebraNew::solve(M, p, w);       // Invoke the linear system solver.
  return P;
}

//=================================================================================================

/** Class to evaluate the Polya vector fields and Polya potentials for various complex functions
w = f(z). It implements a bunch of functions that can evaluate the Polya vector field and potential
for certain common elementary functions directly via analytic expressions and it also facilitates 
the numerical etsimation of the Polya potential for functions, where such analytic formulas are not
(yet) available. The evaluation of functions based on analytic expressions come in two flavors: One 
that takes two input parameters and returns a value. That variant computes the potential. And one 
that takes two input parameters and two output parameters by pointer. That variant computes the 
vector field. */

template<class T>
class rsPolyaPotentialEvaluator
{

public:

  using Complex = std::complex<T>;

  // f(z) = 1/z, has pole at z=0
  static T    reciprocal(T x, T y) { return 0.5 * rsLog(x*x + y*y); }
  static void reciprocal(T x, T y, T* u, T * v) { T s = 1/(x*x + y*y); *u = s * x; *v = s * y; }

  // f(z) = z^2
  static T    square(T x, T y) { return x*x*x/3 - x*y*y; } 
  static void square(T x, T y, T* u, T* v) { *u = x*x - y*y; *v = -2*x*y; }

  // f(z) = z^n
  static T    power(T x, T y, int n);
  static void power(T x, T y, int n, T* u, T* v);
  // ToDo:
  // -Figure out and document what the limits for n are. In the implementation we work with static
  //  arrays of some fixed length given by some maxN. Maybe implement the functions using a 
  //  workspace and let a convenience function use a std::vector for that workspace.

  // f(z) = e^z = exp(z)
  static T    exp(T x, T y) { return rsExp(x)*rsCos(y); }
  static void exp(T x, T y, T* u, T* v) { *u = rsExp(x)*rsCos(y); *v = -rsExp(x)*rsSin(y); }

  // f(z) = e^(i*z)
  static T    exp_i(T x, T y) { return exp(-y, x); }
  static void exp_i(T x, T y, T* u, T* v) { exp(-y, x, u, v); }
  // Verify these formulas!

  // f(z) = sin(z)
  static T    sin(T x, T y) { return -rsCos(x)*rsCosh(y); }
  static void sin(T x, T y, T* u, T* v) { *u = rsCosh(y)*rsSin(x); *v = -rsCos(x)*rsSinh(y); }


  // zerosAt_1_m1
  // f(z) = (z-1) * (z+1), u(x,y) = x^2 - y^2 - 1, v(x,y) = -2*x*y, P(x,y) = (1/3)*x^3 - x*y^2 - x
  //static T    zerosAt_1_m1(T x, T y) { return x*x*x/3 - x*y*y - x; } 



  // -Implement more functions to compute Polya vector fields and potentials for other kinds of 
  //  functions. Create contour plots of such functions.


  /** Given a complex function w = f(z) where f is assumed to be holomorphic, this function 
  produces a data matrix of the Polya potential of this function by means of function evaluation 
  and numerical estimation of a potential for the so produced Polya vector field data. The 
  estimation algorithm is rather expensive, works only for smallish numbers for the number of 
  samples (say, something like numSamplesX * numSamplesY = 1000, like 20x50 or 30x30) such that for
  producing larger sized images, it has to be used in conjunction with upsampling. Using it on a
  full resolution grid of, say, 500x500 pixels is still out of the question. It is also still 
  somewhat unreliable with regard to convergence and the results are only approximate anyway. So, 
  if it is at all possible to produce the Polya potential data by an analytic expression, it's 
  better to do it that way. This function should be used only as last resort if it's not reasonably
  possible to find an analytic solution. The function f needs to be holomorphic because otherwise, 
  a Polya potential does not even exist. */
  static rsMatrix<T> estimatePolyaPotential(std::function<Complex (Complex z)> f, 
    T xMin, T xMax, T yMin, T yMax, int numSamplesX, int numSamplesY);
  // ToDo: 
  // -Handle meromorphic functions (= mostly holomorpic but with isolated poles). I think, we 
  //  could do this by introducing upper and lower clipping thresholds in the evaluation of the 
  //  Polya vector field to clip/tame the infinities at the poles. The estimated Polya potential 
  //  will then be wrong in these regions. I think, it may look linear there because the clipped 
  //  vector field is then constant there. But depends on how exactly we clip. We could simply clip 
  //  real and imag seperately but we could also clip the magnitude and retain the phase.

};
// Needs tests.

// The expressions were found with SageMath using the code below. In the example, we use 
// w = f(z) = 1/z. For other functions, just change the "w = 1 / z" line. In cases where you get
// different expressions for U and V, they will differ only in the terms that depend on one 
// variable only. For example, for w = z^2, you'll get (U,V) = (1/3*x^3 - x*y^2, -x*y^2). U differs
// from V, but only by a term that depends only on x but not on y. That's the "integration 
// constant" that has to be added to V when v is integrated with respect to y. It's a constant with
// respect to y but depends on x. It comes about because u contains a term that depends only on x, 
// namely x^2. That's why we always compute both integrals. Often, they will be equal. But if they 
// do differ, we have to collect also the difference terms into out final expression for the 
// potential. Just always collect all the common terms that depend on both variables (these should 
// always match in the two expressions) and then add to that the difference terms from both 
// expressions, if any are present. These difference terms should always depend only on one of the 
// variables (on x in U, on y in V).
//
// var("x y")
// assume(x, "real")
// assume(y, "real")
// z = x + I*y
// w = 1 / z               # function of interest
// u =  w.real() 
// v = -w.imag()
// U = integral(u, x)
// V = integral(v, y)
// u, v, U, V
//
// Results:
// cos(z):  (cos(x)*cosh(y), sin(x)*sinh(y), cosh(y)*sin(x), cosh(y)*sin(x))
// tan(z):  (sin(2*x)/(cos(2*x) + cosh(2*y)), -sinh(2*y)/(cos(2*x) + cosh(2*y)),
//          -1/2*log(cos(2*x) + cosh(2*y)),-1/2*log(cos(2*x) + cosh(2*y)))
// sinh(z): (cos(y)*sinh(x), -cosh(x)*sin(y), cos(y)*cosh(x), cos(y)*cosh(x)) 
// cosh(z): (cos(y)*cosh(x), -sin(y)*sinh(x), cos(y)*sinh(x), cos(y)*sinh(x))
// tanh(z): (sinh(2*x)/(cos(2*y) + cosh(2*x)), -sin(2*y)/(cos(2*y) + cosh(2*x)),
//          1/2*log(cos(2*y) + cosh(2*x)), 1/2*log(cos(2*y) + cosh(2*x)))
// sqrt(z): (sqrt(abs(x + I*y))*cos(1/2*arctan2(y, x)), -sqrt(abs(x + I*y))*sin(1/2*arctan2(y, x)))
//          The integrals didn't succeed -> Try it by hand or other CAS!
// (z-1)*(z+1): (x^2 - y^2 - 1, -2*x*y, 1/3*x^3 - x*y^2 - x, -x*y^2)
//
// ToDo: 
// -Do also z^p for general p (real or complex)
// -Trying to let sage assume that p is a positive integer doesn't seem to change anything. In 
//  that case, we get polynomials with coeffs obtained from binomial coeffs. See zeta paper.


/** Computes the Polya vector field u(x,y), v(x,y) for the integer power function f(z) = z^n 
where n is an integer. */
template<class T>
void rsPolyaPotentialEvaluator<T>::power(T x, T y, int n, T* u, T* v)
{
  static const int maxN = 20;
  T   c[maxN];        // polynomial coeffs for u
  int xP[maxN];       // exponents/powers for x values
  int yP[maxN];       // exponents/powers for y values
  int m;              // number of terms

  // Compute real part:
  m  = rsRealCoeffsComplexPower(abs(n), c, xP, yP);
  *u = rsEvaluateBivariatePolynomial(x, y, m, c, xP, yP);

  // Compute imag part (negated because we want the Polya vector field):
  m  = rsImagCoeffsComplexPower(abs(n), c, xP, yP);
  *v = -rsEvaluateBivariatePolynomial(x, y, m, c, xP, yP);

  // For negative exponents, the result has to be divided by u^2 + v^2:
  T s = 1;
  if(n == -1)
  {
    s = 1 / (x*x + y*y);
    *u = s*x;
    *v = s*y;
  }
  else if(n < 0)
  {
    s  = 1 / (*u * *u + *v * *v);
    *u *=  s;
    *v *= -s;  // Minus is needed. But why?
  }
}

/** Computes the Polya potential P(x,y) for the integer power function f(z) = z^n 
where n is an integer. */
template<class T>
T rsPolyaPotentialEvaluator<T>::power(T x, T y, int n)
{
  static const int maxN = 20;
  T   c[maxN];        // polynomial coeffs for u
  int xP[maxN];       // exponents/powers for x values
  int yP[maxN];       // exponents/powers for y values
  int m;              // number of terms

  if(n >= 0)
  {
    m = rsPotentialCoeffsComplexPower(n, c, xP, yP);
    return rsEvaluateBivariatePolynomial(x, y, m, c, xP, yP);
  }
  else if(n == -1)
  {
    return 0.5 * log(x*x + y*y);
  }
  else
  {
    n = -n;  // make sign positive
    m = rsRealCoeffsComplexPower(n-1, c, xP, yP); 
    T num = -rsEvaluateBivariatePolynomial(x, y, m, c, xP, yP);
    T den = (n-1) * pow(x*x + y*y, n-1);
    return num / den;
  }
}


template<class T>
rsMatrix<T> rsPolyaPotentialEvaluator<T>::estimatePolyaPotential(
  std::function<Complex (Complex z)> f,
  T xMin, T xMax, T yMin, T yMax, int Nx, int Ny)
{
  // Create Polya vector field data (maybe factor out):
  rsMatrix<T> u(Nx, Ny), v(Nx, Ny);
  T dx = (xMax - xMin) / Nx;
  T dy = (yMax - yMin) / Ny;
  for(int j = 0; j < Ny; j++) {
    T y = yMin + j*dy;
    for(int i = 0; i < Nx; i++) {
      T x = xMin + i*dx;
      Complex z(x, y);
      Complex w = f(z);
      u(i, j) =  real(w);
      v(i, j) = -imag(w); }}

  // Create the Polya potential from the Polya vector field: 
  rsMatrixView<T> um(Nx, Ny, u.getDataPointer());
  rsMatrixView<T> vm(Nx, Ny, v.getDataPointer());
  rsMatrix<T> P = rsNumericPotentialSparse(um, vm, dx, dy);
  return P;

  // For tests. Uses dense matrix implementation. Practical only for very small problems, i.e. 
  // small Nx, Ny:
  //rsMatrix<T> P = rsNumericPotential(um, vm, dx, dy);  
  //plotMatrix(P, true);   // for test
  //plotMatrix(P, false);  // for test
}



//=================================================================================================

/** A class for plotting height maps ...TBC...  */

template<class T>
class rsHeightMapPlotter
{

public:

  //-----------------------------------------------------------------------------------------------
  // \name Setup

  /** Sets scaling factors for the final image to be produced. The idea is that the actual 
  evaluation of the potential might have to done on a grid that is coarser than the final image due
  to it being too compuational expensive to directly evaluate it on a finer grid. We would the 
  evaluate it on a coarser grid and obtain the final image using interpolation of this coarse grid 
  data.  ...tbc... */
  void setImageScaling(int newScaleX, int newScaleY) { scaleX = newScaleX; scaleY = newScaleY; }


  //-----------------------------------------------------------------------------------------------
  // \name Plotting

  /** Given a matrix z of height values and a range of x- and y-values such that 
  z(i, j) = height(x_i, y_i) where x_i = xMin + (xMax-xMin) / Nx where Nx is the number of samples 
  of the height z along the x-axis (given by z.numRows) and likewise for y_i, this function 
  produces an image of that height data by converting the matrix to an image, possibly scaling it
  up according to our scaling settings and post-processing the result by drawing in contour lines, 
  etc. (ToDo: later we may also draw in the coordinate axes)   */
  rsImage<T> getHeightMapImage(const rsMatrix<T> z, T xMin, T xMax, T yMin, T yMax);


  rsImage<T> getHeightMapImage(const std::function<T(T x, T y)>& f, 
    T xMin, T xMax, T yMin, T yMax, int w, int h);


  // ToDo:
  // -Add a function that does the same thing but instead of taking a data matrix, it takes a 
  //  std::function<T(T, T)> with two inputs and one output. Eventually, we want to pass functions 
  //  like rsPolyaPotentialEvaluator::square (wrapped into std::function) to that


  /** A helper function to convert data from rsMatrix to rsImage. */
  static rsImage<T> rsMatrixToImage(const rsMatrixView<T>& mat, bool normalize);

  /** Post processes raw image data of the function values (normalization, scaling, drawing of 
  contour lines, etc.) */
  rsImage<T> postProcess(const rsImage<T> img, T xMin, T xMax, T yMin, T yMax);



protected:

  int scaleX = 1, scaleY = 1;

};
// Notes:
// -I considered to give it two template parameters TPix, TVal like rsImageContourPlotter, but that
//  doesn't seem to make sense here and one common T seems enough. There, we may want to use 
//  TPix != TVal becase the class actually *draws* lines in some user given color. But we don't do
//  that here, so we don't need that complication.
//
// ToDo:
// -see rsImageContourPlotter for API design

template<class T>
rsImage<T> rsHeightMapPlotter<T>::rsMatrixToImage(const rsMatrixView<T>& mat, bool normalize)
{
  int w = mat.getNumRows();
  int h = mat.getNumColumns();
  rsImage<T> img(w, h);
  for(int i = 0; i < w; i++)
    for(int j = 0; j < h; j++)
      img(i, j) = mat(i, j);
  if(normalize)
    rsImageProcessor<T>::normalize(img);
  return img;
  // ToDo:
  // -Maybe give the user options, how the matrix row/cols should be interpreted in terms of pixel
  //  coordinates, i.e. how we map between the two index pairs. At the moment, we interpret the
  //  row index of the matrix as x-coordinate in the image and the column index as y-coordinate. 
  //  That means, when looking at the matrix itself, it represents the transposed/rotated image.
}
template<class T>
rsImage<T> rsHeightMapPlotter<T>::getHeightMapImage(const std::function<T(T x, T y)>& f,
  T xMin, T xMax, T yMin, T yMax, int w, int h)
{
  rsImage<T> img(w, h);
  T dx = (xMax - xMin) / w;
  T dy = (yMax - yMin) / h;
  for(int j = 0; j < h; j++) {
    T y = yMin + j*dy;
    for(int i = 0; i < w; i++) {
      T x = xMin + i*dx;
      img(i, j) = f(x, y); }}
  return postProcess(img, xMin, xMax, yMin, yMax);
}

template<class T>
rsImage<T> rsHeightMapPlotter<T>::getHeightMapImage(const rsMatrix<T> P, 
  T xMin, T xMax, T yMin, T yMax)
{
  // Convert matrix P to image and post-process it by scaling it up to the final resolution and
  // drawing in some contour lines:
  rsImage<T> img = rsMatrixToImage(P, true);
  return postProcess(img, xMin, xMax, yMin, yMax);
}

template<class T>
rsImage<T> rsHeightMapPlotter<T>::postProcess(const rsImage<T> imgIn, 
  T xMin, T xMax, T yMin, T yMax)
{
  rsImage<T> img = imgIn;
  rsImageProcessor<T>::normalize(img);
  if(scaleX > 1 || scaleY > 1)
    img = rsImageProcessor<T>::interpolateBilinear(img, scaleX, scaleY);

  // Plot contour lines:
  int numContourLines = 40;   // make member, give the user a setter for that
  rsImageContourPlotter<T, T> cp;  
  rsImage<T> tmp = img;
  for(int i = 0; i < numContourLines; i++)
  {
    T level = T(i) / T(numContourLines);
    cp.drawContour(tmp, level, img, T(0.1875), true);
  }
  rsImageProcessor<T>::normalize(img);  // May need new normalization after adding contours
  // Maybe in the contour plotter, use a saturating addition when drawing in the pixels. That could 
  // avoid the second normalization and also look better overall.

  return img;

  // Notes:
  // -The xMin, ... parameters are not yet used here but maybe we can use them later to draw 
  //  coordinate axes.
}

//=================================================================================================

/** A class for plotting Polya potentials of complex functions. It pulls together functionality 
from rsHeightMapPlotter (which is a superclass of this) and rsPolyaPotentialEvaluator (which is 
used locally where needed). The class makes it convenient to produce images of Polya potentials of
complex functions. ...tbc...  */

template<class T>
class rsPolyaPotentialPlotter : public rsHeightMapPlotter<T>
{

public:

  using Complex = std::complex<T>; 

  /** Given a complex function w = f(z) and and ranges for the real and imaginary parts of the 
  function argument (xMin, ...) and a number of samples for sampling the function along the real 
  (x) and imaginary (y) axis, this function produces an image of the function's Polya vector 
  field. tbc... */
  rsImage<T> getPolyaPotentialImage(std::function<Complex (Complex z)> f, 
    T xMin, T xMax, T yMin, T yMax, int numSamplesX, int numSamplesY);

};

template<class T>
rsImage<T> rsPolyaPotentialPlotter<T>::getPolyaPotentialImage(
  std::function<Complex (Complex z)> f,
  T xMin, T xMax, T yMin, T yMax, int Nx, int Ny)
{
  rsMatrix<T> P = rsPolyaPotentialEvaluator<T>::estimatePolyaPotential(
    f, xMin, xMax, yMin, yMax, Nx, Ny);              // Find Polya potential numerically
  return rsHeightMapPlotter<T>::getHeightMapImage(
    P, xMin, xMax, yMin, yMax);                      // Convert data to image and post-process
}



//=================================================================================================
/*
Move to some other file:
creating movies from pictures:
https://askubuntu.com/questions/971119/convert-a-sequence-of-ppm-images-to-avi-video
https://superuser.com/questions/624567/how-to-create-a-video-from-images-using-ffmpeg


https://hamelot.io/visualization/using-ffmpeg-to-convert-a-set-of-images-into-a-video/


not good:
ffmpeg -r 60 -f image2 -s 1920x1080 -i pic%04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p test.mp4
ffmpeg -r 25 -f image2 -s 300x300 -i Epidemic_Frame%d.ppm -vcodec libx264 -crf 0 test.mp4
ffmpeg -r 25 -f image2 -s 300x300 -i Epidemic_Frame%d.ppm -vcodec libx264 -crf 0  -pix_fmt rgb24 test.mp4
ffmpeg -r 25 -f image2 -s 100x100 -i Epidemic_Frame%d.ppm -vcodec libx264 -crf 0  -pix_fmt rgb24 test.mp4
  this works with the 100x100 images in the "Box" folder - it seems to be related to the background 
  gradients? with uniform background color, all seems well


copy ffmpeg.exe into the folder where the images reside, open a cmd window in that folder and 
enter:

good:
ffmpeg -r 25 -f image2 -s 100x100 -i Epidemic_Frame%d.ppm -vcodec libx264 -crf 10  -pix_fmt yuv420p SIRP.mp4

ffmpeg -r 25 -i VideoTempFrame%d.ppm -vcodec libx264 -crf 10 -pix_fmt yuv420p


path:
C:\Program Files\ffmpeg\bin

https://www.java.com/en/download/help/path.xml


-Righ-click start
-System
-Advanced
-Environment Variables
-select PATH
-click Edit
-append the new path, sperated by a semicolon from what is already there
-confirm with OK and restart the computer
...i don't know, if it's sufficient to add it to User Settings, PATH, and/or if it has to be added
// to system settings -> Path - i added it to both but that may not be necessarry -> figure out


// i get a video file, but it's garbage whe using -pix_fmt rgb24 (sort of - it's still somewhat 
// recognizable, though)...but -pix_fmt yuv420p seems to work

// https://stackoverflow.com/questions/45929024/ffmpeg-rgb-lossless-conversion-video

ffmpeg gui frontends:
https://www.videohelp.com/software/Avanti
https://www.videohelp.com/software/MediaCoder
https://sourceforge.net/projects/ffmpegyag/

// here is a list:
https://github.com/amiaopensource/ffmpeg-amia-wiki/wiki/3)-Graphical-User-Interface-Applications-using-FFmpeg
http://qwinff.github.io/
http://www.avanti.arrozcru.org/
https://handbrake.fr/
http://www.squared5.com/
https://sourceforge.net/projects/ffmpegyag/


*/
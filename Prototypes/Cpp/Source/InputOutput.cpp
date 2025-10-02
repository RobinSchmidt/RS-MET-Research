
//#################################################################################################
// Plotting with GNUPlotCPP

//=================================================================================================
// Convenience functions for certain types of plots. Maybe move to library, maybe into rs_testing 
// module into TestTools/Plotting.h. Maybe at some point even into GNUPlotCPP itself.
// Maybe (some of) it into rs_testing/TestTools/Plotting.h:

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
  //  assignment of std::vector
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

/** Produces a surface plot ...TBC... */
void plotSurface(GNUPlotter& plt)
{
  plt.addCommand("set style fill solid 1.0 noborder");
  plt.addCommand("set pm3d depthorder noborder");
  plt.addCommand("set pm3d lighting primary 0.6 specular 0.0");
  // https://stackoverflow.com/questions/71490416/how-to-make-the-choice-in-3d-color-palette-in-gnuplot-with-light-effect
  // "primary 0.0" is brighter than "primary 0.5". "primary 1.0" makes the underside completely 
  // black. I think this number adjusts between ambient light and sourced light where 0.0 uses 
  // ambient light only (underside is just as bright as upside) whereas 1.0 makes the underside
  // completely black. I think, "specular 0.0" is good for coarser meshes where the segmentation
  // into quadrilaterals give cues about structure of the surface. For fineer meshes, these cues go
  // away a bit more specular light may be beneficial (or it may not). A setting of 
  // "primary 0.6 specular 0.0" looks good with a coarse mesh. Some specular light (like 0.25) 
  // seems to work well with unipolar darkish color-maps like CB_YlGnBu9m but not so good with 
  // bipolar maps with a whiteish color in the middle like CJ_BuYlRd11. A little bit like 0.1
  // might be OK - but it really depends on the angle. Experimentation is needed...

  //plt.plot3D();
  // This will produce the file but it will be only a wireframe drawing. Interestingly, The 
  // wireframe drawing produces a larger .png file. For the PolyaSurfacePow2.png for the
  // paper, the wirframe file takes 144 kB and the nice rendering 89 kB.

  // Because we can't use plot3D, we add the commands and invoke Gnuplot ourselves:
  plt.setupOutputTerminal();
  plt.addCommand("splot '" + plt.getDataPath() + "' i 0 nonuniform matrix with pm3d notitle");
  plt.invokeGNUPlot();
}
// ToDo:
// -Try to use plt.plot3D instead of the 3 bottom lines. For this, we need to change how the splot
//  command is generated intenally - and I don't know yet how.
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
// -Move that functionality into GNUPlotter ...done in plotContourMap?

void prepareForContourPlot(GNUPlotter& plt,
  const std::vector<float> levels, bool useConstColors = true)
{
  bool drawContours = true;   // ToDo: make this a parameter
  std::string cmd;

  // Add the contour lines:
  if(drawContours && !levels.empty())
  {
    plt.addCommand("set contour");
    cmd = "set cntrparam levels discrete " + std::to_string(levels[0]);
    for(int i = 1; i < levels.size(); i++)
      cmd += "," + std::to_string(levels[i]);
    plt.addCommand(cmd);

    const char c[9] = "AA000000";
    for(int i = 1; i <= 10; i++)  // 1...10
      plt.addCommand("set lt " + std::to_string(i) + " lw 2 lc rgb \"#" + c + "\"" );

    //plt.setGraphColors(c, c, c, c, c, c, c, c, c, c);  
    // Old - does the same thing that we do here in the loop except not including the lw setting.
  }

  // Use constant color fills between the contour lines if desired:
  if(useConstColors && !levels.empty())
  {
    cmd = "set palette maxcolors " + std::to_string(levels.size() - 1);
    plt.addCommand(cmd);
    std::string range = "[" + std::to_string(levels[0]) + ":" + std::to_string(rsLast(levels)) + "]";
    plt.addCommand("set zrange " + range);   // range for z values
    plt.addCommand("set cbrange " + range);  // color bar range
  }

  // Plot:
  plt.addCommand("set pm3d map impl");
  plt.setupOutputTerminal();

  plt.addCommand("splot 'C:/Temp/gnuplotData.dat' i 0 nonuniform matrix w pm3d notitle");
  // use plt.getDataPath


  // Notes:
  // -For the contour line color setting in the  const char c[9] = "AA000000";  assignment, I tried
  //  also 00,44,88,AA,CC. AA looks best for pngcairo. Unfortunately, when embedding the png into a 
  //  pdf in LaTeX, the plots do not look so good anymore. In some plots, the contour lines
  //  disappear completely. But using 88 or 00 doesn't seem to help either. In the pdf, some of 
  //  the plots will just look bad, no matter what setting we use here. Maybe try to manually
  //  export the plots from the GUI application. That is an absolute chore but seems to give better 
  //  results. How Gnuplot inteprets the color setting is a complete mess anyway. The color 
  //  channels are ignored completely, only the alpha channel does something - but in pngcairo, 
  //  there always seems to be some extra transparency on top if the setting. It does not behave at 
  //  all as one would expect it to. Finding a setting that looks good for a pdf is a matter of 
  //  trial and error. ToDo: try to use a somwehat thicker linewidth - yes, using "lw 2" seem to 
  //  give acceptable results also for the pdf. So, using "AA000000" for the color and "lw 2" for 
  //  the linewidth seems to be the way to go.
  //
  // ToDo:
  // -Figure out what happens, we change the loop limits for the "set lt ..." commands. Then 
  //  document it somewhere. Try uing 3..7 instead of 1..10. oK - yeah - in this case, 5 lines
  //  are drawn with the defauls lw 1. But why 1..10? is it because we previously have the set lw
  //  1..10 commands in the commandfile, triggered by setToLightMode()? If we do not previously 
  //  call setToLightMode - what is the behavior then?
  //
  // Questions:
  // -What happens, if the levels are non-equidistant? I guess, in this case, the alignment between
  //  constant color region boundaries and contour lines gets messed up.
}

void plotContours(GNUPlotter& plt, const std::vector<float> levels, bool useConstColors = true)
{
  prepareForContourPlot(plt, levels, useConstColors);
  plt.invokeGNUPlot();
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


//=================================================================================================

/** Baseclass for 2D field plotters. Factors out the stuff that is common to different kinds of 2D 
field plotters. Subclasses are rsFieldPlotter2D for creating arrow plots of 2D vector fields and 
rsContourMapPlotter for creating contour maps of 2D scalar fields. */

template<class T>
class rsFieldPlotter2D
{

public:

  rsFieldPlotter2D() { resetToDefaults(); }


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
  // Let's abbreviate "left" by "L", "right" by "R", etc.. I think, we need T-B = R-L to get an 
  // aspect ratio of 1? Here, we use T-B = R-L = 0.8. -> Figure this out and document it properly.
  // What setting do we need to get a 1:1 aspect ratio, given that the pixel-size is square?


  void setTitle(const std::string& newTitle) { title = newTitle; }


  void setColorPalette(GNUPlotter::ColorPalette newMap, bool reverse)
  { colorMap = newMap; reverseColors = reverse; }

  void setToDarkMode(bool shouldBeDark = true) { dark = shouldBeDark; }

  // Adds a custom command that will be passed to the plotter after the standard commands have been
  // passed. Can be used to set up special plotting options or to override the default behavior:
  void addCommand(const std::string& command) { commands.push_back(command); }
  void clearCommands() { commands.clear(); }
  // Maybe rename to addUserCommand or addCustomCommand, adapt alos clearCommands accordingly

  void addPath(const std::vector<rsVector2D<T>>& path) { paths.push_back(path); }
  void clearPaths() { paths.clear(); }
  // One could think that this means a file path but it's a path to draw. Try to find a better name.
  // Maybe addPolyLine

  void setOutputFileName(const std::string& newName) { outputFileName = newName; }
  // it's only the name, not the full path


  void setupPlotter(GNUPlotter* plt);

  virtual void resetToDefaults();
  // Sets all internal member variables back to their defaults

protected:

  void addPathsToPlot(GNUPlotter* plt);

  // Data setup:
  T xMin, xMax, yMin, yMax;

  // Plotting setup:
  int pixelWidth, pixelHeight;

  double left, right, bottom, top;  // Margins

  //double left     = 0.07;  // Left margin.
  //double right    = 0.87;  // One minus right margin.
  //double bottom   = 0.1;   // Bottom margin
  //double top      = 0.9;   // One minus top margin


  std::string title;


  GNUPlotter::ColorPalette colorMap; // = GNUPlotter::ColorPalette::EF_Viridis;
  bool reverseColors;
  bool dark;


  std::vector<std::string> commands;  // Additional commands set by the user for customization
  // maybe rename to userCommands or customCommands


  std::vector<std::vector<rsVector2D<T>>> paths;
  // An array of paths where each path is given as an array of points p_i = (x,y)_i. These paths 
  // are drawn into the plot over the actual data. In the subclass rsFieldPlotter2D, these paths 
  // could, for example, be used to draw in field lines and in the subclass rsContourMapPlotter, 
  // they could be used to draw in geodesics between points of interest. They could also be used to 
  // draw in other "lines of interest" such as lines of constant curvature or whatever.

  std::string outputFileName;
};


template<class T>
void rsFieldPlotter2D<T>::setupPlotter(GNUPlotter* plt)
{
  // Style setup:
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
  plt->addCommand("set xlabel \"\""); // Use empty string as label to get rid of it
  plt->addCommand("set ylabel \"\""); // ToDo: let the user specify axis labels
  if(outputFileName != "")
    plt->setOutputFilePath("C:/Temp/" + outputFileName); // ToDo: don't hardcode the directory
  // What happens, if we don't prepend a directory? will the file end up the current working 
  // directory or will no file be created at all?

  // Plot customizations:
  for(size_t i = 0; i < commands.size(); i++)
    plt->addCommand(commands[i]);
  // We use use the custom "commands" list to set additional user-defined options and/or to 
  // override the default settings from the style setup above.

  /*
  // Try to draw the paths - but it doesn't seem to work:
  // Add the paths:
  std::string lineAttribs = "lw 2";
  plt->drawLine(lineAttribs, 0,0,  1,1);  // test
  //plt->drawPolyLine("lw 2", { 1,2,2,1 }, { 1,1,2,2 });
  // Produces the command:
  //
  // set arrow from 0,0 to 1,1 nohead 
  //
  // in line 40 of the generated gnuplotcommands.txt file. I don't see any line being drawn in the
  // plot though. Perhaps the attributes are such that it becomes invisible. And/or maybe the 
  // actual contour map is drawn on top the produced line and thus is overdrawing it. in this case,
  // we should try to position the "set arrow ..." command later in the command file. mayb after
  // splot command in line 47. That would imply that we can't do it here in setupPlot but rather
  // must do it in the 2 callers in our 2 subclasses. OK - the attributes seem OK. i think, we
  // indeed need to put these drawing commands after the splot command. That requires some 
  // refactoring.
  // BTW: the drawPolyLine call just produces a bunch of "set arrow " commands. It would perhaps be
  // nicer, if we could just produce a single "set polyline" command for gnuplot. Figure out, if 
  // there is such a thing, i.e. if gnuplot supports polylines.
  */

  // ...
}

template<class T>
void rsFieldPlotter2D<T>::addPathsToPlot(GNUPlotter* plt)
{
  for(size_t i = 0; i < paths.size(); i++) {
    if(paths[i].empty())
      continue;
    T x1 = paths[i][0].x;
    T y1 = paths[i][0].y;
    //std::string attribs = "lw 3 lc rgb \"#408844FF\" front";  // A little transparency
    std::string attribs = "lw 3 lc rgb \"#8844FF\" front";      // Opaque
    for(size_t j = 1; j < paths[i].size(); j++) {
      T x2 = paths[i][j].x;
      T y2 = paths[i][j].y;
      plt->drawLine(attribs, x1, y1, x2, y2);
      x1 = x2;
      y1 = y2; }}

  // Notes:
  // -The "front" is important and it's also important that the commands generated from these calls
  //  appear before the splot command in the command file.
  //
  // ToDo:
  // -Let the caller set the line attributes. Have member std::vector<std::string> pathAttributes 
  //  or maybe just pathColors. Hardcoding a thick purple line is not suitable for general 
  //  purpose use.
  //
  // See:
  // http://www.gnuplot.info/demo/arrowstyle.html
}

template<class T>
void rsFieldPlotter2D<T>::resetToDefaults()
{
  setInputRange(T(0), T(1), T(0), T(1));
  setPixelSize(600, 600);
  setDrawRectangle(0.07, 0.87, 0.1, 0.9);
  setTitle("");
  setColorPalette(GNUPlotter::ColorPalette::EF_Viridis, false);
  setToDarkMode(false);
  clearCommands();
  clearPaths();
  setOutputFileName("");
}

//=================================================================================================

/** Suclass of rsFieldPlotter2D to plot scalar fields as contour map. */

template<class T>
class rsContourMapPlotter : public rsFieldPlotter2D<T>
{

public:

  rsContourMapPlotter() { resetToDefaults(); }


  void resetToDefaults() override;

  void setFunction(const std::function<T(T x, T y)>& newFunction) { f = newFunction; }

  void setOutputRange(T minZ, T maxZ, bool clipDataToRange = true) 
  { 
    zMin = minZ;
    zMax = maxZ;
    clipData = clipDataToRange;
  }
  // Invalid range (with max <= min) triggers automatic range choice, I think
  // Maybe have separate clipping options for lower and upper limit

  void setNumContours(int newNumber) { numContours = newNumber; }

  void setSamplingResolution(int numSamplesX, int numSamplesY) 
  { resX = numSamplesX; resY = numSamplesY; }
  // Maybe, if invalid values are passes (<= 0), use numSamplesX = pixelWidthX, etc.


  void plot();

protected:

  // Data setup:
  std::function<T(T x, T y)> f;
  T zMin, zMax;
  bool clipData;

  // Plotting setup:
  int resX, resY, numContours;
};

template<class T>
void rsContourMapPlotter<T>::resetToDefaults()
{
  rsFieldPlotter2D<T>::resetToDefaults();
  std::function<T(T x, T y)> func = [](T x, T y){ return T(0); };
  setFunction(func);
  setOutputRange(T(0), T(0), true); 
  setNumContours(21);
  setSamplingResolution(101, 101);
}

template<class T>
void rsContourMapPlotter<T>::plot()
{
  // Generate the data and figure out appropriate values for zMin/zMax if the user hasn't given a 
  // valid z-range and generate the array of the contour levels that will drawn in as contour 
  // lines:
  std::vector<T> x, y;
  RAPT::rsMatrix<T> z;
  T minZ = zMin;
  T maxZ = zMax;
  generateMatrixData(f, xMin, xMax, yMin, yMax, resX, resY, x, y, z);
  if(minZ >= maxZ) {
    minZ = z.getMinimum();
    maxZ = z.getMaximum(); }

  //std::vector<T> levels = RAPT::rsRangeLinear(minZ, maxZ, numContours);  // Array of contour levels

  std::vector<T> levels;
  if(numContours > 0)
    levels = RAPT::rsRangeLinear(minZ, maxZ, numContours);  // Array of contour levels

  // Clip the matrix data, if desired:
  if(clipData == true) {
    for(int i = 0; i < z.getNumRows(); i++)
      for(int j = 0; j < z.getNumColumns(); j++)
        z(i, j) = RAPT::rsClip(z(i, j), minZ, maxZ); }
  // ToDo: 
  // -Maybe implement and use a function z.clipToRange(minZ, maxZ);

  GNUPlotter plt;
  plt.addDataMatrixFlat(resX, resY, &x[0], &y[0], z.getDataPointer());
  setupPlotter(&plt);
  addPathsToPlot(&plt);
  plotContours(plt, levels, true);   // Make this a member function! It's currently free.
}

//=================================================================================================

/** Suclass of rsFieldPlotter2D to plot a 2D vector fields as arrow map. */

template<class T>
class rsVectorFieldPlotter : public rsFieldPlotter2D<T>  // maybe rename to ...2D
{

public:

  // ToDo:
  //void resetToDefaults() override;


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

  // ToDo:
  // -Call the function to add the paths to the plot. Test it by drawing some vector field with
  //  field lines.
}

//=================================================================================================

/** Under Construction....
Subclass of rsVectorFieldPlotter that can plot a gradient field of a given scalar field. It 
takes a scalar field as input like the rsContourPlotter class does but doesn't plot the contours 
but instead a numerically evaluated gradient field.  */
template<class T>
class rsGradientFieldPlotter : public rsVectorFieldPlotter<T>
{

public:

  void setFunction(const std::function<T(T x, T y)>& newFunction);

protected:

  std::function<T(T x, T y)> P; // The potential function whose gradient we plot as vector field.
  T hx = T(1)/T(1024);          // Step size for the numeric gradient evaluation in x-direction
  T hy = T(1)/T(1024); 
  
private:

  void setFunction(const std::function<void(T x, T y, T* u, T* v)>& newFunction) {}
  // We make this inherited function unavailable to client code.


};

template<class T>
void rsGradientFieldPlotter<T>::setFunction(const std::function<T(T x, T y)>& newFunction) 
{ 
  P = newFunction;

  // Create a function for evaluating the gradient numerically by a central difference:
  std::function<void(T x, T y, T* u, T* v)> g;
  g = [this](T x, T y, T* u, T* v)
  {
    T hi, lo;
    hi = P(x+hx, y);
    lo = P(x-hx, y);
    *u = (hi-lo) / (2*hx);
    hi = P(x, y+hy);
    lo = P(x, y-hy);
    *v = (hi-lo) / (2*hy);
  };

  // Set the so produced g as vector-field function:
  rsVectorFieldPlotter<T>::setFunction(g);
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


//#################################################################################################
// Image read/write

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

//#################################################################################################
// Console output

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

//#################################################################################################
// Video Generation stuff

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
  // The range of the CRF scale is 0–51, where 0 is lossless, 23 is the default, and 51 is worst 
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

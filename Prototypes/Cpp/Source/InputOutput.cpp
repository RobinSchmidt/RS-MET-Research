
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



//-------------------------------------------------------------------------------------------------

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

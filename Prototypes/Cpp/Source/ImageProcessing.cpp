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
  for a rotation of 90° such that there is no difference between centerSumHorz and centerSumVert 
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

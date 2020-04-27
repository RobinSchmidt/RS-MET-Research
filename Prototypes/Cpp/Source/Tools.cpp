#include "../JuceLibraryCode/JuceHeader.h"
using namespace RAPT;
using namespace rosic;

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
    // can we do better? like first creating a string of backspaces and writing it to cout in one 
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
  // " 56.87%" (the leading whitespace is intentional)
  // for longer videos, we may need a finer indicator - factor out this functionality of writing a 
  // progress indicator to the console - this could be useful in many places - maybe have a class
  // rsConsoleProgressIndicator

  /** Prints the initial 0% in the desired format (i.e. with padding) to the console. */
  void init() const
  {
    std::cout << getPercentageString(0.0);
  }

  /** Deletes the old percentage from the console and prints the new one. */
  void update(int itemsDone, int lastItemIndex) const
  {
    deleteCharactersFromConsole(4);                  // delete old percentage from console
    double percentDone = 100.0 * double(itemsDone) / double(lastItemIndex);
    std::cout << getPercentageString(percentDone);   // write the new percentage to console
  }
  // maybe implement a progress bar using ascii code 178 or 219
  // http://www.asciitable.com/
  // maybe make the function mor general: itemsDone, lastItemIndex

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

  rsVideoRGB(int width, int height/*, int frameRate*/)
  {
    this->width     = width;
    this->height    = height;
    //this->frameRate = frameRate;
  }


  //-----------------------------------------------------------------------------------------------
  /** \name Setup */

  /** Selects, whether or not the pixel values should be clipped to the valid range. If they are
  not clipped, they will wrap around to zero when they overflow the valid range. */
  void setPixelClipping(bool shouldClip) { clipPixelValues = shouldClip; }


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

  /** Writes the temporary .ppm files to the harddisk for the given video. */
  void writeTempFiles(const rsVideoRGB& vid) const
  {
    int numFrames = vid.getNumFrames();

    //std::cout << "Writing ppm files:   0%";
    std::cout << "Writing ppm files: ";
    progressIndicator.init();


    for(int i = 0; i < numFrames; i++) {
      std::string path = getTempFileName(i);
      writeImageToFilePPM(vid.getFrame(i), path.c_str());
      //updateProgressIndicator(i, numFrames-1); 
      progressIndicator.update(i, numFrames-1);
    }
    std::cout << "\n\n";
  }

  /** Combines the temporary .ppm files that have previously writted to disk (via calling 
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
    // and by trial and error, i have commented out those optionas which turned out to be 
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
  // used for the temp files: VideoTempFrame.ppm, VideoTempFrame.ppm, ...

  // have strings for other options: pix_fmt, vcodec, preset:
  // std::string pix_fmt = "yuv420p";
  // std::string vcodec  = "libx264"
  // std::string preset  = "veryslow";


  bool cleanUp = true;

  rsConsoleProgressIndicator progressIndicator;

  //std::string tempDir, outDir, outFileName;
};
// todo: mayb allow to encode int different qualities using the same temporary files, i.e create
// temp files once but rund the encoder several times over the files with different quality 
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

  /** Computes parameters s,t such a point on the plane with these parameters will be have matched 
  x and y coordinates with the given target point. */
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

  /** Computes parameters s,t such a point on the plane with these parameters will be have matched 
  x and z coordinates with the given target point. */
  void getMatchParametersXZ(const rsVector3D<T>& target, T* s, T* t) const
  {
    rsVector3D<T> c = target-u;
    T k =   1 / (v.z*w.x - v.x*w.z);
    *s =  (c.z*w.x - c.x*w.z) * k;
    *t = -(c.z*v.x - c.x*v.z) * k;
  }

  /** Computes parameters s,t such a point on the plane with these parameters will be have matched 
  y and z coordinates with the given target point. */
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


/** Returns a vector that contains a chunk of the given input vector v, starting at index "start" 
with length given by "length". */
template<class T>
inline std::vector<T> rsChunk(const std::vector<T>& v, int start, int length)
{
  rsAssert(length >= 0);
  rsAssert(length-start <= (int) v.size());
  std::vector<T> r(length);
  rsArrayTools::copy(&v[start], &r[0], length);
  return r;
}

//-------------------------------------------------------------------------------------------------

/** Extends rsMultiArray by storing information whether a given index is covariant or contravariant
and a tensor weight which is zero for normal tensors and nonzero for relative tensors. 

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

  int getTensorWeight() const { return weight; }

  /** Returns true, iff indices i and j are of opposite variance type, i.e. one covariant the other
  contravariant. This condition is required for the contraction with respect to i,j make sense. */
  bool areIndicesOpposite(int i, int j) const
  {
    return (covariant[i] && !covariant[j]) || (!covariant[i] && covariant[j]);
  }

  // isScalar, isVector, isCovector, 
  // isOfSameTypeAs(const rsTensor<T>& A) - compares ranks, variances and weights

  //-----------------------------------------------------------------------------------------------
  // \name Operations
  // see rsmultiArrayOld for possible implementations

  /** Returns a tensor that results from contracting this tensor with respect to the given pair of 
  indices.  */
  static rsTensor<T> getContraction(const rsTensor<T>& A, int i, int j)
  {
    rsAssert(A.covariant.size() == 0 || A.areIndicesOpposite(i, j),
      "Indices i,j must have opposite variance for contraction");
    rsAssert(A.shape[i] == A.shape[j], "Summation indices must have the same range");
    rsAssert(A.getRank() >= 2, "Rank must be at least 2.");

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
  product or Kronecker product. */
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


  /** Given a tensor product C = A*B and the right factor B, this function retrieves the left
  factor A. */
  static rsTensor<T> getLeftFactor(const rsTensor<T>& C, const rsTensor<T>& B)
  {
    int offset = 0; 
    // preliminary: later use the first index in B which has a nonzero entry and maybe even later 
    // use an entry that causes the leats rounding errors (i think, we should look for numbers, 
    // whose mantissa has the largest number of zero entries, counting from the right end)

    rsTensor<T> A;      // result
    int rankA = C.getRank() - B.getRank();
    A.setShape(rsChunk(C.getShape(), 0, rankA));
    A.covariant = rsChunk(C.covariant, 0, rankA);

    //for(int


    // ...


    return A;
  }


  /** Given a tensor product C = A*B and the left factor A, this function retrieves the right 
  factor B. */
  /*
  static rsTensor<T> getRightFactor(const rsTensor<T>& C, const rsTensor<T>& A)
  {
    int offset = 0; 
    // preliminary: later use the first index in A which has a nonzero entry and maybe even later 
    // use an entry that causes the leats rounding errors (i think, we should look for numbers, 
    // whose mantissa has the largest number of zero entries, counting from the right end)
  }
  */

  // todo:
  // -leftFactor, rightFactor (retrieve left and right factor from outer product and the respective
  //  other factor
  // -factory functions for special tensors: epsilon, delta (both with optional argument to produce
  //  the generalized versions

protected:

  void adjustToNewShape()
  {
    updateStrides(); 
    updateSize();
    data.resize(getSize());
    updateDataPointer();
  }
  // maybe move to baseclass

  int weight = 0;
  std::vector<bool> covariant; // true if an index is covariant, false if contravariant
    // todo: use a more efficient datastructure - maybe rsFlags64 - this would limit us to tensors
    // up to rank 64 - but that should be crazily more than enough

};


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
  (2) Aufstieg zu den Einsteingleichungen (Michael Ruhrländer)


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
  // todo: use them consistently in the function declarations to reduce clutter

  using FuncSclToVec = std::function<void(T, Vec&)>;
  // function that maps scalars to vectors, i.e defines parametric curves

  using FuncVecToVec = std::function<void(const Vec&, Vec&)>;
  // 1st argument: input vector, 2nd argument: output vector
  // maybe rename to FuncVecToVec

  using FuncVecToMat = std::function<void(const Vec&, Mat&)>;
  // 1st argument: input vector, 2nd argument: output Jacobian matrix
  // maybe rename to FuncVecToMat

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



template<class T>
void cartesianToSpherical(T x, T y, T z, T* r, T* theta, T* phi)
{
  *r     = sqrt(x*x + y*y + z*z);
  *phi   = acos(x / sqrt(x*x + y*y));        // ?= atan2(y,x)
  *theta = acos(z / sqrt(x*x + y*y + z*z));  // == acos(z / *r)
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



/*
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
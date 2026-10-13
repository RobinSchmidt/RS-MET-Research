// Maybe rename file to ScriptsResampling.cpp





bool testUpDownSample2D()
{
  // Under construction. We want to create a lossless pair of upsampling/downsampling filters for
  // images, i.e. implement for 2D what we previuosly did for 1D.

  // For the 2D version, start with the assumption of bilinear interpolation for upsampling. We'll 
  // need a 5x5 kernel for downsampling. Let's express the kernel as:
  //
  //    e f c f e
  //    f d b d f
  //    c b a b c
  //    f d b d f
  //    e f c f e
  //
  // where a = d0, b = d1, c = d2 in the old notation used for the 1D case. Bilinear interpolation 
  // would spread into 4 output pixels. I think the condition that kernel must sum to 1 remains. In 
  // this case, this means: a + 4*(b+c+d+e) + 8*f = 1. Maybe the a + 2(b+c) = 1 condition should 
  // also remain valid because when we apply this to a single line, it should work as before? Maybe 
  // the d,e coeffs should be equal to a,b divided by sqrt(2). The rationale is to make the kernel
  // values dependent on distance from the center. We have 6 coeffs, so we need 5 equations if we 
  // want to treat a as free parameter as before. Maybe this is good:
  //
  //    (1) 1 = a + 2*(b+c)                   as before: d1 = 1 - d0
  //    (2) 0 = c + b/2                       as before: d2 = -d1 / 2
  //    (3) 1 = a + 4*(b+c+d+e) + 8*f         total sum of unity
  //    (4) d = b / sqrt(2)
  //    (5) e = c / sqrt(2)
  //
  // Maybe the 2D version of the kernel can be simply obtained by convolving a horizontal and a 
  // vertical version of the 1D kernel? I think, that is equivalent to to putting weighted copy of
  // the kernel into each row (or column) of the 5x5 kernel where the weight is again taken from 
  // the corresponding column (or row) index. That means, if the 1D kernel is [c b a b c]m then the 
  // 2D version would be:
  //
  //   cc  bc  ac  bc  cc                     [c]
  //   cb  bb  ab  bb  cb                     [b]
  //   ca  ba  aa  ba  ca  =  [c b a b c]  *  [a]
  //   cb  bb  ab  bb  ab                     [b]
  //   cc  bc  ac  bc  cc                     [c]
  //
  // where * denotes convolution...or does it? What we have here looks actually more like a tensor
  // product. ...figure out!
  // 
  // -For upsampling in the 2D case, see also rsImageProcessor<T>::interpolateBilinear. But maybe
  //  implement an optimized version for upsampling by 2. It should give the same result as
  //  interpolateBilinear(const rsImage<T>& img, 2, 2)
  // -I think, the upsampling kernel for linear bilinear interpolation is given by:
  //
  //    0   1/4   0                     1/4  0  1/4
  //   1/4   1   1/4   or maybe it's     0   1   0  
  //    0   1/4   0                     1/4  0  1/4


  bool ok = true;

  using TPix = float;
  using Img  = rsImage<TPix>;
  using Prc  = rsImageProcessor<TPix>;
  using IKM  = rsImageKernelMeasures<TPix>;

  // Try to repeatedly upsample a an impulse centered in a 3x3 image by a factor of 2 using 
  // Prc::interpolateBilinear. I'm interested in how the result looks like. What shape does the 
  // impulse become? A Gaussian blob?

  int numStages = 8;   // number of upsampling stages

  Img img(3, 3);
  img(1, 1) = 1.f;

  std::string name = "Impulse";
  int stage = 0;
  while(true)
  {
    // Write current image to a file:
    std::string path = name + std::to_string(stage) + ".ppm";
    //writeImageToFilePPM(img, path.c_str());

    // We interpret the interpolated image as filter kernel and take some measurments of it. The 
    // goal is to find some measurements that say something about the quality of the resampling. Of
    // particular interest is the isotropy of the filter.
    TPix mean  = IKM::mean(img);               // starts at 1/9, approaches 1/4
    TPix sumH  = IKM::centerSumHorz(img);      // 2^stage
    TPix sumV  = IKM::centerSumVert(img);      // 2^stage
    TPix sumD1 = IKM::centerSumDiagDown(img);  // 1, 1.5, 2.75, 5.375, 10.6875 -> find formula!
    TPix asRat = IKM::aspectRatio(img);        // 1
    TPix anIso = IKM::crossness(img);
    // Maybe the measurements should ignore the boundary pixel which are there only for technical 
    // reasons and are always black, so they don't actually belong to the kernel. Maybe use a 
    // cropped image to do the measurements.


    // Interpolate to next stage or leave loop:
    stage++;
    if(stage > numStages)
      break;
    else
      img = Prc::interpolateBilinear(img, 2, 2);
  }

  // Unit test for the crossness computation with crosses and (pseudo) isotropic kernels of 
  // different sizes. The 3x3 kernels for the straight and diagonal cross and the isotropic one 
  // look like:
  //
  //   0  1  0       1  0  1      k  c  k
  //   1  1  1       0  1  0      c  1  c  
  //   0  1  0       1  0  1      k  c  k
  //  
  // where c = 0.5 and k = c/sqrt(2). The isotropic kernel should give zero crossness for any value
  // of c but we use only 0.5 in the test. The straight cross should give a crossness of +1 and the 
  // diagonal cross a crossness of -1. The "isotropic" kernels of size > 3x3 are not really 
  // isotropic - they are only isotropic with respect to those pixels that actually enter the 
  // crossness computation which is good enough for this test.
  TPix tol = 1.e-6;
  for(int n = 3; n <= 11; n += 2)  // kernel sizes: 3x3, 5x5, ..., 11x11
  {
    img.setSize(n, n);
    int m = (n-1) / 2;             // m: middle

    // A straight cross:
    img.clear();
    for(int i = 0; i < n; i++) {
      img(m, i) = 1;
      img(i, m) = 1; }
    //writeImageToFilePPM(img, "Cross.ppm");
    TPix crs = IKM::crossness(img);   // 1
    ok &= rsIsCloseTo(crs, TPix(+1), tol);

    // A diagonal cross:
    img.clear();
    for(int i = 0; i < n; i++) {
      img(i, i)     = 1;
      img(i, n-1-i) = 1;   }
    //writeImageToFilePPM(img, "CrossDiag.ppm");
    crs = IKM::crossness(img); // -1
    ok &= rsIsCloseTo(crs, TPix(-1), tol);

    // An isotropic double-cross:
    img.clear();
    TPix c = 0.5;
    TPix s = 1.0/sqrt(2);
    for(int i = 0; i < n; i++) {  
      img(m, i) = c;
      img(i, m) = c; }
    for(int i = 0; i < n; i++) {
      img(i, i)     = s*c;
      img(i, n-1-i) = s*c;   }
    img(m, m) = 1;       // center (should not matter but anyway)
    //writeImageToFilePPM(img, "DoubleCross.ppm");
    crs = IKM::crossness(img); // 0
    ok &= rsIsCloseTo(crs, TPix(0), tol);
  }

  rsAssert(ok);
  return ok;


  // Observations:
  // -Using Prc::interpolateBilinear for repeatedly upsampling leads to an output image in which we
  //  don't really see a nice Gaussian circular blob. There's a clearly discernible cross of 
  //  vertical and horizontal bright pixels. See Impulse5.ppm


  // ToDo:
  // -Try to get the same result by upsampling via zero-stuffing and post filtering. I think, we 
  //  could filter vertically and horizontally by the kernel [1 2 1]/2 or use th 2D kernel
  //    1  2  1
  //    2  4  2   / 8    ...check the divisor...might be wrong
  //    1  2  1
  // -Then do the same with an isotropic 3x3 kernel. But I think, the isotropic kernel is not 
  //  separable. I think, the general condition for separability is that each row is a scalar 
  //  multiple of the same "prototype" row where the weights come from the vertical column (or vice
  //  versa, i.e. the columns are multiples of a prototype column weighted by coeffs from the 
  //  horizontal kernel).
  // -Try the magic kernel and binomial kernels.
  //
  // -Try to write an algorithm that takes a 2D kernel as input and factors it into 2 1D kernels
  //  in the least squeres sense. Define the error function:
  //    E = (1/2) sum_{i=1}^M sum_{j=1}^N (v_i h_j - k_{ij})^2
  //  where v_i, h_i are the coeffs of the vertical and horizontal 1D kernels, k_{ij} is the given
  //  2D kernel with height M and width N such that v is of length M and h is of length N. The 
  //  partial derivatives are (verify!): 
  //    d E / d v_i = sum_{j=1}^N h_j * (v_i h_j - k_ij)
  //    d E / d h_j = sum_{i=1}^M v_i * (v_i h_j - k_ij)
  //  Maybe solve the problem by a gradient based optimizer. As initial guess for v_i, h_j use
  //  the constant given by square root of the mean of the input kernel such that the initial
  //  product kernel has the same mean as the input kernel


  // Some notes for designing an isotropic fixed point kernel of size 3x3:
  // -181/128 is a decent approximation of sqrt(7) and 181/256 a decent approximation of 1/sqrt(2)
  //  ...but that may be worthless because we actually need a divisor of 255 because that's what
  //  1 maps to. Or maybe we should map 1 to 256. When the pixels are 8 bit, we will need a 16 bit
  //  accumulator anyway, so it makes sense. The kernel could look like:
  //
  //    [181  128  181]
  //    [128  256  128]   /   256
  //    [181  128  181]
}
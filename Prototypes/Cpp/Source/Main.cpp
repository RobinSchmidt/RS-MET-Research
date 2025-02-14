#include "Experiments.cpp"  // includes Tools.cpp which includes libraries rapt and rosic 
#include "Renderings.cpp"   // ...what about the rs_testing module? Where is this included?

// ToDo: 
// -Bring some order into the code of the experiments. At the moment, they are all in the flat file
//  Experiments.cpp. Split it into ExperimentsMath, ExperimentsData, ExperimentsPhysics/Modeling, 
//  ExperimentsSignal, ExperimentsMisc


int main (int /*argc*/, char* /*argv[]*/)
{

  // Signal Processing:
  //testRandomVectors();               // Stub
  //testKalmanFilter();                // Under construction - doesn't work yet


  // Image processing:
  //testGaussBlurFIR();
  //testGaussBlurIIR();
  //testMultiPass();
  //testImageFilterSlanted();
  //testExponentialBlur();
  //testComplexExponentialBlur();
  //animateComplexExponentialBlur();
  //plotComplexGauss1D();
  //testComplexGaussBlurIIR();
  //testUpDownSample1D();              // up- and downsampling with lossless roundtrip
  //testUpDownSample1D_2();
  //testUpDownSample2D();              // stub, dito but for images


  //epidemic();

  // Tensors and differential geometry:
  //testTensor();
  //testPlane();
  //testManifoldPlane();
  //testManifold1();
  //testManifold2();
  //testManifoldPolar();
  //testManifoldSphere();
  //testManifoldEarth();
  //testGeodesic();
  
  //testSortedSet();

  // Autodiff:
  //testAutoDiff();
  //testAutoDiff2();
  //testAutoDiff3();
  //testAutoDiff4();
  //testAutoDiff5();     // stub
  //testAutoDiffReverse1();
  //testDualComplex();

  //testVectorMultiplication3D();
  //testHermiteInterpolation();

  // PDE solvers on nonuniform meshes:
  //testMeshGeneration();
  //testTransportEquation();
  //testWaveEquation();
  //testBiModalFeedback();
  //testPDE_1stOrder();  // stub
  //testVertexMesh();  // moved to main codbase

  // Geometric algebra:
  //testExteriorAlgebra3D();
  //testGeometricAlgebra();
  //testGeometricAlgebraNesting();

  // Math:
  //eulersNumberViaPascalsTriangle();
  //testBellTriangle();
  //testEulerTransformation();
  //testCesaroSum();
  //testFejerSum();
  //testGreensFunction();  // stub
  //testComplexPolar();    // stub
  //testRationalTrigonometry();
  //testLeveledNumber();   // stub
  //testCommutativeHyperOperations();
  //testNewtonFractal();       // move up to image processing, compare to stuff in main repo - may be redundant
  //testPrimeFactorTable();
  //testPrimesAndMore();
  //testDivisors();
  //testSquarity();
  //testFiniteField();
  //testFieldExtensions();
  //testRingExtensions();
  //testPolynomialQuotientRing();  // stub

  //testSet();
  //testRelation();
  //testNeumannNumbers();
  //testNeumannIntegers();
  //testNeumannRationals();
  //testPairingFunctions();

  //testGeneralizedCollatz();
  //testPowerCommutator();
  //testParticleSystem();
  //testWeightedAverages();
  //testSylvesterMatrix();
  //testBezoutMatrix();

  // Modular forms, lattices, elliptic functions:
  //testModularGroup();         // maybe move down, closer to the other 2x2 matrix stuff
  //testModularForms();
  //testIntegerGroup();
  //testBiPeriodicFunctions();
  // ToDo: 
  // -Maybe move the string-group code over here from the main repo, and write testStringGroup()

  //testAttractors();  // move into a section for physics/models

  //testLiftedPolynomial();
  //testFactoredPolynomial();
  testPolynomialRootCorrespondence1();
  testPolynomialRootCorrespondence2();


  // MIMO Filters:
  //testMimoTransferMatrix();
  //testMimoFilters();        // just a stub - not much in there yet, mostly comments
  //testStateSpaceFilters();

  // Math:
  //testMatrixSqrt();              // Tests functions to compute square root of a matrix
  //test2x2Matrices();               // stub
  //test2x2MatrixCommutation();
  //test2x2MatrixInterpolation();    // stub
  //testQuaternion();
  //testChebychevEconomization();    // stub
  //testGeneratingFunction();
  //testCatalanNumbers();
  //testSmoothCrossFade();
  //testSmoothCrossFade2();
  //testSmoothMax();                   // stub





  //testMerge();  // Trying to find an in-place merge algorithm - not sure, if that's possible, though


  // Riemann zeta function:
  //testPolyaPotenialFormulas();
  //testPolarPotenialFormulas();
  //testRiemannZeta();
  //plotZetaPotential();
  //plotZetaPotentialNumeric();
  //testNumericPotential();
  //testPotentialPlotter();
  //polyaPlotExperiments();
  //polyaGeodesics();


  // String processing:
  //testRegex();


  // Batch creation of plots:
  //makePlotsForPolyaPotentialPaper();  // todo: move to Renderings.cpp

  // Renderings of mathematical art:
  //imgRainbowRadiation();
  //testImageFractalization();


  // Throw away code:
  //testPlotToFile();
  //testDefaultArguments();



  return 0;
}

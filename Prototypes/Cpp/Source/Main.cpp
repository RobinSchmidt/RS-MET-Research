#include "Experiments.cpp"  // includes Tools.cpp which includes libraries rapt and rosic 
#include "Renderings.cpp"   // ...what about the rs_testing module? Where is this included?

// ToDo: 
// -Bring some order into the code of the experiments. At the moment, they are all in the flat file
//  Experiments.cpp. Split it into ExperimentsMath, ExperimentsData, ExperimentsPhysics/Modeling, 
//  ExperimentsSignal, ExperimentsMisc


int main (int /*argc*/, char* /*argv[]*/)
{

  // Signal Processing:
  //testRandomVectors();               // Stub - Produce random vectors with user specified covariance matrix
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

  // Resampling:
  testUpDownSample1D();                // up- and downsampling with lossless roundtrip
  //testUpDownSample2D();              // stub, lossless up/downsampling for images


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
  //testTransportEquationMesh2D();
  //testWaveEquationMesh2D();        // stub
  //testBiModalFeedback();
  //testPDE_1stOrder();      // stub - rename to testPDE_1stOrder_Mesh2D
  //testVertexMesh();  // moved to main codbase - does not exist here anymnore




  // Geometric algebra:
  //testExteriorAlgebra3D();
  //testGeometricAlgebra();
  //testGeometricAlgebraNesting();

  // Math:
  //eulersNumberViaPascalsTriangle();
  //testBellTriangle();
  //testEulerTransformation();
  //testShanksTransformation();   // Could be useful for a nonlinear filter?
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
  //testGcdLcm();                    // stub
  //testSquarity();
  //testCompositeness();
  //testPrimeDecomposition();        // Decompose primes additively
  //testFiniteField();
  //testFieldExtensions();
  //testRingExtensions();
  //testPolynomialQuotientRing();  // stub

  //testSet();
  //testRelation();
  //testSetBirthing();
  //testSetSorting();
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
  //testMatrixMulAdapter();              // stub
  //testGeneralizedMatrixOperations();   // Generalize matrix add/mul to arbitrary shaped matrices

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
  //testPolynomialRootCorrespondence1();
  //testPolynomialRootCorrespondence2();


  // MIMO Filters:
  //testMimoTransferMatrix();
  //testMimoFilters();           // just a stub - not much in there yet, mostly comments
  //testStateSpaceFilters();
  testWaveGuides();              // Under construction - maybe move this near the PDE stuff


  // Math:
  //testMatrixSqrt();              // Tests functions to compute square root of a matrix
  //test2x2Matrices();               // stub
  //test2x2MatrixCommutation();
  //test2x2MatrixInterpolation();    // stub
  //testQuaternion();
  //testChebychevExpansion();        // stub
  //testChebychevEconomization();    // stub
  //testGeneratingFunction();
  //testCatalanNumbers();
  //testSmoothCrossFade();
  //testSmoothCrossFade2();
  //testSmoothMax();                   // stub
  //testNewtonOptimizer1D();
  //testFourierTrafo2D();
  //testKroneckerTrafo2D();
  //testDiscreteCalculus();             // stub
  //testIntervalArithmetic();             // stub
  //testContinuedFractions();



  // Misc:
  //testMerge();  // Trying to find an in-place merge algorithm - not sure, if that's possible, though


  //testGaussIntRoots();                  // stub

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

//#include "Tools.cpp"  // this includes rapt and rosic
#include "Experiments.cpp"  // includes Tools.cpp which includes libraries rapt and rosic 
                            // ...what about the rs_testing module? wher is this included?

int main (int /*argc*/, char* /*argv[]*/)
{
  /*
  // try to delete and replace characters on the on the console programatically:
  unsigned char back = 8;     // backspace see http://www.asciitable.com/
  std::cout << "abcd";        // write abcd
  std::cout << back << back;  // delete cd
  std::cout << "CD";          // write CD - should result in abCD - seems to work
  getchar();
  // OK - this can be deleted. It has been integrated into class rsConsoleProgressIndicator
  // in Tools.cpp
  */

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


  //testEulerTransformation();
  //testGreensFunction();  // stub
  //testComplexPolar();    // stub
  //testRationalTrigonometry();
  //testLeveledNumber();
  //testNewtonFractal();       // move up to image processing, compare to stuff in main repo - may be redundant
  //testPrimeFactorTable();
  //testPrimesAndMore();
  //testFiniteField();     // stub
  //testGeneralizedCollatz();
  //testPowerCommutator();
  //testParticleSystem();
  //testWeightedAverages();

  // Modular forms, lattices, elliptic functions:
  //testModularGroup();         // maybe move down, closer to the other 2x2 matrix stuff
  //testModularForms();
  //testIntegerGroup();
  //testBiPeriodicFunctions();
  // ToDo: 
  // -Maybe move the string-group code over here from the main repo, and write testStringGroup()

  //testAttractors();

  // MIMO Filters:
  //testMimoTransferMatrix();
  //testMimoFilters();        // just a stub - not much in there yet, mostly comments
  //testStateSpaceFilters();

  //test2x2Matrices();               // stub
  //test2x2MatrixInterpolation();    // stub
  //testQuaternion();
  //testChebychevEconomization();  // stub

  //testGeneratingFunction();
  //testCatalanNumbers();
  //testSmoothCrossFade();
  //testMerge();


  // Riemann zeta function:
  //testPolyaPotenialFormulas();
  //testRiemannZeta();
  //plotZetaPotential();
  //plotZetaPotentialNumeric();
  //testNumericPotential();
  //testPotentialPlotter();
  polyaPlotExperiments();
  polyaGeodesics();
  makePlotsForPolyaPaper();



  return 0;
}

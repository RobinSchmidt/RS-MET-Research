//#include "Tools.cpp"  // this includes rapt and rosic
#include "Experiments.cpp"  // includes Tools.cpp

int main (int /*argc*/, char* /*argv[]*/)
{
  /*
  // try to delete and replace characters on the on the console programatically:
  unsigned char back = 8;     // backspace see http://www.asciitable.com/
  std::cout << "abcd";        // write abcd
  std::cout << back << back;  // delete cd
  std::cout << "CD";          // write CD - should result in abCD - seems to work
  getchar();
  */

  //testGaussBlurFIR();
  //testGaussBlurIIR();
  //testMultiPass();
  //testImageFilterSlanted();
  //testExponentialBlur();
  //testComplexExponentialBlur();
  //animateComplexExponentialBlur();
  //plotComplexGauss1D();
  //testComplexGaussBlurIIR();
  //epidemic();

  //testTensor();
  //testPlane();
  //testManifoldPlane();
  //testManifold1();
  //testManifold2();
  //testManifoldPolar();
  //testManifoldSphere();
  //testManifoldEarth();
  
  //testSortedSet();
  //testAutoDiff();
  //testAutoDiff2();
  //testAutoDiff3();
  //testAutoDiff4();
  //testAutoDiff5();     // stub
  //testAutoDiffReverse1();
  //testDualComplex();

  //testVectorMultiplication3D();
  //testHermiteInterpolation();
  //testMeshGeneration();
  //testTransportEquation();
  //testWaveEquation();
  //testBiModalFeedback();
  //testPDE_1stOrder();  // stub
  //testVertexMesh();  // moved to main codbase
  //testExteriorAlgebra3D();
  //testGeometricAlgebra();
  //testEulerTransformation();
  //testGreensFunction();  // stub
  //testComplexPolar();    // stub
  //testRationalTrigonometry();
  //testLeveledNumber();
  //testNewtonFractal();
  //testPrimeFactorTable();
  //testPrimesAndMore();
  //testGeneralizedCollatz();
  //testParticleSystem();
  //testWeightedAverages();
  //testModularGroup();
  //testModularForms();

  //testAttractors();

  //testMimoTransferMatrix();
  //testMimoFilters();        // just a stub - not much in there yet, mostly comments
  //testStateSpaceFilters();

  //test2x2Matrices();  // stub
  //testQuaternion();
  //testChebychevEconomization();  // stub

  //testGeneratingFunction();
  //testCatalanNumbers();
  testSmoothCrossFade();


  return 0;
}

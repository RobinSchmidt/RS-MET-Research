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
  //testAutoDiffReverse1();
  //testDualComplex();
  //testVectorMultiplication3D();
  //testHermiteInterpolation();
  testMeshGeneration();
  //testVertexMesh();  // moved to main codbase


  return 0;
}

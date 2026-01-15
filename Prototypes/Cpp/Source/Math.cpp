
// ToDo:
//
// - Move code from Tools.cpp to here - but not all math related code should go into this file. We 
//   want to have some finer granularity. For example, all the geometric algebra related stuff 
//   should get its own file, etc.


//=================================================================================================

/** Under Construction. Just a stub at the moment

A class for polynomial roots finding. It implements various methods such as the Laguerre method,
Jenkis-Traub, etc. as well as the closed form formulas for polynomials where such formuals exist,
namely polynomials of degrees 1..4. ...TBC...

*/

template<class TCoef, class TArg>   // Maybe rename TArg to TRoot ...or maybe not
class rsPolynomialRootFinder
{

public:


protected:


};

// ToDo:
//
// - We want to factor out anything that has to do with polynomial root finding from the class
//   rsPolynomial in the RAPT library. This includes all the stuff in the "Roots (Low Level)" 
//   section. Somewhere in the experiments or prototypes, there must also already be some code 
//   where I attempted to implement the formulas for cubic (and maybe even quartic?) roots. That
//   stuff should also go into this class.
//
// - The private repo has a textfile Documents/MathNotes/CubicAndQuarticEquations.txt where I have
//   noted some formulas, derivations, links about the closed form formulas. Look that up for help
//   for implementing them. Also, this video is very good:
//   https://www.youtube.com/watch?v=o8UNhs2OaG8  How to Solve ANY Cubic or Quartic Equation!
//
// - Maybe try to implement this class with the help from Copilot. It seems to be a suitable task
//   to tackle with AI assistance because it has a well defined and managable scope without being
//   too trivial. The AI probably knows the algorithms better than what I could figure out.




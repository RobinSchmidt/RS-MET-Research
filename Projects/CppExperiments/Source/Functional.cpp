

class Statistics
{
public:
  static double normalDistribution(double x, double mu, double sigma)
  {
    static const double pi = 3.14;   // we are grossly imprecise here
    double s22 = 2 * sigma * sigma;
    double xm  = x - mu;
    return (1./sqrt(s22*pi)) * exp(-xm*xm / s22);
  }
};

void testFunctionShortcuts()
{
  // Use a lambda function to define a shortcut to an otherwise verbose function call (the compiler
  // will optimize it away):
  double mu = 5, sigma = 2;
  auto normal_5_2 = [=](double x)->double{ return Statistics::normalDistribution(x, mu, sigma); };
  double y = normal_5_2(3);
  // Using such a shortcut is useful, when we need to call a function with a long name many times 
  // (and not inside a loop). We save a lot of redundant typing by using a shortcut compared to 
  // writing the full name out for each individual call. It's a bit similar to using things like
  // using Vec = std::vector<complex<double>>;  which is equivalent to 
  // typedef std::vector<complex<double>> Vec;  ...but the former is considered more modern
  
  // make a closure: std::function<...> getNormalDistribution(mu, sigma) return lambda-function

}


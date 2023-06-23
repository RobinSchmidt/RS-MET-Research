// Code is taken from this post:
// https://stackoverflow.com/questions/41549533/riemann-zeta-function-with-complex-argument

const long double LOWER_THRESHOLD = 1.0e-6;
const long double UPPER_BOUND = 1.0e+4;
const int MAXNUM = 100;

std::complex<long double> zeta(const std::complex<long double>& s)
{
  std::complex<long double> a_arr[MAXNUM + 1];
  std::complex<long double> half(0.5, 0.0);
  std::complex<long double> one(1.0, 0.0);
  std::complex<long double> two(2.0, 0.0);
  std::complex<long double> rev(-1.0, 0.0);
  std::complex<long double> sum(0.0, 0.0);
  std::complex<long double> prev(1.0e+20, 0.0);

  a_arr[0] = half / (one - std::pow(two, (one - s))); //initialize with a_0 = 0.5 / (1 - 2^(1-s))
  sum += a_arr[0];

  for (int n = 1; n <= MAXNUM; n++)
  {
    std::complex<long double> nCplx(n, 0.0); //complex index

    for (int k = 0; k < n; k++)
    {
      std::complex<long double> kCplx(k, 0.0); //complex index

      a_arr[k] *= half * (nCplx / (nCplx - kCplx));
      sum += a_arr[k];
    }

    a_arr[n] = (rev * a_arr[n - 1] * std::pow((nCplx / (nCplx + one)), s) / nCplx);
    sum += a_arr[n];


    if (std::abs(prev - sum) < LOWER_THRESHOLD)//If the difference is less than or equal to the threshold value, it is considered to be convergent and the calculation is terminated.
      break;

    if (std::abs(sum) > UPPER_BOUND)//doesn't work for large values, so it gets terminated when it exceeds UPPER_BOUND
      break;

    prev = sum;
  }

  return sum;
}
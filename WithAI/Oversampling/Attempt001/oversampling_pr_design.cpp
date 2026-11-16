// Build: g++ -O2 -std=c++17 oversampling_pr_design.cpp -o pr_design
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>
#include <algorithm>

using Vec = std::vector<double>;
using Mat = std::vector<Vec>;

// ---------------- adjustable parameters ----------------
static const int    L           = 4;           // oversampling factor
static const int    N_U         = 8 * L + 1;   // length of u (odd -> linear phase)
static const int    N_D         = 16 * L + 1;  // length of d (odd)
static const double KAISER_BETA = 8.0;         // window shape for u
static const double TRANSITION  = 0.2;         // relative transition width around pi/L, used for d
// --------------------------------------------------------

static const double PI = 3.14159265358979323846;

// Modified Bessel function I0 (series), for the Kaiser window
static double besselI0(double x)
{
    double sum = 1.0, term = 1.0, q = x * x / 4.0;
    for (int k = 1; k < 200; ++k) {
        term *= q / (double(k) * k);
        sum += term;
        if (term < 1e-18 * sum) break;
    }
    return sum;
}

static double sinc(double x)
{
    return x == 0.0 ? 1.0 : std::sin(PI * x) / (PI * x);
}

// Kaiser-windowed sinc, cutoff pi/L, DC gain L
Vec designU(int L, int nTaps, double beta)
{
    Vec u(nTaps);
    double mid = (nTaps - 1) / 2.0, sum = 0.0;
    double i0b = besselI0(beta);
    for (int i = 0; i < nTaps; ++i) {
        double n = i - mid;
        double r = (nTaps > 1) ? n / mid : 0.0;
        double w = besselI0(beta * std::sqrt(std::max(0.0, 1.0 - r * r))) / i0b;
        u[i] = sinc(n / L) * w;
        sum += u[i];
    }
    for (double& v : u) v *= L / sum;
    return u;
}

// Solve A x = b by Gaussian elimination with partial pivoting
Vec solveLinear(Mat A, Vec b)
{
    const int n = (int)b.size();
    for (int col = 0; col < n; ++col) {
        int piv = col;
        for (int r = col + 1; r < n; ++r)
            if (std::fabs(A[r][col]) > std::fabs(A[piv][col])) piv = r;
        if (std::fabs(A[piv][col]) < 1e-300)
            throw std::runtime_error("Singular KKT matrix");
        std::swap(A[piv], A[col]);
        std::swap(b[piv], b[col]);
        for (int r = col + 1; r < n; ++r) {
            double f = A[r][col] / A[col][col];
            if (f == 0.0) continue;
            for (int c = col; c < n; ++c) A[r][c] -= f * A[col][c];
            b[r] -= f * b[col];
        }
    }
    Vec x(n);
    for (int r = n - 1; r >= 0; --r) {
        double s = b[r];
        for (int c = r + 1; c < n; ++c) s -= A[r][c] * x[c];
        x[r] = s / A[r][r];
    }
    return x;
}

// Minimize stopband energy of d subject to (u*d)[c + L*n] = delta[n]
Vec designD(const Vec& u, int L, int nTaps, double transition)
{
    const int nU = (int)u.size();
    const int nH = nU + nTaps - 1;
    const int c  = (nH - 1) / 2;

    std::vector<int> idx;
    for (int p = c % L; p < nH; p += L) idx.push_back(p);
    const int m = (int)idx.size();
    if (m >= nTaps)
        throw std::runtime_error("Too many constraints; increase N_D or reduce N_U.");

    const double wPass = (PI / L) * (1.0 - transition);
    const double wStop = 2.0 * PI / L - wPass;

    const int K = nTaps + m;
    Mat A(K, Vec(K, 0.0));
    Vec rhs(K, 0.0);

    // Q block: stopband energy matrix
    for (int i = 0; i < nTaps; ++i)
        for (int j = 0; j < nTaps; ++j) {
            int diff = i - j;
            A[i][j] = (diff == 0) ? (PI - wStop)
                                  : (std::sin(diff * PI) - std::sin(diff * wStop)) / diff;
        }
    for (int i = 0; i < nTaps; ++i) A[i][i] += 1e-12;

    // Constraint blocks: C[m][j] = u[idx[m] - j]
    for (int r = 0; r < m; ++r) {
        for (int j = 0; j < nTaps; ++j) {
            int i = idx[r] - j;
            if (i >= 0 && i < nU) {
                A[nTaps + r][j] = u[i];
                A[j][nTaps + r] = u[i];
            }
        }
        rhs[nTaps + r] = (idx[r] == c) ? 1.0 : 0.0;
    }

    Vec sol = solveLinear(A, rhs);
    return Vec(sol.begin(), sol.begin() + nTaps);
}

Vec convolve(const Vec& a, const Vec& b)
{
    Vec y(a.size() + b.size() - 1, 0.0);
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] == 0.0) continue;   // skips the zero-stuffed samples
        for (size_t j = 0; j < b.size(); ++j) y[i + j] += a[i] * b[j];
    }
    return y;
}

// upsample -> (identity waveshaper) -> downsample, compensating the total delay
Vec roundtrip(const Vec& x, const Vec& u, const Vec& d, int L)
{
    Vec up(x.size() * L, 0.0);
    for (size_t i = 0; i < x.size(); ++i) up[i * L] = x[i];
    up = convolve(up, u);
    // <-- waveshaper would go here, e.g. for (double& s : up) s = std::tanh(s);
    Vec y = convolve(up, d);
    size_t delay = (u.size() - 1) / 2 + (d.size() - 1) / 2;
    Vec out(x.size());
    for (size_t n = 0; n < x.size(); ++n) out[n] = y[delay + n * L];
    return out;
}

static double magDb(const Vec& h, double w)
{
    std::complex<double> s = 0.0;
    for (size_t k = 0; k < h.size(); ++k)
        s += h[k] * std::exp(std::complex<double>(0.0, -w * double(k)));
    return 20.0 * std::log10(std::abs(s) + 1e-300);
}

static double worstStopbandDb(const Vec& h, double wStop)
{
    double worst = -1e300;
    const int num = 4096;
    for (int i = 0; i < num; ++i) {
        double w = PI * i / (num - 1);
        if (w >= wStop) worst = std::max(worst, magDb(h, w));
    }
    return worst;
}

int main()
{
    try {
        Vec u = designU(L, N_U, KAISER_BETA);
        Vec d = designD(u, L, N_D, TRANSITION);

        std::mt19937 rng(0);
        std::normal_distribution<double> dist(0.0, 1.0);
        Vec x(4096);
        for (double& v : x) v = dist(rng);

        Vec y = roundtrip(x, u, d, L);
        double maxErr = 0.0;
        for (size_t i = 0; i < x.size(); ++i)
            maxErr = std::max(maxErr, std::fabs(y[i] - x[i]));

        double sumU = 0.0, sumD = 0.0;
        for (double v : u) sumU += v;
        for (double v : d) sumD += v;
        const double wStop = 2.0 * PI / L - (PI / L) * (1.0 - TRANSITION);

        std::printf("L=%d, len(u)=%zu, len(d)=%zu\n", L, u.size(), d.size());
        std::printf("max roundtrip error: %g\n", maxErr);
        std::printf("u: DC gain %.4f (should be %d), worst stopband %.1f dB\n",
                    sumU, L, worstStopbandDb(u, wStop));
        std::printf("d: DC gain %.4f, worst stopband %.1f dB\n",
                    sumD, worstStopbandDb(d, wStop));
    }
    catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
    return 0;
}
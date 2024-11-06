// From here:
// https://gist.github.com/hollance/91309bd54b2ac9663d4315e130c5ccd9


inline float denormalsToZero(float x) {
  return (std::abs(x) < 1e-15f) ? 0.0f : x;
}

/**
  mystran's Bi-quadratic filter

  https://www.kvraudio.com/forum/viewtopic.php?p=4836443#p4836443

  This is not a direct form topology! Essentially it is modified coupled form
  (and not too different from classic digital SVF).

  Unlike the direct form biquads, this implementation does not blow up under
  heavy modulation. However, keep in mind that biquads don't make very musical
  filters and should only be for simple filtering tasks.
 */
class BiQuadFilter {
public:
  BiQuadFilter() : t0(1.0f), t1(0.0f), t2(0.0f), e(0.0f), f(0.0f) { }

  void setCoefficients(float t0, float t1, float t2, float e, float f) {
    this->t0 = t0;
    this->t1 = t1;
    this->t2 = t2;
    this->e = e;
    this->f = f;
  }

  void setCoefficients(double a0, double a1, double a2, double b0, double b1, double b2) {
    a0 /= b0;
    a1 /= b0;
    a2 /= b0;
    b1 /= b0;
    b2 /= b0;

    f = b2;
    e = std::sqrt(1.0f + b1 + b2);

    t0 = a0 / e;
    t1 = -a2 / e;
    t2 = (a2 + a1 + a0) / (e * e);
  }

  void setCoefficients(const BiQuadFilter &other) {
    t0 = other.t0;
    t1 = other.t1;
    t2 = other.t2;
    e  = other.e;
    f  = other.f;
  }

  void reset() {
    z1 = z2 = 0.0f;
  }

  inline float processSample(float x) {
    const float tmp = denormalsToZero(z1 * f + (x - z2) * e);
    const float y = tmp * t0 + z1 * t1 + z2 * t2;
    z2 = denormalsToZero(tmp * e + z2);
    z1 = tmp;
    return y;
  }

private:
  float t0, t1, t2, e, f;
  float z1, z2;
};
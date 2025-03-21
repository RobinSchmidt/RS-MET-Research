
// This code was posted by Elan here:
// https://github.com/RobinSchmidt/RS-MET/discussions/324

class Attractor
{

public:

  void setSampleRate(double newSampleRate)
  {
    sampleRate = newSampleRate;
    update();
  }

  void setFrequency(double v)
  {
    frequency = v;
    update();
  }

  /** Sets the internal state of the system consisting of the 3 state variables x, y, z. */
  void setState(double inX, double inY, double inZ)
  {
    x = inX;
    y = inY;
    z = inZ;
  }

  /* Because each attractor has a different set of constants, we use an array to set values 
  arbitrarily */
  void setConstant(uint index, double value)
  {
    C[index] = value;
  }

  double getX()
  {
    return x;
  }

  double getY()
  {
    return y;
  }

  double getZ()
  {
    return z;
  }

  double getStepSize() const { return h; }

  void inc(){}

  void reset(){}

protected:

  
  void update() {  h = frequency / sampleRate; }

  double C[10]; // array of constants
  double sampleRate = 44100;
  double frequency = 100;
  double h = frequency / sampleRate;
  double x, y, z, w;
  double dx, dy, dz, dw;
};


class Lorentz : public Attractor
{
  uint sigma = 0;
  uint rho   = 1;
  uint beta  = 2;

public:

  Lorentz()
  {
    C[sigma] = 10.0;
    C[rho]   = 28.0;
    C[beta]  = 8.0/3.0;

    reset();
  }

  void inc()
  {
    dx = C[sigma] * (y - x);
    dy = x * (C[rho] - z) - y;
    dz = x * y - C[beta] * z;

    x += h * dx;
    y += h * dy;
    z += h * dz;
  }

  void reset()
  {
    x = 0.5;
    y = 0.0;
    z = 0.0;
  }
};

class Aizawa : public Attractor // http://www.3d-meier.de/tut19/Seite3.html
{
  uint a = 0;
  uint b = 1;
  uint c = 2;
  uint d = 3;
  uint e = 4;
  uint f = 5;

public:

  Aizawa()
  {
    C[a] = 0.95;
    C[b] = 0.7;
    C[c] = 0.6;
    C[d] = 3.5;
    C[e] = 0.25;
    C[f] = 0.1;

    reset();
  }

  void inc()
  {
    dx = (z - C[b]) * x - C[d] * y;
    dy = C[d] * x + (z - C[b]) * y;
    dz = C[c] + C[a] * z - (pow(z, 3) / 3.0) - (x*x + y*y) * (1 + C[e] * z) + C[f] * z * pow(x, 3);

    x += h * dx;
    y += h * dy;
    z += h * dz;
  }

  void reset()
  {
    x = .1;
    y = 0;
    z = 0;
  }
};

class Bouali : public Attractor // http://www.3d-meier.de/tut19/Seite5.html
{
  uint a = 0;
  uint s = 1;

  Bouali()
  {
    C[a] = 0.3;
    C[s] = 1.0;

    reset();
  }

  void inc()
  {
    dx = x * (4-y) + C[a] * z;
    dy = -y * (1 - x * x);
    dz = -x * (1.5 - C[s] * z) -0.05 * z;

    x += h * dx;
    y += h * dy;
    z += h * dz;
  }

  void reset()
  {
    x = 1;
    y = 0.1;
    z = 0.1;
  }
};

class ChenLee : public Attractor // http://www.3d-meier.de/tut19/Seite8.html
{
  uint a = 0;
  uint b = 1;
  uint c = 2;

  ChenLee()
  {
    C[a] = 5;
    C[b] = -10;
    C[c] = -0.38;

    reset();
  }

  void inc()
  {
    dx = C[a] * x - y * z;
    dy = C[b] * y + x * z;
    dz = C[c] * z + x*y/3.0;

    x += h * dx;
    y += h * dy;
    z += h * dz;
  }

  void reset()
  {
    x = 1;
    y = 0;
    z = 4.5;
  }
};

class DequanLi : public Attractor // http://www.3d-meier.de/tut19/Seite9.html
{
  uint a = 0;
  uint c = 1;
  uint d = 2;
  uint e = 3;
  uint k = 4;
  uint f = 5;

  DequanLi()
  {
    C[a] = 40;
    C[c] = 1.833;
    C[d] = 0.16;
    C[e] = 0.65;
    C[k] = 55;
    C[f] = 20;

    reset();
  }

  void inc()
  {
    dx = C[a] * (y - x) + C[d] * x * z;
    double dy = C[k] * x + C[f] * y - x * z;
    dz = C[c] * z + x * y - C[e] * pow(x, 2);

    x += h * dx;
    y += h * dy;
    z += h * dz;
  }

  void reset()
  {
    x = .349;
    y = 0;
    z = -.16;
  }
};

class DenTSUCS2 : public Attractor // http://www.3d-meier.de/tut19/Seite43.html
{

public:

  DenTSUCS2()
  {
    C[a] = 40;
    C[b] = 55;
    C[c] = 1.833;
    C[d] = 0.16;
    C[e] = 0.65;
    C[f] = 20;

    reset();
  }

  /** Performs a forward Euler step. */
  void stepEuler()
  {
    computeDerivatives(x, y, z, &dx, &dy, &dz);
    x += h*dx; y += h*dy; z += h*dz;
  }

  /** Performs a step using the midpoint rule. */
  void stepMidpoint()
  {
    computeDerivatives(x,          y,          z,          &dx, &dy, &dz);
    computeDerivatives(x+0.5*h*dx, y+0.5*h*dy, z+0.5*h*dz, &dx, &dy, &dz);
    x += h*dx; y += h*dy; z += h*dz;
  }


  void reset()
  {
    x =  0.1;
    y =  1.0;
    z = -0.1;
  }

protected:

  void computeDerivatives(double x, double y, double z, double* dx, double* dy, double* dz)
  {
    *dx = C[a] * (y - x) + C[d] * x * z;
    *dy = C[b] * x - x * z + C[f] * y;
    *dz = C[c] * z + x * y - C[e] * pow(x, 2);
  }
  // todo: 
  // -maybe factor out into a purely virtual baseclass function

  uint a = 0;
  uint b = 1;
  uint c = 2;
  uint d = 3;
  uint e = 4;
  uint f = 5;

};

class DenGenesioTesi : public Attractor // http://www.3d-meier.de/tut19/Seite11.html
{
  uint a = 0;
  uint b = 1;
  uint c = 2;

  DenGenesioTesi()
  {
    C[a] = .44;
    C[b] = 1.1;
    C[c] = 1.0;

    reset();
  }

  void inc()
  {
    dx = y;
    dy = z;
    dz = -C[c] * x - C[b] * y - C[a] * z + pow(x, 2);

    x += h * dx;
    y += h * dy;
    z += h * dz;
  }

  void reset()
  {
    x = .1;
    y = .1;
    z = .1;
  }
};

class Hadley : public Attractor // http://www.3d-meier.de/tut19/Seite12.html
{
  uint a = 0;
  uint b = 1;
  uint f = 2;
  uint g = 3;

  Hadley()
  {
    C[a] = .2;
    C[b] = 4;
    C[f] = 8;
    C[g] = 1;

    reset();
  }

  void inc()
  {
    dx = -pow(y, 2) - pow(z, 2) - C[a] * x + C[a] * f;
    dy = x*y - C[b]*x*z - y + C[g];
    dz = C[b]*x*y + x*z - z;

    x += h * dx;
    y += h * dy;
    z += h * dz;
  }

  void reset()
  {
    x = .1;
    y = 0;
    z = 0;
  }
};

class Halvorsen : public Attractor // http://www.3d-meier.de/tut19/Seite13.html
{
  uint a = 0;

  Halvorsen()
  {
    C[a] = 1.4;

    reset();
  }

  void inc()
  {
    dx = -C[a]*x - 4*y - 4*z - pow(y, 2);
    dy = -C[a]*y - 4*z - 4*x - pow(z, 2);
    dz = -C[a]*z - 4*x - 4*y - pow(x, 2);

    x += h * dx;
    y += h * dy;
    z += h * dz;
  }

  void reset()
  {
    x = 1;
    y = 0;
    z = 0;
  }
};

class HyperchaoticQi : public Attractor // http://www.3d-meier.de/tut19/Seite90.html
{
  uint a = 0;
  uint b = 1;
  uint c = 2;
  uint d = 3;
  uint e = 4;
  uint f = 5;

  HyperchaoticQi()
  {
    C[a] = 50;
    C[b] = 24;
    C[c] = 13;
    C[d] = 8;
    C[e] = 33;
    C[f] = 30;

    reset();
  }

  void inc()
  {
    dx = C[a] * (y - x) + y * z;
    dy = C[b] * (x + y) - x * z;
    dz = -C[c]*z - C[e]*w + x*y;
    dw = -C[d]*w + C[f]*z + x*y;

    x += h * dx;
    y += h * dy;
    z += h * dz;
    w += h * dw;
  }

  void reset()
  {
    x = 10;
    y = 15;
    z = 20;
    w = 22;
  }
};

class Dadra : public Attractor //http://www.3d-meier.de/tut19/Seite77.html
{
  uint p = 0;
  uint q = 1;
  uint r = 2;
  uint s = 3;
  uint e = 4;

  Dadra()
  {
    C[p] = 3;
    C[q] = 2.7;
    C[r] = 1.7;
    C[s] = 2;
    C[e] = 9;

    reset();
  }

  void inc()
  {
    dx = y - C[p]*x + C[q]*y*z;
    dy = C[r]*y - x*z + z;
    dz = C[s]*x*y - C[e]*z;

    x += h * dx;
    y += h * dy;
    z += h * dz;
  }

  void reset()
  {
    x = .1;
    y = .1;
    z = .1;
  }
};

class DenSprottLinzJ : public Attractor // http://www.3d-meier.de/tut19/Seite29.html
{
  uint a = 0;

  DenSprottLinzJ()
  {
    C[a] = 2;

    reset();
  }

  void inc()
  {
    dx = C[a]*z;
    dy = -C[a]*y + z;
    dz = -x + y + y*y;

    x += h * dx;
    y += h * dy;
    z += h * dz;
  }

  void reset()
  {
    x = .1;
    y = .1;
    z = .1;
  }
};

class LiuChen : public Attractor // http://www.3d-meier.de/tut19/Seite46.html
{
  uint a = 0;
  uint b = 1;
  uint c = 2;
  uint d = 3;
  uint e = 4;
  uint f = 5;
  uint g = 6;

  LiuChen()
  {
    C[a] = 2.4;
    C[b] = -3.78;
    C[c] = 14;
    C[d] = -11;
    C[e] = 4;
    C[f] = 5.58;
    C[g] = -1;

    reset();
  }

  void inc()
  {
    dx = C[a]*y + C[b]*x + C[c]*y*z;
    dy = C[d]*y - z + C[e]*x*z;
    dz = C[f]*z + C[g]*x*y;

    x += h * dx;
    y += h * dy;
    z += h * dz;
  }

  void reset()
  {
    x = 1;
    y = 3;
    z = 5;
  }
};

class MultiSprottC : public Attractor // http://www.3d-meier.de/tut19/Seite192.html
{
  uint a = 0;
  uint b = 1;

  MultiSprottC()
  {
    C[a] = 2.5;
    C[b] = 1.5;

    reset();
  }

  void inc()
  {
    dx = C[a] * (y - x);
    dy = x * z;
    dz = C[b] - y*y;

    x += h * dx;
    y += h * dy;
    z += h * dz;
  }

  void reset()
  {
    x = .1;
    y = .1;
    z = .1;
  }
};

class SprottLinz : public Attractor // http://www.3d-meier.de/tut19/Seite20.html
{
  SprottLinz()
  {
    reset();
  }

  void inc()
  {
    dx = y;
    dy = -x + y*z;
    dz = 1 - y*y;

    x += h * dx;
    y += h * dy;
    z += h * dz;
  }

  void reset()
  {
    x = .1;
    y = .1;
    z = .1;
  }
};

class Thomas : public Attractor // http://www.3d-meier.de/tut19/Seite20.html
{
  uint b = 0;

  Thomas()
  {
    C[b] = .19;

    reset();
  }

  void inc()
  {
    dx = -C[b]*x + sin(y);
    dy = -C[b]*y + sin(z);
    dz = -C[b]*z + sin(x);

    x += h * dx;
    y += h * dy;
    z += h * dz;
  }

  void reset()
  {
    x = .1;
    y = 0;
    z = 0;
  }
};

class LorenzMod1 : public  Attractor // http://www.3d-meier.de/tut19/Seite79.html
{
  uint a = 0;
  uint b = 1;
  uint c = 2;
  uint d = 3;

  LorenzMod1()
  {
    C[a] = 0.1;
    C[b] = 4;
    C[c] = 14;
    C[d] = 0.08;

    reset();
  }

  void inc()
  {
    dx = -C[a]*x + y*y - z*z + C[a]*C[c];
    dy = x * (y - C[b]*z) + C[d];
    dz = z + x * (C[b]*y + z);

    x += h * dx;
    y += h * dy;
    z += h * dz;
  }

  void reset()
  {
    x = .1;
    y = .1;
    z = .1;
  }
};

class LorXYZ15 : public Attractor // https://arxiv.org/ftp/arxiv/papers/1409/1409.7842.pdf
{
  uint a = 0;
  uint b = 1;
  uint c = 2;

  LorXYZ15()
  {
    C[a] = 2;
    C[b] = 0.82942;
    C[c] = 7.178;

    reset();
  }

  void inc()
  {
    double sinY = sin(y);

    dx = z + sin(pow(y, C[a])) * (z + sinY);
    dy = z + sin(x*y) * (sinY - C[b]);
    dz = x * (sinY - C[c]);

    x += h * dx;
    y += h * dy;
    z += h * dz;
  }

  void reset()
  {
    x = .1;
    y = .1;
    z = .1;
  }
};

class WangSun : public Attractor // http://www.3d-meier.de/tut19/Seite99.html
{
  uint a = 0;
  uint b = 1;
  uint c = 2;
  uint d = 3;
  uint e = 4;
  uint f = 5;

  WangSun()
  {
    C[a] = .2;
    C[b] = -.01;
    C[c] = 1;
    C[d] = -0.4;
    C[e] = -1;
    C[f] = -1;

    reset();
  }

  void inc()
  {
    dx = x*C[a] + C[c]*y*z;
    dy = C[b]*x + C[d]*y - x*z;
    dz = C[e]*z + C[f]*x*y;

    x += h * dx;
    y += h * dy;
    z += h * dz;
  }

  void reset()
  {
    x = .3;
    y = .1;
    z = 1;
  }
};

class HindenmarshRose : public Attractor // https://en.wikipedia.org/wiki/Hindmarsh%E2%80%93Rose_model
{
  uint a = 0;
  uint b = 1;
  uint c = 2;
  uint d = 3;
  uint r = 4;
  uint xR = 5;
  uint s = 6;
  uint I = 7;

  HindenmarshRose()
  {
    C[a] = 1;
    C[b] = 3;
    C[c] = 1;
    C[d] = 5;
    C[r] = .005;
    C[s] = 4.0;
    C[xR] = -1.6;
    C[I] = 3.25;

    reset();
  }

  void inc()
  {
    dx = y + (C[b] * x * x) - (C[a] * x * x * x) - z + C[I];
    dy = C[c] - (5 * x * x) - y;
    dz = C[r] * (C[s] * (x - C[xR]) - z);

    x += h * dx;
    y += h * dy;
    z += h * dz;
  }

  void reset()
  {
    x = -1.6; // membrane potential
    y = 4.0;
    z = 2.75;
  }
};


//-------------------------------------------------------------------------------------------------
// Here are my own ideas:

class Rotor : public Attractor
{

public:

  Rotor()
  {
    setAngularVelocities(2.0, 3.0, 5.0);
    reset();
  }

  void setAngularVelocities(double wx, double wy, double wz)
  {
    C[0] = wx;
    C[1] = wy;
    C[2] = wz;
  }

  void inc()
  {
    // Assign dx,dy,dz with the contributions from the rotation:
    calcRotationTerms();

    // Check that v = (dx,dy,dz) is orthogonal to r = (x,y,z):
    double c = x*dx + y*dy + z*dz;
    // This should be (close to) zero. If it turns out that this doesn't work well due to numerical
    // errors, one trick could be to orthogonalize it. I think, this can be done by computing the 
    // projection of v onto r and subtract that from v - or something. Like in Gram-Schmidt 
    // orthogonalization - but just for one vector.


    updatePosition();


    // As a cheap trick, we renormalize r to ensure that we really did a pure rotation:
    renormalize();
  }

  void reset()
  {
    x = 1.0;
    y = 0.0;
    z = 0.0;
  }


protected:


  void updatePosition()
  {
    x += h * dx;
    y += h * dy;
    z += h * dz;
  }

  void calcRotationTerms()
  {
    double wx = C[0];
    double wy = C[1];
    double wz = C[2];
    dx = wy*z - wz*y;
    dy = wz*x - wx*z;
    dz = wx*y - wy*x;
  }

  void renormalize()
  {
    double s = 1 / sqrt(x*x + y*y + z*z);
    //double s = 2.0 / (1 + x*x + y*y + z*z);  // Test

    // Test - normalize only when the vector is too big:
    //s = RAPT::rsMin(s, 1.0);
    //if(s > 1) s = 1;

    x *= s;
    y *= s;
    z *= s;
  }

};



class ChaoticRotor : public Rotor
{

  // Hmm - so far, I've not yet found any parameter setup that leads to chaotic behavior. It always
  // seems to be either quasi-periodic or run into a fixed point.

public:


  void inc()
  {
    calcRotationTerms();

    // Chaos parameters:
    double a =  -14.0;
    double b =   -5.0;
    double c = -200.0;
    double d =  +20.0;
    // Interesting settings:
    // (-12.5,0,0,0),(-14,2,-100,0),(-14,2,-200,20)


    //double e =   10.0;
    //dx /= (1 + e*x*x*y*z);
    //dy /= (1 + e*y*y*x*z);
    //dz /= (1 + e*z*z*x*y);


    double f = 0.0;


    // Apply nonlinear terms:
    dx += a*y*z + b*x*x + c*x*y*z + d*x*x*x + f;
    dy += a*x*z + b*y*y + c*x*y*z + d*y*y*y + f;
    dz += a*x*y + b*z*z + c*x*y*z + d*z*z*z + f;
    // Maybe the terms are too much symmetrically placed for chaos to ensue? Deleting the + c... 
    // stuff in dz += doesn't seem to help.

    updatePosition();
    renormalize();

    // Without renormalization, it explodes. Maybe try saturation instead of renormalization. Or 
    // renomralize conditionally - like only the the point goes out of the sphere
  }




};



/*

this video has a few more at 1:57

https://www.youtube.com/watch?v=wZvFKcQ_3Rc&list=PLw5b-jhi9cCD_YQgijiHnodoD0G91lu4c&index=6

*/
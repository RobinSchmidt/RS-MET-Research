
//=================================================================================================

/** A data structure for optimizing the computation of partial derivatives on irregular meshes. 
The mesh is given as a graph and the data stored in this graph is used to derive coefficients with
which the partial derivatives u_x, u_y at some vertex i can be computed as weighted sum of the 
vertex value itself and the values at its neighbors. These weights are stored internally in the
rsStencilMesh2D object when you call the member function computeCoeffsFromMesh. This can be done 
once and for all for a given mesh and after that, subsequent gradient computations can be computed 
by the gradient method which is (supposedly) cheaper than calling the corresponding function
rsNumericDifferentiator<T>::gradient2D(mesh, ...). The result should be the same up to roundoff 
errors. */

template<class T>
class rsStencilMesh2D
{

public:

  /** Computes the coefficients (or weights) for forming the weighted sum of the vertex value and
  its neighbors for the estimate of the partial derivatives. */
  void computeCoeffsFromMesh(const rsGraph<rsVector2D<T>, T>& mesh);

  /** Computes estimates of the partial derivatives u_x, u_y given function values u. */
  void gradient(const T* u, T* u_x, T* u_y);

  /** Given the function values u, this computes the partial derivatives u_x and u_y at the 
  grid-point with index i. Note that u should point to the start of the u-array whereas u_x and u_y
  should point to the i-th elements of the respective arrays. This makes sense because... */
  void gradient(const T* u, int i, T* u_x, T* u_y);

  // todo: hessian, laplacian, etc.


  int getNumNodes() const { return numNodes; }
  int getNumEdges() const { return numEdges; }


  int getNumNeighbors(int i) const { return numNeighbors[i]; }
  T   getSelfWeightX( int i) const { return selfWeightsX[i]; }
  T   getSelfWeightY( int i) const { return selfWeightsY[i]; }

  int getNeighborIndex(  int i, int k) const { return neighborIndices[ starts[i] + k]; }
  T   getNeighborWeightX(int i, int k) const { return neighborWeightsX[starts[i] + k]; }
  T   getNeighborWeightY(int i, int k) const { return neighborWeightsY[starts[i] + k]; }


protected:


  // internal:


  // These arrays are all numNodes long:
  int numNodes = 0;
  std::vector<int> numNeighbors; // entry is the number of neighbors of node i
  std::vector<T>   selfWeightsX, selfWeightsY; //
  std::vector<int> starts;  
  // entries are start-indices of the sections with values belonging to node i 
  // in the neighborIndices, etc arrays - maybe rename to offsets

  // These arrays are all numEdges long:
  int numEdges = 0;              // sum of entries of numNeighbors array
  std::vector<int> neighborIndices;
  std::vector<T>   neighborWeightsX, neighborWeightsY;
};

template<class T>
void rsStencilMesh2D<T>::computeCoeffsFromMesh(const rsGraph<rsVector2D<T>, T>& mesh)
{
  using Vec2 = rsVector2D<T>;

  // Allocate memory and set offsets:
  numNodes = mesh.getNumVertices();
  numNeighbors.resize(numNodes);
  selfWeightsX.resize(numNodes);
  selfWeightsY.resize(numNodes);
  starts.resize(numNodes);

  for(int i = 0; i < numNodes; i++)
    numNeighbors[i] = mesh.getNumEdges(i);
  starts[0] = 0;
  for(int i = 1; i < numNodes; i++)
    starts[i] = starts[i-1] + numNeighbors[i-1];
  numEdges = rsSum(numNeighbors);

  neighborIndices.resize(numEdges);
  neighborWeightsX.resize(numEdges);
  neighborWeightsY.resize(numEdges);
 
  // Compute the coefficients:
  for(int i = 0; i < numNodes; i++)
  {
    const Vec2& vi = mesh.getVertexData(i);

    // Compute least squares matrix A and its inverse M:
    T a11 = T(0), a12 = T(0), a22 = T(0);
    T sx = T(0), sy = T(0);
    for(int k = 0; k < numNeighbors[i]; k++)
    {
      int j = mesh.getEdgeTarget(i, k);         // index of current neighbor of vi
      const Vec2& vj = mesh.getVertexData(j);   // current neighbor of vi
      Vec2 dv = vj - vi;                        // difference vector
      T    w  = mesh.getEdgeData(i, k);         // edge weight
      a11 += w * dv.x * dv.x;
      a12 += w * dv.x * dv.y;
      a22 += w * dv.y * dv.y;
      sx  += w * dv.x;
      sy  += w * dv.y;
    }
    rsMatrix2x2 A(a11, a12, a12, a22);
    rsMatrix2x2 M = A.getInverse();
    T m11 = M.a, m12 = M.b, m22 = M.d;
    // try to optimize: get rid of using rsMatrix2x2


    selfWeightsX[i] = -(m11 * sx + m12 * sy);
    selfWeightsY[i] = -(m12 * sx + m22 * sy);   // verify!
    for(int k = 0; k < numNeighbors[i]; k++)
    {
      int j = mesh.getEdgeTarget(i, k);         // index of current neighbor of vi
      const Vec2& vj = mesh.getVertexData(j);   // current neighbor of vi
      Vec2 dv = vj - vi;                        // difference vector
      T    w  = mesh.getEdgeData(i, k);         // edge weight
      neighborWeightsX[starts[i] + k] = w * (m11 * dv.x + m12 * dv.y);
      neighborWeightsY[starts[i] + k] = w * (m12 * dv.x + m22 * dv.y);  // verify!
      neighborIndices [starts[i] + k] = j;
    }
    // maybe the sum of the neighbor-weights plus the self-weight should sum up to zero? ...they
    // seem to indeed do...so the self-weight should always be minus the sum of the neighbor 
    // weights - maybe it's numerically more precise to use that sum, such that the cancellation
    // is perfect?
  }
}

template<class T>
void rsStencilMesh2D<T>::gradient(const T* u, int i, T* u_x, T* u_y)
{
  *u_x = getSelfWeightX(i) * u[i];
  *u_y = getSelfWeightY(i) * u[i];
  for(int k = 0; k < getNumNeighbors(i); k++) {
    int j = getNeighborIndex(i, k);
    *u_x += getNeighborWeightX(i, k) * u[j];
    *u_y += getNeighborWeightY(i, k) * u[j]; }
}

template<class T>
void rsStencilMesh2D<T>::gradient(const T* u, T* u_x, T* u_y)
{
  for(int i = 0; i < numNodes; i++)
    gradient(u, i, &u_x[i], &u_y[i]);
}

//=================================================================================================

template<class T> 
class rsParticleSystem2D
{

public:

  // Lifetime:
  rsParticleSystem2D(int numParticles);

  // Setup:
  void setNumParticles(int newNumber);
  void setMasses(std::vector<T>& newMasses);
  void setPositions(std::vector<rsVector2D<T>>& newPositions);

  // Processing:
  void computeForcesNaive(std::vector<rsVector2D<T>>& forces); // for referencce in unit tests, inefficient
  void computeForcesFast(std::vector<rsVector2D<T>>& forces); 
  // Maybe the forces vector doesn't need to be stored as member. it can be passed in as
  // parameter. That makes it also convenient to test the force computation

protected:

  std::vector<rsVector2D<T>> positions, velocities;
  std::vector<T> masses;

};

template<class T> 
rsParticleSystem2D<T>::rsParticleSystem2D(int numParticles)
{
  setNumParticles(numParticles);
}

template<class T> 
void rsParticleSystem2D<T>::setNumParticles(int newNumber)
{
  positions.resize(newNumber);
  velocities.resize(newNumber);
  //forces.resize(newNumber);
  masses.resize(newNumber);
}

template<class T> 
void rsParticleSystem2D<T>::setMasses(std::vector<T>& newMasses)
{
  rsAssert(newMasses.size() == masses.size());
  rsCopy(newMasses, masses);
}

template<class T> 
void rsParticleSystem2D<T>::setPositions(std::vector<rsVector2D<T>>& newPositions)
{
  rsAssert(newPositions.size() == positions.size());
  rsCopy(newPositions, positions);
}


template<class T> 
void rsParticleSystem2D<T>::computeForcesNaive(std::vector<rsVector2D<T>>& forces)
{
  int N = positions.size(); 
  rsAssert((int)velocities.size() == N);
  rsAssert((int)forces.size() == N);

  for(int i = 0; i < N; i++)
  {
    forces[i] = 0;
    rsVector2D<T> p = positions[i];

    // Loop over all other particles and sum up their contributions to the total force that acts
    // on the current particle i:
    for(int j = 0; j < N; j++)
    {
      if(j != i)
      {
        rsVector2D<T> q = positions[j] - p; // delta-vector

        // ToDo: factor out, use modified gravity laws:
        T d = rsNorm(q);
        forces[i] += masses[j] * masses[i] * q / (d*d*d);
        // we use d^3 and not d^2 because q in the numerator also still contains a factor of d, 
        // i.e. q is not normalized to unit length

        int dummy = 0;

      }
    }
  }
}
// maybe it should take the positions as parameter, too?

template<class T> 
void rsParticleSystem2D<T>::computeForcesFast(std::vector<rsVector2D<T>>& forces)
{
  int N = positions.size(); 
  rsAssert((int)velocities.size() == N);
  rsAssert((int)forces.size() == N);

  T M = 0;                     // total mass of all particles (maybe precompute - it's a constant!)
  rsVector2D<T> S(0, 0);       // sum of all position vectors weighted by the mass of the particle
  for(int i = 0; i < N; i++)
  {
    M += masses[i];
    S += masses[i] * positions[i];
  }
  // By the way: the center of mass of all particles would now be given by S/M. But we don't need 
  // that in further computations. We will only need the weighted sum itself.

  for(int i = 0; i < N; i++)
  {
    T M_i = M - masses[i];
    // sum of all masses except the i-th

    rsVector2D<T> S_i = S - masses[i] * positions[i];  
    // Weighted sum of all positions expcept the i-th

    rsVector2D<T> C_i = S_i / M_i;
    // Center of mass of all particles except the i-th

    rsVector2D<T> Q = C_i - positions[i];
    // Difference vector between current particle i and the center of mass of all other particles

    T D = rsNorm(Q);
    // Distance from i-th particle to center of mass of all others

    forces[i] = M_i * masses[i] * Q / (D*D*D);
    // Force excerted on i-th particle by all the others
  }


  // Idea:
  // -We first compute the center of mass of all particles. This is an O(N) operation.
  // -For each particle, we subtract out its contribution to the center of mass which gives the
  //  center of all other masses. We treat this center of all other masses as a single replacement
  //  mass that acts as a stand-in for all others. I'm not sure, if that is supposed to work. Maybe
  //  only for very specific force laws like Hooke's but not for the (nonlinear) gravitational law?
  //  But maybe the law can be taken into account when we compute S?
}

//=================================================================================================

/** Implements a digital waveguide based on a so called "bidirectional" delay line which is 
basically a particular configuration of two delay lines with mutual cross feedback. A digital 
waveguide can be seen as an efficient method to implement a numerical solver for the 1D wave 
equation. This is a partial differential equation (PDE) which describes the motion of a string 
(like a guitar or violin string) or the pressure waves in a column of air in a long cylindrical 
bore (like a flute or organ pipe). Conical pipes can also be modeled with a waveguide. In our
member function names and in the documentation, we may occasionally nevertheless refer to the 
mental image of a string, even though the waveguide could model some other kind of physical 
system. The string is taken to be the prototypical example of a 1D system in which waves 
propagate.

The class is supposed to be used from some sort of driver code which can interact with the 
waveguide through an API that lets the driver inject inputs and/or pick up outputs at arbitrary 
positions along the string. The driver code may also trigger scattering (i.e. partial reflection 
and transmission) at arbitrary positions along the string. More complex drivers may even 
orchestrate a whole network of several interconnected waveguides and/or interactions of waveguides
with other algorithms such as nonlinear models of string or bore excitation (such as bowing, reeds,
lips, etc.). This class is supposed to be the basic building block of waveguide based physical 
models of musical instruments but it does not work like a normal DSP block (such as oscillators, 
filters, etc.) in its own right. It has an API that is a bit different from those kinds of DSP 
units. To see how a very simple driver could look like, look at the subclass rsWaveGuideFilter 
which gives you the normal API that you expect from a DSP building block for signal filtering
(i.e. getSample(), etc.) and uses a waveguide internally to do its work. ...TBC...
 
References:

  PASP: Physical Audio Signal Processing (Julius O. Smith), https://ccrma.stanford.edu/~jos/pasp

*/

template<class TSig, class TPar>
class rsWaveGuide
{

public:


  //-----------------------------------------------------------------------------------------------
  // \name Lifetime

  /** Default constructor. Initially allocates an unrealistically small amount delay line 
  memory. ...TBC...  */
  rsWaveGuide()
  {
    setMaxStringLength(M);
  }
  // ToDo: Maybe do not yet allocate here at all. Rationale: The client code will likely want to
  // allocate more memory anyway so it would just be a wasted allocation. And in the great scheme
  // of things, we may create a lot of waveguide objects in instrument simulations, so it may 
  // actually matter a bit in terms of initialization costs.

  ~rsWaveGuide()
  {
    int dummy = 0;
  }
  // The explicitly defined destructor exists only to set a breakpoint here for debugging so we can
  // track when a waveguide object gets destroyed. That's also why we have the dummy instruction 
  // here. In Visual Studio, it often doesn't work right when trying to set breakpoints at closing 
  // braces. Sometimes it works fine but sometimes it doesn't - I don't know why.

  //-----------------------------------------------------------------------------------------------
  // \name Setup

  /** Sets the maximum length of the string. In a realtime context, this should be called on setup
  time (not during realtime operation) because it may re-allocate memory for the delay lines. */
  void setMaxStringLength(int newMaxLength) 
  { 
    delay1.setMaxDelayInSamples(newMaxLength); 
    delay2.setMaxDelayInSamples(newMaxLength);
    M = rsMin(M, newMaxLength);
    updateDelaySettings();
  }
  // Maybe call it just setMaxLength. It does not necessarily represent a string although that is 
  // the most intuitive way of visualizing it. Or maybe setMaxNumSpatialSamples

  /** Sets the length of the virtual string in terms of spatial samples. If you set it to some 
  number M, the output signal of the waveguide will produce a signal with a period of 2*M. */
  void setStringLength(int newLength)   
  { 
    M = newLength; 
    updateDelaySettings();  
  }
  // Rename to setLength or setNumSpatialSamples
  
  /** Sets up the state of the waveguide by distributing the given shape appropriately into the
  delay lines for the right- and left-traveling wave. This can be used to set an initial condition 
  for displacement (or velocity) to emulate pluck (or struck) strings. Whether the given newState 
  represents displacement or velocity or something else is up to the interpretation of the caller.
  If you use the waveguide to represent displacement waves, then the given newState would represent
  the initialshape of the string (before it is released) and we would emulate a plucked string. If 
  the waveguide represents velocity waves, then the given newState would represent the initial 
  velocity distribution of the string at the moment of striking it. */
  void setState(const TSig* newState, int stateSize);
  // ToDo: 
  // -Write a similar function but instead of directly writing the newState into the delay 
  //  lines, add it to what's already there. That could be used to (crudely) emulating to strike
  //  the string while it already is in motion.
  // -Rename to something like setDisplacementState. Have a similar function for setVeloityState
  //  and maybe also functions that set initial displacement and velocity at the same time. I 
  //  think, the translation from traveling waves is: y[m] = yR[m] + yL[m], 
  //  v[m] = yR[m] - yL[m] or v[m] = yL[m] - yR[m]. We need to invert these relations to go from
  //  displacement and velocity (y,v) to right and left traveling (displacement) waves yR,yL. But:
  //  what if we do not want to represent displacement waves but rather velocity or force waves?
  // -Maybe have also functions that directly init the traveling waves components, i.e. without 
  //  conversion from physical variables.



  //-----------------------------------------------------------------------------------------------
  // \name Inquiry

  //const RAPT::rsDelay<TSig>& getDelayLine1() const { return delay1; }

  //const RAPT::rsDelay<TSig>& getDelayLine2() const { return delay2; }
  // Maybe try to avoid exposing these internals to the outside. Client code should not directly 
  // access these delay lines! I think, we currently need this access just for plotting purposes
  // for R&D anyway. Maybe that can be solved in a better way. Maybe we can declare the plotting
  // functions as friend. A bit ugly but still better than exposing the internals to everyone. Or
  // maybe the plotting functions could be based on calling (to be written) functions like 
  // extractOutputAt() or extractForwardOutputAt(int m), extractBackwardOutputAt(int m) or
  // extractOutputsAt(int m, TSig* yR, TSig* yR)

  /** Returns the number M of spatial samples along this waveguide (i.e. the "string"). This equals
  the amount of delay that is used in both of the delay lines (each one uses M samples of delay). 
  When the waveguide is terminated at both ends such that traveling waves get reflected there, then 
  the temporal periodicity of recirculating standing waves, will be given by 2*M samples. This 
  temporal periodicity is, of course, independent from the location m where the signal is picked 
  up. When, in addition to reflections at the ends, we also introduce scattering (i.e. partial 
  reflection and transmission) somewhere along the string, the temporal patterns in the output 
  signal become more complicated... ...TBC... */
  int getLength() const { return M; }
  // Maybe rename to getNumSpatialSamples(). But that's rather long. Well - maybe not. getLength()
  // seems actually fine.

  bool isValidIndex(int m) const { return (m >= 0 && m <= M); }

  bool isStableScatterCoeff(TPar k) const { return rsAbs(k) <= TPar(1); }


  //-----------------------------------------------------------------------------------------------
  // \name Processing

  /** Resets the waveguide to its initial state. Clears the content of the delay lines. */
  void reset() { delay1.reset(); delay2.reset(); }

  /** Injects an input into the waveguide at the given location m. This writes the given input 
  sample value */
  inline void injectInputAt(TSig input, int m);
  // Maybe swap the parameters. Maybe m should come first for consistency with the other member
  // functions. But check also how other classes from RAPT handle it - especially rsDelay. 
  // Replicate the conventions used there. Hmm - OK - there, we pass the value first (in
  // writeInputAt(), addToInputAt(), etc.), so maybe
  // we should adapt the other functions like setTravelingWavesAt. Or maybe chaneg the convention
  // in rsDelay. But that requires a lot of care because it's already used all over the place.
  // Check also, if we have other classes where we can write values into specific locations and
  // see how we do it there - for example in rsMatrix, rsArrayTools, etc. Establish a consistent
  // convention - either index-first or value-first - for such getters and setters throughout the 
  // library. I tend to thing that index-first is nicer liguistically because the "...At" in the 
  // function names already suggests that the thing that comes next is a description of a location.
  // OK: rsMatrix::setRow, rsPolynomial::setCoeff also use the index-first convention. I think, we
  // should really change it in rsDelay. Maybe try to do this with assistence from GitHub Copilot
  // at some point. Seems to be a perfect task to do with AI. Then we could also check, if there
  // are any other classes in the library that need similar adjustments.

  /** Extracts an output sample of the physical wave variable (such as displacement, pressure, 
  etc.) from the waveguide at the given location m. This just pulls out the rightward and leftward
  traveling wave components from the delay lines and adds them */
  inline TSig extractOutputAt(int m) const;

  /** Writes a pair of the non-physical traveling wave variables into the waveguide at the spatial
  sample position m where yR represets the rightward going traveling wave and yL the leftward going
  traveling wave. */
  inline void setTravelingWavesAt(int m, TSig yR, TSig yL);

  /** Extracts a pair of the non-physical traveling wave variables from the waveguide at the 
  spatial sample position m and stores them in yR and yL where yR represets the rightward going 
  traveling wave and yL the leftward going traveling wave. */
  inline void getTravelingWavesAt(int m, TSig* yR, TSig* yL) const;

  /** Implements reflections at the left and right end of the waveguide using the given 
  reflection coefficients. */
  inline void reflectAtEnds(TPar reflectLeft, TPar reflectRight);
  // ToDo: Document the physical interpretation of these coeffs in terms of boundary conditions,
  // stability, etc.


  /** Implements a Kelly-Lochbaum scattering junction at the spatial location m with the reflection
  coefficient k. For an impedance change from R1 to R2 when going from left to right, the coeff
  can be computed as k = (R2-R1)/(R2+R1). See PASP, page 562 or here:
    https://ccrma.stanford.edu/~jos/pasp/Kelly_Lochbaum_Scattering_Junctions.html
    https://ccrma.stanford.edu/~jos/pasp/Reflection_Coefficient.html
  The function can be called inside the per-sample computations for example immediately before
  calling reflectAtEnds() which actually performs a quite similar computation just without the
  transmission part. */
  inline void scatterAt_KL(int m, TPar k);
  // ToDo: 
  // -Document in which way this form scattering can be  seen as "lossless". See PASP, pg 561.
  //  There, it says that "signal power is conserved at the junction".


  /** Implements scattering junction at the spatial location m with the reflection coefficient k 
  that is suitable when the delay lines contain root-power waves as described in PASP, pg 559, 
  Eq C.53 and here:
    https://ccrma.stanford.edu/~jos/pasp/Root_Power_Waves.html
    https://ccrma.stanford.edu/~jos/pasp/Normalized_Scattering_Junctions.html
  On page 572, the book says: "a more precise term would be normalized wave scattering junction"
  so that's why we use the suffix NW here for "normalized wave". The coefficient k in this context
  can be interpreted as the sine of an angle in a 2D rotation. That is: Let k = sin(w) such that 
  w = asin(k). Then the wave gets reflected with a factor of sin(w) and transmitted with a factor 
  of cos(w). We also have that cos(w) = sqrt(1 - k^2) because sin^2(w) + cos^2(w) = 1 for any w. */
  inline void scatterAt_NW(int m, TPar k) { scatterAt_NW(m, k, rsSqrt(TPar(1) - k*k)); }

  /** A two parameter version of scatterAt_NW(int m, TPar k) that takes the sine s and cosine c of 
  an angle as parameters rather than the reflection coeff k. It will be more efficient to 
  precompute these s,c values once and then use this function to do the actual (per sample) 
  scattering because the one parameter version involves computing a square root. ...TBC... */
  inline void scatterAt_NW(int m, TPar s, TPar c);

  /** Steps the time forward by one sample instant. This basically moves/advances the pointers in 
  the delay lines. */
  inline void stepTime();


  //-----------------------------------------------------------------------------------------------
  // \name Static Member Functions. Here, we implement some general formulas that are relevant in
  // the context of digital waveguide synthesis

  /** Computes the reflection coefficient for an impedance step from a wave impedance of R1 in the
  left section of a string to a wave impedance of R2 in the right section of the string. See:
    https://ccrma.stanford.edu/~jos/pasp/Reflection_Coefficient.html
  This coefficient can be used with the function scatterAt_KL() to implement the scattering that 
  occurs at impedance steps along the string.  */
  static TPar reflectionCoeff(TPar R1, TPar R2) { return (R2 - R1) / (R2 + R1); }
  // ToDo: 
  // -Maybe we could also write a function that directly computes the s,c coeffs for a 
  //  normalized wave scattering junction (see below). Maybe there are some optimization 
  //  opportunities in these calculations (not sure, though).

  // ToDo: implement C.69,C.127 (transformerCoeff or transformerTurnsRatio). There's also some
  // talk about gyrators and dualizers



  //-----------------------------------------------------------------------------------------------

protected:

  /** Updates the settings of our two delay lines according to the user parameter M. */
  void updateDelaySettings();
  // Rename to updateDelays or setupDelays Or maybe get rid entirely. It doesn't really do much.

  RAPT::rsDelay<TSig> delay1, delay2;  // The two delay lines that make up the waveguide
  int  M = 30;                         // Length of the delay lines. 
  //TPar R = TPar(1);                    // Wave impedance of the string (not used yet)
 
  // Notes:
  //
  // - The member variable M represents the length of the waveguide in spatial samples. It is 
  //   actually redundant because it is always equal to the delay length of the two delay lines.
  //   However, it makes sense to cache it here because it's used a lot in per sample computations
  //   and pulling it out from one of the delays every time we need it (e.g. via 
  //   delay1.getDelayInSamples()) may be more expensive because the delay lines themselves 
  //   actually do not directly store their lengths either but compute it on demand as difference 
  //   of the tapIn and tapOut pointers so such a call would always trigger a little calculation.
  //
  //
  // ToDo:
  //
  // - Maybe add a self-test function like isStateConsistent() that verifies that the two delay 
  //   lines have the same length and that this length is equal to M. This can be used for unit 
  //   tests and in assertions to catch bugs during development. Maybe we should also expect the
  //   tap-pointers of the two delay lines to be somehow in sync? If so, we should verify that in 
  //   the consistency check as well. 
  // 
  // - By the way: What happens when the delay lines have unequal lengths? Could that be an 
  //   interesting extension? It wouldn't be physically meaningful but maybe it's musically useful
  //   nonetheless? Or maybe it actually could be interpreted physically, for example in terms of 
  //   the wave velocity being direction dependent? The medium being anisotropic in some weird way?
  //   I think normal anisotropy does not distiguish between forward and backward travel, i.e. the 
  //   speed depends only on direction in the sense of an angle of a line but not on orientation in
  //   the sense of the tip of an arrow. But maybe there are media where this is different? I 
  //   think, acoustic waves in moving media move differently when going with or against the flow. 
  //   Like when there is some current (like a wind in the air), the speed of sound is higher into
  //   the direction of the wind compared to against the wind? Or maybe it was the other way 
  //   around?
  //
  // - Maybe in addition to extractOutputAt which computes the sum of the traveling wave variables,
  //   provide also a function to extract the difference. Maybe rename extractOutputAt to
  //   extractOutputSumAt. 
};

template<class TSig, class TPar>
void rsWaveGuide<TSig, TPar>::setState(const TSig* newState, int stateSize)
{
  rsAssert(stateSize == M); // Is this correct or should it be M+1?
  // ToDo: Maybe be a bit more liberal here. If the passed state buffer has a too small size, just
  // fill the rest with zeros and if it has a too larger size, ignore the final portion of it

  for(int m = 0; m < M; m++)
  {
    TSig x = TSig(0.5) * newState[m];
    delay1.writeInputAt(x,   m);
    delay2.writeInputAt(x, M-m);
  }
}

template<class TSig, class TPar>
inline void rsWaveGuide<TSig, TPar>::injectInputAt(TSig in, int m)
{
  TSig x = TSig(0.5) * in;               // Signal goes into both delay lines with weight 0.5
  setTravelingWavesAt(m, x, x);
}

template<class TSig, class TPar>
inline TSig rsWaveGuide<TSig, TPar>::extractOutputAt(int m) const
{
  TSig yR, yL;                           // Non-physical right- and left going traveling waves
  getTravelingWavesAt(m, &yR, &yL);      // Extract traveling waves variables at spatial sample m
  return yR + yL;                        // The physical wave variable is their sum

  // ToDo:
  //
  // - Figure out if their difference yR - yL (or yL - yR) also have physical interpretations, 
  //   perhaps as "dual" wave variables? Like, for example, when the primal wave variable as 
  //   computed here via the sum is displacement, the dual could be velocity? If that is the case,
  //   then maybe we should have a member function extractDualOutputAt(int m) as well. If it is not
  //   the case, then document that too such that we will not have to wonder about it again later.
}

template<class TSig, class TPar>
inline void rsWaveGuide<TSig, TPar>::setTravelingWavesAt(int m, TSig yR, TSig yL)
{
  rsAssert(isValidIndex(m), "Index out of range in rsWaveGuide::setTravelingWavesAt");

  delay1.addToInputAt(yR,   m);          // Index used as is for right going wave
  delay2.addToInputAt(yL, M-m);          // Index must be reflected for left going wave
}

template<class TSig, class TPar>
inline void rsWaveGuide<TSig, TPar>::getTravelingWavesAt(int m, TSig* yR, TSig* yL) const
{
  rsAssert(isValidIndex(m), "Index out of range in rsWaveGuide::getTravelingWavesAt");

  *yR = delay1.readOutputAt(  m);        // Index used as is for right going wave
  *yL = delay2.readOutputAt(M-m);        // Index must be reflected for left going wave
}

template<class TSig, class TPar>
inline void rsWaveGuide<TSig, TPar>::reflectAtEnds(TPar kL, TPar kR)
{
  rsAssert(isStableScatterCoeff(kL) && isStableScatterCoeff(kR), 
    "Unstable reflection coeff in rsWaveGuide::reflectAtEnds");

  // Implement the mutual crossfeedback using the reflection coefficients:
  TSig ref1 = delay1.readOutput();       // Right going wave reflected at right end
  TSig ref2 = delay2.readOutput();       // Left going wave reflected at left end
  delay1.writeInput(kL * ref2);          // Reflection at left end
  delay2.writeInput(kR * ref1);          // Reflection at right end

  // Maybe rename the coeffs to rL,rR and the signals ref1/2 to yR,yL
  // Or maybe use kL,kR to emphasize the similarity to scattering (done)
}

template<class TSig, class TPar>
inline void rsWaveGuide<TSig, TPar>::scatterAt_KL(int m, TPar k)
{
  // Sanity checks:
  rsAssert(m >= 0        && m <= M,        "Scatter point out of range");
  rsAssert(k >= TPar(-1) && k <= TPar(+1), "Scattering coeff out of stable range");

  // Read delay line contents from top and bottom rail:
  TSig uTL = delay1.readOutputAt(  m);   // TL: top-left,      f^+_{i-1}
  TSig uBR = delay2.readOutputAt(M-m);   // BR: bottom-right,  f^-_i

  // Compute the scattered signals (see to PASP, page 564 and 570, Fig. C.17 and C.20):
  TSig uTR = (1+k)*uTL - k*uBR;          // Upper rail transmission + reflection
  TSig uBL = (1-k)*uBR + k*uTL;          // Lower rail transmission + reflection

  // Write the scattered signals back into the delay lines at the appropriate places:
  delay1.writeInputAt(uTR,   m);         // TR: top-right,     f^+_i
  delay2.writeInputAt(uBL, M-m);         // BL: bottom-left,   f^-_{i-1}

  // Verify all of this! Implement also the one-multiply form from page 571 using the alpha 
  // parameter. This one can be generalized to junctions of more than 2 waveguides.

  // See: https://ccrma.stanford.edu/~jos/pasp/Kelly_Lochbaum_Scattering_Junctions.html

  // ToDo: https://ccrma.stanford.edu/~jos/pasp/One_Multiply_Scattering_Junctions.html

  // Maybe factor out functions isValidIndex(m), isStableScatterCoeff(k) for the assertions and
  // use similar assertions whereever it makes sense.

  // Use y instead of u. Make notation consistent with reflectAtEnds, extractOutputsAt, etc.
}

template<class TSig, class TPar>
inline void rsWaveGuide<TSig, TPar>::scatterAt_NW(int m, TPar s, TPar c)
{
  // Sanity checks:
  rsAssert(isValidIndex(m),         "Scatter point out of range");
  rsAssert(isStableScatterCoeff(s), "Scattering coeff out of stable range");
  // Maybe check also that s^2 + c^2 = 1 up to some tolerance. Maybe have a function like
  // rsIsSinCosPair(s, c, tol) for such a check somewhere in the library. It could also check
  // that -1 <= s,c <= +1 in addition to s^2 + c^2 = 1. Then we could get rid of the
  // call to isStableScatterCoeff(s) above because that check would be included.

  // Read delay line contents from top and bottom rail:
  TSig uTL = delay1.readOutputAt(  m);   // TL: top-left,      f^+_{i-1}
  TSig uBR = delay2.readOutputAt(M-m);   // BR: bottom-right,  f^-_i

  // Compute the scattered signals (see to PASP, page 572, Fig. C.22 and Eq. C.66):
  TSig uTR = c*uTL - s*uBR;              // Upper rail transmission + reflection
  TSig uBL = c*uBR + s*uTL;              // Lower rail transmission + reflection

  // Write the scattered signals back into the delay lines at the appropriate places:
  delay1.writeInputAt(uTR,   m);         // TR: top-right,     f^+_i
  delay2.writeInputAt(uBL, M-m);         // BL: bottom-left,   f^-_{i-1}

  // See: https://ccrma.stanford.edu/~jos/pasp/Normalized_Scattering_Junctions.html
}

template<class TSig, class TPar>
inline void rsWaveGuide<TSig, TPar>::stepTime()
{
  delay1.incrementTapPointers();
  delay2.incrementTapPointers();
}

template<class TSig, class TPar>
void rsWaveGuide<TSig, TPar>::updateDelaySettings()
{
  delay1.setDelayInSamples(M);
  delay2.setDelayInSamples(M);
}

//-------------------------------------------------------------------------------------------------
// Free (i.e. non-member) functions for helping with research, development and debugging using 
// class rsWaveGuide:

template<class TSig, class TPar>
void rsPlotWaveGuideContent(const rsWaveGuide<TSig, TPar>& wg)
{
  int M = wg.getLength();
  std::vector<TSig> yR(M+1), yL(M+1);
  for(int m = 0; m <= M; m++)
    wg.getTravelingWavesAt(m, &yR[m], &yL[m]);
  rsPlotVectors(yR, yL);

  // ToDo:
  // 
  // - Document why the vector length of yR, yL is M+1 rather than M. I think, it is because
  //   M is the maximum amount of delay that is used and a delay line with a max delay of M needs
  //   to store M+1 signal values - at least when we use it like here (where at time n, we may want
  //   to write into before we read out of it) - I think. -> Verify! But whatever the correct 
  //   explanation may be from an algorithmic point of view, it is indeed true that the delay 
  //   lines in rsWaveGuide store M+1 signal values as can be seen for example in the 
  //   implementation of rsDelay::setMaxDelayInSamples(). ...but verify that, too! See also 
  //   comments in rsPlotDelayLineContent().
  //
  // - Maybe plot also the sum of the contents of both delay lines because it is that sum that 
  //   represents our actual physical signal. However - showing only the sum may hide some 
  //   undesirable effects. For example, there may be situations in which the signals inside the
  //   waveguides grow without bound while their sum stays bounded. If the difference also has 
  //   some relevance, we may plot that, too.
}

//=================================================================================================

/** Implements a waveguide based filter with adjustable length, driving point and pickup point. 
It's implemented as a subclass of rsWaveGuide and thereby provides example code how a suitable 
driver code for the waveguide class could look like in a very simple case (perhaps the simplest 
possible). The resulting filter can be used like a regular filter, i.e. it provides the usual
getSample(TSig in) API like many other filter classes in RAPT do. Spectrally, the filter can be 
seen as a particular kind of comb filter that emulates the effects that occur when driving a string
at a particular driving point and listening to it at a particular pick-up point. I think, this will
result in two series of comb frequencies and is equivalent to two comb filters in series (Verify! 
See PASP book. Maybe it's even more complicated - the length should also play a role). ...TBC... */

template<class TSig, class TPar>
class rsWaveGuideFilter : public rsWaveGuide<TSig, TPar>
{

public:

  using Base = rsWaveGuide<TSig, TPar>;    // For convenience
  
  //-----------------------------------------------------------------------------------------------
  // \name Setup

  /** Sets the position along the string (in terms of spatial samples) at which the external 
  output signal is injected into the string, i.e. the point where the string is driven by an 
  excitation signal. */
  void setDrivingPoint(int newLoaction) {  mIn = newLoaction; }
  // ToDo: make sure that 0 <= mIn <= M

  /** Sets the position along the string (in terms of spatial samples) at which we extract the 
  output signal, i.e. the point where e pick up that string's movement. */
  void setPickUpPoint(int newLocation)  {  mOut = newLocation; }
    // ToDo: make sure that 0 <= mOut <= M

  /** Sets the reflection coefficients for the left and right boundary, i.e. the left and right 
  ends of the string. Setting both to -1 corresponds to a boundary condition where both ends are
  fixed to zero displacement. ...TBC... */
  void setReflectionCoeffs(TPar leftEnd, TPar rightEnd)
  {
    reflectLeft  = leftEnd; 
    reflectRight = rightEnd;
    // ToDo: Maybe assert that they are in the stable range (inside +-1). Maybe clip them to that
    // range. But maybe this class is too low-level to have such safeguarding here. But maybe
    // we should have assertions.
  }


  //-----------------------------------------------------------------------------------------------
  // \name Processing

  /** Produces one output sample at a time. */
  TSig getSample(TSig in) { return getSampleInExRef(in); }
  // Can be used to easily switch between the two variants of the algorithm via changing one line
  // of code. The different algorithms may show different behaviors in ertain edge cases when
  // the input tap is at one of the boundaries. In these cases, the exact order of the operations
  // in the per sample computation matters.  ...TBC... ToDo: Figure out and document if having the
  // pick up point at a boundary (while the driving point is on the interior) will also lead to
  // different behavior of the different variants. The default behavior should follow the 
  // priciple of least astonishment. Following that, we need to pick the most appropriate version 
  // of the algorith here. I think, it's InExRef but this needs to be verified.


  /** Computes an output sample using the operation order: inject -> extract -> reflect. */
  TSig getSampleInExRef(TSig in);
  // Maybe rename to getSample_IER, etc.

  /** Computes an output sample using the operation order: inject -> reflect -> extract. */
  TSig getSampleInRefEx(TSig in);

  /** Computes an output sample using the operation order: extract -> inject -> reflect. */
  TSig getSampleExInRef(TSig in);

  /** Computes an output sample using the operation order: extract -> reflect -> inject. */
  TSig getSampleExRefIn(TSig in);

  /** Computes an output sample using the operation order: reflect -> inject -> extract. */
  TSig getSampleRefInEx(TSig in);

  /** Computes an output sample using the operation order: reflect -> exctract -> inject. */
  TSig getSampleRefExIn(TSig in);

  // We have 6 versions because we have 3 operations (inject, extract, reflect) that we want to 
  // permute in all possible ways. That leads to 3! = 6 different possible algorithms which will 
  // produce the same results in the typical case where the string is being driven somewhere along
  // its length but may behave differently in edge cases where we are driving the string directly 
  // at a boundary point.


  /** Injects the given input signal into the waveguide at the driving point which can be set up 
  via setDrivingPoint(). Injection of a signal into the waveguide entails distributing it equally
  into both delay lines with weight 0.5. It's called from the various getSampleXXX() methods.  */
  inline void injectInput(TSig in) { Base::injectInputAt(in, mIn); }

  /** Extracts one physical output sample from the waveguide by reading out the delay lines that
  store the right- and left going waves and adds them up. The point along the string at which the
  signal is picked up can be set by setPickUpPoint(). It's called from the various getSampleXXX() 
  methods. */
  inline TSig extractOutput() const { return Base::extractOutputAt(mOut); }
 
  /** Performs the reflections at the left and right boundaries. This is one of the steps in the
  per sample algorithm, so it's called from the various getSampleXXX() methods. It implements the 
  mutual crossfeedback between the two delay lines using our reflection coefficients. */
  inline void reflectAtEnds() { Base::reflectAtEnds(reflectLeft, reflectRight); }


protected:

  // Reflection coefficients:
  TPar reflectLeft  = TPar(-1);        // Reflection coefficient at left boundary
  TPar reflectRight = TPar(-1);        // Reflection coefficient at right boundary
  // Using -1, i.e. inverting reflections, corresponds to a fixed end boundary condition (aka 
  // Dirichlet boundary condition.). This seems to be a reasonable default behavior.

  // Driving and pickup points:
  int mIn  =  7;                       // Driving point for input
  int mOut = 11;                       // Pick up point for output
  // ToDo: Find better default values. These were chosen out of convenience during R&D.
  // Maybe rename to drivingPoint, pickupPoint


  // ToDo: 
  // 
  // - Maybe the subclass needs to override set(Max)StringLength in order to limit mIn, mOut to the
  //   range 0..M. But this sort of polymorphism will then only work at compile time which is fine
  //   for my typical use cases but it may be dangerous when some client code assumes that this 
  //   polymorphism also works dynamically, i.e. at runtime. But making set(Max)StringLength 
  //   virtual in the baseclass just for this may be not worth it in terms of performance hit. I 
  //   mean, it's actually not such a big deal but the general philosophy of my library is to avoid
  //   virtual functions in the low level DSP and number crunching classes and rsWaveGuide 
  //   qualifies as such. Maybe we can just document that the client code should take care. Maybe 
  //   we can somehow make it somewhat safe to have mIn, mOut outside the range 0..M. By 
  //   "somewhat", I mean that in such cases, we may get weird audio output but no access 
  //   violations. It should be considered a bug in the driver code when it tries to set mIn or 
  //   mOut outside the range 0..M. It may actually already be "somewhat" safe in that manner. 
  //   Verify and document that! Or maybe we should not subclass rsWaveGuide but instead keep
  //   a member of that type. Maybe that is generally the better way to do it. It also generalizes
  //   better to more complex situations like networks of waveguides. We would then here also have 
  //   functions like set(Max)StringLength that would just delegate to calls in the embedded object
  //   and could also do the necessary additional stuff like limiting mIn, mOut.
  //
  // - Maybe find a more specific name for this class. A waveguide with multiple driving points
  //   and multiple pickup points is also a filter. Maybe rsWaveGuideFilter_In1_Out1
  //
  // - Maybe the reflection stuff can go into an intermediate class like rsTerminatedWaveGuide
  //   or rsWaveGuideTerminated. Rationale: We may want to re-use the reflection facilities
  //   without necessarily having the input/output injection/extraction stuff implemented in 
  //   the particular ways done here. For example, one could imagine using waveguides with
  //   multiple injection and extraction (i.e. driving and pickup) points or even networks of
  //   (terminated) waveguides.
  //
  // - Implement the usual getTransferFunction() and getTransferFunctionAt() member functions like
  //   those we have in rsDelay, rsAllpassComb, etc.. The former should return an object of type
  //   rsSparseRationalFunction or maybe some subclass thereof. I think, I have made a subclass
  //   specifically for transfer functions of digital filters. I think it does some 
  //   canonicalizations differently than the general purpose class, e.g. normalizing to 
  //   a[0] = 1 rather than a[N] = 1 or something. Not sure, what the name of that was or where it
  //   is. That can be looked up in the experimental code for the allpass-comb stuff in the main 
  //   repo. 
  // 
  // - I think, it doesn't make sense to let class rsWaveGuide have such transfer function 
  //   computation functions because the waveguide itself doesn't yet have well-defined inputs and
  //   outputs and also does not yet have any opinion about (i.e. settings for) how to do 
  //   reflections and possibly also scattering. These are all things that will determine the 
  //   ultimate transfer function that any waveguide based filter will realize. However, the driver
  //   code may need some way to inquire certain waveguide settings that determine the transfer 
  //   function. The most obvious thing being the length M which we already can inquire via 
  //   getLength() but maybe there is more (or will be in the future). The API of rsWaveGuide 
  //   should support all the required inquiries and the actual computation of the transfer 
  //   function should then be done by the driver, incorporating its knowledge about how it 
  //   actually uses the waveguide.
};

template<class TSig, class TPar>
TSig rsWaveGuideFilter<TSig, TPar>::getSampleInExRef(TSig in)
{
  injectInput(in);             // Inject
  TSig out = extractOutput();  // Extract
  reflectAtEnds();             // Reflect
  stepTime();
  return out;
}

template<class TSig, class TPar>
TSig rsWaveGuideFilter<TSig, TPar>::getSampleInRefEx(TSig in)
{
  injectInput(in);             // Inject
  reflectAtEnds();             // Reflect
  TSig out = extractOutput();  // Extract
  stepTime();
  return out;
}

template<class TSig, class TPar>
TSig rsWaveGuideFilter<TSig, TPar>::getSampleExInRef(TSig in)
{
  TSig out = extractOutput();  // Extract
  injectInput(in);             // Inject
  reflectAtEnds();             // Reflect
  stepTime();
  return out;
}

template<class TSig, class TPar>
TSig rsWaveGuideFilter<TSig, TPar>::getSampleExRefIn(TSig in)
{
  TSig out = extractOutput();  // Extract
  reflectAtEnds();             // Reflect
  injectInput(in);             // Inject
  stepTime();
  return out;
}

template<class TSig, class TPar>
TSig rsWaveGuideFilter<TSig, TPar>::getSampleRefInEx(TSig in)
{
  reflectAtEnds();             // Reflect
  injectInput(in);             // Inject
  TSig out = extractOutput();  // Extract
  stepTime();
  return out;
}

template<class TSig, class TPar>
TSig rsWaveGuideFilter<TSig, TPar>::getSampleRefExIn(TSig in)
{
  reflectAtEnds();             // Reflect
  TSig out = extractOutput();  // Extract
  injectInput(in);             // Inject
  stepTime();
  return out;

  // Maybe we need to write Base::stepTime(); because some compilers complain when accessing
  // inherited members in class templates without such a qualification. I'm not sure, if this 
  // applies to member functions but I have it seen happening for member variables.
}

/*
ToDo:

- Document the physical interpretation and spectral and temporal effects of the settings. I think,
  the initial delay for the spike to show up in the output is given by mOut-mIn. Document also
  what comb-filtering effects we should expect. I think, driving the string at M/p will not be 
  able to excite modes that have their maxima at locations k * M/p. Putting the pickup at M/p
  will make us unable to pick up such modes. I think, this is a controllability/observability thing
  in the jargon of control systems. So, I think we should expect to see two series of notches in
  the generally harmonic spectrum.

- Look at some reference implementations. Maybe check out the STK and/or maybe there
  are some MatLab files associated with the PASP book

- Instead of taking the leapfrog algo as ground truth, try to theoretically figure out what
  sort of signal we should expect when driving the string at the boundary and compare that to
  the actual computed results of the various algorithms.

- Maybe have a function to compute the total stored energy. Maybe that function needs different
  implementations depending on what physical quantity we represent with the traveling wave 
  variable (displacement, force, velocity, pressure, etc.). Maybe we need also some functions to 
  convert the wave variables between different quantities. For example, force can be converted to
  velocity via R = f/v  ->  v = f/R, f = v*R where R is the wave impedance. (Verify! Cite source.).
  Maybe such conversion functions should take the impedance as argument (we don't want to store it
  as a member here - but maybe we can do it in a subclass, though.).

- Maybe implement transformers, gyrators and dualizers (see PASP, pg 616 ff). Maybe they can act at
  a point inside the waveguide or on the whole waveguide state (i.e. maybe implement both). The
  transformer should perhaps be based on a function rsDelay::scaleContentAt because it would be
  inefficient to use readOutputAt() and writeInputAt(). It should perhaps be a function 
  transformAt() or applyTransformerAt() similar to scatterAt()

- Figure out what happens if we do not reflect the waves at all, i.e. use reflection coeffs of zero
  or just leave out the reflection step. How could we interpret such a situation physcially? Maybe 
  it corresponds to a string that just continues to go on after the (now missing) termination. It 
  would correspond to a conceptually infinitely long string but we just do not really care or 
  emulate what is going on in the portions of the string that are beyond our view window. The 
  traveling wave signals would just travel out of sight and get lost, so to speak.

- Maybe parametrize the waveguide class with a template parameter for the delay class to use such 
  that we can use the same waveguide code for integer delay lines as well as interpolating delay
  lines. Eventually, we want to be able to create waveguides of non-integer length using different
  (user adjustable) interpolation methods (Lagrange, Thiran, Hermite, sinc, etc.)

- Analyze the eigenvalues of the Kelly-Lochbaum scattering matrix given by A = [1+k,-k; k,1-k] 
  (verify!). What are the genral conditions for a lossless scattering matrix anyway? All 
  eigenvalues must have absolute value of 1, maybe? I can see why the power-normalized scattering
  marix is lossless. It's a rotation matrix, after all. But in the case of Kelly-Lochbaum, it's not
  so obvious. Maybe it actually isn't lossless, after all? But nah! That seems implausible. I 
  think, that would imply instability (or decay) of the whole system (i.e. the waveguide (network)
  that uses Kelly-Lochbaum scattering). Or would it? Or maybe there is some sort of pole-zero 
  cancellation going on? Figure this out and document it.

- In PASP, pg 277 or here https://ccrma.stanford.edu/~jos/pasp/Summary_Lumped_Modeling.html
  it is said that "when the mass is in contact with the string, it creates a scattering junction on
  the string having reflection and transmission coefficients that are first-order filters". So that
  means that when we want to create a general framework for waveguide based synthesis, the 
  rsWaveGuide class needs some way of implementing (temporary, e.g. time-varying) scattering 
  junctions with filters instead of just coeffs. The time-varying aspect should be handled by the
  driver code and the scattering filters should be kept in the driver, too. But the rsWaveGuide 
  class needs some sensibe API, to let the driver pass in a filter object. Maybe the scatterAt
  functions should take a pointer to a filter object or something. The class for that filter should
  be general enough to handle all sorts of scattering filters and mesh nicely with the rest of the
  library. Maybe a pointer to rsBiquadChain or rsStateVariableFilterChain or something like that
  could be appropriate. The same should be possible for the reflections at the ends.

- Maybe in order to facilitate arbitrary signal manipulations in the driver (such as reading, then 
  filtering, then writing back - as needed in filtered scattering), the class rsWaveGuide should
  have functions like getTravelingWavesAt(), which it already has, and also a corresponding
  setTravelingWavesAt(int m, TSig yR, TSig yL) that would write the traveling wave variables.

*/

//=================================================================================================

/** Under construction. Just a stub at the moment.

Implements a network of an arbitrary number of waveguides that are arbitrarily interconnected by 
means of scattering junctions. */

template<class TSig, class TPar>
class rsWaveGuideNetwork  // Nickname: ScatterNet
{

public:

  //-----------------------------------------------------------------------------------------------
  // \name Setup

  void addWaveGuide(int maxLength, int initialLength);
  // Maybe the initial length should be optional and default to the max-length? We could implement
  // this by giving the parameter a default value of -1 and then inside the function, check for -1
  // and if it is -1 indeed, we would use maxLength for the initialization, otherwise the passed
  // value. Maybe it should also take an impedance as parameter?

  //void setWaveGuideLength(int index, int newLength);
  // For updating the length during operation

  void addScatterJunction(int i1, int m1, int i2, int m2, TPar k);
  // Maybe the scatter coeff k should be optional, defaulting to some invlaid value (like nan) and 
  // if it is nan in the call, we will ignore it and instead compute the coeff from the impedances
  // of the two waveguides. That requires to give rsWaveGuide a member for the impedance. In 
  // contexts where we don't need it, we can just ignore it.


  //-----------------------------------------------------------------------------------------------
  // \name Inquiry

  int getNumWaveGuides() const { return (int) waveGuides.size(); }

  //int getWaveGuideLength(int i) const { return waveguides[i].getLength(); }

  const rsWaveGuide<TSig, TPar>* getWaveGuide(int i) const 
  { 
    return waveGuides[i].get();
    //return waveGuides[i];
  }

  //-----------------------------------------------------------------------------------------------
  // \name Processing

  inline void injectInputAt(int wgIndex, int position, TSig signalValue);

  inline TSig extractOutputAt(int wgIndex, int position);








protected:

  //std::vector<rsWaveGuide<TSig, TPar>> waveGuides;   // Old
  std::vector<std::unique_ptr<rsWaveGuide<TSig, TPar>>> waveGuides;  // New
  // The vector of waveguide objects needs to use (smart) pointers because the class rsWaveGuide
  // contains members of type rsDelay which are not copyable. This leads to memory errors when 
  // trying to use a vector of objects because std::vector may have to copy stuff when it's 
  // re-allocating. 
  // 
  // OK - we will now get a compiler error when we try to copy an rsDelayLine object. This was 
  // achieved by deleting the copy constructor and copy assignment operator. 

  struct Junction
  {
    Junction(int index1, int position1, int index2, int position2, TPar scatterCoeff)
      : i1(index1), m1(position1), i2(index2), m2(position2), k(scatterCoeff) { }

    int  i1, i2;           // Indices of the two involved waveguides
    int  m1, m2;           // Spatial sample indices in the two waveguides
    TPar k;                // Scattering coefficient
    //rsMatrix4x4<TPar> A;  // Scattering matrix
    //rsMatrix2x2<TPar> A;  // Scattering matrix
  };
  // Special cases: i := i1 == i2 and m := m1 == m2 represents scattering within the single 
  // waveguide with index i at location m. If additionally m == 0 or m == M and k == 1 or k == -1, 
  // then this implements a reflection at one of the ends.

  std::vector<Junction> junctions;

  // Maybe have different kinds of junctions. Maybe some that scatter only conditionally, for 
  // example, when the distance between the waves in the two involved strings is below some 
  // threshhold. That would realize a nonlinearity that switches the scattering on and off based
  // on the current positions of both strings. Maybe the hard switch can also be somehow softened
  // to "fade in" the scattering continuously when the strings approach one another.

};

template<class TSig, class TPar>
void rsWaveGuideNetwork<TSig, TPar>::addWaveGuide(int maxLength, int initialLength)
{
  auto wg = std::make_unique<rsWaveGuide<TSig, TPar>>();
  wg->setMaxStringLength(maxLength);
  wg->setStringLength(initialLength);
  //wg->setImpedance(impedance);          // Maybe add later
  waveGuides.push_back(std::move(wg));
}

template<class TSig, class TPar>
void rsWaveGuideNetwork<TSig, TPar>::addScatterJunction(int i1, int m1, int i2, int m2, TPar k)
{
  Junction j(i1, m1, i2, m2, k);
  junctions.push_back(j);
}


/** Plots the contents of all the waveguides in the given waveguide network. */
template<class TSig, class TPar>
void rsPlotContents(const rsWaveGuideNetwork<TSig, TPar>& wgn)
{
  GNUPlotter plt;
  std::vector<TSig> yR, yL;
  int num = wgn.getNumWaveGuides();
  for(int i = 0; i < num; i++)
  {
    const rsWaveGuide<TSig, TPar>* wg = wgn.getWaveGuide(i);
    int M = wg->getLength(); 
    yR.resize(M+1);             // ToDo: Document why +1, see rsPlotWaveGuideContent
    yL.resize(M+1);
    for(int m = 0; m <= M; m++)
      wg->getTravelingWavesAt(m, &yR[m], &yL[m]);
    plt.addDataArrays(M+1, &yR[0]);
    plt.addDataArrays(M+1, &yL[0]);
  }
  plt.plot();


  // Notes:
  //
  // - We can't use a single call to plt.addDataArrays(M+1, &yR[0], &yL[0]); instead of the two 
  //   calls plt.addDataArrays(M+1, &yR[0]); plt.addDataArrays(M+1, &yL[0]); because that would 
  //   mean to interpret yR as x-axis values and yL as corresponding y-axis values, which is the 
  //   wrong interpretation of the data for our purposes here. Instead, we want the data to be 
  //   interpreted as two independent sets of y-axis values that shall be plotted against the 
  //   index. This is what two calls with just 1 array argument achieve. When addDataArrays()
  //   receives just one array, it automatically assumes that we want to plot the content of the
  //   given array against the index which is indeed what we want here.
}


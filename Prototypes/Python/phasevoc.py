"""
Phase Vocoder

@author: Robin Schmidt, www.rs-met.com
"""

from math import ceil, pi
from matplotlib import pylab as plt
from numpy import (zeros, linspace, sin, cos, transpose, flipud, angle, shape,
                   conj, real, allclose, ones)
from numpy.random import seed, random                
from numpy.fft import fft, ifft, fftshift


def spectrogram(x, w, hop, pad=1):
    """
    Complex spectrogram.

    Computes the complex spectrogram of signal "x" using window "w", a hopsize
    of "hop". The FFT transform size is given by pad*length(w) where "pad" is a
    zero-padding factor which should be a power of 2. The default is 1 which
    amounts to no zero-padding. In this case, there will be len(w)/2
    non-redundant spectral bins (only the positive frequencies are passed in
    the returned matrix).

    Parameters
    ----------
    x : numpy array
      input signal
    w : numpy array
      window to be used
    hop : integer
      hopsize, distance between successive frames (in samples)
    nBins : integer
      number of FFT frequency bins, should be at least len(w)/2

    Returns
    -------
    s : 2D numpy array of complex numbers
      complex spectrogram
    """
    assert(even(len(w)))               # blocksize must be even
    N = len(x)                         # number of samples
    B = len(w)                         # blocksize
    H = hop                            # hopsize
    F = ceil((N+H-1)/H)                # number of frames
    M = pad*B                          # FFT size
    K = M/2+1                          # number of non-redundant FFT bins   
    s = zeros((F,K), dtype=complex)    # allocate spectrogram
    n = 0                              # center of current block
    for i in range(0, F):              # loop over blocks/frames
        si     = stft(x, n, w, M)      # M-point STFT, centered at n
        s[i,:] = si[0:K]               # store positive frequencies
        n     += H                     # hop to next block
    return 2*s / sum(w)                # normalize


def synthesize(s, w, hop, N):
    """
    Synthesize signal from complex spectrogram.

    Notes:
    -maybe rename to spectrosynth
    -normalization works only, if the analysis- and synthesis-window  are the
     same
    -maybe factor out the normalization into a function
    """
    assert(even(len(w)))               # blocksize must be even
    F, K = shape(s)                    # number of frames and bins    
    B  = len(w)                        # blocksize
    H  = hop                           # hopsize
    Ny = (F-1) * H + B                 # number of samples - verify
    M  = 2*(K-1)                       # FFT size
    y  = zeros(Ny)                     # allocate signal
    si = zeros(M, dtype=complex)       # allocate STFT buffer
    n  = B/2                           # center of current block
    #print(F, K, M, Ny)                 # !!! debug !!!
    for i in range(0, F):              # loop over blocks/frames
        si = fftsym(s[i,:], M)         # symmetrized STFT
        si = ifft(si)                  # reconstructed grain, use fftshift
        g  = w*real(si)                # preliminary - si might be longer than
                                       # w due to zero-padding - write function
        ola(y, n, g)                   # overlap/add
        n += H                         # hop to next block
    y *= 4*H*sum(w) / (3*B)            # scale
    return y[B/2:B/2+N]                # truncate start and end


def demod(Nx, Ny, wa, Ha, ws, Hs):
    """
    Computes a demodulation signal for a resynthesized signal.
    
    Due to the fact that overlapping windows may not sum up to unity at all
    sample-instants, there might be an amplitude modulation present in the
    resynthesized signal. This function computes a gain signal that removes
    this amplitude modulation, when multiplied into the resynthesized signal.
    The signal is simply computed by feeding a DC signal through the 
    analysis/resynthesis roundtrip and taking the reciprocal of the result.
    
    Notes:
    The implementation is conceptually simple but could be made much more
    efficient by avoiding the FFT/IFFT roundtrip of each block, which is
    supposed to be unity operation in this context because no modification is
    made in the short-time spectra.
    It seems to work well only for Ha == Hs at the moment
    """
    k = ones(Nx)
    s = spectrogram(k, wa, Ha, 1)
    k = synthesize( s, ws, Hs, Ny)
    return 1 / k


#------------------------------------------------------------------------------
# spectrogram manipulation:


def magdev(m):
    """
    Magnitude deviatogram.
    
    Idea: If there's a stable sinusoid, the magnitude of a given bin should not
    vary too much from frame to frame. Therefore, we expect the magnitude in
    a given frame (and bin) to be the average between its left and right 
    neighbour frame (at the same bin). The relative deviation of the actual
    magnitude from that expected value is taken to be the magnitude deviation
    and considered as a measure for the nonsinusoidality of this pixel.
    
    ...
    todo: 
    take some caution when me == 0
    use linear extrapolation for expected values at n=0, n=N-1
    
    """
    N, K = shape(m)                    # number of frames and bins
    d = zeros((N, K))                  # allocate deviatogram
    for n in range(1, N-1):
        for k in range(1, K-1):
            mnk = m[n, k]            
            me  = (m[n-1, k] + m[n+1, k]) / 2   # expected value
            dnk = abs(mnk-me) / me                 # relative deviation 
            d[n, k] = dnk
            #d[n, k] = dnk**2                       # maybe use abs or **2
    return d
    
    


def sinusoidality(m, p, n, k):
    """
    Compute sinusoidality of a spectrogram value n, k.
    
    Given a magnitude spectrogram m and a corresponding phasogram p, this
    function computes a value between 0 and 1 (inklusive) that determines how
    much sinusoidal the value at frame index n and bin index k looks. When used 
    as multipplier, the value can be used for separation of stable sinusoidal
    partials from other signal components.
    
    Parameters
    ----------
    m : numpy 2D array
      spectrogram magnitudes
    p : numpy 2D array
      spectrogram phases
    n : integer
      frame index
    k : integer
      bin index

    Returns
    -------
    s : real
      sinusoidality value between 0 and 1
    """
    # Lokk compare magnitudes and phases at n, k with values at (n-1, k), 
    # (n+1, k), (n-1, k-1), etc. ....
    
    mnk = m[n, k]
    pnk = p[n, k]
    
    return 1  # preliminary
    


def only_sines(s):
    """
    Remove non-sinusoidal components from spectrogram
    
    under construction
    
    Given complex spectrogram s, this function returns another spectrogram in
    which only stable sinusoids are retained.
    """
    N, K = shape(s)                    # number of frames and bins
    m = abs(s)                         # magnitudes
    p = angle(s)                       # phases
    y = zeros((N,K), dtype=complex)    # allocate output spectrogram
    for n in range(0, N):
        for k in range (0, K):
            y[n, k] = s[n, k] * sinusoidality(m, p, n, k)
    return y

# maybe write one (or more) sinusoidality functions that return a value between
# 0 and 1 indicating how much sinuosidal the bin looks - more generally, we
# could write indicator functions that return multiplier values

# for production code in C++, we should avoid recalculations of magnitudes and
# phases -> reduces these computations by factor 3


#------------------------------------------------------------------------------
# Helper functions (maybe move to other file tools.py or something)


def reverse(a):
    """
    Reverse array a.
    """
    return a[::-1]


def even(n):
    return n % 2 == 0


def fftbins(N): # move to file ffthelp.py
    """
    Returns the number of non-redundant FFT bins for FFT size N.
    """
    if even(N):
        return (N/2)+1
    else:
        return (N+1)/2


def ffttrunc(s): # move to file ffthelp.py
    """
    Returns non-redundant part of the FFT of a real signal.
    """
    return s[0:fftbins(len(s))]


def fftsym(s, N): # move to file ffthelp.py
    """
    Symmetrizes a spectrum for inverse FFT.

    Given a DFT spectrum s of a real signal, containing only non-redundant
    bins, this function returns the full conjugate-symmetrized spectrum
    suitable for passing to the inverse FFT.
    """
    K = len(s)
    Y = zeros(N, dtype=complex)
    Y[0:K] = s
    if even(N):
        Y[K:N] = reverse(conj(s[1:K-1]))
    else:
        Y[K:N] = reverse(conj(s[1:K]))        
    return Y


def ola(y, n, g):
    """
    Overlap/adds grain g into y centered at sample n.
    
    todo: make this function work, if n is such that we would have to write out
    of range (truncate the grain accordingly in this case)
    """
    assert(even(len(g)))               # grain length must be even
    L = len(g)
    s = n-L/2                          # start index
    e = n+L/2                          # end index
    #print(s,e)
    y[s:e] += g


def stft(x, n, w, M):
    """
    Short Time Fourier Transform
    
    Returns the short time Fourier transform of signal x centered at sample n
    using window w and FFT size M. M should be >= len(w).
    """
    g = padded_grain(x, n, w, M)  # windowed, zero padded grain
    s = fft(g)                    # short time spectrum (maybe use fftshift)
    #s = fft(fftshift(g))          # maybe use rfft (for real input)
    return s


def padded_grain(x, n, w, M):
    """
    Extract grain from signal.

    Extracts a grain centered at "n" from "x". A chunk of length given by the
    window-length will be cut out of the signal, multiplied by the window and
    possibly zero padded at start and end to match the desired length "M".
    """
    L = len(w)
    c = w * chunk(x, n-L/2, L)
    return zero_pad(c, M)


def zero_pad(x, M):
    """
    Zero pads array x symmetrically to length M.
    """
    L = len(x)
    if M == L:
        return x         # no zero padding
    y = zeros(M)
    w = (M-L)/2          # start index for writing into the grain
    y[w:w+L] = x
    return y


def chunk(a, start, length):
    """
    Extract chunk from array

    Cuts a chunk of given "length" starting at "start" out of array "a".
    If start < 0 and/or start+length >= len(x), the chunk will be filled up
    with zeros, as if the array is conceptually extended to plus/minus
    infinity with zeros.
    """
    L = length
    n = start
    N = len(a)
    if n >= 0 and n+L < N:
        return a[n:n+L]       # no out-of-bounds indices
    c = zeros(L)
    w = 0                     # write-start index into chunk c
    if n < 0:
        w  = -n               # shift write-start
        L -= -n               # cut length
        n  = 0                # shift read-start
    if n+L >= N:
        L -= n+L-N            # cut length
    if L > 0:
        c[w:w+L] = a[n:n+L]   # copy chunk from a to c
    return c


def hanning(N):
    """
    Periodic Hanning window function for phase vocoder.

    Returns a Hanning window of length "N" that is specifically tweaked for use
    in the phase vocoder. It is periodic with a period of N samples, which
    means w[0] = 0 and w[N-1] = w[1] such that the nominal, non-existing sample
    w[N] = w[0]. Typical Hanning windows have w[0] = w[N-1] which is not
    appropriate for the phase vocoder, because the overlapped windows would
    not add up to a constant.
    """
    w = linspace(0, N-1, N)
    w = (1-cos((2*pi/N)*w))/2
    return w


#------------------------------------------------------------------------------
# test- and experimentation functions (move to other file):


def sine(fn, a, p, N):  # maybe move to test_phasevoc
    """
    Create sinewave

    Creates a sinewave with normalized frequency fn (= 2*fs/fs), with amplitude
    a and initial phase p. Intended to be used as testsignal.
    """
    s = a * sin(pi*fn * linspace(0, N-1, N) + p)
    return s


def noisysine(f, p, an, N):
    """
    Noisy sine wave
    """
    xs = sine(f, 1, p, N)       # sine
    xn = an * (2*random(N)-1)   # noise
    return xs + xn


def plot_matrix(m):
    mp = flipud(transpose(m))
    plt.imshow(mp, cmap='gray')
    

def show_spectrogram(s):
    mag = abs(s)
    plot_matrix(mag)


def show_phasogram(s):
    phs = angle(s)
    plot_matrix(phs)


def test_fftsym(N): # move to file ffthelp.py
    """
    Tests the FFT-buffer symmetrization
    """
    seed(1)
    x = random(N)                # input buffer
    X = fft(x)                   # FFT spectrum of x
    s = ffttrunc(X)              # non-redundant part
    Y = fftsym(s, N)             # reconstruct symmetric FFT buffer
    y = ifft(Y)                  # reconstruct signal
    plt.plot(s)
    plt.plot(X)
    plt.plot(Y)
    assert(allclose(X,Y))
    assert(allclose(x,y))
    
    
def test_roundtrip(N, f, fs, p=pi/2, B=512, H=128):
    """
    Analyzes and resynthesizes a sinusoid
    todo: make these function work with normalized frequencies for consistency.
    renam to test_roundtrip, maybe return a bool - true if y == x
    """    
    w = hanning(B)
    x = sine(2*f/fs, 1, p, N) 

    s  = spectrogram(x, w, H, 1)    # analysis
    y  = synthesize( s, w, H, N)    # resynthesis
    y *= demod(N, N, w, H, w, H)    # demodulation
    # maybe factor out a synthesize_raw function
   
    #m = abs(s)                      # magnitudes   
    #show_spectrogram(s)
    #show_phasogram(s)
    #plt.plot(w)   
    plt.plot(x)
    plt.plot(y)


def test_magdev(N, f, an, B=512, H=128, P=1):
    """
    Test drawing of a magnitude deviatogram.
    """
    x  = noisysine(f, 0, an, N)
    sx = spectrogram(x, w, H, P)    # analysis
    md = magdev(abs(sx))            # magnitude deviatogram
    #plot_matrix(abs(sx))
    #plot_matrix(md)
    #plt.plot(x)



def test_separate1(N, fn, an):
    """
    Test separation of sinusoid from noise.
    
    an: noise amplitude
    B:  block size
    H:  hop size
    P:  zero padding factor
    """
    xs = sine(fn, 1, 0, N)          # sine
    xn = an * (2*random(N)-1)       # noise
    x  = xs + xn
    sx = spectrogram(x, w, H, P)    # analysis
    sy = only_sines(sx)
    
    #show_spectrogram(sx)
    show_spectrogram(sy)
    #plt.plot(x)


# if the module is run as script:
if __name__ == "__main__":
    seed(1)
    f  = 0.01          # normalized frequency
    an = 0.02           # noise amplitude
    N  = 20000         # number of samples
    B  = 512           # block size
    H  = 128            # hop size
    P  = 1             # zero padding factor
    w  = hanning(B)    # window 

    # a test for plotting amagnitude deviatogram:
    x  = noisysine(f, 0, an, N)
    s  = spectrogram(x, w, H, P)    # analysis
    sa = abs(sx)
    md = magdev(sa)                 # magnitude deviatogram
    #plt.plot(x)
    #plot_matrix(sa)
    plot_matrix(md**0.3)
    
    #test_roundtrip(2000, 100, 44100, pi/4, 512, 128)
    #test_separate1(40000, 0.15, 0.5)
    #test_magdev(40000, 0.2, 1.0, 512, 128, 1)
    



# Notes:
# When using the Hanning window as defined here for analysis and resynthesis,
# and a hopSize of blockSize/2^N (N>=2), the demod function should only 
# modify the start and end (in the middle, there should be no modulation
# present anyway)
   
# ToDo: 
# -make sure that the phase values are with respect to the center of the
#  windows - use fftshift
# -try different analysis and synthesis windows and hopsizes
# -use k for bin index, n for sample index, j for frame index
# figure out how to use grayscale colormap
   
# for the decision, if a spectrogram value s(j, k) belongs to a sinusoid, 
# compare it to a local neighbourhood centered at s(j, k) - compare magnitude
# and phase to a range of expected values that is determined by the
# neighbourhood values

   
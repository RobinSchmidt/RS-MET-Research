PROSampler - Perfect Reconstruction OverSampler
===============================================
by Robin Schmidt (www.rs-met.com)

Introduction
------------

I present a Python library for the design of FIR lowpass filters that are suitable for the upsampling and downsampling steps within an oversampled digital signal processing (DSP) algorithm. The library is the result of a personal research project that I conducted with AI assistence from Claude Sonnet 5.5 with medium effort level. I started it with the following (slightly corrected and redacted) initial prompt:

> I write audio signal processing algorithms for musical purposes. In this context, it is often necessary to use oversampling, i.e. to first upsample an incoming signal to a higher sample rate, then do the actual processing at that higher rate, then downsample it to the original sample rate again. Both, up- and downsampling steps involve lowpass filtering the signal. In the case of upsampling, the lowpass is called anti-imaging lowpass, in the case of downsampling, it's called anti-aliasing lowpass. What I want to create is a pair of FIR upsampling and downsampling filter kernels (let's denote them by u[n] and d[n] respectively), such that an upsampling-downsampling roundtrip is an exact identity operation (up to floating point roundoff errors). For concreteness, let's assume I want to implement a waveshaping distortion effect. So, the processing pipeline would be:
>
>     input -> upsample -> waveshaper -> downsample -> output
>
> The goal is that my oversampled distortion algorithm is completely transparent as long as the waveshaper that sits in between the upsampler and the downsampler is itself an identity function (i.e. does nothing). Is there any known literature for this particular type of filter design problem?

From this prompt, you can understand the problem that I was trying to solve. It turns out the problem is indeed solvable and there exists prior art for designing filter pairs that have the features that we require here. However, these known results have not yet - to my knowledge (TODO: Figure out!) - applied to the exact oversampling situation that we are dealing with here. The solution to this problem is embodied in the filter design library that is the result of this research project. 

For reference, the chat sessions are publically shared here:

https://claude.ai/share/e5f52b28-a86d-47da-be86-2d92bd6e0786  
https://claude.ai/share/0bdb366b-24fb-41dc-a51f-fc4121a8f88c


Research
--------

Claude answered affirmatively and explained to me that the condition for perfect reconstruction, i.e. a lossless upsampling-downsampling roundtrip, is that the convolution product `h[n] = u[n] * d[n]`  of the upsampling and downsampling kernels u and d respectively must be a "Nyquist(L) filter" where L is the desired oversampling factor and the Nyquist(L) condition means that the center sample must be 1 and every sample that is a multiple of L away from the center must be 0. We assume an odd filter length for h here such that there exists a center sample. 

If we start with `u[n]` given, for example produced by a standard method such as a windowed sinc design, finding a corresponding partner kernel `d[n]` is a constrained linear optimization problem. The constraints are embodied in the required Nyquist-L condition and the optimization objective is the frequency response of the filter. ...TBC...


<!--

ToDo:

- Explain the settings in which such Nyquist(L) filters have been used so far. I think mostly in eprfect reconstruction filter banks (QMF, etc.) where a full bandwidth signal gets split into several subbands which are then individually downsampled. The goal in this setting is that adding up these downsampled subbands after upsampling them again gives the original signal back (verify!). So, with 2 subbands, the processing pipeline would be

          |---1-d[n]-- -> D2 
 input ---|                   ....not sure....finish this picture - and be better at ASCII art
          |----d[n]--- -> D2

- Explain that for a given fixed u, the problem of designing a corresponding partner kernel d is constrained linear optimization problem (with (linear)) equality constraints, I think. 

- Explain the design "pipeline" (see chat session 1)

- Explain the two methods to solve the constrained linear optimization problem (KKT, null-space). I think, the null-space method is preferable and the KKT method exists only because a previous session produced it but it turned out to be actually inferior due to numerical instability (verify!)

- Explain the two different objectiove functions that we initially used. I think, they are 
  -stopband energy (via an analytic formula)
  -stopband energy by grid sampling (?) ...verify


- Explain that an optimal solution for u and d in the sense of a minimal maximum deviation from a given filter specification (in terms of passband flatness, transition width, stopband rejection) is a nonlinear optimization problem (verify!)

-->
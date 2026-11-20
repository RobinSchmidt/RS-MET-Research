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

from which you can understand the problem statement. It turns out the problem is indeed solvable and there exists prior art for designing such filter pairs. However, these known results have not yet - to my knowledge (TODO: Figure out!) - applied to the exact oversampling situation that we are dealing with here. The solution to this problem is embodied in the filter design library that is the result of this project. 

For reference, the chat sessions are publically shared here:

https://claude.ai/share/e5f52b28-a86d-47da-be86-2d92bd6e0786  
https://claude.ai/share/0bdb366b-24fb-41dc-a51f-fc4121a8f88c


Research
--------
Some investigations about desigining a pair of filters for upsampling and downsampling that allows
for a transparent oversampling process, i.e. one in which the upsample -> downsample roundtrip is
an identity operation up to numeric roundofff errors.



Chat with Claude Sonnet 5.5, 2026-10-13
https://github.com/copilot/share/c25e12ac-49a0-88d5-8943-b80804806057
...Its gone! GitHub deletes them after some time!! WTF!!! Try to re-create it and then export it!
But maybe this time, do it in claude.ai. There, the chats are stored indefinitely. This was the
first prompt:

I write audio signal processing algorithms for musical purposes. In this context, it is often necessary to use oversampling, i.e. to first upsample an incoming signal to a higher sample rate, then do the actual processing at that higher rate, then downsample it to the original sample rate again. Both, up- and downsampling steps involve lowpass filtering the signal. In the case of upsampling, the lowpass is called anti-imaging lowpass, in the case of downsampling, it's called anti-aliasing lowpass. What I want to create is a pair of FIR upsampling and downsampling filter kernels (let's denote them by u[n] and d[n] respectively), such that an upsampling-downsampling roundtrip is an exact identity operations (up to floating point roundoff errors). For concreteness, let's assume I want to implement a waveshaping distortion effect. So, the processing would be:

  input -> upsample -> waveshaper -> downsample -> output

The goal is that my oversampled distortion algorithm is completely transparent as long as the waveshaper that sits in between the upsampler and the downsampler is itself an identity function (i.e. does nothing). Is there any known literature for this particular type of filter design problem?





Follow up questions:



-Why is the downsampling kernel roughly twice as long as the upsampling kernel? Is this a general rule?

-What is a "root-Nyquist design"?

-Can you also give me a script that gives me the "u = d = g where g * g is Nyquist(L), the "matched filter" pair". Could this even be done using the same helper methods from the script before such that they can be re-used? If so, that would be preferable. If not, nevermind - just do it the way it's supposed to be done.

-You said, in the case of the waveshaper, the quality of both filters matters - which I totally agree with. But are there any arguments for one of the filters being more important in this context? And what about other context? In what situations would the upsampling kernel matter more and in what would the downsampling kernel matter more?

-Can you elaborate on the statement "Joint optimization of u and d, or a minimax criterion, would need an iterative method such as alternating between the two.". How exactly would that be done?

-Are there any libraries specifically for that purpose?

-Is it also possible to design u and d such that they are both the same kernel, i.e. u = d?
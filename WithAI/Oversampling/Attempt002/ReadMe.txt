This is my 2nd attempt at the problem. This time, I use claude.ai directly rather than GitHub 
Copilot. The model was Sonnet 5.5 with medium effort level. The chat is publically shared here:

https://claude.ai/share/e5f52b28-a86d-47da-be86-2d92bd6e0786

In the "Open questions" section in the file: oversampling_research_summary.md, the following ideas
seem to be very interesting to explore later:

- Item 2: Using a minimax objective. Maybe this will result in quiripple responses that achieve an
  even better stopband attenuation.

- Item 7: Transition band shaping.

- Maybe we could use 3 different weighting factors for the errors in the passband, transition band 
  and stopband. I guess, we now use weights [1, 0, 1]? The code has an option for a transition band
  weight and Claude recommends a value of 0.3, so I guess in this case, it would be [1, 0.3, 1]?
  Verify this!

- Maybe we could try to enforce similar magnitude responses for u and d by penalizing differences
  in the magnitude responses.

- Maybe we could also provide a convenient code exporter that exports a ready-to-use C++ class with
  hardcoded coeffs that uses an efficient polyphase implementation. It should have the usual 
  getSample() functions. Maybe the classes should be named like Upsampler_L4_90_75 and 
  Downsampler_L4_90_75 for a pair with L=4, passband = 0.9 (90 %) and rejection = 75 dB. Or maybe 
  instead of passband and rejection, it should give the lengths of u and d like 
  Upsampler_L4_U63_D127. And/or maybe we should be able to export C implementations, too. And maybe 
  Rust. 

- Maybe we could explore IIR designs, too?
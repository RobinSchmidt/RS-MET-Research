This is my 2nd attempt at the problem. This time, I use claude.ai directly rather than GitHub 
Copilot. The model was Sonnet 5.5 with medium effort level. The chat is publically shared here:

https://claude.ai/share/e5f52b28-a86d-47da-be86-2d92bd6e0786

In the file:  oversampling_research_summary.md  the "Open questions" 2 (minimax objective) and 7 
(transition band shaping) seem to be most interesting. Maybe we could also provide a convenient
code exporter that exports a ready-to-use C++ class with hardcoded coeffs that uses an efficient
polyphase implementation. It should have the usual getSample() functions. Maybe the classes should 
be named like Upsampler_L4_90_75 and Downsampler_L4_90_75 for a pair with L=4, passband = 0.9 (90 %)
and rejection = 75 dB. Or maybe instead of passband and rejection, it should give the lengths of u 
and d like Upsampler_L4_U63_D127. And/or maybe we should be able to export C implementations, too.
And maybe Rust. Maybe we could explore IIR designs, too?
This is my 2nd attempt at the problem. This time, I use claude.ai directly rather than GitHub 
Copilot. The model was Sonnet 5.5 with medium effort level. The chat is publically shared here:

https://claude.ai/chat/b98c9ff2-aaed-467d-a295-d17944ea98bf    my private link
https://claude.ai/share/e5f52b28-a86d-47da-be86-2d92bd6e0786   the published shared link

This chat has now grown enormously large such that any single follow up question will burn through
tokens like there is no tomorrow. To continue, we need to start a fresh session. Google Gemini
recommended to request a summary with the following prompt (I modified it a bit):

This chat has grown too long to meaningfully continue here. We need to start a new session with a fresh context window. Stop all other tasks and generate a compact, high-density "Context Handoff" of our research so far so I can continue in a fresh chat without losing progress. 

Keep it strictly factual, concise, using these exact sections:
1. **Core Research Goal:** What I am trying to figure out or decide.
2. **Confirmed Findings & Key Facts:** Bullet points of established facts, figures, sources, or conclusions we reached.
3. **Ruled Out / Dead Ends:** Options or assumptions we already dismissed and why.
4. **Constraints & Context:** My specific requirements, setup, or rules you must follow.
5. **Next Immediate Question:** The exact question or subtopic we were about to explore next.

Do not include conversational filler, greetings, or explanations.


...and then copy the result of that into the prompt of the new session. Or export the response to an
.md file with tools like saveAI or Inkpour and link the generated document in the prompt of the 
fresh chat.





Some possible follow up questions:

- The doc of the prosampler library says:
    "a windowed-sinc u is itself (nearly) Nyquist(L), so d = delta is an exact partner."
  Why is it only "nearly" Nyquist(L) and not exactly? What is the error and why does it occur? Maybe
  the center sample is not exactly 1?


In the "Open questions" section in the file: oversampling_research_summary.md, the following ideas
seem to be very interesting to explore later:

- Item 2: Using a minimax objective. Maybe this will result in equiripple responses that achieve an
  even better stopband attenuation.

- Item 7: Transition band shaping. Maybe we could prescribe a desired shape for the transition. But
  what shape? In the simplest case, it could just go linearly down from 1 to 0. But we may also try
  to use functions that are closer to what such FIR filters actually tend to want to do. Maybe try
  functions like  f(x) = 1 - x^p, f(x) = (cos(0.5*pi*x))^(1/p)  for some parameter p >= 1. See:
  https://www.desmos.com/calculator/pe0oopida3
  
- Item 3: Explore the "Root-Nyquist design" route. I think, this is what Kimi tried in Attempt004.
  

Here are a couple of my own additional ideas that can be explored further:

- If we do this transition band shaping, we may also look at how the prescribed transition shape 
  impacts the step response of the filter. It may be desirable in certain contexts to have a step 
  response that doesn't overshoot and ring too much. Maybe by prescribing a suitable transition band
  shape, we may indirectly influence the amount of overshoot and ringing. Alternatively we may try 
  to minimize ringing more directly by adding a term that penalizes the ringing into the 
  optimization objective.

- Maybe we could use 3 different weighting factors for the errors in the passband, transition band 
  and stopband. I guess, we now use weights [1, 0, 1]? The code has an option for a transition band
  weight ("transition_weight") and Claude recommends a value of 0.3, so I guess in this case, it 
  would be [1, 0.3, 1]? Verify this!

- Maybe we could try to enforce similar magnitude responses for u and d by penalizing differences
  in the magnitude responses. But maybe together with the linear phase constraint, this will again 
  boil down to enforcing d = u which was found to result in significantly worse magnitude responses
  for a given filter length. The additional degrees of freedom when allowing d != u really unlocks
  further optimization potential. But what if we drop the linear phase constraint, i.e. allow 
  asymmetric u and d again?

- Maybe we could also provide a convenient code exporter that exports a ready-to-use C++ class with
  hardcoded coeffs that uses an efficient polyphase implementation. It should have the usual 
  getSample() functions. Maybe the classes should be named like Upsampler_L4_90_75 and 
  Downsampler_L4_90_75 for a pair with L=4, passband = 0.9 (90 %) and rejection = 75 dB. Or maybe 
  instead of passband and rejection, it should give the lengths of u and d like 
  Upsampler_L4_U63_D127. And/or maybe we should be able to export C implementations, too. And maybe 
  Rust. 

- Maybe we could explore IIR designs, too?

- Maybe the library could have a "demos" folder that demonstrates the usage of the library in a 
  somewhat realistic scenario. The demo should design a filter pair, plot the impulse-, step- and 
  magnitude-responses and optionally export C/C++/Rust code with an optimized polyphase 
  implementation. It should be easy to modify the design parameters in the demo such that it can be
  used as default starting point.  ...work on that is in progress...
  
----------------------------------------------------------------------------------------------------


Verify this:

When we use the kernels u,d specifically for upsampling and downsampling rather than for filtering
arbitrary signals at the oversampled rate, we unlock an optimization that is known as "polyphase 
filtering" in the DSP literature. It works as follows:

Polyphase upsampling takes advantage of the fact that only every L-th sample of the zero-stuffed 
input signal is nonzero. It avoids computing the zero contributions from the zero samples to the 
FIR convolution sum. 

Polyphase downsampling takes advantage of the fact that in the decimation stage, we will throw away 
all samples except every L-th anyway. It avoids to compute these extra samples that we will not need
anyway.

So, the polyphase upsampling and downsampling stages merge the filter-then-upsample and
filter-then-downsample steps into a single operation.
Some explorations of Anti Derivative Anti Aliasing.

Chats:

With Claude Sonnet 5.5, 2026-10-19:
https://github.com/copilot/share/8a5f133e-0880-8c77-9152-aa40240a4015


Digest:

The notation is as follows: f is the original waveshaping function, F1 is integrated once, F2 
integrated twice, etc.


This is an interesting reply to my question about using tabulated numerically integrated functions:

"Instead of integrating numerically and storing F, approximate f itself by a piecewise polynomial, such as a spline or minimax fits per segment. The antiderivatives F1 and F2 are then exact piecewise polynomials, with no numerical integration error. You evaluate them with Horner's scheme per segment, and you can keep the segments continuous."


To my question about how to choose the integration constants:

"Set F1(0) = 0 and F2(0) = 0, and make F2′ = F1 and F1′ = f hold exactly, including at the joins. Then initialise the state to zero and check that the processor outputs zero for a long run of zero input. A quick unit test is also to compare the regular branch with the fallback branch near the threshold. Any mismatch points to an inconsistent constant."

It may make sense to choose then in such a way that the antiderivatives have a value of zero at x=0.
Additionally, they need to be consistent, i.e. F2' must be F1 exactly.



At the start, it says:

"
The ADAA approach:

- Treat x(t) as a continuous, piecewise-linear interpolation of the samples.

- Apply f to that continuous signal, giving y(t) = f(x(t)).

- Low-pass filter y(t) with a one-sample-wide moving-average (box) filter, then sample the result.
"

which makes me think that maybe using the simple linear interpolation approach, i.e. using a 3 
sample long upsampling filter u[n] together with a 5 sample long downsampling filter d[n] (for 2x
oversampling) might be viable for transparent oversampling as well. Try it!


Some investigations about desigining a pair of filters for upsampling and downsampling that allows
for a transparent oversampling process, i.e. one in which the upsample -> downsample roundtrip is
an identity operation up to numeric roundofff errors.



Chat with Claude Sonnet 5.5, 2026-10-13
https://github.com/copilot/share/c25e12ac-49a0-88d5-8943-b80804806057

Follow up questions:

-Why is the downsampling kernel roughly twice as long as the upsampling kernel? Is this a general rule?

-What is a "root-Nyquist design"?

-Can you also give me a script that gives me the "u = d = g where g * g is Nyquist(L), the "matched filter" pair". Could this even be done using the same helper methods from the script before such that they can be re-used? If so, that would be preferable. If not, nevermind - just do it the way it's supposed to be done.

-You said, in the case of the waveshaper, the quality of both filters matters - which I totally agree with. But are there any arguments for one of the filters being more important in this context? And what about other context? In what situations would the upsampling kernel matter more and in what would the downsampling kernel matter more?

-Can you elaborate on the statement "Joint optimization of u and d, or a minimax criterion, would need an iterative method such as alternating between the two.". How exactly would that be done?

-Are there any libraries specifically for that purpose?

-Is it also possible to design u and d such that they are both the same kernel, i.e. u = d?
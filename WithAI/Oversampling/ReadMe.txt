This folder has a bunch of subfolders where I keep the AI chat transcripts and code artifacts of 
various attempts at the same problem.

Attempt001
My very first attempt using my GitHub Copilot subscription with the model Claude Sonnet 5.5 at 
medium effort level. Unfortunately, the chat itself got lost because GitHub deletes them after 28
days of inactivity which I wasn't aware of. If you want to keep them, you need to export them to .md
files. But I did keep the generated code artifacts, i.e. the Python and C++ scripts.

Attempt002  <--- This is the one that succeeded most
This is a second attempt with Claude Sonnet 5.5 at medium effort but this time I did it directly on
claude.ai. Not only will the chat be retained there indefinitely but it has also the additional 
advantage that Sonnet can directly execute code there such that it can actually try out its own 
scripts, debug them etc. That was not possible in a chat on github, so the generated code snippets
from there were untested.n In this second attempt, my goal was to recreate the results from the
first but this time, Sonnet took another approach for the optimization problem. The chat actually 
went rather deep and we ended up with a small Python library. ...TBC...

Attempt003
In the third attempt, I went back to GitHub Copilot but this time using the model GPT-5.3-Codex with
default effort level. This time, I made sure to export the chat to a .md file so it doesn't get lost
again. Codex gave pretty concise replies and code. If that code works, it might be the best option.
Codex also directly offered to write a Python script with a convenient API.

Attempt004
In the fourth attempt, I gave the same initial prompt to Kimi K3 at default effort in GitHub 
Copilot. Kimi chose the spectral factorization route but then got stuck at some point.


Some resources:

- Multirate Systems and Filter Banks  by  Vaidyanathan
  https://www.amazon.de/-/en/Multirate-Systems-Filter-Banks-Vaidyanathanm/dp/0136057187
  https://authors.library.caltech.edu/records/rmhds-22q28
  All models cited this reference so it seems to be an important main source.

- Multirate Digital Signal Processing  by  Crochiere, Rabiner
This repo contains prototypes, experiments, collected information, code snippets and libraries from 
various sources for research and devlopment. It's my main playground to try out new ideas which may 
or may not at some point be moved over into the main RS-MET repo depending on if and when I consider 
them generally useful and mature enough. This includes experimental signal processing algorithms but 
also stuff that I do in a more recreational mode such as mathematical art rendering and image 
processing functions that I write just for fun. It contains also code where I familiarize myself 
with new mathematical or algorithmic ideas or try out some of my own ideas. It's a bit messy here in 
this repo because this is the place where I just mess around with ideas and hobby projects. 


The main active C++ project is in the Visual Studio solution:

  Prototypes/Cpp/Builds/VisualStudio2019/Experiments.sln
  
In order to build it projects, the main RS-MET repo must sit locally in the same folder as this
RS-MET-Research repo. If it still doesn't work, it might be the case that RS-MET repo has to be 
switched to a more recently updated branch. The most up to date branch there is always the "work" 
branch. That project is similar to the 

   RS-MET\Tests\TestsRosicAndRapt\Builds\VisualStudio2019

project in my main repo. It builds a command line application that runs one or more experimental 
script-like functions that are either supposed to be run in the debugger to inspect variables 
and/or produce some output files. Which script is run is decided by what function calls you 
uncomment in the Main.cpp.


There's also some activity in:

  Notes\LatexDocuments

where I work on some latex documents where I explain some ideas. Not much to see there yet, though.
The C++ project is also used to render the figures for the latex documents. This is explained in 
more detail in the local ReadMe file there.




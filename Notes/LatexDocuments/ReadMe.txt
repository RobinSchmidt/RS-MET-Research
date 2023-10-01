In order to be able to render the LaTex documents with all the figures, you'll first need to render
the plots and put the results into the MathFigures subfolder. The plots are created with GNUPlotCPP
and this repo contains to the code to produce the plots in the project:

  Prototypes\Cpp

There is, for example, a function

  makePlotsForPolyaPotentialPaper()

and running this function will produce all the plots for the paper "The Polya Potential of Complex 
Functions". To re-generate the plots, I recommend to just open the solution:

  Prototypes/Cpp/Builds/VisualStudio2019/Experiments.sln

in Viusal Studio and in Main.cpp look for the call to this function. It may be commented out, so
you need to uncomment it and maybe comment out other stuff that you currently don't want be 
executed. Navigate into the function and see the documentation there for further instructions.

The idea is that the so generated graphics files are not and shall never be part of this repo 
because having lots of media files in the repo would blow its size up unreasonably, especially when 
the content of these media files may be updated from time to time (due to, say, changing some 
plotting style setting like axis labels, colors, pixel-size, etc.). In order to render the pdf paper 
from the latex source, these image files must first be re-generated and then moved put in the 
appropriate folder "Notes/LatexDocuments/MathFigures" after the repo has been freshly cloned from 
the github server to the local development machine.

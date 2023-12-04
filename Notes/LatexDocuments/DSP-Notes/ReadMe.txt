This folder contains some older notes and papers about DSP topics. The papers do not compile by 
themselves. Instead, the "shell" document OneAtATime.tex should be loaded and compiled after the 
desired, to-be-compiled paper has been included. I did it this way back then in order to set up all
the style settings consistently in the shell document such that all papers get the same style 
without having to set it up redundantly in each paper and quite possibly diverging when some changes
are not applied to all papers in sync.

When the plots for the GradientsFromDirectionalDerivatives.tex need to be re-generated, the code for
this is in the main RS-MET codebase in the function meshGradientErrorVsDistance() in the 
TestsRosicAndRapt project. Eventually, this plot generation stuff should go into the same place 
where we generate the plots for the newer papers.
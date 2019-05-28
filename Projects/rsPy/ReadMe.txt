Project to create python bindings for the RS-MET signal processing libraries 
rapt and rosic.

To build it successfully, the BoostPython juce module needs to include the 
header file pyconfig.h which is supposed to reside somewhere in the python 
installation folder. The folder where this file resides needs to be added to
the "Header Search Paths" field in the exporter settings in the .jucer file.
On my machine, the path is:

C:\Users\Rob\Anaconda3\include

but may obviously be different on other people's machines. So if you get an
include error for this file, change the path there and re-generate the visual
studio project. If you don't use Projucer, change the path directly in visual 
studio.



todo: maybe at some stage, distribute it as a proper python module...maybe 
AudiPy or something - should work together with NumPy/SciPy
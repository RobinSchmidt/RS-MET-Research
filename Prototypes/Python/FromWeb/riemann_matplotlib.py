# http://homepages.math.uic.edu/~jan/mcs507f13/riemann_matplotlib.py

# L-16 MCS 507 Wed 2 Oct 2013 : riemann_matplotlib.py

"""
This script uses pyplot of matplotlib to plot
the Riemann surface of the cubed root.
"""

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

F = plt.figure()
A = F.gca(projection='3d')
UR = np.arange(-1, 1, 0.01)
VR = np.arange(-1, 1, 0.01)
U, V = np.meshgrid(UR, VR)
X = U**3 - 3*U*V**2
Y = 2*U**2*V - V**3

# scale colors so that in [0,1]
CV = (1.0 + V)/2

S = A.plot_surface(X, Y, U, facecolors=cm.jet(CV))

plt.show()

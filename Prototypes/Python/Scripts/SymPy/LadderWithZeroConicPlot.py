"""
Plots the two conic section equations that arise from designing a first order
filter (1 pole, 1 zero) to be used in a series connection...

p2 is always a hyperbola that goes trough 0 and 1. For continuity around 
w=0.5*pi, we seem to have to use the lower arm (the one that goes through 0).
However, that solution tends to grow large for small w.
..hmm - this soltuion is very small around w=pi/2 
but with the upper arm, there's a intersection only for certain values of w
..maybe the 1 pole / 1 zero idea is not really good, after all and we should 
stick to 1 pole / 0 zero
"""

from sympy import plot_implicit, symbols, sin, cos, tan, pi

w = 0.45*pi               # cutoff frequency

# intermediate parameters:
s = sin(w)
c = cos(w)
t = tan((w-pi)/4)

#define symbols and equations:
x, y = symbols('x y')  # x=b0, y=b1
a    = x+y-1           # a = a1 = b0+b1-1
e1   = 2*(1-c)*(x+y-x*y-1) + x**2 + y**2
e2   = t*(x+a*y+(y+a*x)*c) - s*(a*x-y)

# plot the two conics:
p1 = plot_implicit(e1,(x,-2,2),(y,-2,2))
p2 = plot_implicit(e2,(x,-2,2),(y,-2,2))
#p2 = plot_implicit(e2)

"""
Demonstrates implicit plotting. We plot a conic section equation of the form:
A*x**2 + B*x*y + C*y**2 + D*x + E*y + F = 0

"""

from sympy import plot_implicit, symbols
A =  1
B =  1
C = -1       # +1: ellipse, -1: hyperbola
D =  1
E =  1
F = -1/2
x, y = symbols('x y')
p = plot_implicit(A*x**2 + B*x*y + C*y**2 + D*x + E*y + F)

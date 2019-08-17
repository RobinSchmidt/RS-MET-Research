"""
Script for solving the equations for a soft-clipper that behaves as identity
function between a given threshold t < 1, as a polynomial
f(x) = a + b*x + c*x^2 + d*x^3
between t < x < 1 and the constant function 1 above 1, where the polynomial
has matched derivatives at the transition points
"""

from sympy import *
a,b,c,d,t = symbols('a b c d t');

# define equations to be solved:
e1 = a + b + c + d - 1              # f(1)  = 1 = a + b + c + d
e2 = b + 2*c + 3*d                  # f'(1) = 0 = b + 2c + 3d
e3 = a + b*t + c*t**2 + d*t**3 - t  # f(t)  = t = a + b*t + c*t^2 + d*t^3
e4 = b + 2*c*t + 3*d*t**2 - 1       # f'(t) = 1 = b + 2c*t + 3d*t^2

# solve the set of equations:
r = solve([e1,e2,e3,e4],[a,b,c,d])

# to show the result, type r at the prompt -  it gives (already optimized):
# k = 1 / (t**2 - 2*t + 1)
# d = k * (-1) = -k
# b = k * (1 - 4*t)
# a = k * t**2
# c = k * (1 + 2*t)



"""
{d: -1/        (t**2 - 2*t + 1),
 b: (-4*t + 1)/(t**2 - 2*t + 1),
 a: t**2/      (t**2 - 2*t + 1),
 c: (2*t + 1)/ (t**2 - 2*t + 1)}
"""


"""
Script for solving the equations for a soft-clipper that behaves as identity
function between a given threshold t < 1, as a polynomial
y = a*x + b*x^3 + c*x^5 + d*x^7 
between t < x < 1 and the constant function 1 above 1, where the polynomial
has matched derivatives at the transition points
"""

"""
from sympy import *
x,y,a,b,c,d,t = symbols('y a b c d t');

y  = a*x + b*x**3 + c*x**5 + d*x**7; # y(x)
yp = diff(y, x);                     # y'(x)

# solve the system:
# y(t)  = t -> y  - t = 0
# y(1)  = 1 -> y  - 1 = 0
# y'(t) = 1 -> yp - 1 = 0
# y'(1) = 0 -> yp     = 0
# for a, b, c, d:
r = solve([y-x,y-1,yp-1,yp],[a,b,c,d])

#damn - that's wrong
"""




#x = Symbol('x')
#t = Symbol('t')         # threshold

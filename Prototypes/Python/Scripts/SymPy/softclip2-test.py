"""
We consider a 4th order polynomial:
p(x) = k*x + p2*x^2 + p3*x^3 + p4*x^4
By construction, we have: p(0)=0, p'(0)=k
We require: p(1)=1, p'(1)=0, p monotonic on the interval 0...1

...

Let's choose some particular value of p2...

We get the system of 2 equations in 2 unknwowns:
p (1)=1: b0**2+(a0-b0+2*b0*b1)/2+(b1*b1+2*a0*a1-2*b0*b1)/3+(a1*a1-b1*b1)/4-1=0
p'(1)=0: b0**2+(a0-b0+2*b0*b1)  +(b1*b1+2*a0*a1-2*b0*b1)  +(a1*a1-b1*b1)    =0


the system is solved using sympy.

...maybe we need a 3rd equation
"""

from sympy import *

# declare symbols:
b0, b1, k = symbols('b0 b1 k')

# define equations to be solved:
e1=k+(b1**2+2*(b0-2*b0*b1)*(sqrt(b1**2-(3-2*k)*4))-2*b0*b1)+((sqrt(b1**2-(3-2*k)*4))**2-b1**2)

# solve the set of equations (enter r at the prompt to see it):
r = solve([e1],[b1])
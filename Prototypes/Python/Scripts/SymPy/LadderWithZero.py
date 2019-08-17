# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 17:32:26 2016

@author: Rob
maybe declare b0, b1 nonnegative
"""

from sympy import *
b0,b1,s,c,t = symbols('b0 b1 s c t');

# define equations to be solved:
e1 = (2-2*c)*(b0+b1-b0*b1-1) + b0**2 + b1**2
e2 = t*( b0 + (b0+b1-1)*b1 + (b1+(b0+b1-1)*b0)*c ) - s*((b0+b1-1)*b0-b1)

# solve the set of equations:
r = solve([e1,e2],[b0,b1])
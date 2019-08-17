# -*- coding: utf-8 -*-

def wallis_pi(N):
    num = 1
    den = 1
    for j in range(1, N):
        num *= 4*j*j
        den *= 4*j*j-1
    return 2 * num/den
    
    
pi = wallis_pi(10000)
print(pi)


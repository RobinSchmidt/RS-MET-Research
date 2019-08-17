︠565a870f-9cb6-48ca-a36e-dcad98408ea1s︠
reset()
from PolySymb import * # operations on lists of symbols/expressions

var("a b c d p q r")   # symbolic polynomial coefficients
P = [d,c,b,a]          # list of coeffs of polynomial P
Q = [r,q,p]            # list of coeffs of polynomial Q
PQ = polymul(P,Q)      # product of PQ via convolving lists
PQ
︡23d5d196-5910-4812-92ef-621b7a71d51b︡{"stdout":"(a, b, c, d, p, q, r)\n"}︡{"stdout":"[d*r, d*q + c*r, d*p + c*q + b*r, c*p + b*q + a*r, b*p + a*q, a*p]\n"}︡{"done":true}︡
︠435d5d4b-8484-4eb9-9cec-c00a0fb3a2cf︠
S = polyadd(P,Q)
S
︡919db203-82fa-459f-9c08-9eb7eb6d62e4︡{"stdout":"[d + r, c + q, b + p, a]\n"}︡{"done":true}︡
︠0f2fc10b-c8f2-47a9-acb7-9b4fc59db944︠
T = polymul([2,3,-1,2],[1,1,-2])
T = polyadd(T,[3,4,-2])
T
︡a5e32db7-e65e-40ac-a485-883d2889a840︡{"stdout":"[5, 9, -4, -5, 4, -4]\n"}︡{"done":true}︡
︠c5d4b3f3-dfcd-49f3-93c1-e745b64b313es︠
D = [2,-1,5,7,-3,2]
Q = [2,-3,6,2]
R = [3,1,4,-5,3]   # seems to work, if R has 0 or 1 elements, fails with 2 or more
P = polymul(Q,D)
P
P = polyadd(P,R)
P
polydiv(P,D)
#polydiv([7,-7,29,-8,4,65,-10,6,4],[2,-1,5,7,-3,2])
︡6efa239d-dfac-4c95-a17f-cf62963e9cdb︡{"stdout":"[4, -8, 25, -3, 1, 65, -10, 6, 4]\n"}︡{"stdout":"[7, -7, 29, -8, 4, 65, -10, 6, 4]\n"}︡{"stdout":"([2, -3, 6, 2, 0, 0, 0, 0, 0], [3, 1, 4, -5, 3, 0, 0, 0, 0])\n"}︡{"done":true}︡
︠c1d80832-feac-4e08-83a4-66b7bed460cb︠
polymul([1,2,-2,3,-1,-3,-5,4,-3,-1],[2,-3,1,2,-1])
# {2,1,-9,16,-10,-6,6,15,-28,4,13,-11,1,1}
︡0f2d8677-6f12-45ae-87dd-1f3446131ef2︡{"stdout":"[2, 1, -9, 16, -10, -6, 6, 15, -28, 4, 13, -11, 1, 1]\n"}︡{"done":true}︡
︠cf0f59d6-f4b2-4696-8166-1242c39a2f29︠
polyadd([2,-3,1,2,-1],[1,2,-2,3,-1,-3,-5,4,-3,-1])
︡733f6a05-2742-46b5-a0b5-4875bff7e8c1︡{"stdout":"[3, -1, -1, 5, -2, -3, -5, 4, -3, -1]\n"}︡{"done":true}︡
︠90ea8692-d860-4b34-af28-31244fff5999︠










︠e6a96d13-5056-4734-8c95-3fb0a28598b8s︠
%typeset_mode True
# Numerical approximation of pi:
pi.n(digits=30) # alternative: N(pi, digits=30)

︡3f68b66f-780d-462d-8641-dcb91815ff44︡{"html":"<div align='center'>$\\displaystyle 3.14159265358979323846264338328$</div>"}︡{"done":true}︡
︠f39f8945-fca4-4117-9f15-902a461afbf6s︠
# solve equation y = f(x) for x:
_ = var("a b c d x y")          # _ = to supress output
solve(y == (a*x+b)/(c*x+d), x)
︡81d66fab-cd51-4b1a-8658-dc7317e11fd1︡{"html":"<div align='center'>[$\\displaystyle x = -\\frac{d y - b}{c y - a}$]</div>"}︡{"done":true}︡
︠48e080e7-b634-4f16-b8fb-9eabfb2f91eds︠
# solve a system of 2 equations:
_ = var("a b x y p q r s t u")
f(x,y) = a*x + b*y       # define bivariate model function
eq1 = f(p,q) == r        # 1st requirement
eq2 = f(s,t) == u        # 2nd requirement
solve([eq1,eq2],[a,b])   # find model parameters a,b
︡226cba23-78b7-41a7-bb92-df801253ab3f︡{"html":"<div align='center'>[[$\\displaystyle a = -\\frac{r t - q u}{q s - p t}$, $\\displaystyle b = \\frac{r s - p u}{q s - p t}$]]</div>"}︡{"done":true}︡
︠d9373502-c923-4731-b1c1-83b9570eb644s︠
# Taylor series expansion of exponential function:
taylor(exp(x),x,0,6) # 6th order taylor polynomial of e^x around 0
︡782a374d-f22d-4351-9343-a9d7c4d2d5ae︡{"html":"<div align='center'>$\\displaystyle \\frac{1}{720} \\, x^{6} + \\frac{1}{120} \\, x^{5} + \\frac{1}{24} \\, x^{4} + \\frac{1}{6} \\, x^{3} + \\frac{1}{2} \\, x^{2} + x + 1$</div>"}︡{"done":true}︡
︠204028a9-e2a9-46d9-a5af-c44ad64c07a3s︠
# Finding coefficients for the reduced cubic equation z^3 = p*z + q:
_= var("a b c d z")             # declare symbolic variables
f(x) = a*x^3 + b*x^2 + c*x + d  # define a symbolic function
g = f.subs(x = z - b/(3*a))     # substitue z-b/(3*a) for x, removes x^2 term
h = g/a                         # divide resulting expression by a
h.collect(z)                    # collect terms with respect to z
︡0c2a0450-8322-4542-b61e-feb59838597b︡{"html":"<div align='center'>$\\displaystyle x \\ {\\mapsto}\\ z^{3} - \\frac{1}{3} \\, z {\\left(\\frac{b^{2}}{a^{2}} - \\frac{3 \\, c}{a}\\right)} + \\frac{2 \\, b^{3}}{27 \\, a^{3}} - \\frac{b c}{3 \\, a^{2}} + \\frac{d}{a}$</div>"}︡{"done":true}︡
︠d706bc8e-8f3f-4eea-8b1a-fc11cfdd484cs︠
# -p equals the z^1 coeff and -q the z^0 coeff, print coeffs only:
h = h.collect(z)                # collect again, but this time re-assign h
h.coefficients(z)               # gives coefficients for powers of z
︡8ba73e29-c8f7-41d1-8260-91e721235923︡{"html":"<div align='center'>[[$\\displaystyle x \\ {\\mapsto}\\ \\frac{2 \\, b^{3}}{27 \\, a^{3}} - \\frac{b c}{3 \\, a^{2}} + \\frac{d}{a}$, $\\displaystyle x \\ {\\mapsto}\\ 0$], [$\\displaystyle x \\ {\\mapsto}\\ -\\frac{b^{2}}{3 \\, a^{2}} + \\frac{c}{a}$, $\\displaystyle x \\ {\\mapsto}\\ 1$], [$\\displaystyle x \\ {\\mapsto}\\ 1$, $\\displaystyle x \\ {\\mapsto}\\ 3$]]</div>"}︡{"done":true}︡
︠52b57d57-986b-4792-9ed4-11da9e4f9f8bs︠
# Expansion of polynomial given in product form with symbolic roots:
reset()                         # reset all variable definitions
_= var("x k x1 x2 x3")          # x, scale factor and roots
f(x) = k*(x-x1)*(x-x2)*(x-x3)   # define polynomial in product form
fe = expand(f); fe              # expand to sum/coefficient form
︡3ccfd2ec-0365-4695-9a54-656fec61e0db︡{"html":"<div align='center'>$\\displaystyle x \\ {\\mapsto}\\ k x^{3} - k x^{2} x_{1} - k x^{2} x_{2} + k x x_{1} x_{2} - k x^{2} x_{3} + k x x_{1} x_{3} + k x x_{2} x_{3} - k x_{1} x_{2} x_{3}$</div>"}︡{"done":true}︡
︠31cae724-7a25-4c17-87e0-4491c71b9956s︠
# Factor it again, _ accesses the result of most recent computation:
#factor(_) # sometimes, this fails in typeset mode
factor(fe)
︡299a3c22-fab8-4029-a496-f1f835d95366︡{"html":"<div align='center'>$\\displaystyle x \\ {\\mapsto}\\ k {\\left(x - x_{1}\\right)} {\\left(x - x_{2}\\right)} {\\left(x - x_{3}\\right)}$</div>"}︡{"done":true}︡
︠bd3ac992-6e53-4ffe-9657-e6c3ca5fb872s︠
# Polynomial with integer coeffs instead of symbols:
reset()
f(x) = 3*(x^3 - 2*x - 4)        # define polynomial with integer coeffs
factor(f)                       # will factor it into a real and quadratic term
︡3bff847d-2e70-44fb-ba5a-f5831e07d6cf︡{"html":"<div align='center'>$\\displaystyle x \\ {\\mapsto}\\ 3 \\, {\\left(x^{2} + 2 \\, x + 2\\right)} {\\left(x - 2\\right)}$</div>"}︡{"done":true}︡
︠5c1cc485-5789-4b78-953c-99150b3663d0s︠
f.roots()                       # this actually produces the complex roots
︡caa1adb6-0814-404e-b625-a8a8990b475e︡{"html":"<div align='center'>[($\\displaystyle -i - 1$, $\\displaystyle 1$), ($\\displaystyle i - 1$, $\\displaystyle 1$), ($\\displaystyle 2$, $\\displaystyle 1$)]</div>"}︡{"done":true}︡
︠c20065ee-4926-4065-a76e-600c71335ddes︠
f.coefficient(x^3)              # returns the leading coefficient (scale factor)
︡a9a0b2f5-67f1-43a5-a773-9996deaa1cb1︡{"html":"<div align='center'>$\\displaystyle x \\ {\\mapsto}\\ 3$</div>"}︡{"done":true}︡
︠1148ddb6-ab0a-42cb-99ec-fd64a1087a5c︠










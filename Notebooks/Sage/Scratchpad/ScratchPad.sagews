︠26ffe7e8-bd3e-45bf-8f52-8369694796ffs︠
%typeset_mode True
# solve a system of 2 equations:
var("a b c x y p q r s t u")
f(x,y) = a*x^2 + b*y^2 + c*x*y  # define bivariate model function
eq1 = f(p,q) == r               # 1st requirement
eq2 = f(s,t) == u               # 2nd requirement
f,eq1,eq2                       # show function and equations
solve([eq1,eq2],[a,b])          # find model parameters a,b
︡ae75bb1e-7159-4591-bdc1-a871dae7608a︡{"html":"<div align='center'>($\\displaystyle a$, $\\displaystyle b$, $\\displaystyle c$, $\\displaystyle x$, $\\displaystyle y$, $\\displaystyle p$, $\\displaystyle q$, $\\displaystyle r$, $\\displaystyle s$, $\\displaystyle t$, $\\displaystyle u$)</div>"}︡{"html":"<div align='center'>($\\displaystyle \\left( x, y \\right) \\ {\\mapsto} \\ a x^{2} + c x y + b y^{2}$, $\\displaystyle a p^{2} + c p q + b q^{2} = r$, $\\displaystyle a s^{2} + c s t + b t^{2} = u$)</div>"}︡{"html":"<div align='center'>[[$\\displaystyle a = \\frac{c p q t^{2} - {\\left(c s t - u\\right)} q^{2} - r t^{2}}{q^{2} s^{2} - p^{2} t^{2}}$, $\\displaystyle b = -\\frac{c p q s^{2} - {\\left(c s t - u\\right)} p^{2} - r s^{2}}{q^{2} s^{2} - p^{2} t^{2}}$]]</div>"}︡{"done":true}︡
︠74cae113-50dd-431f-8b02-4b4324206b19︠
#1234567890123456789012345678901234567890123456789012345678901234567890
︠f5adad2a-ce2d-4d1b-bb41-4ea4fa4c50c8︠










︠bb046896-1f57-43e5-9fda-2d5538eff83bs︠
%typeset_mode True
%var a, b, c, d, e, t
f(t) = sqrt(c + b*t + a*t^2)
assume(a > 0)
assume(b > 0)
assume(c > 0)
assume(4*a*c-b^2 > 0)
integrate(f(t), t,0,1)
︡6b8f28c6-8715-47ed-afdc-9c060e5a425e︡{"html":"<div align='center'>$\\displaystyle -\\frac{2 \\, a b \\sqrt{c} - {\\left(b^{2} - 4 \\, a c\\right)} \\sqrt{a} \\operatorname{arsinh}\\left(\\frac{b}{\\sqrt{-b^{2} + 4 \\, a c}}\\right)}{8 \\, a^{2}} - \\frac{{\\left(b^{2} - 4 \\, a c\\right)} \\sqrt{a} \\operatorname{arsinh}\\left(\\frac{2 \\, a + b}{\\sqrt{-b^{2} + 4 \\, a c}}\\right) - 2 \\, {\\left(2 \\, a^{2} + a b\\right)} \\sqrt{a + b + c}}{8 \\, a^{2}}$</div>"}︡{"done":true}︡
︠9a3685c9-8bd3-49ed-bbdc-e6c6523b6553s︠
f(t) = sqrt(d + c*t + b*t^2 + a*t^3)
assume(d > 0)
integrate(f(t), t,0,1, algorithm='mathematica_free')
︡bde0d1ea-967a-41be-bbbe-48746f2bb891︡{"stderr":"Error in lines 1-1\n"}︡{"stderr":"Traceback (most recent call last):\n  File \"/cocalc/lib/python2.7/site-packages/smc_sagews/sage_server.py\", line 1188, in execute\n    flags=compile_flags) in namespace, locals\n  File \"\", line 1, in <module>\nNameError: name 'd' is not defined\n"}︡{"done":true}︡
︠1b186962-f32f-40d2-8a2c-43076f2b5d82︠










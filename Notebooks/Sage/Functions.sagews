︠d4e580ad-28f8-47a7-b080-71c1909882f3s︠
# trigonometric functions:
p =  plot(sin(x), 0, 20, color="blue")
p += plot(cos(x), 0, 20, color ="green")
p += plot(tan(x), 0, 20, ymin=-3, ymax=3, color ="red", detect_poles="show")
p.show()
︡afcc6b78-11e8-4f68-9252-b4e058a8f97a︡{"file":{"filename":"/home/user/.sage/temp/project-7ec714f4-e576-4527-8964-5f7122fd6d99/176/tmp_BYWA5n.svg","show":true,"text":null,"uuid":"b42f8b82-9d07-4c86-bc63-791c69a58187"},"once":false}︡{"done":true}︡
︠6b0d2136-badd-4889-91c4-dc131e3c6658s︠
# hyperbolic functions:
xmin= -2; xmax = 2
p  = plot(sinh(x), xmin, xmax, color="blue")
p += plot(cosh(x), xmin, xmax, color="green")
p += plot(tanh(x), xmin, xmax, color="red")
p += plot(exp(x),  xmin, xmax, color="black") # exp as reference
p.show()
︡d3202b09-c933-4bdb-a8be-1ba830488aaa︡{"file":{"filename":"/home/user/.sage/temp/project-7ec714f4-e576-4527-8964-5f7122fd6d99/176/tmp_IcuzUg.svg","show":true,"text":null,"uuid":"f3d5d865-63dc-41f2-9486-d42c33321d0d"},"once":false}︡{"done":true}︡
︠d76a6576-01a5-4c15-b86b-a9f7fa12db88︠










︠887b062b-3f41-403f-977c-a1191a437bfc︠
plot(sin(2*pi*x))
︡077c12ca-130d-49a3-af55-83b2cf6f0c43︡{"file":{"filename":"/home/user/.sage/temp/project-7ec714f4-e576-4527-8964-5f7122fd6d99/3345/tmp_p_NEAp.svg","show":true,"text":null,"uuid":"5051242d-fd7d-408b-938c-760911508005"},"once":false}︡{"done":true}︡
︠d57f4d36-122d-45c4-a044-36f025a786c9s︠
p = plot([sin(x),cos(x)],xmin=-10,xmax=+10,ymin=-1.2,ymax=+1.2)
show(p)
︡a43f9479-035c-4a5d-a757-a240efcb0599︡{"file":{"filename":"/home/user/.sage/temp/project-7ec714f4-e576-4527-8964-5f7122fd6d99/144/tmp_HukxFP.svg","show":true,"text":null,"uuid":"ce9925b8-ea1b-426b-a426-01d2929e4341"},"once":false}︡{"done":true}︡
︠9024ae39-3ab5-4bbe-922d-ca35821f44b6s︠
# Repeated convolutions of a square-pulse function with itself:
f = piecewise([((0,1),1)])        # define piecewise function f
f1 = f.convolution(f)             # f1 is convolution of f with itself
f2 = f1.convolution(f)            # f2 is convolution of f1 with f
f3 = f2.convolution(f)            # etc.
plot([f,f1,f2,f3],xmin=0,xmax=4)
︡c39f68d2-9142-4477-8888-144531fdf068︡{"stdout":"verbose 0 (3749: plot.py, generate_plot_points) WARNING: When plotting, failed to evaluate function at 150 points."}︡{"stdout":"\n"}︡{"stdout":"verbose 0 (3749: plot.py, generate_plot_points) Last error message: 'point 3.98030150754 is not in the domain'\n"}︡{"stdout":"verbose 0 (3749: plot.py, generate_plot_points) WARNING: When plotting, failed to evaluate function at 100 points."}︡{"stdout":"\n"}︡{"stdout":"verbose 0 (3749: plot.py, generate_plot_points) Last error message: 'point 3.98030150754 is not in the domain'\n"}︡{"stdout":"verbose 0 (3749: plot.py, generate_plot_points) WARNING: When plotting, failed to evaluate function at 51 points."}︡{"stdout":"\n"}︡{"stdout":"verbose 0 (3749: plot.py, generate_plot_points) Last error message: 'point 3.98030150754 is not in the domain'\n"}︡{"file":{"filename":"/home/user/.sage/temp/project-7ec714f4-e576-4527-8964-5f7122fd6d99/939/tmp_0bWYA5.svg","show":true,"text":null,"uuid":"f74a1d58-7e4b-4b42-b551-9420865914f1"},"once":false}︡{"done":true}︡
︠827cd8b6-530d-4a70-aa4c-41f4d231a020s︠
@interact
def f(n=[0..4], s=(1..5), c=Color("red")):
    var("x")
    show(plot(sin(n+x^s), -pi, pi, color=c))
︡94f9ef54-48c8-4eaf-b5ef-e2c5f232b5c6︡{"interact":{"controls":[{"button_classes":null,"buttons":true,"control_type":"selector","default":0,"label":"n","lbls":["0","1","2","3","4"],"ncols":null,"nrows":null,"var":"n","width":null},{"animate":true,"control_type":"slider","default":0,"display_value":true,"label":"s","vals":["1","2","3","4","5"],"var":"s","width":null},{"Color":"<class 'sage.plot.colors.Color'>","control_type":"color-selector","default":"#ff0000","hide_box":false,"label":"c","readonly":false,"var":"c","widget":null}],"flicker":false,"id":"7c9a42fa-76e6-45ef-85b4-12f0f094d2ef","layout":[[["n",12,null]],[["s",12,null]],[["c",12,null]],[["",12,null]]],"style":"None"}}︡{"done":true}︡
︠5eb452a7-13bc-47d2-be7d-62197665c286s︠
# Implicit plot of two conic sections
var("x y")
eq1 = x^2/2 + y^2*2 + x*y + y + x/4 == 2 # ellipse
eq2 = x^2   - y^2   + x*y + x + 2*y == 1 # hyperbola
G = Graphics()
G += implicit_plot(eq1, (x,-4,4),(y,-4,4)) 
G += implicit_plot(eq2, (x,-4,4),(y,-4,4)) 
plot(G)
︡910cd5a7-d7d6-42b3-a823-49c62f22ee81︡{"stdout":"(x, y)\n"}︡{"file":{"filename":"/home/user/.sage/temp/project-7ec714f4-e576-4527-8964-5f7122fd6d99/108/tmp_sqqUdI.svg","show":true,"text":null,"uuid":"3e13613f-afa3-4727-856a-c62e2d856f11"},"once":false}︡{"done":true}︡
︠3860cba5-2b97-488e-8ace-628fdad92aca︠











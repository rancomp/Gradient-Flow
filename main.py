from flow import Flow
from sympy.abc import x,y,z #Defining symbolic parameters for sympy and for the function so that I can calc the gradient at any point

#example code. Just set func as a function. the variables x,y,z are reserved for symbolic algebra.

func = 0.3*z
flow = Flow(func)
_, indices = flow.k_nns(10)
normals = flow.calc_normal_LMS(10)

print(flow.scatter[0])
print(normals[0])

for i in range(5):
    print(i)
    flow.step_flow(10)

flow.plot_scatter(True)

print(flow.scatter[0])
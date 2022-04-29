# define several functions: 
## original functions of f(x,y) and g(x,y)
def f(x, y):
    return x*x + y*y - 1
def g(x, y):
    return x*x*x - y
## partial differentiate functions of f on x and y
def df_x(x, y):
    return 2*x
def df_y(x, y):
    return 2*y
## partial differentiate functions of g on x and y
def dg_x(x, y):
    return 3*x*x
def dg_y(x, y):
    return -1

# define linear function solution: Gauss elimination method
def Gauss2D(a, b):
    # solve ax=b, a is 2x2 matrix and b is 2x1
    ## upper triangularize
    a[1][0] = a[1][0] / a[0][0]
    a[1][1] = a[1][1] - a[1][0] * a[0][1]
    b[1] = b[1] - a[1][0] * b[0]
    ## back substitution
    b[1] = b[1] / a[1][1]
    b[0] = b[0] - a[0][1] * b[1]
    b[0] = b[0] / a[0][0]
    return b

# solve non-linear functions
xy_0 = [1, 1]
xy_1 = [0.8, 0.6] # initial value
d_xy = [1, 1]
while max(abs(d_xy[0]), abs(d_xy[1])) > 1e-5:
    xy_0 = xy_1
    a = [[df_x(xy_0[0], xy_0[1]), df_y(xy_0[0], xy_0[1])],
         [dg_x(xy_0[0], xy_0[1]), dg_y(xy_0[0], xy_0[1])]]
    b = [-f(xy_0[0], xy_0[1]), -g(xy_0[0], xy_0[1])]
    d_xy = Gauss2D(a, b)
    xy_1[0] = xy_0[0] + d_xy[0]
    xy_1[1] = xy_0[1] + d_xy[1]
    print("|delta_x|, |delta_y|: ")
    print(abs(d_xy[0]), abs(d_xy[1]))
print("\nfinal result: ", xy_1)
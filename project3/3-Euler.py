# define f and g: du/dt = f, dv/dt = g
def f(u, v):
    return 0.09*u*(1-u/20) - 0.45*u*v
def g(u, v):
    return 0.06*v*(1-v/15) - 0.001*u*v

# optimized euler method, according to the problem
def euler(x0, y0, z0, x, N):
# interval [x0, x]
# partition the interval into N smaller intervals
# y0 = y(x0), z0 = z(x0)
# should solve (y1,z1),...,(yN,zN) in [x0, x]
    h = (x - x0) / N # length of the smaller interval
    for i in range(1, N + 1) :
        delta_y1 = y0 + h * f(y0, z0)
        delta_z1 = z0 + h * g(y0, z0)
        y1 = y0 + (h/2) * (f(y0, z0) + f(delta_y1, delta_z1))
        z1 = z0 + (h/2) * (g(y0, z0) + g(delta_y1, delta_z1))
        print("k = {}, yk = {}, zk = {}".format(i, y1, z1))
        y0 = y1
        z0 = z1

# handle input and output
print("the start of interval is 0, please enter the")
print("end of interval(b) and number of partitions(N)")
print("in the format b,N :")
str = input()
splited_str = str.split(',')
b = float(splited_str[0])
N = int(splited_str[1])
euler(0, 1.6, 1.2, b, N)

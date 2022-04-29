import math

# dy/dx = f(x,y)
def f(x, y):
    return y * math.sin(x * math.pi)

def rungeKutta(x0, y0, x, n):
# interval [x0, x]
# partition the interval into n smaller intervals
# y(x0) = y0
# should solve y1,...,yn in [x0, x]
    h = (x - x0) / n # length of the smaller interval
    for i in range(1, n + 1):
        k1 = f(x0, y0)
        k2 = f(x0 + h, y0 + h * k1)
        y = y0 + 0.5 * h * (k1 + k2)
        print("k = {}, yk = {}".format(i, y))
        y0 = y

# handle input and output
print("the start of interval is 0, please enter the")
print("end of interval(b) and number of partitions(n)")
print("in the format b,n :")
str = input()
splited_str = str.split(',')
b = float(splited_str[0])
n = int(splited_str[1])
rungeKutta(0, 1, b, n)

import math
import sys
# preparation
eps = float(input("Please enter the precision: "))
line = input("Please enter the integration interval in the format a,b: ")
splited = line.split(",")
a = int(splited[0])
b = int(splited[1])
func = input("Please enter the function name(supported functions are ln(x)): ")

# exception handling
if func != "ln(x)":
    print("Invalid function name!")
    sys.exit(1)

# initialization
n = 2
h_n = (b - a) / n
T_2 = math.log((a+b)/2) + math.log(a)/2 + math.log(b)/2
T_2 *= h_n
T_1 = T_2 + 100
global point_list
point_list = [a, (a+b)/2, b]

def H_n (h_n):
    # this function is to calculate the newly inserted point and add them to the point list
    result = 0
    global point_list
    length = len(point_list)
    bound = 2 * length - 2
    i = 0
    while i < bound:
        x_l = point_list[i]
        x_r = point_list[i+1]
        x_m = (x_l + x_r) / 2
        result += math.log(x_m)
        point_list.insert(i+1, x_m)
        i += 2
    result *= h_n
    return result

# iteration
while abs(T_1 - T_2) > eps:
    T_1 = T_2
    H = H_n(h_n)
    T_2 = (T_1 + H) / 2
    h_n /= 2
    n *= 2

print("The integration result: ", T_2)
print("The division n is: ", int(n/2))
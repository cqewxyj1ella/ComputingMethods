# Problem discription: input a pair of n+1 points: (x0,y0),(x1,y1),(x2,y2)...(xn,yn); and an unknown point: given x, but y is to be estimated
# Algorithm: Lagrange Insertion
# input n points: 1920,105711 1930,123203 1940,131669 1950,150697 1960,179323 1970,203212
# n+1 th point(for estimating error): 1910,91772

# global variables
global x_list, y_list
global insert_length

# handle input
def preprocess_input_insert():
    print("please enter n poins with the format of: x1,y1 x2,y2 ... xn,yn")
    print("(out of pairs, split by blank space; in the pairs, split by comma):")
    raw_readline = input()
    # input preprocessing
    pairs = raw_readline.split(' ')
    global insert_length
    insert_length = len(pairs)  # actually there are n+1 points(x1,y1 x2,y2 ... xn,yn xn+1,yn+1)
    global x_list, y_list
    x_list = []
    y_list = []
    for pair in pairs:
        pair = pair.split(',')
        x_list.append(int(pair[0]))
        y_list.append(int(pair[1]))

# estimate for unknown point
def handle_unkown_point(x_array, y_array, length, x):
    fx = 0
    for i in range(length):
        tmp = 1
        x_i = x_array[i]
        for j in range(i):
            x_j = x_array[j]
            tmp = tmp * (x - x_j) / (x_i - x_j)
        for j in range(i+1,length,1):
            x_j = x_array[j]
            tmp = tmp * (x - x_j) / (x_i - x_j)
        y_i = y_array[i]
        fx = fx + tmp*y_i
    return fx

if __name__=="__main__":
    preprocess_input_insert()
    while True:    
        str_x = input("please enter a new point to be estimated(its x value); enter 'exit' to leave: ")
        if str_x == "exit":
            break
        x = int(str_x)
        # calculate using first n inserting points:
        first_x = []
        first_y = []
        for i in range(0,insert_length-1,1):
            first_x.append(x_list[i])
            first_y.append(y_list[i])
        first_result = handle_unkown_point(x_list, y_list, insert_length-1, x)
        print("first result is: {}".format(first_result))
        # calculate using last n inserting points:
        last_x = []
        last_y = []
        for i in range(1,insert_length,1):
            last_x.append(x_list[i])
            last_y.append(y_list[i])
        last_result = handle_unkown_point(last_x, last_y, insert_length-1, x)
        print("last result is: {}".format(last_result))
        # estimating the error
        error = (x - x_list[0]) / (x_list[0] - x_list[insert_length-1])
        error = error * (first_result - last_result)
        print("the absolute error is: {}".format(error))
        relative_error = error / first_result
        print("the relative error is: {}".format(relative_error))


# Problem discription: input a pair of n+1 points: (x0,y0),(x1,y1),(x2,y2)...(xn,yn); and an unknown point: given x, but y is to be estimated
# Algorithm: Newton Insertion
# input n points: 1920,105711 1930,123203 1940,131669 1950,150697 1960,179323 1970,203212

# global variables
global x_list, y_list, g_list
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

# create difference quotient table from n pairs of points
def difference_table(x_list, y_list, length):
    # initialize
    global g_list
    g_list = []
    for i in range(length):
        g_list.append(y_list[i])
    # calculate difference quotient for n points
    for k in range(1,length,1):
        for j in range(length-1,k-1,-1):
            g_list[j] = (g_list[j] - g_list[j-1]) / (x_list[j] - x_list[j-k])

# estimate for unkown point
def handle_unkown_point(u, length):
    # initialize
    t = 1
    newton = g_list[0]
    for k in range(1,length,1):
        t = t * (u - x_list[k-1])
        newton = newton + t * g_list[k]
    return newton

if __name__=="__main__":
    preprocess_input_insert()
    difference_table(x_list, y_list, insert_length)
    while True:    
        str_x = input("please enter a new point to be estimated(its x value); enter 'exit' to leave: ")
        if str_x == "exit":
            break
        x = float(str_x)
        result = handle_unkown_point(x, insert_length)
        print("the result is: {}".format(result))

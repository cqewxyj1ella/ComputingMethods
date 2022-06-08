from math import pi, sin, log, pow
import numpy as np
# numpy has many methods on matrices,
# so that I don't need to write them by myself


list_jacob_error_ln = []
# stores the error (log form) in Jacob iter: 2-norm(uh-ue)


def Jacobi_Iter(A, b, ue, e=1e-10):
    # Jacobi iteration functions
    '''
    A = D + A - D, where D is diagonal of A
    x2 = R*x1 + g
    R = I - D^-1 * A
    g = D^-1 * b
    '''
    # initialization
    n = np.shape(A)[0]  # get the size of A: nxn
    I = np.matrix(np.identity(n))  # get an identity matrix
    D = np.matrix(np.zeros((n, n)))
    # get content of D
    for i in range(n):
        D[i, i] = A[i, i]  # D is the diagonal matrix
    # calculate iteration matrix
    R = I - (D.getI() * A)
    g = D.getI() * b
    x1 = np.matrix(np.zeros((n, 1)))
    x2 = np.matrix(np.ones((n, 1)))
    # x1 and x2 are the iteration variable
    iter_times = 0
    while abs(np.max(x1-x2)) > e:
        # np.max can give infinity norm of a vector (max element of it)
        x1 = x2
        x2 = R * x1 + g
        iter_times = iter_times + 1
    # after iteration, converged
    error = np.linalg.norm(x2 - ue)
    list_jacob_error_ln.append(log(error))
    print(f"Jacobi iteration: {iter_times}, error: {error}")


list_gauss_error_ln = []
# stores the error (log form) in Gauss-Seidel iter: 2-norm(uh-ue)


def Gauss_Seidel_Iter(A, b, ue, e=1e-10):
    # Gauss-Seidel iteration function
    '''
    A = D + L + U, where D is diagonal of A, 
    L is down triangle of A, U is up triangle of A
    x2 = S*x1 + f
    S = -(D + L)^-1 * U
    f = (D + L)^-1 * b
    '''
    # initialization
    n = np.shape(A)[0]  # get the size of A: nxn
    D = np.matrix(np.zeros((n, n)))
    L = np.matrix(np.zeros((n, n)))
    U = np.matrix(np.zeros((n, n)))
    # get content of D, L, U
    for i in range(n):
        for j in range(n):
            if (i == j):  # diagonal
                D[i, j] = A[i, j]
            elif (i < j):  # up triangle
                U[i, j] = A[i, j]
            else:  # i > j, down triangle
                L[i, j] = A[i, j]
    # calculate iteration matrix
    DL_inverse = (D + L).getI()
    S = - (DL_inverse * U)
    f = DL_inverse * b
    x1 = np.matrix(np.zeros((n, 1)))
    x2 = np.matrix(np.ones((n, 1)))
    # x1 and x2 are the iteration variable
    iter_times = 0
    while abs(np.max(x1-x2)) > e:
        # np.max can give infinity norm of a vector (max element of it)
        x1 = x2
        x2 = S * x1 + f
        iter_times = iter_times + 1
    # after iteration, converged
    error = np.linalg.norm(x2 - ue)
    list_gauss_error_ln.append(log(error))
    print(f"Gauss iteration: {iter_times}, error: {error}")
    return 0


list_h_ln = []
# stores every h (in log form)


def Q2():
    # Problem 2
    def f(x):
        return pi*pi*sin(pi*x)

    def u_precise(x):
        # precise solution of u(x)
        return sin(pi*x)
    for n in [10, 20, 40, 80, 160]:
        print("n = ", n)
        # initialization
        A = np.matrix(np.zeros((n-1, n-1)))
        # A is a matrix with every element to be 0, shape of (n-1)x(n-1)
        b = np.matrix(np.zeros((n-1, 1)))
        ue = np.matrix(np.zeros((n-1, 1)))
        h = 1/n
        list_h_ln.append(log(h))
        for i in range(n-1):
            if i == 0:  # the first row
                A[0, 1] = -1/h/h
            elif i == n-2:  # the last row
                A[i, i-1] = -1/h/h
            else:  # rows in the middle
                A[i, i-1] = -1/h/h
                A[i, i+1] = -1/h/h
            A[i, i] = 2/h/h
            b[i, 0] = f((i+1)*h)
            ue[i, 0] = u_precise((i+1)*h)
        # iteration
        Jacobi_Iter(A, b, ue, 1e-10)
        Gauss_Seidel_Iter(A, b, ue, 1e-10)


def Least_Squares_Method(x: list, y: list):
    '''
    use ln() to transform h and eh into x and y 
    use the formulus on page 50
    '''
    # some preparation
    m = len(x)
    x_array = np.array(x)
    y_array = np.array(y)
    x_sum = np.sum(x_array)
    y_sum = np.sum(y_array)
    x_sqrt_sum = sum([x_*x_ for x_ in x])  # \sum x^2
    x_y_sum = sum([x[i]*y[i] for i in range(len(x))])  # \sum x*y
    x_sum_sqrt = pow(x_sum, 2)
    # calculate c0, c1
    denominator = m * x_sqrt_sum - x_sum_sqrt
    c0 = (x_sqrt_sum * y_sum - x_sum * x_y_sum) / denominator
    c1 = (m * x_y_sum - x_sum * y_sum) / denominator
    return c0, c1


def Q3():
    '''
    call least squares method to calculate for 
    Jacobi and Gauss-Seidel iteration separately
    '''
    c0_J, c1_J = Least_Squares_Method(list_h_ln, list_jacob_error_ln)
    print(f"Jacobi: c0 = {c0_J}, c1 = {c1_J}")
    c0_G, c1_G = Least_Squares_Method(list_h_ln, list_gauss_error_ln)
    print(f"Gauss-Seidel: c0 = {c0_G}, c1 = {c1_G}")


list_newton_error_ln = []
# stores the error (log form) in Newton iter: 2-norm(uh-ue)


def Newton_Iter(e=1e-8):
    def f(x):
        return pi*pi*sin(pi*x) + pow(sin(pi*x), 3)

    def u_precise(x):
        # precise solution of u(x)
        return sin(pi*x)

    def Create_G(X, h):
        # given last time X, create G(X) array
        n = len(X)
        G = np.matrix(np.zeros((n, 1)))
        for i in range(n):
            G[i, 0] = 2 * X[i, 0] + pow(X[i, 0], 3) * pow(h, 2) \
                - f((i+1)*h) * pow(h, 2)
            if (i != 0):
                G[i, 0] -= X[i-1, 0]
            if (i != n-1):
                G[i, 0] -= X[i+1, 0]
        return G

    list_n_ln = []  # stores every n (in log form)
    for n in [10, 20, 40, 80, 160]:
        print("n = ", n)
        list_n_ln.append(log(n))
        # initialize Jacobi matrix and u_precise
        h = 1/n
        J = np.matrix(np.zeros((n-1, n-1)))
        ue = np.matrix(np.zeros((n-1, 1)))
        for i in range(n-1):
            if (i != 0):
                J[i, i-1] = -1
            if (i != n-2):
                J[i, i+1] = -1
            J[i, i] = 2
            ue[i, 0] = u_precise((i+1)*h)
        # x1 and x2 are the iteration variable
        X = np.matrix(np.zeros((n-1, 1)))
        dx = np.matrix(np.ones((n-1, 1)))
        iter_times = 0
        while abs(np.max(dx)) > e:
            # update Jacobi matrix
            for i in range(n-1):
                J[i, i] = 2 + 3*pow(X[i, 0]*h, 2)
            G = Create_G(X, h)
            # solve function J*dx=-G(X)
            dx = -J.getI() * G
            X += dx
            iter_times += 1
        # after iteration, converged
        error = np.linalg.norm(X - ue)
        list_newton_error_ln.append(log(error))
        print(f"Newton iteration: {iter_times}, error: {error}")

    # least squares method -> convergence order
    c0_N, c1_N = Least_Squares_Method(list_n_ln, list_newton_error_ln)
    print(f"Newton: c0 = {c0_N}, c1 = {c1_N}")


def Q6():
    Newton_Iter(1e-8)


if __name__ == "__main__":
    # Q2()
    # Q3()
    Q6()

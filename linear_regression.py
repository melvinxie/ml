"""Gradient descent for linear regression with one variable."""

from scipy.misc import derivative

def hypothesis(t0, t1, x):
    return t0 * x + t1

def J(t0, t1, x, y):
    """Cost function."""
    m = len(x)
    return sum((hypothesis(t0, t1, x[i]) - y[i])**2 for i in range(m)) / 2 / m

def gradient_descent(t00, t10, x, y):
    E = 1e-6
    m = len(x)
    a = 0.1
    t0 = t00
    t1 = t10
    while True:
        def J1(theta0):
            return J(theta0, t1, x, y)
        def J2(theta1):
            return J(t0, theta1, x, y)
        new_t0 = t0 - a * derivative(J1, t0, dx=E)
        new_t1 = t1 - a * derivative(J2, t1, dx=E)
        if abs(new_t0 - t0) < E and abs(new_t1 - t1) < E:
          break
        t0, t1 = new_t0, new_t1
    return t0, t1

x = [0, 1, 2, 3]
y = [1, 3, 5, 7]
print gradient_descent(1, 0, x, y)  # t0, t1 converge to 2, 1

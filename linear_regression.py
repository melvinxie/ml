"""Gradient descent for linear regression."""

from copy import copy
from numpy import dot
from scipy.misc import derivative

def h(t, x):
    return dot(t, x)

def J(t, x, y):
    """Cost function."""
    m = len(x)
    return sum((h(t, x[i]) - y[i])**2 for i in range(m)) / 2 / m

def gradient_descent(t0, x , y):
    E = 1e-6
    m = len(x)
    a = 0.1
    t = t0
    while True:
        done = True
        new_t = []
        for i in range(len(t)):
            def Ji(theta):
                t_current = copy(t)
                t_current[i] = theta
                return J(t_current, x, y)
            new_t.append(t[i] - a * derivative(Ji, t[i], dx=E))
            if abs(new_t[i] - t[i]) > E:
                done = False
        if done:
            break
        t = new_t
    return t

t0 = [0, 1, 2]
x = [[1, 0, 0], [1, 2, 1], [1, 3, 2], [1, 4, 3]]
y = [1, 6, 9, 12]  # y = 1 + 2 * x1 + x2
print gradient_descent(t0, x, y)  # t converges to [1, 2, 1]

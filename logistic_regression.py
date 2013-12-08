"""Logistic regression."""

from gradient_descent import gradient_descent
from math import exp, log
from numpy import dot

def h(t, x):
    """Hypothesis sigmoid function."""
    return 1 / (1 + exp(-dot(t, x)))

def J(t, x, y):
    """Simplified cost function."""
    m = len(x)
    return -sum(y[i] * log(h(t, x[i])) + (1 - y[i]) * log(1 - h(t, x[i]))
                for i in range(m)) / m

x = [[1, -2], [1, -1], [1, 1], [1, 2]]
y = [0.12, 0.27, 0.73, 0.88]
# y = [0, 0, 1, 1]
t0 = [1, 2]
print gradient_descent(h, x, y, t0)  # Not working

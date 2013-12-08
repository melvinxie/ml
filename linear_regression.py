"""Linear regression."""

from gradient_descent import gradient_descent
from numpy import dot

def h(t, x):
    return dot(t, x)

def J(t, x, y):
    """Cost function."""
    m = len(x)
    return sum((h(t, x[i]) - y[i])**2 for i in range(m)) / 2 / m

x = [[1, 0, 0], [1, 2, 1], [1, 3, 2], [1, 4, 3]]
y = [1, 6, 9, 12]  # y = 1 + 2 * x1 + x2
t0 = [0, 1, 2]
print gradient_descent(h, x, y, t0)  # t converges to [1, 2, 1]

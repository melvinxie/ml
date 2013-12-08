"""Gradient descent for linear regression and logistic regression."""

def gradient_descent(h, x , y, t0):
    E = 1e-6
    m = len(x)
    a = 0.1
    t = t0
    while True:
        done = True
        new_t = []
        for j in range(len(t)):
            # Derivative of the cost function J for t[j].
            d = sum((h(t, x[i]) - y[i]) * x[i][j] for i in range(m)) / m
            new_t.append(t[j] - a * d)
            if abs(new_t[j] - t[j]) > E:
                done = False
        if done:
            break
        t = new_t
    return t

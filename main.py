#GEO5017 Assignment 1: Linear Regression
#Dimitris Mantas 5836670
#Maria Luisa Tarozzo Kawasaki 5620341
#Dennis Lagendijk 4587693

import numpy as np

# Observations
P = np.array([[+2.00, +0.00, +1.00],
              [+1.08, +1.68, +2.38],
              [-0.83, +1.82, +2.49],
              [-1.97, +0.28, +2.15],
              [-1.31, -1.51, +2.59],
              [+0.57, -1.91, +4.32]], dtype=float)

# Question 1. Plot the trajectory through these data points with your tool of choice.
#TODO Insert the code we used for plotting

# Question 2. Linear Regression Exercises

# Time constant speed
T1 = np.array([[x ** n for n in range(2)] for x in [1, 2, 3, 4, 5, 6]]).reshape(len(P), -1)

# Time constant acc
T2 = np.array([[x ** n for n in range(3)] for x in [1, 2, 3, 4, 5, 6]]).reshape(len(P), -1)

W_cons_speed = np.ones((2, 3))

W_cons_acc = np.ones((3, 3))

def gradient(objective_arguments, time):
    # dE/dW = 2X(XW-T) using the chain rule, but the dimensions of X and (XW-T) are not compatible, so X is transposed.
    return 2 * time.T @ (time @ objective_arguments - P)

def gradient_descent(objective_arguments, objective_gradient, time, learning_rate=1e-5, max_num_iterations=1e+10,
                     step_eps=1e-15):
    for i in range(int(max_num_iterations)):
        step = learning_rate * objective_gradient(objective_arguments, time)
        if np.amax(np.abs(step)) < step_eps:
            break
        objective_arguments -= step
    return objective_arguments

model_weights_cons_speed = gradient_descent(W_cons_speed, gradient, T1)

model_weights_cons_acc = gradient_descent(W_cons_acc, gradient, T2)

# Compute residual error
def res_error(time,weight):
    return np.sum((P - time @ weight) ** 2)

res_error_cons_speed = res_error(T1,W_cons_speed)

# Lower error compared to constant speed
res_error_cons_acc = res_error(T2,W_cons_acc)

print("Model Weights for costant speed: \n", model_weights_cons_speed)
print("Residual Error for costant speed: \n", res_error_cons_speed)

print("Model Weights for costant acceleration: \n", model_weights_cons_acc)
print("Residual Error for costant acceleration: \n", res_error_cons_acc)

# GEO5017-Linear-Regression

## Project description

This project was developed as a solution for the assignment 1 in the Machine Learning course GEO5017 lectured at TU Delft.
The main aim of this program is to solve linear regression exercises proposed in the assignment.

## Dependencies.

The code uses the Numpy library (https://numpy.org/) and Matplotlib (https://matplotlib.org/). Install by:

`pip install -r requirements.txt`

## How to use this code

This program plot the trajectory and computes both the weight model and residual error for a set of known coordinates in the 3D space, in a constant speed and in a constant acceleration. The program works by taking a known set of coordinates stored as a numpy array.The set of points given in the assignment is called P and can be seen in the top part of the program under Observations.

For computing the mentioned values, functions for gradient, gradient descendent and residual error computation were developed:

`def gradient(objective_arguments, time):
    # dE/dW = 2X(XW-T) using the chain rule, but the dimensions of X and (XW-T) are not compatible, so X is transposed.
    return 2 * time.T @ (time @ objective_arguments - P)`
    
`def gradient_descent(objective_arguments, objective_gradient, time, learning_rate=1e-5, max_num_iterations=1e+10,
                     step_eps=1e-15):
    for i in range(int(max_num_iterations)):
        step = learning_rate * objective_gradient(objective_arguments, time)
        if np.amax(np.abs(step)) < step_eps:
            break
        objective_arguments -= step
    return objective_arguments`
    
`def res_error(time,weight):
    return np.sum((P - time @ weight) ** 2)`

The output of the program is a ploted trajectory (Question 1) and print statements of the model weights and residual error for both constant speed and constant acceleration (Question 2).

Run this code using:

`python main.py`

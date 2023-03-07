# GEO5017-Linear-Regression

## Project description

This project was developed as a solution for the assignment 1 in the Machine Learning course GEO5017 lectured at TU Delft.
The main aim of this program is to solve linear regression exercises proposed in the assignment.

## Dependencies.

The code uses the Numpy library (https://numpy.org/) and Matplotlib (https://matplotlib.org/). Install by:

`pip install -r requirements.txt`

## How to use this code

This program plot the trajectory and computes both the weight model and residual error for a set of known coordinates in the 3D space, in a constant speed and in a constant acceleration. The program works by taking a known set of coordinates stored as a numpy array.The set of points given in the assignment is called P and can be seen in the top part of the program under Observations.

For computing the mentioned values, functions for gradient, gradient descendent and residual error computation were developed.

The output of the program is a ploted trajectory (Question 1) and print statements of the model weights and residual error for both constant speed and constant acceleration (Question 2).

Run this code using:

`python main.py`

# GEO5017-Linear-Regression

## Project description

This project was developed as a solution for the assignment 1 in the Machine Learning course GEO5017 lectured at TU Delft.
The main aim of this program is to solve linear regression exercises proposed in the assignment, which is in short to create a regression model with a 3D plot of the observed trajectory of a quadcopter over time.

## Dependencies.

To run the script, the user needs to have numpy, matplotlib, and math installed. Install by:

`pip install -r requirements.txt`

## How to use this code

The script has several functions:

`build_independent_variable_matrix`: Constructs the independent variable matrix of the regression model based on its desired order.

`build_model_weights`: Constructs an initial weight matrix of the regression model based on its desired order.

`make_plot`: Creates a 3D plot of the data represented by the dependent and independent variable matrices and save it to a file.

In addition to these functions, the script also sets various parameters for the matplotlib plotting environment, including the plotting style, paper size, font size, marker size, and line width. The mpl.rcParams dictionary is used to set these parameters.

In case the user has problems with these parameters, a line in the script can be uncommented to set parameters back to default:

`mpl.rcParams.update(mpl.rcParamsDefault)`

Run this code using:

`python main.py`

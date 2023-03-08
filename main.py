import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Restore the default plotting style.
mpl.rcdefaults()

# Specify the plotting style.
mpl.rcParams["figure.constrained_layout.use"] = True
mpl.rcParams["figure.dpi"] = 600
# Specify the paper size.
paper_size = [21.0,  # W
              29.7,  # H
              ]
# Specify the left and right margins.
x_margins = [02.5,  # L
             02.5,  # R
             ]
p_width = (paper_size[0] - x_margins[0] - x_margins[1]) / 2.54
mpl.rcParams["figure.figsize"] = [p_width, p_width * paper_size[0] / paper_size[1]]
mpl.rcParams["figure.labelsize"] = 10
mpl.rcParams["figure.titlesize"] = 12
mpl.rcParams["font.size"] = 10
mpl.rcParams["legend.edgecolor"] = "#BFBFBF"
mpl.rcParams["grid.color"] = "#BFBFBF"
mpl.rcParams["legend.fancybox"] = False
mpl.rcParams["legend.fontsize"] = 8
mpl.rcParams["lines.markersize"] = (0.5 * (mpl.rcParams["figure.dpi"] / 25.4)) ** 2  # d=1 mm
mpl.rcParams["savefig.transparent"] = True
mpl.rcParams["lines.linewidth"] = 0.1 * math.sqrt(mpl.rcParams["lines.markersize"])
mpl.rcParams["scatter.marker"] = "."
# Computer Modern Serif
mpl.rcParams["font.family"] = "serif"
mpl.rcParams[
    "text.latex.preamble"] = r"\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}\usepackage{amsmath, amssymb, siunitx, upgreek}"
mpl.rcParams["text.usetex"] = True
mpl.rcParams["xtick.labelsize"] = 8
mpl.rcParams["ytick.labelsize"] = 8
mpl.rcParams['axes3d.xaxis.panecolor'] = "#FFFFFF"
mpl.rcParams['axes3d.yaxis.panecolor'] = "#FFFFFF"
mpl.rcParams['axes3d.zaxis.panecolor'] = "#FFFFFF"


def build_independent_variable_matrix(dependent_variable_matrix: np.ndarray, model_order: int) -> np.ndarray:
    """
    Constructs the independent variable matrix of the regression model based on its desired order.

    :param dependent_variable_matrix:
        The dependent variable matrix (i.e., position).
    :param model_order:
        The order of the model.
    :return:
        The independent variable matrix (i.e., time).
    """
    return np.array(
        [[x ** n for n in range(model_order + 1)] for x in range(1, len(dependent_variable_matrix) + 1)]).reshape(
        len(X), -1)


def build_model_weights(dependent_variable_matrix: np.ndarray, model_order: int) -> np.ndarray:
    """
    Constructs an initial weight matrix of the regression model based on its desired order.

    :param dependent_variable_matrix:
        The dependent variable matrix (i.e., position).
    :param model_order:
        The order of the model.
    :return:
        The weight matrix.
    """
    # The initialization values have been chosen arbitrarily.
    return np.ones((model_order + 1, dependent_variable_matrix.shape[1]))


def make_plot(filename: str, dependent_variable_matrix: np.ndarray, independent_variable_matrix: np.ndarray) -> None:
    """
    TODO
    '
    :param filename:
        The output file path.
    :param dependent_variable_matrix:
        The dependent variable matrix of the regression model (i.e., position).
    :param independent_variable_matrix:
        The dependent variable matrix of the regression model (i.e., time).
    """
    # Visualize the observed trajectory of the quadcopter over time.
    ax = plt.figure().add_subplot(projection='3d')
    # Set the camera a front isometric view.
    ax.view_init(elev=math.degrees(math.asin(1 / math.sqrt(3))), azim=-45)
    # Scatter the observed positions.
    p = ax.scatter(dependent_variable_matrix[:, 0], dependent_variable_matrix[:, 1], dependent_variable_matrix[:, 2],
                   alpha=1, c=list(range(1, len(dependent_variable_matrix) + 1)),
                   cmap=plt.get_cmap('viridis', len(independent_variable_matrix)),
                   # This parameter is required when using Axes3D.
                   s=mpl.rcParams["lines.markersize"])
    # Plot the observed trajectory.
    ax.plot(dependent_variable_matrix[:, 0], dependent_variable_matrix[:, 1], dependent_variable_matrix[:, 2],
            c="#404040")
    # Project the trajectory to the X-Y plane.
    ax.plot(dependent_variable_matrix[:, 0], dependent_variable_matrix[:, 1], c="#808080", ls="--",
            marker=mpl.rcParams["scatter.marker"],
            # NOTE - Do not change the value of this parameter!
            ms=0.08 * mpl.rcParams["lines.markersize"])
    # Disable the automatic axis scaling functionality.
    min_y, max_y = plt.ylim()
    plt.ylim(min_y, max_y)
    # Project the trajectory to the X-Z plane.
    ax.plot(dependent_variable_matrix[:, 0], np.full_like(dependent_variable_matrix[:, 0], max_y),
            dependent_variable_matrix[:, 2], c="#808080", ls="--",
            marker=mpl.rcParams["scatter.marker"],
            # NOTE - Do not change the value of this parameter!
            ms=0.08 * mpl.rcParams["lines.markersize"])
    # Set the primary axis labels.
    ax.set_xlabel(r"\textbf{X} (\si{\meter)")
    ax.set_ylabel(r"\textbf{Y} (\si{\meter)")
    ax.set_zlabel(r"\textbf{Z} (\si{\meter)")
    # Add a color bar to display time information.
    cb = plt.colorbar(p, label=r"\textbf{Time} (\si{\second)",
                      # NOTE - Do not change the value of this parameter!
                      pad=0.2,
                      # NOTE - Do not change the value of this parameter!
                      ticks=1 + (np.arange(len(independent_variable_matrix)) + 0.5) * (
                              len(independent_variable_matrix) - 1) / len(independent_variable_matrix))
    # Set the tick labels
    cb.ax.set_yticklabels(list(range(1, len(independent_variable_matrix) + 1)))
    # Hide the vertical axis ticks.
    cb.ax.axes.tick_params(length=0)
    # Add a legend.
    plt.legend(["Position", "3D Trajectory", "2D Trajectory"],
               # NOTE - Do not change the value of this parameter!
               bbox_to_anchor=(0.5, -0.2), loc="lower center", ncol=3)

    plt.savefig(filename)


def objective(dependent_variable_matrix: np.ndarray, independent_variable_matrix: np.ndarray,
              model_weights: np.ndarray) -> float:
    """
    Evaluates the objective function of the underlying optimization problem.

    :param dependent_variable_matrix:
        The dependent variable matrix of the regression model (i.e., position).
    :param independent_variable_matrix:
        The dependent variable matrix of the regression model (i.e., time).
    :param model_weights:
        The weight matrix of the regression model.
    :return:
        The value of the objective function.
    """
    # noinspection PyTypeChecker
    return np.sum((dependent_variable_matrix - independent_variable_matrix @ model_weights) ** 2)


def objective_gradient(dependent_variable_matrix: np.ndarray, independent_variable_matrix: np.ndarray,
                       model_weights: np.ndarray) -> np.ndarray:
    """
    Evaluates the gradient of the objective function of the underlying optimization problem.

    :param dependent_variable_matrix:
        The observations.
    :param independent_variable_matrix:
        The dependent variable matrix of the regression model (i.e., time).
    :param model_weights:
        The model weight matrix of the regression model.
    :return:
        The gradient of the objective function.
    """
    return 2 * independent_variable_matrix.T @ (independent_variable_matrix @ model_weights - dependent_variable_matrix)


def gradient_descent(gradient, dependent_variable_matrix: np.ndarray, independent_variable_matrix: np.ndarray,
                     model_weights: np.ndarray, learning_rate=1e-5, max_num_iterations=1e+10, step_eps=1e-15) -> None:
    """
    TODO

    :param gradient:
    :param dependent_variable_matrix:
    :param independent_variable_matrix:
    :param model_weights:
    :param learning_rate:
    :param max_num_iterations:
    :param step_eps:
    """
    for _ in range(int(max_num_iterations)):
        step = learning_rate * gradient(dependent_variable_matrix, independent_variable_matrix, model_weights)
        if np.amax(np.abs(step)) < step_eps:
            break
        model_weights -= step


# Dependent Variable Matrix
X = np.array(
    [[+2.00, +0.00, +1.00], [+1.08, +1.68, +2.38], [-0.83, +1.82, +2.49], [-1.97, +0.28, +2.15], [-1.31, -1.51, +2.59],
     [+0.57, -1.91, +4.32]])

# Linear Model
T1 = build_independent_variable_matrix(X, 1)
W1 = build_model_weights(X, 1)

make_plot("observedQuadcopterTrajectory", X, build_independent_variable_matrix(X, 1))

gradient_descent(objective_gradient, X, T1, W1)
print("Optimal Model Weights:\n", W1)
print("Residual Error:\n", objective(X, T1, W1))

print('Speed:\n', np.linalg.norm(W1[0, :]))

# Square Model
T2 = build_independent_variable_matrix(X, 2)
W2 = build_model_weights(X, 2)

gradient_descent(objective_gradient, X, T2, W2)
print("Optimal Model Weights:\n", W2)
print("Residual Error:\n", objective(X, T2, W2))

next_timestep = np.array([1, 7, 49])
print('Next Position:\n', next_timestep @ W2)

make_plot("estimatedQuadcopterTrajectory", np.vstack((T2, next_timestep)) @ W2, np.vstack((T2, next_timestep)))

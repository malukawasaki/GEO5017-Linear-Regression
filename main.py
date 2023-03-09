import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Restore the default plotting style.
mpl.rcdefaults()

# Specify the plotting style.
# Figure Settings
# DPI
mpl.rcParams["figure.dpi"] = 600
# Layout
# mpl.rcParams["figure.constrained_layout.use"]: True
# Size Constraints
# Paper Size
paper_size = [21.0,  # W (cm)
              29.7]  # H (cm)
# Margins
plr_margins = [2.5,  # L (cm)
               2.5]  # R (cm)
# Width
graph_width = (paper_size[0] - plr_margins[0] - plr_margins[1]) / 2.54
# Size
mpl.rcParams["figure.figsize"] = [graph_width, graph_width *
                                  # Paper Aspect Ratio
                                  paper_size[0] / paper_size[1]]
# Marker
mpl.rcParams["scatter.marker"] = "."
# Marker Radius
# 0.3 mm
mpl.rcParams["lines.markersize"] = (0.3 * (mpl.rcParams["figure.dpi"] / 25.4)) ** 2
# Line Width
# 0.1 mm
mpl.rcParams["lines.linewidth"] = 0.1 * math.sqrt(mpl.rcParams["lines.markersize"])
# LaTeX Compiler Settings
mpl.rcParams["text.usetex"] = True
# Font
mpl.rcParams["font.family"] = "Computer Modern Roman"
# Font Size
# Global
mpl.rcParams["font.size"] = 10
# Titles
mpl.rcParams["axes.titlesize"] = 10
# Label Ticks
mpl.rcParams["xtick.labelsize"] = 8
mpl.rcParams["ytick.labelsize"] = 8
# Preamble
mpl.rcParams["text.latex.preamble"] = "\\usepackage{siunitx}"
# Writer Settings
# Whitespace
mpl.rcParams["savefig.bbox"] = "tight"
# Alpha Channel
mpl.rcParams["savefig.transparent"] = True


# Enable this command if you encounter any graphics-related issues.
# mpl.rcParams.update(mpl.rcParamsDefault)


def build_independent_variable_matrix(dependent_variable_matrix: np.ndarray, model_order: int) -> np.ndarray:
    """
    Constructs the independent variable matrix of the regression model based on its desired order.

    Args:
        dependent_variable_matrix:
            The dependent variable matrix of the model.
        model_order:
            The order of the model.

    Returns:
        The independent variable matrix of the model.
    """
    return np.array(
        [[x ** n for n in range(model_order + 1)] for x in range(1, len(dependent_variable_matrix) + 1)]).reshape(
        len(dependent_variable_matrix), -1)


def build_weight_matrix(dependent_variable_matrix: np.ndarray, model_order: int) -> np.ndarray:
    """
    Constructs an initial, non-optimal weight matrix of the regression model based on its desired order.

    Args:
        dependent_variable_matrix:
            The dependent variable matrix of the model.
        model_order:
            The order of the model.

    Returns:
        The weight matrix of the model.
    """
    # The initialization values have been chosen arbitrarily.
    return np.ones((model_order + 1, dependent_variable_matrix.shape[1]))


def subplot(ax, dependent_variable_matrix, independent_variable_matrix):
    # Scatter the observed positions.
    p = ax.scatter(dependent_variable_matrix[:, 0], dependent_variable_matrix[:, 1], dependent_variable_matrix[:, 2],
                   alpha=1, c=list(range(1, len(dependent_variable_matrix) + 1)),
                   cmap=plt.get_cmap('viridis', len(independent_variable_matrix)),
                   # This parameter is required when using Axes3D.
                   s=mpl.rcParams["lines.markersize"])
    # Plot the observed trajectory.
    ax.plot(dependent_variable_matrix[:, 0], dependent_variable_matrix[:, 1], dependent_variable_matrix[:, 2],
            c="#000000")
    # Set the primary axis labels.
    ax.set_xlabel(r"\textbf{X} (\si{\meter)")
    ax.set_ylabel(r"\textbf{Y} (\si{\meter)")
    ax.set_zlabel(r"\textbf{Z} (\si{\meter)")
    return p


def plot_model(filename: str, dependent_variable_matrix: np.ndarray, independent_variable_matrix: np.ndarray) -> None:
    """
    Generates a plot of the regression model and saves the corresponding figure to disk.

    Args:
        filename:
            The relative or absolute output file path.
        dependent_variable_matrix:
            The dependent variable matrix of the model.
        independent_variable_matrix:
            The independent variable matrix of the model.
    """
    # Visualize the observed trajectory of the quadrocopter.
    # Create a 2x2 array of 3D subplots.
    fig, axs = plt.subplots(2, 2, subplot_kw={'projection': '3d'})
    # Specify the padding.
    # NOTE - Do not change the value of this parameter!
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    # Generate each subplot.
    for ax, i in zip(axs.flat, [0, 1, 2, 3]):
        # Set the camera to an isometric view.
        ax.view_init(azim=45 + i * 90, elev=math.degrees(math.asin(1 / math.sqrt(3))))
        # ax.set_box_aspect(aspect=None, zoom=0.9)

        sp = subplot(ax, dependent_variable_matrix, independent_variable_matrix)
        # Add a label specifying the azimuth of the camera.
        ax.set_title(r"$\alpha = {}$".format(45 + i * 90) + r"\si{\degree}")
    # Add a common color bar to display time information.
    cb = plt.colorbar(sp, ax=axs.flat, label=r"\textbf{Time} (\si{\second)",
                      # NOTE - Do not change the value of this parameter!
                      pad=0.2,
                      # NOTE - Do not change the value of this parameter!
                      ticks=1 + (np.arange(len(independent_variable_matrix)) + 0.5) * (
                              len(independent_variable_matrix) - 1) / len(independent_variable_matrix))
    # Set the tick labels
    cb.ax.set_yticklabels(list(range(1, len(independent_variable_matrix) + 1)))
    # Hide the vertical axis ticks.
    cb.ax.axes.tick_params(length=0)

    plt.savefig(filename)


def objective(dependent_variable_matrix: np.ndarray, independent_variable_matrix: np.ndarray,
              weight_matrix: np.ndarray) -> float:
    """
    Evaluates the objective function of the underlying optimization problem.

    Args:
        dependent_variable_matrix:
            The dependent variable matrix of the regression model.
        independent_variable_matrix:
            The independent variable matrix of the regression model.
        weight_matrix:
            The weight matrix of the regression model.

    Returns:
        The value of the objective function.
    """
    # noinspection PyTypeChecker
    return np.sum((dependent_variable_matrix - independent_variable_matrix @ weight_matrix) ** 2)


def objective_gradient(dependent_variable_matrix: np.ndarray, independent_variable_matrix: np.ndarray,
                       weight_matrix: np.ndarray) -> np.ndarray:
    """
    Evaluates the gradient of the objective function of the underlying optimization problem.

    Args:
        dependent_variable_matrix:
            The dependent variable matrix of the regression model.
        independent_variable_matrix:
            The independent variable matrix of the regression model.
        weight_matrix:
            The weight matrix of the regression model.

    Returns:
        The gradient of the objective function.
    """
    return 2 * independent_variable_matrix.T @ (independent_variable_matrix @ weight_matrix - dependent_variable_matrix)


def gradient_descent(gradient, dependent_variable_matrix: np.ndarray, independent_variable_matrix: np.ndarray,
                     weight_matrix: np.ndarray, learning_rate: float = 1e-5, iter_eps: float = 1e+10,
                     step_eps: float = 1e-15) -> None:
    """
    Estimates the optimal weight matrix of the regression model by solving the underlying optimization problem.

    Args:
        gradient:
            The gradient of the objective function of the problem.
        dependent_variable_matrix:
            The dependent variable matrix of the regression model.
        independent_variable_matrix:
            The independent variable matrix of the regression model.
        weight_matrix:
            The weight matrix of the regression model.
        learning_rate:
            The learning rate.
        iter_eps:
            The maximum allowed number of iterations.
        step_eps:
            The minimum allowed iteration step.

    Note:
        This function modifies the weight matrix of the regression model!
    """
    for _ in range(int(iter_eps)):
        step = learning_rate * gradient(dependent_variable_matrix, independent_variable_matrix, weight_matrix)
        if np.amax(np.abs(step)) < step_eps:
            break
        weight_matrix -= step


def main():
    # Dependent Variable Matrix
    X = np.array(
        [[+2.00, +0.00, +1.00], [+1.08, +1.68, +2.38], [-0.83, +1.82, +2.49], [-1.97, +0.28, +2.15],
         [-1.31, -1.51, +2.59],
         [+0.57, -1.91, +4.32]])

    # Linear Model - Constant Speed
    T1 = build_independent_variable_matrix(X, 1)
    W1 = build_weight_matrix(X, 1)

    plot_model("observedQuadcopterTrajectory", X, build_independent_variable_matrix(X, 1))

    gradient_descent(objective_gradient, X, T1, W1)
    print("Linear Model - Optimal Model Weights:\n", W1)
    print("Linear Model - Residual Error:\n", objective(X, T1, W1))

    print('Linear Model - Speed:\n', np.linalg.norm(W1[1, :]))

    # Quadratic Model - Constant Acceleration
    T2 = build_independent_variable_matrix(X, 2)
    W2 = build_weight_matrix(X, 2)

    gradient_descent(objective_gradient, X, T2, W2)
    print("Quadratic Model - Optimal Model Weights:\n", W2)
    print("Quadratic Model - Residual Error:\n", objective(X, T2, W2))

    next_timestep = np.array([1, 7, 49])
    print('Quadratic Model - Next Position:\n', next_timestep @ W2)

    plot_model("estimatedQuadcopterTrajectory", np.vstack((T2, next_timestep)) @ W2, np.vstack((T2, next_timestep)))


if __name__ == '__main__':
    main()

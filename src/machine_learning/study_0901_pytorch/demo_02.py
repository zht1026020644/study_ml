import numpy as np


def compute_error_for_line_given_points(b, w, points):
    """
    Computes the error between the points in the given line.

    Parameters
    ----------
    b : array_like
        Array of shape (n_points, )
    w : array_like
        Array ofparam b:
    :param w:
    :param points:
    :return:
    """
    totalError = 0
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (w * x + b)) ** 2
    return totalError / float(len(points))


def step_gradient(b_current, w_current, points, learningRate):
    """
    Performs a single step gradient descent.

    Parameters
    ----------
    b_current : array_like
        Array of shape (n_points, )
    w_current : array_like
        Array ofparam b_current:
    :param w_current:
    :param points:
    :param learningRate:
    :return:
    """
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2 / N) * (y - (w_current * x + b_current))
        w_gradient += -(2 / N) * x * (y - (w_current * x + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_w = w_current - (learningRate * w_gradient)
    return [new_b, new_w]


def gradient_descent_runner(points, starting_b, starting_w, learningRate, num_iterations):
    """
    Performs a single gradient descent.

    Parameters
    ----------
    points:
    :param starting_b:
    :param starting_w:
    :param learningRate:
    :param
    """
    b = starting_b
    w = starting_w
    for i in range(num_iterations):
        b, w = step_gradient(b, w, np.array(points), learningRate)
    return [b, w]


def run():
    points = np.random.rand(100, 2)
    learningRate = 0.0001
    init_b = 0
    init_w = 0
    num_iterations = 1000
    print("Starting gradient descent at b = {0}, w={1}, error = {2}".format(init_b, init_w,
                                                                            compute_error_for_line_given_points(init_b,
                                                                                                                init_w,
                                                                                                                points)))
    print("Running...")
    [b, w] = gradient_descent_runner(points, init_b, init_w, learningRate, num_iterations)
    print("After {0} iterations b = {1}, m={2}, error = {3}".format(num_iterations, b, w,
                                                                    compute_error_for_line_given_points(b, w, points)))


if __name__ == "__main__":
    run()

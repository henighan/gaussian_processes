from typing import Tuple
import random
import numpy as np
from matplotlib import pyplot as plt


def main(n_points: int = 300, n_frequencies: int = 3, training_set_size: int = 10):
    assert training_set_size < n_points
    x, y = generate_ground_truth_data(n_points=n_points, n_frequencies=n_frequencies)
    train_inds = np.random.choice(n_points, size=training_set_size, replace=False)
    y_train = y[train_inds]
    sigma = np.array([[radial_basis_fn(x1, x2) for x1 in x] for x2 in x])
    sigma_train_train = sigma[train_inds, :][:, train_inds]
    # Do inference at all points, including the training points
    sigma_test_test = sigma
    sigma_train_test = sigma[train_inds, :]
    sigma_test_train = sigma_train_test.transpose()
    predicted_means = sigma_test_train @ np.linalg.inv(sigma_train_train) @ y_train
    predicted_covariance = (
        sigma_test_test
        - sigma_test_train @ np.linalg.inv(sigma_train_train) @ sigma_train_test
    )
    predicted_variances = np.diag(predicted_covariance)
    generate_plot(
        x=x,
        y=y,
        train_inds=train_inds,
        predicted_means=predicted_means,
        predicted_variances=predicted_variances,
    )


def radial_basis_fn(x1, x2, sigma=1.0, l=0.06):
    """ Radial Basis Kernel """
    return (sigma ** 2) * np.exp(-((x1 - x2) ** 2) / 2 / l ** 2)


def generate_ground_truth_data(
    n_points: int = 300,
    n_frequencies: int = 3,
    frequency_range: Tuple[float, float] = (0.5, 3.0),
    amplitude_range: Tuple[float, float] = (0.1, 1.0),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates 1d curve by sampling `n_frquencies` sine curves
    :param n_points: number of points on curve
    :param n_frequencies: Number of sine curves to superimpose
    :param frequency_range: Range to sample sine frequencies from
    :param amplitude_range: Range to sample sine amplitudes from
    """
    xmin, xmax = 0.0, 1.0
    x = np.linspace(xmin, xmax, num=n_points)
    y = np.zeros_like(x)
    for _ in range(n_frequencies):
        sampled_frequency = random.uniform(*frequency_range)
        sampled_amplitude = random.uniform(*amplitude_range)
        y += sampled_amplitude * np.sin(2 * np.pi * sampled_frequency * x)
    return x, y


def generate_plot(
    x: np.ndarray,
    y: np.ndarray,
    train_inds: np.ndarray,
    predicted_means: np.ndarray,
    predicted_variances: np.ndarray,
):
    """ Plot Predictons """
    x_train, y_train = x[train_inds], y[train_inds]
    plt.plot(
        x_train, y_train, linestyle="", marker="o", label="training data", color="r"
    )
    plt.plot(x, y, label="true", color="b", linestyle="--")
    plt.fill_between(
        x,
        predicted_means - predicted_variances,
        predicted_means + predicted_variances,
        color="gray",
        alpha=0.2,
    )
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


if __name__ == "__main__":
    main()

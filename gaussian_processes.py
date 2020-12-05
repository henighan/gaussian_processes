import random
import numpy as np
from matplotlib import pyplot as plt


def radial_basis_fn(x1, x2, sigma=1.0, l=0.03):
    return (sigma ** 2) * np.exp(-((x1 - x2) ** 2) / 2 / l ** 2)


def main():
    n_frequencies = 3
    n_points_to_plot = 300
    training_set_size = 10
    assert training_set_size < n_points_to_plot
    min_frequency, max_frequency = 0.5, 3.0
    xmin, xmax = 0.0, 1.0
    x = np.linspace(xmin, xmax, num=n_points_to_plot)
    amplitude_min, amplitude_max = 0.1, 1.0
    y = np.zeros_like(x)
    for _ in range(n_frequencies):
        sampled_frequency = random.uniform(min_frequency, max_frequency)
        sampled_amplitude = random.uniform(amplitude_min, amplitude_max)
        y += sampled_amplitude * np.sin(2 * np.pi * sampled_frequency * x)

    train_inds = np.random.choice(
        n_points_to_plot, size=training_set_size, replace=False
    )
    test_inds = np.array(list(set(range(n_points_to_plot)).difference(set(train_inds))))
    x_train = x[train_inds]
    y_train = y[train_inds]
    sigma = np.array([[radial_basis_fn(x1, x2) for x1 in x] for x2 in x])
    sigma_train_train = sigma[train_inds, :][:, train_inds]
    sigma_test_test = sigma[test_inds, :][:, test_inds]
    sigma_train_test = sigma[train_inds, :][:, test_inds]
    sigma_test_train = sigma_train_test.transpose()
    predicted_means = sigma[:, train_inds] @ np.linalg.inv(sigma_train_train) @ y_train
    predicted_covariance = (
        sigma
        - sigma[:, train_inds] @ np.linalg.inv(sigma_train_train) @ sigma[train_inds, :]
    )
    predicted_sigmas = np.diag(predicted_covariance)
    plt.plot(
        x_train, y_train, linestyle="", marker="o", label="training data", color="r"
    )
    plt.plot(x, y, label="true", color="b", linestyle="--")
    # plt.plot(x, predicted_means, label="predicted")
    plt.fill_between(
        x,
        predicted_means - predicted_sigmas,
        predicted_means + predicted_sigmas,
        color="gray",
        alpha=0.2,
    )
    plt.legend()
    # plt.imshow(sigma)
    # plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()

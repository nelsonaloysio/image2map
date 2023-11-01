"""
Self-Organizing Map base class.

.. autosummary::

    SOM

.. rubric:: Classes
"""

import os.path as osp
import random
from time import time
from typing import Literal, Optional, Union

import numpy as np

from . import functional as F
from .plot import plot_neurons, plot_weights

USE_NUMPY = True

ALIAS_DISTANCE = {
    "manhattan": "l1",
    "euclidean": "l2",
}

FUNC_DECAY = {
    "exp": F.exp_decay,
    "linear": F.linear_decay,
}

FUNC_DISTANCE = {
    "l1": F.manhattan_dist,
    "l2": F.euclidean_dist,
    "chebyshev": F.chebyshev_dist,
}


class SOM():
    """
    Kohonen's Self-Organizing Map base class.

    This class serves as a base class for the Kohonen Network (Self-Organizing Map)
    implementation. It provides a structure for initializing parameters and defining
    common methods, leaving loading and processing logic to be implemented in subclasses.

    .. rubric:: Example

    Training a :class:`SOM` with images from a directory, on a grid topology of ``(10, 10)`` units:

    .. code-block:: python

        >>> from image2map import SOM, utils
        >>>
        >>> som = SOM(
        ...     k_units=100,             # Number of units (neurons) in the SOM.
        ...     k_shape=(10, 10),        # Shape of the output map.
        ...     k_dist="l2",             # Distance among units ('l1', 'l2', 'chebyshev').
        ...     n_inputs=None,           # Number of input features.
        ...     n_shape=None,            # Shape of the input data.
        ...     topology="grid",         # Topology of the SOM ('GRID', 'MESH', 'LINE', 'RING').
        ...     unit_topology="square",  # Type of tiling for the SOM ('SQUARE', 'HEX').
        ...     radius_max=None,         # Maximum radius for neighborhood function.
        ...     radius_min=0,            # Minimum radius for neighborhood function.
        ...     radius_rate=None,        # Decay function type ('exp', 'lin').
        ...     radius_decay="exp",      # Decay rate (constant).
        ...     alpha_max=0.1,           # Initial learning rate.
        ...     alpha_min=0.01,          # Final learning rate.
        ...     alpha_rate=None,         # Decay function type ('exp', 'lin').
        ...     alpha_decay="exp",       # Decay rate (constant).
        ...     phi="lap",               # Neighborhood function type ('lap', 'exp', 'sqd', 'lin').
        ...     k=1.0,                   # Scaling factor.
        ...     sigma=1.0,               # Standard deviation.
        ...     seed=42                  # Random seed.
        ... )
        >>>
        >>> X = utils.load_images("/path/to/input/images")
        >>> X_train, X_test = utils.split_ts(X, 0.8)
        >>>
        >>> # MNIST dataset
        >>> # X_train, y_train, X_test, y_test = utils.load_mnist("/path/to/input/mnist")
        >>>
        >>> X_train = utils.minmax_scale(X_train, (-1, 1))
        >>> X_test = utils.minmax_scale(X_test, (-1, 1))
        >>>
        >>> som.fit(X)
        >>> som.init_neurons()
        >>> som.init_weights(-1, 1)
        >>>
        >>> som.train(X_train, epochs=100)

    :param k_units: Number of units (neurons) in the SOM.
    :param k_shape: Shape of the output map (height, width).
    :param k_dist: Distance metric for neighbor selection ('l2', 'l1', 'chebyshev').
    :param n_inputs: Shape of the input data (height, width, channels).
    :param n_shape: Shape of the output map (height, width).
    :param topology: Topology of the SOM ('grid', 'mesh', 'line', 'ring').
    :param unit_topology: Type of tiling for the SOM ('square', 'hex').
    :param radius_max: Maximum (initial) radius for neighborhood function.
        If unset, radius is set to the maximum distance between any two neurons.
    :param radius_min: Minimum (final) radius for neighborhood function.
        Default: ``1``.
    :param radius_rate: Constant decay rate for neighborhood radius (optional).
        If unset, radius decays based on the number of epochs.
    :param radius_decay: Function for radius decay ('exp', 'linear').
    :param alpha_max: Maximum (initial) learning rate. Default: ``0.1``.
    :param alpha_min: Minimum (final) learning rate. Default: ``0.01``.
    :param alpha_rate: Constant decay rate for learning rate (optional).
        If unset, learning rate decays based on the number of epochs.
    :param alpha_decay: Function for learning rate decay ('exp', 'linear').
    :param phi: Neighborhood influence function ('lap', 'exp', 'sqd', 'linear').
    :param k: Scaling factor for exponential neighborhood function. Optional.
    :param sigma: Standard deviation of neighborhood width for Gaussian
        Laplacian function. Optional.
    :param seed: Random seed for initialization.
    """
    plot_neurons = plot_neurons
    plot_weights = plot_weights

    def __init__(
        self,
        k_units: int,
        k_shape: tuple,
        k_dist: Literal["l1", "l2", "euclidean", "manhattan", "chebyshev"] = "l2",
        n_inputs: Optional[tuple] = None,
        n_shape: Optional[tuple] = None,
        topology: Literal["grid", "mesh", "line", "ring"] = "grid",
        unit_topology: Literal["square", "hex"] = "square",
        radius_max: Optional[Union[float, int]] = None,
        radius_min: Union[float, int] = 0,
        radius_rate: Optional[float] = None,
        radius_decay: Literal["exp", "linear"] = "exp",
        alpha_max: Union[float, int] = 0.1,
        alpha_min: Union[float, int] = 0.01,
        alpha_rate: Optional[float] = None,
        alpha_decay: Literal["exp", "linear"] = "exp",
        phi: Literal["lap", "exp", "sqd", "linear"] = "lap",
        k: float = None,
        sigma: float = None,
        # k: float = 1.0,
        # sigma: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:

        self.k_units = k_units
        self.k_shape = k_shape
        self.k_dist = k_dist.lower()
        self.n_inputs = n_inputs
        self.n_shape = n_shape
        self.topology = topology.upper()
        self.unit_topology = unit_topology.upper()

        self.radius_max = radius_max
        self.radius_min = radius_min
        self.radius_rate = radius_rate
        self.radius_decay = radius_decay

        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.alpha_rate = alpha_rate
        self.alpha_decay = alpha_decay

        self.phi = phi
        self.k = k
        self.sigma = sigma
        self.seed = seed

        self.W: Optional[Union[list, np.ndarray]] = None  # Weight vectors.
        self.Y: Optional[Union[list, np.ndarray]] = None  # Neurons positions.

        # if self.topology in ("MESH", "RING"):
        #     raise NotImplementedError(f"Topology {self.topology} not implemented.")
        if self.unit_topology == "HEX":
            raise NotImplementedError(f"Unit topology {self.unit_topology} not implemented.")

        self.init_neurons()

    def __repr__(self) -> str:
        """
        Returns a string representation of the Self-Organizing Map.
        """
        return (f"SOM(n_inputs={self.n_inputs}, k_units={self.k_units}, "
                f"topology='{self.topology}', "
                f"unit_topology='{self.unit_topology}', "
                f"n_shape={self.n_shape}, "
                f"k_shape={self.k_shape})")

    def __str__(self) -> str:
        """
        Returns a string representation of the Kohonen Network.
        """
        return str(self.__repr__())

    def fit(self, X: Union[list, np.ndarray]) -> None:
        """
        Fit model parameters (inputs, shape) to data.

        :param X: Input data.
        """
        x = np.array(X[0])

        n_inputs = x.size
        n_shape = tuple(x.shape)

        if self.topology in ("GRID", "MESH"):
            dims = int(n_inputs ** (1/2))
            if dims*dims == n_inputs:
                n_shape = (dims, dims)

        self.n_inputs = n_inputs
        self.n_shape = n_shape

    def predict(self, X: list, bmu: bool = True) -> list:
        """
        Predicts the best matching unit (BMU) for each input vector.

        :param X: Input data as a list of feature vectors.
        :param bmu: Whether to return the best matching unit (BMU) or all units.
            Default: ``True``.
        """
        assert self.W is not None
        assert self.Y is not None

        units = []
        for x in X:
            dists = [F.cosine_similarity(x, self.W[j]) for j in range(len(self.W))]
            j0 = F.argmax(dists) if bmu else np.argsort(dists)[::-1]
            units.append(j0)

        return units

    def train(self, X: list, epochs: int, save_weights: Optional[str] = None) -> None:
        """
        Train network on input data.

        :param X: Input data as a list of feature vectors.
        :param epochs: Number of training epochs.
        :param save_weights: Directory to save weights (optional).
        """
        if self.W is None:
            self.init_weights()

        if self.Y is None:
            self.init_neurons()

        if self.radius_max is None:
            self.radius_max = self.k_units
            if self.topology in ("GRID", "MESH"):
                self.radius_max = int(self.k_units ** .5)

        self.epoch = getattr(self, "epoch", 0)
        self.radius = getattr(self, "radius", self.radius_max)
        self.alpha = getattr(self, "alpha", self.alpha_max)

        if epochs:
            epochs += self.epoch

        if save_weights and osp.isdir(save_weights):
            path = osp.join(save_weights, f"weights_epoch_0.npy")
            self.save_weights(path)

        t0 = time()
        while True:
            j0s = []
            for i in range(len(X)):
                dists = [F.cosine_similarity(X[i], self.W[j]) for j in range(len(self.W))]
                j0 = F.argmax(dists)
                j0s.append(j0)

            # NOTE: uncomment to update weights after training set iteration
            # for i in range(len(X)):
                j0 = j0s[i]
                self.update_weights(X[i], j0, lr=self.alpha, phi=1)
                neighbors, dists = self.unit_dists(j0, r=self.radius)

                for j, dist in zip(neighbors, dists):
                    # NOTE: self.radius (dynamic) != self.radius_max (constant)
                    phi = F.phi(dist, self.radius,
                                k=self.k, sigma=self.sigma, decay=self.phi)
                    self.update_weights(X[i], j, lr=self.alpha, phi=phi)

            print("Epoch {}/{} (r={}, a={})".format(
                self.epoch+1,
                epochs or 'inf',
                f"{self.radius:.3f}",
                f"{self.alpha:.3f}"
            ), end="\r")

            at = self.epoch if self.alpha_rate else (self.epoch/epochs)
            rt = self.epoch if self.radius_rate else (self.epoch/epochs)

            self.alpha = F.decay(at, self.alpha_rate, self.alpha_max, self.alpha_min,
                                 decay=self.alpha_decay)

            self.radius = F.decay(rt, self.radius_rate, self.radius_max, self.radius_min,
                                  decay=self.radius_decay)

            if save_weights:
                path = save_weights
                if osp.isdir(path):
                    path = osp.join(path, f"weights_epoch_{self.epoch}.npy")
                self.save_weights(path)

            self.epoch += 1
            if (epochs and self.epoch == epochs)\
            or (self.alpha == self.alpha_min and self.radius == self.radius_min):
                break

        print(f"\nFinished training in {time()-t0:.3f}s.")

    def init_neurons(self) -> None:
        """
        Initializes neuron positions.
        """
        assert self.k_units
        assert self.topology in ("LINE", "RING", "GRID", "MESH")
        assert type(self.k_shape) == tuple or self.k_shape == self.k_units

        if self.topology in ("LINE", "RING"):
            Y = np.arange(self.k_shape)
        elif self.topology in ("GRID", "MESH"):
            num = int(self.k_units**0.5)
            assert num == self.k_shape[0] == self.k_shape[1]
            Y = np.array(np.meshgrid(*[np.linspace(0, d-1, num=num) for d in (self.k_shape)]))
            Y = Y.T.reshape(-1, len(self.k_shape))

        assert len(Y) == self.k_units

        self.Y = Y

        if not USE_NUMPY:
            self.Y = self.Y.tolist()

    def init_weights(self, wmin: float = -1, wmax: float = 1) -> None:
        """
        Initializes neuron weights.

        :param wmin: Minimum weight value. Default: ``-1``.
        :param wmax: Maximum weight value. Default: ``1``.
        """
        assert self.n_inputs
        assert self.topology in ("LINE", "RING", "GRID", "MESH")
        assert type(self.k_shape) == tuple or self.k_shape == self.k_units

        if USE_NUMPY:
            self.W = np.random.uniform(
                wmin,
                wmax,
                (self.k_units, self.n_inputs)
            )
        else:
            self.W = [
                [random.uniform(wmin, wmax) for _ in range(self.n_inputs)]
                for _ in range(self.k_units)
            ]

    def unit_dists(self, j0: int, r: Optional[float] = None) -> Union[list, tuple]:
        """
        Returns the distances from :math:`j_0` to all other neurons.

        If radius :math:`r` is defined, returns a tuple of (units, distances).

        :param j0: Index of the neuron.
        :param r: Radius for neighborhood. Optional.
        """
        assert j0 < self.k_units, f"Expected j0 < {self.k_units}, got {j0}."

        func = FUNC_DISTANCE[self.k_dist]

        if self.topology in ("GRID", "MESH"):
            dists = [func(self.Y[j0], self.Y[j]) for j in range(self.k_units)]
        elif self.topology in ("LINE", "RING"):
            dists = [abs(self.Y[j0] - self.Y[j]) for j in range(self.k_units)]

        if r is not None:
            units = []
            for j, dist in enumerate(dists):
                if j != j0 and dist <= r:
                    units.append(j)
            dists = [dists[j] for j in units]
            return (np.array(units), np.array(dists)) if USE_NUMPY else (units, dists)

        return np.array(dists) if USE_NUMPY else dists

    def update_weights(self, x: list, j: int, lr: int, phi: float = 1.0) -> None:
        """
        Updates the weights of a specific unit in the SOM.

        .. math::

            \\Delta w_j = \\phi \\cdot \\alpha \\cdot (x - w_j),

        where :math:`x` is the input vector,
        :math:`w_j` is the weight vector of unit :math:`j`,
        :math:`\\alpha` is the learning rate,
        and :math:`\\phi` is the Neighbor influence.

        :param x: Input vector.
        :param j: Index of the unit to update.
        :param lr: Learning rate :math:`\\alpha`.
        :param phi: Neighbor influence :math:`\\phi`.
        """
        if USE_NUMPY:
            delta_w = (x - self.W[j]) * lr * phi
            self.W[j] += delta_w
        else:
            delta_w = [(x[i] - self.W[j][i]) * lr * phi for i in range(len(x))]
            self.W[j] = [self.W[j][i] + delta_w[i] for i in range(len(delta_w))]

    def load_weights(self, path: str) -> None:
        """
        Loads the weights from a file.

        :param path: Path to load the weights from.
        """
        self.W = np.load(path)
        if not USE_NUMPY:
            self.W = self.W.tolist()

    def save_weights(self, path: str) -> None:
        """
        Saves the weights to a file.

        :param path: Path to save the weights.
        """
        np.save(path, self.W)

    def reset_alpha(self):
        """
        Resets current learning rate value.
        """
        self.__dict__.pop("alpha", None)

    def reset_epoch(self):
        """
        Resets current epoch value.
        """
        self.__dict__.pop("epoch", None)

    def reset_radius(self):
        """
        Resets current iteration radius value.
        """
        self.__dict__.pop("radius", None)

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value: Optional[int]):
        assert value is None or type(value) == int, \
            f"Expected int or None, got {type(value)}."
        np.random.seed(value)
        random.seed(value)
        self._seed = value

    @property
    def k_dist(self):
        return self._k_dist

    @k_dist.setter
    def k_dist(self, value):
        value = value.lower()
        value = ALIAS_DISTANCE.get(value, value)
        assert value in FUNC_DISTANCE, f"Distance '{value}' not in {list(FUNC_DISTANCE.keys())}."
        self._k_dist = value

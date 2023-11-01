"""
Utility functions, file handling, and random number generation.

.. autosummary::

    minmax_scale
    fsread
    lsdir
    load_images
    load_mnist
    split_ts
    dims
    .. randomfloat

.. rubric:: Functions
"""

import gzip
import os
import os.path as osp
import random
from typing import Any, Optional

import numpy as np
from PIL import Image

LIST_DIR_SORT_KEY = lambda x: int(x.split("_")[1].split(".")[0])
LIST_DIR_FILTER_KEY = lambda x: [f for f in x if not f.startswith(".")]
MNIST_BASE_URL = ""


def minmax_scale(x: list, scale: tuple = (-1, 1), reverse: bool = False) -> list:
    """
    Returns min-max normalized (scaled) values in the given range. Expressed by:

    .. math::

        X^{\\prime} = \\frac{x - X_{min}}{X_{max} - X_{min}}
        \\cdot (y_{max} - y_{min}) + y_{min},

    where :math:`X_{min}` and :math:`X_{max}`, are the minimum and maximum input values,
    and :math:`y_{min}` and :math:`y_{max}` are the minimum and maximum output values
    given by the range in ``scale``.

    :param x: Arrays to be normalized.
    :param scale: Range for normalization. Default: ``(-1, 1)``.
    :param reverse: If True, reverses minimum and maximum.
    """
    new_min, new_max = min(scale), max(scale)

    if reverse:
        new_min, new_max = new_max, new_min

    if type(x) == list:
        x = [x] if type(x[0]) in (int, float) else x
        assert 0 < len(dims(x)) < 3
        old_min = min([xi for i in range(len(x)) for xi in x[i]])
        old_max = max([xi for i in range(len(x)) for xi in x[i]])
        x = [[((xi - old_min) / ((old_max - old_min) or 1)) for xi in x[i]] for i in range(len(x))]
        x = [[xi * (new_max - new_min) + new_min for xi in x[i]] for i in range(len(x))]
        return x

    old_min, old_max = x.min(), x.max()
    x = ((x - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
    return x


def fsread(*files) -> np.ndarray:
    """
    Reads a stream of files and returns content as list or numpy arrays.

    :param f: Path to the file.
    """
    return np.array([
        np.array(Image.open(f).convert("L").getdata())
        for f in files
    ])


def lsdir(
    *paths,
    filter_key: Any = LIST_DIR_FILTER_KEY,
    sort_key: Optional[Any] = LIST_DIR_SORT_KEY,
) -> list[str]:
    """
    Returns a list of files in a given directory.

    :param path: Path to the directory.
    :param sort_key: Function to sort files, default is by filename.
    :param filter_key: Function to filter files, default excludes hidden files.
    """
    ls = []
    for path in paths:
        ls.extend([
            osp.join(path, f)
            for f in sorted(filter_key(os.listdir(path)), key=sort_key)
            if osp.isfile(osp.join(path, f))
        ])
    return ls


def load_images(root: str) -> np.ndarray:
    """
    Returns training and test set arrays from files in path.

    :param root: Path to the directory containing the files.
    """
    return fsread(*lsdir(root))


def load_mnist(root: str) -> tuple:
    """
    Returns MNIST dataset.

    :param root: Root directory with the dataset files.

    :note: Returns ``(train_images, train_labels, test_images, test_labels)``.
    """
    file_names = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz"
    }

    mnist = []
    for file, name in file_names.items():
        path = osp.join(root, name)

        # if not osp.isfile(path):
        #     url = f"{MNIST_BASE_URL}/{name}"
        #     print(f"Downloading: {url}...")
        #     request.urlretrieve(url, path)

        with gzip.open(path, "rb") as f:
            mnist.append(
                np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28*28)
                if file.endswith("images") else
                np.frombuffer(f.read(), np.uint8, offset=8)
            )

    return tuple(mnist)


def split_ts(X: np.ndarray, size: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Splits the input data into training and test sets.

    :param X: Input data as NumPy array.
    :param size: Proportion to use for training. Default: ``1.0``.

    :note: Returns ``(X_train, X_test)``.
    """
    assert 0 <= size <= 1, "Size must be between 0 and 1."

    if type(X) == list:
        index = list(range(len(X)))
        random.shuffle(index)
        train_size = int(len(X) * size)
        X_train = [X[i] for i in index[:train_size]]
        X_test = [X[i] for i in index[train_size:]]
        return X_train, X_test

    index = np.random.permutation(len(X))
    train_size = int(len(X) * size)
    X_train = X[index[:train_size]]
    X_test = X[index[train_size:]]
    return X_train, X_test


def dims(x: list) -> tuple:
    """
    Returns input array dimensions.

    :param x: Input array.
    """
    dims, x0 = [], x
    while "__len__" in x0.__dir__() and len(x0):
        if x0 is x0[0]:
            break
        dims.append(len(x0))
        x0 = x0[0]
    return tuple(dims)

'''
def randomfloat(max_value: float = 1.0, min_value: float = 0.0, chunksize: int = 1) -> float:
    """
    Returns a random float value.

    Reads ``chunksize`` bytes from ``/dev/urandom`` and scales them to the specified range.

    :param max_value: Maximum value of the random number.
    :param min_value: Minimum value of the random number.
    :param chunksize: Number of bytes to read from ``/dev/urandom``. Default is ``1``.
    """
    assert os.path.exists("/dev/urandom"), "/dev/urandom does not exist: is the system Unix-like?"
    with open("/dev/urandom", "rb") as f:
        r = [ord(f.read(1)) for chunk in range(chunksize)]
    return (reduce(mul, r) / (255 ** chunksize)) * (max_value - min_value) + min_value
'''

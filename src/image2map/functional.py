"""
Mathematical functions used by the neural network.

.. autosummary::

    argmax
    cosine_similarity
    dist
    euclidean_dist
    manhattan_dist
    chebyshev_dist
    phi
    lap_phi
    exp_phi
    sqd_phi
    linear_phi
    decay
    exp_decay
    linear_decay

.. rubric:: Functions
"""

from math import e
from typing import Union

import numpy as np

from .math import dot, norm

# e = 2.718281828459045


def argmax(values: Union[list, np.ndarray]) -> int:
    """
    Returns the index of the maximum value from a list of values,
    corresponding to the

    :param values: List of values.
    """
    if type(values) == list:
        return values.index(max(values))
    return np.argmax(values)


def cosine_similarity(u: Union[list, np.ndarray], v: Union[list, np.ndarray]) -> float:
    """
    Returns the cosine similarity between two vectors.

    Defined by the square root of the sum of squared differences between vector
    elements (their dot product or inner product) divided by the product of their
    Euclidean norms (magnitudes), in order to ensure their comparability.

    .. math::

        \\textnormal{cos}(u, v) = \\frac{\\langle u, v \\rangle}{||u|| \\cdot ||v||}

    Values closer to unity (:math:`1`) indicate higher similarity between vectors.

    :param u: Input vector.
    :param v: Weight vector.
    """
    assert len(u) == len(v)
    assert type(u) == type(v)

    if type(u) == list:
        return dot(u, v) / ((norm(u) * norm(v)) or 1)

    return np.dot(u, v) / ((np.linalg.norm(u) * np.linalg.norm(v)) or 1)


def dist(u, v, metric="euclidean") -> float:
    """
    Computes the distance between two vectors using the specified metric.

    .. image:: ../../../docs/assets/fig-dist.png
       :alt: Distance Metrics

    |

    :param u: First input vector.
    :param v: Second input vector.
    :param metric: Distance metric to use.

        - ``'l2'``: Euclidean distance, see: :func:`~euclidean_dist`

        - ``'l1'``: Manhattan distance, see: :func:`~manhattan_dist`

        - ``'chebyshev'``: Chebyshev distance, see: :func:`~chebyshev_dist`

    """
    assert metric in ("euclidean", "l2", "manhattan", "l1", "chebyshev")

    if metric in ("euclidean", "l2"):
        return euclidean_dist(u, v)
    if metric in ("manhattan", "l1"):
        return manhattan_dist(u, v)
    if metric == "chebyshev":
        return chebyshev_dist(u, v)


def euclidean_dist(u: Union[list, np.ndarray], v: Union[list, np.ndarray]) -> float:
    """
    Returns the Euclidean (L2) distance between two vectors.

    .. math::

        \\textnormal{euclidean_dist}(u, v) = \\sqrt {\\sum_{i=1}^{n}(u_i - v_i)^2}

    :param u: Input vector.
    :param v: Weight vector.
    """
    assert len(u) == len(v)
    assert type(u) == type(v)

    if type(u) == list:
        return sum((u[i] - v[i]) ** 2 for i in range(len(u))) ** 0.5

    return np.sqrt(np.power(u-v, 2).sum())


def manhattan_dist(u: Union[list, np.ndarray], v: Union[list, np.ndarray]) -> float:
    """
    Returns the Manhattan (L1) distance between two vectors.

    .. math::

        \\textnormal{manhattan_dist}(u, v) = \\sum_{i=1}^{n}(|u_i - v_i|)

    :param u: Input vector.
    :param v: Weight vector.
    """
    assert len(u) == len(v)
    assert type(u) == type(v)

    if type(u) == list:
        return sum(abs(u[i] - v[i]) for i in range(len(u)))

    return np.abs(u-v).sum()


def chebyshev_dist(u: Union[list, np.ndarray], v: Union[list, np.ndarray]) -> float:
    """
    Returns the Chebyshev distance between two vectors.

    .. math::

        \\textnormal{chebyshev_dist}(u, v) = \\textnormal{max}(|u_i - v_i|)

    :param y0: Position of the first point.
    :param y: Position of the second point.
    """
    assert len(u) == len(v)
    assert type(u) == type(v)

    if type(u) == list:
        return max(abs(u[i] - v[i]) for i in range(len(u)))

    return np.abs(u-v).max()


def phi(d: float, r: float, k: float = None, sigma: float = None, decay: str = "linear") -> float:
    """
    Returns neighbor influence factor :math:`\\phi`.

    This function is employed in the learning rule to compute the weight variation
    for neighboring neurons to the best matching unit.

    .. image:: ../../../docs/assets/fig-phi.png
       :alt: Neighborhood Influence Function

    |

    .. note::

        By default, :math:`\\sigma=\\sqrt{r}` for ``decay='lap'`` and :math:`k=r` for ``decay='exp'``.

    :param decay: Influence decay function.

        - ``'lap'``: Laplacian function, see: :func:`~lap_phi`

        - ``'exp'``: Exponential function, see: :func:`~exp_phi`

        - ``'sqd'``: Squared function, see: :func:`~sqd_phi`

        - ``'linear'``: Linear function, see: :func:`~linear_phi`

    :param d: Distance value.
    :param r: Radius value.
    :param k: Scaling factor for the ``'exp'`` function.
        Optional.
    :param sigma: Scaling factor for the ``'lap'`` function.
        Optional.
    """
    assert decay in ("lap", "exp", "sqd", "linear")

    if decay == "lap":
        return lap_phi(d, sigma=sigma or (r**.5))
    if decay == "exp":
        return exp_phi(d, r, k=(k or r))
    if decay == "sqd":
        return sqd_phi(d, r)
    if decay == "linear":
        return linear_phi(d, r)


def lap_phi(d: float, sigma: float) -> float:
    """
    Returns neighbor influence factor :math:`\\phi`.
    Employs a gaussian laplacian function:

    .. math::

        \\phi = \\text{exp} \\left( -\\frac{d^2}{2\\sigma^2} \\right),

    where :math:`\\sigma` = ``sigma`` is the standard deviation,
    :math:`d` is the distance and :math:`r` is the radius.

    :param d: Distance value.
    :param sigma: Standard deviation value.
    """
    return e ** (-(d*d) / (2*sigma))


def exp_phi(d: float, r: float, k: float) -> float:
    """
    Returns neighbor influence factor :math:`\\phi`.
    Employs an exponential function:

    .. math::

        \\phi = \\exp \\left( -k \\, \\frac{d^2}{r^2} \\right),

    where :math:`k` is a scaling factor,
    :math:`d` is the distance and :math:`r` is the radius.

    :param d: Distance value.
    :param r: Radius value.
    :param k: Scaling factor.
    """
    return e ** (-k * (d*d) / (r*r))


def sqd_phi(d: float, r: float) -> float:
    """
    Returns neighbor influence factor :math:`\\phi`.
    Employs a quadratic function:

    .. math::

        \\phi = 1 - \\frac{d^2}{r^2},

    where :math:`d` is the distance and :math:`r` is the radius.

    :param d: Distance value.
    :param r: Radius value.
    """
    return 1 - ((d*d) / (r*r))


def linear_phi(d: float, r: float) -> float:
    """
    Returns neighbor influence factor :math:`\\phi`.
    Employs a linear function:

    .. math::

        \\phi = 1 - \\frac{|d|}{r},

    where :math:`k` is a scaling factor.

    :param d: Distance value.
    :param r: Radius value.
    """
    return 1 - (abs(d) / r)


def decay(
    t: int,
    rate: float = None,
    xmax: float = None,
    xmin: float = None,
    n: float = None,
    decay: str = "linear",
) -> float:
    """
    Returns updated value based on decay function.

    .. image:: ../../../docs/assets/fig-decay.png
       :alt: Decay Functions

    |

    :param t: Time step.
        If rate is unset, decay assumes :math:`t \\in [0, 1].`
        If rate is set, decay assumes :math:`t \\in [0, T)`,
        where :math:`T` is the total number of epochs.
    :param rate: Decay rate constant :math:`\\lambda`.
    :param xmax: Maximum value.
    :param xmin: Minimum value.
    :param n: Time decay exponent, if ``rate`` is unset.
        Default: ``1``.
    :param decay: Decay function type.

        - ``'exp'``: Exponential function, see: :func:`~exp_decay`

        - ``'linear'``: Linear function, see: :func:`~linear_decay`

    """
    assert decay in ("exp", "linear")

    if rate:
        # Decay based on epoch and constant rate lambda.
        assert t >= 0
        assert 0 <= rate <= 1

        if decay == "linear":
            x = max(xmax - (xmax * rate * t), xmin)
        elif decay == "exp":
            x = max(xmax * e ** (-rate * t), xmin)

    else:
        # Decay based on epoch and total number of epochs.
        assert 0 <= t <= 1
        assert n is None or n > 0

        if decay == "linear":
            x = linear_decay(t, xmax, xmin, n=n or 1)
        elif decay == "exp":
            x = exp_decay(t, xmax, xmin, n=n or 1)

    return x


def exp_decay(t: float, xmax: float = None, xmin: float = None, n: float = 1) -> float:
    """
    Returns exponential decay factor. Expressed by:

    .. math::

        x^{\prime} = (1-t)^{n} \cdot e^{-t}.

    If :math:`x_{max}` and :math:`x_{min}` are provided, the result is normalized to the range:

    .. math::

        x^{\prime}_{\\text{norm}} = (x_{max} - x_{min}) \cdot x^{\prime} + x_{min}.

    :param t: Time in [0, 1].
    :param xmax: Maximum value. Optional.
    :param xmin: Minimum value. Optional.
    :param n: Time decay exponent. Default: ``1``.
    """
    assert 0 <= t <= 1
    if xmin is not None and xmax is not None:
        return ((xmax - xmin) * ((1-t)**n) * (e**-t)) + xmin
    return ((1-t)**n) * (e**-t)


def linear_decay(t: float, xmax: float = None, xmin: float = None, n: float = 1) -> float:
    """
    Returns linear decay factor. Expressed by:

    .. math::

        x^{\prime} = (1-t)^{n}.

    If :math:`x_{max}` and :math:`x_{min}` are provided, the result is normalized to the range:

    .. math::

        x^{\prime}_{\\text{norm}} = (x_{max} - x_{min}) \cdot x^{\prime} + x_{min}.

    :param t: Time in [0, 1].
    :param xmax: Maximum value. Optional.
    :param xmin: Minimum value. Optional.
    :param n: Time decay exponent. Default: ``1``.
    """
    assert 0 <= t <= 1
    if xmin is not None and xmax is not None:
        return ((xmax - xmin) * ((1-t)**n)) + xmin
    return (1-t)**n

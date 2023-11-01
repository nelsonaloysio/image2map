"""
Inner logic and general functions.

.. autosummary::

    dot
    norm
    cos
    exp
    sqrt
    angle

.. rubric:: Functions
"""

from math import acos, degrees, e

# e = 2.718281828459045


def dot(u: list, v: list) -> float:
    """
    Returns inner product (scalar or dot product) between two vectors. Expressed by:

    .. math::

        \\langle u, v \\rangle = \\sum_{i=1}^{n} u_i v_i.

    Or, in terms of magnitudes and the angle :math:`\\theta` formed between vectors:

    .. math::

        u \\cdot v = ||u|| \\cdot ||v|| \\cdot \\text{cos}(\\theta),
        \\quad \\text{where} \\quad
        \\text{cos}(\\theta) = \\frac{\\langle u, v \\rangle}{||u|| \\cdot ||v||}.

    .. note::

        Normalizing the input vectors (see :func:`~image2map.math.norm`)
        is recommended before computing the dot product to ensure vectors
        with different magnitudes result in a meaningful similarity score.

    :param u: First input vector.
    :param v: Second input vector.
    """
    return sum(ui * vi for ui, vi in zip(u, v))
    # return norm(u) * norm(v) * cos(u, v)


def norm(u: list) -> float:
    """
    Returns the Euclidean norm (magnitude) of a vector :math:`u`. Expressed by:

    .. math::

        ||u|| = \\sqrt{\\sum_{i=1}^{n} u_i^2}.

    :param u: First input vector.
    """
    return sum(ui ** 2 for ui in u) ** 0.5


def cos(u: list, v: list):
    """
    Returns the cosine of two vectors :math:`u` and :math:`v`. Expressed by:

    .. math::

        \\text{cos}(u, v) = \\frac{\\langle u, v \\rangle}{||u|| \\cdot ||v||}.

    :param u: First input vector.
    :param v: Second input vector.
    """
    return dot(u, v) / ((norm(u) * norm(v)) or 1)


def exp(t: float) -> float:
    """
    Returns the exponential value of :math:`t` with base set to :math:`e`.

    :param t: Exponent value.
    """
    return e ** t


def sqrt(n: float) -> float:
    """
    Returns square root of :math:`n`.
    """
    return n ** 0.5


def angle(u: list, v: list):
    """
    Returns the angle formed by vectors :math:`u` and :math:`v`. Expressed by:

    .. math::

        \\theta = \\arccos\\left(\\frac{\\langle u, v \\rangle}{||u|| \\cdot ||v||}\\right).

    :param u: First input vector.
    :param v: Second input vector.
    """
    return degrees(acos(cos(u, v)))

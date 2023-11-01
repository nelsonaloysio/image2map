"""
Kohonen Self-Organizing Map (SOM) implementation for image loading, processing, and mapping.

.. rubric:: Modules

.. autosummary::

   functional
   math
   plot
   som
   utils

For more information on each module, please refer to the respective documentation.
"""

from .__version__ import __version__
from .plot import *
from .som import SOM
from .utils import *

__all__ = ("SOM",)

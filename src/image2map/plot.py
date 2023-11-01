"""
Plotting functions for visualizing images and maps.

.. autosummary::

    plot_image
    plot_neurons
    plot_weights
    subplots
    animate
    getax

.. rubric:: Functions
"""

from math import isnan
from typing import Optional, Union

import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from .utils import minmax_scale

FIGSIZE = (5, 5)
INTERPOLATION = None
CMAP = "Greys_r"


def plot_image(
    x: list,
    title: str = "",
    reshape: bool = True,
    figsize: tuple = FIGSIZE,
    cmap: str = CMAP,
    interpolation: str = INTERPOLATION,
    fig: Optional[plt.Figure] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Returns a matplotlib figure with a plot of images.

    :param x: The input data (list of images).
    :param title: The title of the plot (optional).
    :param reshape: Whether to reshape the input data into a grid.
    :param figsize: The size of the figure (default: (5, 5)).
    :param cmap: The colormap to use for the plot (default: "Greys_r").
    :param interpolation: The interpolation method for the plot (default: None).
    :param fig: An existing figure to plot on (default: None).
    """
    if reshape is True:
        n_dims = int(len(x) ** (1/2))
        reshape = (n_dims, n_dims) if n_dims**2 <= len(x) else (1, len(x))

    if fig is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax = fig.axes

    plt.tight_layout()
    ax.axis("off")
    ax.imshow(x.reshape(reshape), cmap=cmap, interpolation=interpolation)
    ax.set_title(title)
    plt.close()
    return fig, ax


def plot_neurons(
    self,
    j0: Optional[int] = None,
    r: Optional[float] = None,
    connect_neighbors: bool = True,
    size: int = 50,
    title: str = "",
    figsize: tuple = FIGSIZE,
    fig: plt.Figure = None,
    output: Optional[str] = None,
) -> plt.Figure:
    """
    Plots neuron positions.

    :param j0: The index of the neuron to highlight.
    :param r: The radius for highlighting neighboring neurons.
    :param connect_neighbors: Whether to draw neuron connections.
    :param size: The size of the neuron markers.
    :param title: The title of the plot.
    :param figsize: The size of the figure.
    :param fig: An existing figure to plot on.
    :param output: The output file path for saving the figure.
    """
    assert j0 is None or j0 < self.k_units

    fig, ax = plt.subplots(figsize=(figsize))
    krow, kcol = (self.k_units, 1)

    x, y = self.Y, [0 for _ in range(self.k_units)]
    if self.topology in ("GRID", "MESH"):
        y = -self.Y[:, 0]  # Invert y-axis to match plot_weights.
        x = self.Y[:, 1]
        krow, kcol = self.k_shape

    color = ["r" for _ in range(self.k_units)]
    if j0 is not None:
        x0, y0 = (x[j0], y[j0])
        color[j0] = "b"

    if r:
        assert j0 is not None
        neighbors, dists = self.unit_dists(j0, r=r)
        # Highlight neighboring units.
        for j, d in zip(neighbors, dists):
            color[j] = "lightblue"
        # Draw the neighborhood radius.
        if self.k_dist in ("l2", "euclidean"):
            patch = plt.Circle(
                (x0, y0),
                radius=r,
                color="b",
                fill=False,
                linestyle="--",
                linewidth=1,
            )
        elif self.k_dist in ("l1", "manhattan"):
            patch = plt.Polygon(
                ((x0-r, y0), (x0, y0+r),
                 (x0+r, y0), (x0, y0-r)),
                color="b",
                fill=False,
                linestyle="--",
                linewidth=1,
            )
        else:
            patch = plt.Rectangle(
                (x0 - r, y0 - r),
                width=2*r,
                height=2*r,
                color="b",
                fill=False,
                linestyle="--",
                linewidth=1,
            )
        ax.add_patch(patch)  # plt.gca().add_patch(patch)

    ax.scatter(x, y, c=color, s=size, edgecolors="black", marker="o", zorder=1)

    if connect_neighbors:
        style = patches.ConnectionStyle.Arc3(rad=-0.1)

        # Draw lines connecting neighbors.
        for i in range(self.k_units):
            xi, yi = (x[i], y[i]) if kcol > 1 else (x[i], 0)
            for j in (i+1, i+kcol):
                if j < self.k_units:
                    xj, yj = (x[j], y[j]) if kcol > 1 else (x[j], 0)
                    for (dx, dy) in ((xj-xi, 0), (0, yj-yi)):
                        if xi + dx < krow and yi + dy < kcol:
                            ax.plot(
                                [xi, xi + dx], [yi, yi + dy],
                                color="black", linewidth=0.5, zorder=0,
                            )

        # Draw curves connecting neighbors.
        if self.topology in ("RING", "MESH"):
            coords = [(0, yi, x[-1], yi) for yi in set(y)]
            if self.topology == "MESH":
                coords += [(xi, 0, xi, y[-1]) for xi in set(x)]
            for xi, yi, dx, dy in coords:
                patch = patches.FancyArrowPatch(
                    (xi, yi), (dx, dy),
                    connectionstyle=style, color="black", linewidth=0.1, zorder=0
                )
                ax.add_patch(patch)  # plt.gca().add_patch(patch)

    plt.axis("off")
    plt.title(title)

    if output:
        fig.savefig(output, bbox_inches="tight", dpi=300)

    plt.close()
    return fig


def plot_weights(
    self,
    W: list = None,
    j0: int = None,
    r: Optional[float] = None,
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    title: str = "",
    figsize: tuple = FIGSIZE,
    output: Optional[str] = None,
) -> plt.Figure:
    """
    Plots output map with neuron weights.

    :param W: Weights to plot (optional).
    :param j0: Best matching unit (optional).
    :param r: Neighborhood radius (optional).
    :param nrows: Number of rows for the output map (optional).
    :param ncols: Number of columns for the output map (optional).
    :param title: Title for the plot (optional).
    :param figsize: Figure size (optional).
    :param fig: Existing figure to plot on (optional).
    :param output: The output file path for saving the figure.
    """
    assert j0 is None or j0 < self.k_units

    if nrows and not ncols:
        ncols = 1
    elif ncols and not nrows:
        nrows = 1
    elif not nrows and not ncols:
        nrows, ncols = (1, self.k_units)
        if self.topology in ("GRID", "MESH"):
            nrows, ncols = self.k_shape

    W = self.W if W is None else W
    fig, axs = subplots(W, nrows=nrows, ncols=ncols, figsize=figsize, title=title)

    if j0 is not None:
        alpha, amin, amax = 0.5, 0.15, 0.35
        # Get axis for the selected unit.
        ax = getax(axs, j0, nrows, ncols)
        # TODO: fix wdim and hdim.
        bbox = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        wdim, hdim = int(bbox.width*fig.dpi), int(bbox.height*fig.dpi)
        # Highlight the selected unit.
        rect = patches.Rectangle(
            (-1, -1), wdim, hdim,
            linewidth=0,
            edgecolor="red",
            facecolor="red",
            alpha=alpha
        )
        ax.add_patch(rect)

    if r:
        # Highlight neighboring units to j0.
        assert j0 is not None
        neighbors, dists = self.unit_dists(j0, r=r)
        # Apply min-max scaling to the distances.
        alphas = minmax_scale(dists, (amin, amax), reverse=True)
        # Highlight neighboring units.
        for j, a in zip(neighbors, alphas):
            rect = patches.Rectangle(
                (-1, -1), wdim, hdim,
                linewidth=0,
                edgecolor="red",
                facecolor="red",
                alpha=amax if isnan(a) else a,
            )
            fig.axes[j].add_patch(rect)

    if output:
        fig.savefig(output, bbox_inches="tight", dpi=300)

    plt.close()
    return fig


def subplots(
    X: list,
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    title: Optional[str] = None,
    figsize: tuple = FIGSIZE,
    cmap: str = CMAP,
    interpolation: str = INTERPOLATION,
    reshape: bool = True,
    fig: Optional[plt.Figure] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Returns a matplotlib figure with subplots of images.

    :param X: The input data (list of images).
    :param nrows: The number of rows in the subplot grid (optional).
    :param ncols: The number of columns in the subplot grid (optional).
    :param title: The title of the plot (optional).
    :param figsize: The size of the figure (optional).
    :param cmap: The colormap to use (optional).
    :param interpolation: The interpolation method to use (optional).
    :param reshape: Whether to reshape the images (optional).
    :param fig: An existing figure to plot on (optional).
    """
    if nrows is None and ncols is None:
        k_dims = int(len(X)**.5)
        nrows, ncols = (k_dims, k_dims) if k_dims**2 == len(X) else (1, len(X))

    assert nrows and ncols

    if reshape is True:
        n_dims = int(len(X[0])**.5)
        reshape = (n_dims, n_dims) if nrows*ncols == len(X) else (1, len(X[0]))

    if fig is None:
        fig, axs = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols)
    else:
        axs = fig.axes

    for i in range(nrows*ncols):
        ax = getax(axs, i, nrows, ncols)
        # Clear axis and plot the image.
        if len(X) < i:
            ax.axis("off")
            continue
        ax.imshow(
            np.array(X[i]).reshape(reshape),
            cmap=cmap,
            interpolation=interpolation,
        )
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.suptitle(title)
    plt.close()
    return fig, axs


def animate(
    Z: list,
    figsize: tuple = FIGSIZE,
    cmap: str = CMAP,
    interval: int = 30,
    interpolation: str = INTERPOLATION,
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    fig: Optional[plt.Figure] = None,
    output: Optional[str] = None,
) -> None:
    """
    Returns a matplotlib animation.

    :param Z: The data to animate.
    :param figsize: The size of the figure.
    :param cmap: The colormap to use.
    :param interval: The interval between frames in milliseconds.
    :param interpolation: The interpolation method to use.
    :param nrows: The number of rows in the subplot grid.
    :param ncols: The number of columns in the subplot grid.
    :param fig: An existing figure to plot on (optional).
    :param output: The path to save the animation (optional).
    """
    if nrows is None and ncols is None:
        nrows = ncols = int(len(Z[0]) ** (1/2))

    if fig is None:
        fig, axs = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols)
    else:
        axs = fig.axs

    dims = int(len(Z[0][0])**(1/2))

    def fnc(epoch: int):
        for i in range(len(Z[epoch])):
            ax = getax(axs, i, nrows, ncols)
            # Clear axis and plot the image.
            ax.clear()
            ax.imshow(
                np.array(Z[epoch][i]).reshape(dims, dims),
                cmap=cmap,
                interpolation=interpolation,
            )
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.suptitle(f"Epoch {epoch+1}/{len(Z)}")
        print(f"Epoch {epoch+1}/{len(Z)}", end="\r")

    ani = animation.FuncAnimation(
        fig,
        fnc,
        frames=range(len(Z)),
        interval=interval,
        blit=False,  # blitting can't be used with Figure artists
    )

    if output:
        ani.save(output)
        # fps=30
        # writer=animation.FFMpegWriter(fps=30, extra_args=[])
        # writer=animation.PillowWriter(fps=30)

    # plt.close()  # Closing the figure is permitted after saving/display.
    return ani


def getax(
    axs: Union[plt.Axes, list],
    i: int,
    nrows: int,
    ncols: int,
) -> plt.Axes:
    """
    Returns the appropriate axis for the current subplot.

    Auxiliary function to handle different axis configurations.

    :param axs: The axes to retrieve the subplot from.
    :param i: The index of the subplot.
    :param nrows: The number of rows in the subplot grid.
    :param ncols: The number of columns in the subplot grid.
    """
    h, v = (i // ncols, i % ncols)

    return (
        axs if nrows == ncols == 1 else
        axs[(h, v) if (nrows > 1 and ncols > 1) else h if nrows > 1 else v]
    )

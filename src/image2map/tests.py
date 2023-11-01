"""
Unit test functions.

.. autosummary::

    plot_decay_fig
    plot_phi_fig
    plot_dist_fig

.. rubric:: Functions
"""

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import color_sequences

from . import functional as F
from .som import SOM

colors = color_sequences["tab10"]
colors = [colors[0], colors[3], colors[1], colors[2], colors[4]]


def plot_grid_mesh_fig(output=None):
    """ Returns figure comparing grid and mesh topologies. """
    k_units = 25
    krow, kcol = 5, 5

    fig, axs = plt.subplots(figsize=(5.5, 3.5), nrows=1, ncols=2, sharey=True)
    # plt.suptitle(f"Comparison of 2-dimensional topologies with (5,5) units")

    som = SOM(k_units=k_units, k_shape=(krow, kcol))
    x, y = som.Y[:, 0], som.Y[:, 1]

    for _, topology in enumerate(("GRID", "MESH")):
        ax = axs[_]
        ax.scatter(x, y, c="r", s=50, edgecolors="black", marker="o", zorder=1)

        # Draw lines connecting neighbors.
        for i in range(som.k_units):
            xi, yi = (x[i], y[i]) if kcol > 1 else (x[i], 0)
            for j in (i+1, i+kcol):
                if j < som.k_units:
                    xj, yj = (x[j], y[j]) if kcol > 1 else (x[j], 0)
                    for (dx, dy) in ((xj-xi, 0), (0, yj-yi)):
                        if xi + dx < krow and yi + dy < kcol:
                            ax.plot(
                                [xi, xi + dx], [yi, yi + dy],
                                color="black", linewidth=0.5, zorder=0,
                            )

        # Draw curves connecting neighbors.
        if topology == "MESH":
            style = patches.ConnectionStyle.Arc3(rad=-0.1)
            coords = [(0, yi, x[-1], yi) for yi in set(y)]
            coords += [(xi, 0, xi, y[-1]) for xi in set(x)]
            print(coords)
            for xi, yi, dx, dy in coords:
                patch = patches.FancyArrowPatch(
                    (xi, yi), (dx, dy),
                    connectionstyle=style, color="black", linewidth=0.1, zorder=0
                )
                ax.add_patch(patch)  # plt.gca().add_patch(patch)

        ax.set_title(f"'{topology}'")
        ax.axis("off")

    plt.tight_layout()

    if output:
        plt.savefig(output, bbox_inches="tight", dpi=300)

    # plt.close()
    return fig


def plot_decay_fig(epochs=500, xmin=0, xmax=1, rate=0.01, output=None):
    """ Returns figure comparing decay functions. """
    fig, axs = plt.subplots(figsize=(6, 3.5), nrows=1, ncols=2, sharey=True)
    # plt.suptitle(f"Decay factor over epochs for ($x_{{min}}$, $x_{{max}}$) = ({xmin}, {xmax})")

    # Decay with constant rate.
    pd.DataFrame({
        'linear':
            [F.decay(t, rate=rate, xmax=xmax, xmin=xmin, decay='linear') for t in range(epochs+1)],
        'exp':
            [F.decay(t, rate=rate, xmax=xmax, xmin=xmin, decay='exp') for t in range(epochs+1)],
    }).plot(
        title=f"With decay rate $\\lambda$={rate}",
        color=colors,
        xlabel="Epoch",
        ylabel="Decay factor",
        xlim=(0, epochs),
        grid=True,
        ax=axs[1],
    )

    # Normalized decay.
    pd.DataFrame({
        'linear':
            [F.decay(i/epochs, xmax=xmax, xmin=xmin, decay='linear') for i in range(epochs+1)],
        'exp':
            [F.decay(i/epochs, xmax=xmax, xmin=xmin, decay='exp') for i in range(epochs+1)],
        'sqd':
            [F.decay(i/epochs, xmax=xmax, xmin=xmin, decay='linear', n=2) for i in range(epochs+1)],
        'cubic':
            [F.decay(i/epochs, xmax=xmax, xmin=xmin, decay='linear', n=3) for i in range(epochs+1)],
    }).plot(
        title="Normalized over epochs",
        color=colors,
        xlabel="Epoch",
        ylabel="Decay factor",
        xlim=(0, epochs),
        grid=True,
        ax=axs[0],
    )

    plt.tight_layout()

    if output:
        plt.savefig(output, bbox_inches="tight", dpi=300)

    # plt.close()
    return fig


def plot_phi_fig(r=10, smooth=1, output=None):
    """ Returns figure comparing phi functions. """
    fig, axs = plt.subplots(figsize=(11, 3.5), nrows=1, ncols=3, sharey=True)
    # plt.suptitle(f"Neuron excitatory field based on distance and radius ($r$={r})")

    dists = [d/smooth for d in range(((r+1)*smooth))]
    decays = [("linear", "exp", "sqd", "lap"),
              ("exp ($k$=$r$)", "exp ($k$=$r/2$)", "exp ($k$=$\\sqrt{r}$)"),
              ("lap ($\\sigma$=$r$)", "lap ($\\sigma$=$r/2$)", "lap ($\\sigma$=$\\sqrt{r}$)")]
    k_sigmas = [(None, None, None, None), (r, r/2, r**.5), (r, r/2, r**.5)]
    titles = ["Influence decay", "Exponential decay ('exp')", "Laplacian decay ('lap')"]

    for i, (decay_, k_sigma_) in enumerate(zip(decays, k_sigmas)):
        df = pd.DataFrame({
            decay.split()[-1].strip('()'):
                [F.phi(d, r, k=k_sigma, sigma=k_sigma, decay=decay.split()[0]) for d in dists]
            for decay, k_sigma in zip(decay_, k_sigma_)
        })
        # Include both directions.
        df_ = df[1:].copy()
        df_.index = [i*-1 for i in df_.index]
        df_ = pd.concat([df_[::-1], df])
        # Plot 1-dimensional graph.
        df_.plot(
            xlabel="Distance",
            ylabel="$\\phi$",
            title=titles[i],
            color=colors if i == 0 else colors[1] if i == 1 else colors[3],
            grid=True,
            xlim=(-r, r),
            xticks=[-r, -r/2, 0, r/2, r],
            style="-" if i == 0 else ["-", "--", ":"],
            ax=axs[i]
        )
        axs[i].legend(loc="lower center", fontsize=10)

    plt.tight_layout()

    if output:
        plt.savefig(output, bbox_inches="tight", dpi=300)

    # plt.close()
    return fig


def plot_dist_fig(j0=55, r=2.5, connect_neighbors=True, output=None):
    """ Returns figure comparing distance functions. """
    som = SOM(k_units=100, k_shape=(10,10))
    fig, axs = plt.subplots(figsize=(9, 3.5), nrows=1, ncols=3, sharey=True)
    # plt.suptitle(f"Comparison of unit distance functions ($r$={r})")

    krow, kcol = (som.k_units, 1)

    x, y = som.Y, [0 for _ in range(som.k_units)]
    if som.topology in ("GRID", "MESH"):
        y = -som.Y[:, 0]  # Invert x-axis to match plot_weights.
        x = som.Y[:, 1]
        krow, kcol = som.k_shape
    x0, y0 = (x[j0], y[j0])

    k_dist_ = som.k_dist
    for i, k_dist in enumerate(("l2", "l1", "chebyshev")):
        som.k_dist = k_dist
        ax = axs[i]

        color = ["r" for _ in range(som.k_units)]
        if j0 is not None:
            color[j0] = "b"

        if r:
            assert j0 is not None
            neighbors, dists = som.unit_dists(j0, r=r)
            # Highlight neighboring units.
            for j, d in zip(neighbors, dists):
                color[j] = "lightblue"
            # Draw the neighborhood radius.
            if som.k_dist in ("l2", "euclidean"):
                patch = plt.Circle(
                    (x0, y0),
                    radius=r,
                    color="b",
                    fill=False,
                    linestyle="--",
                    linewidth=1,
                )
            elif som.k_dist in ("l1", "manhattan"):
                patch = plt.Polygon(
                    ((x0-r, y0), (x0, y0+r),
                     (x0+r, y0), (x0, y0-r)),
                    color="b",
                    fill=False,
                    linestyle="--",
                    linewidth=1,
                )
            elif som.k_dist == "chebyshev":
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

        ax.scatter(x, y, c=color, s=50, edgecolors="black", marker="o", zorder=1)

        if connect_neighbors:
            style = patches.ConnectionStyle.Arc3(rad=-0.1)

            # Draw lines connecting neighbors.
            for i in range(som.k_units):
                xi, yi = (x[i], y[i]) if kcol > 1 else (x[i], 0)
                for j in (i+1, i+kcol):
                    if j < som.k_units:
                        xj, yj = (x[j], y[j]) if kcol > 1 else (x[j], 0)
                        for (dx, dy) in ((xj-xi, 0), (0, yj-yi)):
                            if xi + dx < krow and yi + dy < kcol:
                                ax.plot(
                                    [xi, xi + dx], [yi, yi + dy],
                                    color="black", linewidth=0.5, zorder=0,
                                )

            # Draw curves connecting neighbors.
            if som.topology in ("RING", "MESH"):
                coords = [(0, yi, x[-1], yi) for yi in set(y)]
                if som.topology == "MESH":
                    coords += [(xi, 0, xi, y[-1]) for xi in set(x)]
                for xi, yi, dx, dy in coords:
                    patch = patches.FancyArrowPatch(
                        (xi, yi), (dx, dy),
                        connectionstyle=style, color="black", linewidth=0.1, zorder=0
                    )
                    ax.add_patch(patch)  # plt.gca().add_patch(patch)

        label = "Euclidean" if som.k_dist in ("l2", "euclidean") else "Manhattan" if som.k_dist in ("l1", "manhattan") else "Chebyshev"
        ax.set_title(f"{label} distance")
        ax.set_xticks([])
        ax.set_yticks([])

    legend = {"b": f"Selected unit ($j$={j0})", "lightblue": "Neighboring units", "r": "Other units"}
    handles = [patches.Patch(color=color, label=label) for color, label in legend.items()]
    fig.legend(handles=handles, ncols=3, loc="lower center",  bbox_to_anchor=(.5, -.1))

    plt.tight_layout()

    if output:
        plt.savefig(output, bbox_inches="tight", dpi=300)

    # plt.close()
    return fig
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from scipy.spatial import voronoi_plot_2d
from persim import plot_diagrams
import scipy.spatial as sp
from scipy.spatial import Voronoi
from cycler import cycler

# Define plotting style
custom_colors = ['red', '#1f77b4', '#ff7f0e', '#9467bd', '#2ca02c', '#8c564b']

# Update Matplotlib's default color cycle globally
plt.rcParams['axes.prop_cycle'] = cycler(color=custom_colors)

plotting_style = {
    "text.usetex": True,  # Enable LaTeX rendering
    "font.family": "serif",  # Use a serif font (default LaTeX font)
    "font.serif": ["Computer Modern Roman"],  # Specify LaTeX font family
    "axes.labelsize": 16,  # Font size for axis labels
    "font.size": 16,  # Font size for the whole plot
    "legend.fontsize": 14,  # Font size for the legend
    "xtick.labelsize": 14,  # Font size for x-axis ticks
    "ytick.labelsize": 14,  # Font size for y-axis ticks
    "axes.titlesize": 16,  # Font size for the title
    "figure.titlesize": 22,  # Font size for the figure title
}
plt.rcParams.update(plotting_style)


# Time-series plots
def plot_run_chart(ax, t, signal, label=None, title=None, linestyle='-', color=None):
    ax.plot(t, signal, label=label, linestyle=linestyle, color=color)
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Run Chart")

    ax.legend(loc="upper left")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")

# todo: quantity_type as argument (caller), each quantity has a dictionary, title name, axes names, units, scale
# todo: make universal plotters with normalization and other logic: time-series, 2d, 3d, time-series metrics?,
#  and topological signatures (these might go to metric class, for now i can make a caller function)
# todo: add scale=None, to scale the plot to the scale ay limits.
# todo: when adding to classes use these common plotters with arguments like default axes_labels, labels, etc

# def plot_running_average(ax, t, running_average, linestyle="--"):
#     ax.plot(t, running_average, label="Running Average", linestyle=linestyle)
#
#     ax.legend(loc="upper left")
#     ax.set_xlabel("Time")
#     ax.set_ylabel("Value")


# 2D visualization plots
def plot_2d(ax, embedding, delay, signal_type, title=None):
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    # get max and min values for x and y from ax
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    # rescale plot
    embedding[:, 0] = (embedding[:, 0] - np.min(embedding[:, 0])) / (
                np.max(embedding[:, 0]) - np.min(embedding[:, 0])) * (x_max - x_min) + x_min
    embedding[:, 1] = (embedding[:, 1] - np.min(embedding[:, 1])) / (
                np.max(embedding[:, 1]) - np.min(embedding[:, 1])) * (y_max - y_min) + y_min
    ax.scatter(embedding[:, 0], embedding[:, 1], s=10, alpha=0.8, label="Cartesian")
    if title:
        ax.set_title(f"{signal_type.replace('_', ' ').title()}: {title}")
    else:
        ax.set_title(f"2D Embedding ({signal_type})")

    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")


def plot_tsne(ax, embedding, delay, signal_type, seed=42, title=None):
    ax.scatter(embedding[:, 0], embedding[:, 1], s=10, label="t-SNE")
    if title:
        ax.set_title(f"{signal_type.replace('_', ' ').title()}: {title}")
    else:
        ax.set_title(f"t-SNE Embedding ({signal_type})")
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")


def plot_voronoi(embedding_2d, ax, signal_type=None, title=None):
    vor = Voronoi(embedding_2d)
    voronoi_plot_2d(vor, ax=ax)
    if title:
        ax.set_title(f"{signal_type.replace('_', ' ').title()}: {title}")
    else:
        ax.set_title(f"Voronoi Diagram ({signal_type})")


def plot_vrd_complex(simplices, samples, epsilon, ax=None):
    ax.triplot(samples[:, 0], samples[:, 1], simplices)
    ax.plot(samples[:, 0], samples[:, 1], 'o', markersize=4)
    ax.set_title(rf"{len(simplices)} simplices, $\epsilon$ = {epsilon}")


def plot_3d(ax, embedding, delay, signal_type):
    ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], s=10, alpha=0.8)
    ax.set_title(f"3D Embedding (Signal: {signal_type}, d={delay})")
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")

# def plot_landscape(landscape, ax, delay, signal_type):
#     title = f"Landscape (Signal: {signal_type}, d={delay})"
#     ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
#     for depth, pairs in enumerate(landscape.values):
#         x_vals = [p[0] for p in pairs]
#         y_vals = [p[1] for p in pairs]
#         ax.plot(x_vals, y_vals, label=f"Layer {depth + 1}")
#     ax.set_title(title)
#     ax.set_xlabel("Birth-Death Scale")
#     ax.set_ylabel("Landscape Value")
#     ax.legend()


# Topological Signatures
def plot_landscapes(landscape, axs, n_layers=10, homologies=None):
    landscape = np.array(landscape).reshape((landscape.shape[0] // n_layers, n_layers, landscape.shape[-1]))
    smooth_color_palette = plt.get_cmap("cividis")

    for layer in range(landscape.shape[1]):  # Number of layers
        if homologies is None:
            homologies = range(landscape.shape[0])
        for i, homology_dim in enumerate(homologies):
            axs[i].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
            x_values = np.linspace(0, 1, landscape.shape[-1])  # Scaled x-axis
            axs[i].plot(x_values, landscape[homology_dim, layer], label=f'Layer {layer + 1}',
                        color=smooth_color_palette(100 * layer))
            axs[i].set_title(f'Persistence Landscapes for $H_{homology_dim}$')
            axs[i].set_xlabel('Feature')
            axs[i].set_ylabel('Landscape Value')

    # for ax in axs:
    #     ax.legend()


def plot_silhouettes(silhouette, ax, scale=1):
    for hom_dim in range(silhouette.shape[0]):
        ax.plot(
            np.linspace(0, scale, silhouette.shape[1]),
            silhouette[hom_dim],
            label=rf"$H_{hom_dim}$"
        )
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.set_title(f"Silhouettes")
    ax.set_xlabel("Filtration Value")
    ax.set_ylabel("Silhouette Value")
    ax.legend()


def plot_betti_curves(betti_curve, ax, scale=1):
    for hom_dim in range(betti_curve.shape[0]):
        ax.plot(
            np.linspace(0, scale, betti_curve.shape[1]),
            betti_curve[hom_dim],
            label=rf"$H_{hom_dim}$",
            color=f"C{hom_dim}"
        )
    ax.set_title(f"Betti Curves")
    ax.set_xlabel("Filtration Value")
    ax.set_ylabel("Betti Number")
    ax.legend()


def plot_persistence_images(persistence_image, axs):
    for hom_dim in range(persistence_image.shape[0]):
        img = persistence_image[hom_dim]
        axs[hom_dim].imshow(img, cmap="viridis", origin="lower")
        axs[hom_dim].set_title(rf"Persistence Image for $H_{hom_dim}$")
        axs[hom_dim].set_xlabel("Birth")
        axs[hom_dim].set_ylabel("Persistence")


def plot_heat_kernels(heat_images, axs):
    for hom_dim in range(heat_images.shape[0]):
        img = heat_images[hom_dim]
        axs[hom_dim].imshow(img, origin="lower", cmap="hot", extent=[0, 1, 0, 1])
        axs[hom_dim].set_title(rf"Heat Kernel Image for $H_{hom_dim}$")
        axs[hom_dim].set_xlabel("Filtration Birth")
        axs[hom_dim].set_ylabel("Persistence")
        # fig.colorbar(im, ax=axs[3 + hom_dim])


def plot_persistence_diagram(diagram, ax, title="Persistence Diagrams", normalize=True):
    """
    Plot persistence diagrams with optional normalization and infinity point handling.

    Parameters:
    - diagram: ndarray of shape (n_features, 3)
      Persistence diagram containing birth-death-dimension triples.
    - ax: matplotlib axis
      Axis to plot the diagram.
    - title: str, default "Persistence Diagrams"
      Title for the plot.
    - normalize: bool, default True
      If True, normalize the diagram so the filtration values are scaled to [0, 1].
      Also removes infinity points for normalization.
    """
    h_dims = int(np.max(diagram[:, 2]) + 1)
    diagrams = [diagram[diagram[:, 2] == i] for i in range(h_dims)]

    if normalize:
        finite_values = np.concatenate([dgm[:, :2].flatten() for dgm in diagrams if len(dgm) > 0])
        min_val, max_val = finite_values.min(), finite_values.max()
        scale = max_val - min_val
        if scale > 0:
            for i in range(h_dims):
                diagrams[i][:, :2] = (diagrams[i][:, :2] - min_val) / scale
        diagrams = [dgm[np.isfinite(dgm[:, 1])] for dgm in diagrams]
        plot_range = [0, 1, 0, 1]
    else:
        diagrams[0] = np.append(diagrams[0], np.array([[0, np.inf, 0]]), axis=0)
        plot_range = None

    plot_diagrams(diagrams, ax=ax, xy_range=plot_range)
    plt.rcParams.update(plotting_style)
    ax.set_title(title)


def plot_barcode(dgm, ax):
    colors = ["tab:blue", "tab:red", "tab:green", "tab:orange"]
    h_dims = len(np.unique(dgm[:, 2]))
    for dim in range(h_dims):
        dgm_dim = dgm[dgm[:, 2] == dim]  # Filter for the current homology dimension
        if dim == 0:
            dgm_dim = np.append(dgm_dim, np.array([[0, np.inf, 0]]), axis=0)
        for i, (birth, death, _) in enumerate(dgm_dim):
            if np.isinf(death):
                death = 1.1 * np.max(dgm[:, 1])
            ax.plot([birth, death], [i + dim * 10, i + dim * 10], lw=2, color=colors[dim],
                    label=rf"$H_{dim}$" if i == 0 else None)

    # Formatting the plot
    ax.set_title("Persistence Barcodes")
    ax.set_xlabel("Filtration Value")
    ax.set_ylabel("Feature Index")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)


# Homology time-series plots
def plot_topological_signature_time_series(embedding_indices, metrics, length, signal_type, ax, homologies, title=None):
    if not isinstance(ax, (list, np.ndarray)):
        ax = [ax]
    for j, (metric, values) in enumerate(metrics.items()):
        axis = ax[j] if len(ax) > 1 else ax[0]
        # values[ values > 1e6 ] = np.nan
        # values[ values < 0 ] = 0
        rescaled_values = (values[:, homologies] - np.min(values[:, homologies], axis=0)) / (
                    np.max(values[:, homologies], axis=0) - np.min(values[:, homologies], axis=0))
        axis.plot(embedding_indices / length, rescaled_values,
                  label=[rf"$H_{homologies[i]}$" for i in range(len(homologies))])
        if not title:
            axis.set_title(f"{metric} ({signal_type.replace('_', ' ')})")
        else:
            axis.set_title(title)
        axis.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        axis.legend(loc="upper left")


# Utilities
def save_plot(filename, fig=None, dpi=500, folder="../figures/"):
    os.makedirs(folder, exist_ok=True)

    if fig is None:
        fig = plt.gcf()

    for ax in fig.axes:
        ax.title.set_color('white')  # Set white title
        ax.xaxis.label.set_color('white')  # Set white x-axis label
        ax.yaxis.label.set_color('white')  # Set white y-axis label
        ax.tick_params(axis='x', colors='white')  # Set white x-ticks
        ax.tick_params(axis='y', colors='white')  # Set white y-ticks
        for spine in ax.spines.values():  # Set white spines (edges)
            spine.set_edgecolor('white')

    with plt.rc_context({
        'savefig.facecolor': 'none',
        'savefig.transparent': True,
    }):
        fig.savefig(os.path.join(folder, filename), dpi=dpi)


# todo: make metrics classes, so I can use them in pytorch pipelings,
#  and so that plot is a class function, docs - description of metrics and plots, use print info in notebooks

# def plot_voronoi(ax, embedding, signal_type, delay, seed=42):
#     tsne = TSNE(n_components=2, perplexity=30, random_state=seed)
#     embedding_2d = tsne.fit_transform(embedding)
#     vor = Voronoi(embedding_2d)
#     ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c="blue", s=10, label="t-SNE Points")
#     voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors="orange")
#     ax.set_title(f"t-SNE with Voronoi Diagram (Signal: {signal_type}, d={delay})")
#     ax.set_xlabel("t-SNE Dimension 1")
#     ax.set_ylabel("t-SNE Dimension 2")
#     ax.legend()

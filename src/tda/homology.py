import numpy as np

from ripser import ripser
from persim import plot_diagrams
from persim.landscapes.exact import PersLandscapeExact, PersLandscapeApprox
from gtda.homology import VietorisRipsPersistence
import scipy.spatial as sp

from gtda.diagrams import PersistenceEntropy, Amplitude


# def tda_analysis(segment, prices, window_size, delay, ax=None):
#     log_prices = np.log(prices[segment[0]:segment[1] + 1])
#     embeddings = np.array([log_prices[i:i + window_size * delay:delay]
#                            for i in range(len(log_prices) - window_size * delay)])
#     diagrams = ripser(embeddings.reshape(-1, 1), maxdim=1)['dgms']
#     if ax:
#         plot_diagrams(diagrams, ax=ax)
#     return diagrams

def compute_topological_metrics(embedding_sequence, metrics_list, homologies_show):
    vr = VietorisRipsPersistence(homology_dimensions=homologies_show, n_jobs=-1)
    diagrams = vr.fit_transform(embedding_sequence)

    metrics = {}
    for metric, metric_params in metrics_list.items():
        if metric == "entropy":
            from gtda.diagrams import PersistenceEntropy
            persistence_entropy = PersistenceEntropy(normalize=False, nan_fill_value=None, n_jobs=-1)
            entropy_values = persistence_entropy.fit_transform(diagrams)
            metrics["Entropy"] = entropy_values
        else:
            amplitude = Amplitude(metric=metric, metric_params=metric_params, n_jobs=-1)
            amplitude_values = amplitude.fit_transform(diagrams)
            metrics[metric.capitalize() + " Norm"] = amplitude_values

    return metrics


def tda_analysis(segment, prices, window_size, delay, ax=None):
    log_prices = np.log(prices[segment[0]:segment[1] + 1])

    if len(log_prices) < window_size * delay:
        raise ValueError(f"Segment {segment} is too short for the given window_size={window_size} and delay={delay}.")

    embeddings = np.array([log_prices[i:i + window_size * delay:delay]
                           for i in range(len(log_prices) - window_size * delay)])
    diagrams = ripser(embeddings.reshape(-1, 1), maxdim=1)['dgms']

    if ax:
        plot_diagrams(diagrams, ax=ax)

    return diagrams


def compute_persistent_homology(embeddings, max_dim=1):
    """
    Compute persistent homology for the given time-series data.

    Parameters:
    - data: The input time-series data (1D array).
    - window_size: Size of the sliding window for embeddings.
    - delay: Time delay between points in the embedding.
    - max_dim: Maximum homology dimension to compute.

    Returns:
    - Diagrams of persistent homology.
    """
    # Validate input size

    # Ensure embeddings have the correct shape
    if embeddings.ndim == 1:
        embeddings = embeddings[:, np.newaxis]  # Convert to 2D array

    # Compute persistent homology
    return ripser(embeddings, maxdim=max_dim)['dgms']
    # return ripser(embeddings.reshape(1, -1), maxdim=max_dim)['dgms']


def compute_landscape(diagrams, hom_deg=1):
    return PersLandscapeExact(dgms=diagrams, hom_deg=hom_deg)


def compute_vietoris_rips_delaunay(point_cloud, epsilon):
    delaunay = sp.Delaunay(point_cloud)
    simplices = []

    for simplex in delaunay.simplices:
        vertices = point_cloud[simplex]
        is_valid = True

        for i, p1 in enumerate(vertices):
            for j, p2 in enumerate(vertices):
                print(np.linalg.norm(p1 - p2))
                if i < j and np.linalg.norm(p1 - p2) >= epsilon:
                    is_valid = False
                    break

            if not is_valid:
                break

        if is_valid:
            simplices.append(simplex)

    return simplices

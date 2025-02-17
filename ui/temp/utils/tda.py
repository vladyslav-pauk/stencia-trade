import numpy as np
from ripser import ripser
from scipy.integrate import quad


def compute_landscape(diagrams, hom_deg=1):
    """
    Compute persistence landscapes from a persistence diagram.
    Arguments:
        diagrams: The output of Ripser, containing persistence diagrams.
        hom_deg: Homology degree to use (default is 1).
    Returns:
        List of persistence landscape layers (functions).
    """
    if hom_deg >= len(diagrams):
        raise ValueError(f"Homology degree {hom_deg} is not available in diagrams.")

    landscape = []
    diagram = diagrams[hom_deg]

    for i, (b, d) in enumerate(diagram):
        if d == np.inf:
            continue  # Ignore infinite points
        midpoint = (b + d) / 2

        def piecewise_linear(x, b=b, d=d, midpoint=midpoint):
            if b <= x <= midpoint:
                return x - b
            elif midpoint < x <= d:
                return d - x
            return 0

        landscape.append(piecewise_linear)

    return landscape


def landscape_norm(landscape, p=2, x_min=-np.inf, x_max=np.inf):
    """
    Compute the L_p norm of a persistence landscape.
    Arguments:
        landscape: List of piecewise linear functions.
        p: Order of the norm (default is 2 for L2 norm).
        x_min: Lower bound of the integration range.
        x_max: Upper bound of the integration range.
    Returns:
        Norm of the persistence landscape.
    """
    norm = 0

    for layer in landscape:
        def integrand(x):
            return abs(layer(x)) ** p

        # Integrate over the specified range
        layer_norm, _ = quad(integrand, x_min, x_max)
        norm += layer_norm ** p

    return norm ** (1 / p)


def perform_tda(prices, w, d, N, p=2):
    """
    Perform TDA and compute persistence landscape norms.
    Arguments:
        prices: Time-series data (e.g., stock prices).
        w: Sliding window size.
        d: Time-delay embedding step.
        N: Embedding dimension.
        p: Order of the norm to compute (default is 2 for L2 norm).
    Returns:
        Time series of persistence landscape norms.
    """
    segment_length = len(prices)
    required_length = w + (N - 1) * d

    if segment_length < required_length:
        raise ValueError(f"Segment length ({segment_length}) is smaller than required embedding length ({required_length}).")

    norms = []

    # Generate time-delay embeddings with sliding window
    for t in range(segment_length - required_length + 1):
        # Create embedding for the current window
        embeddings = np.array([
            prices[t + i:t + i + (N - 1) * d + 1:d] for i in range(w)
        ])

        if embeddings.size == 0:
            raise ValueError("Embeddings are empty. Check your TDA parameters.")

        # Compute persistent homology
        diagrams = ripser(embeddings, maxdim=1)['dgms']

        # Compute persistence landscape
        landscape = compute_landscape(diagrams)

        # Compute the norm of the persistence landscape
        norm = landscape_norm(landscape, p=p)
        norms.append(norm)

    return norms
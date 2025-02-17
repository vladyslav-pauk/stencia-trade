import numpy as np
from sklearn.manifold import TSNE
from gtda.time_series import TakensEmbedding, SlidingWindow


def takens_embedding(signal, delay=1, dimension=2):
    """
    Taken's embedding of a time series. Takes collections
    of (possibly multivariate) time series as input,
    applies the Takens embedding algorithm to each independently,
    and returns a corresponding collection of point clouds in Euclidean space
    (or possibly higher-dimensional structures, see `flatten`).

    Components of embedding vectors correspond with values sampled every (delay) steps
    from a moving time window of (dimension) * (delay) width:
        z_t = [x_t, x_{t + delay}, ..., x_{t + (dimension - 1) * delay}].
    Set `stride` to control duration between two consecutive embedded points.
    Delay defines how far apart in time consecutive values for constructing an embedded point are sampled.
    When delay=1, the mean of the embedding is equivalent to the running average of the time series.
    When dimension=1, the embedding is equivalent to the original time series.

    The output is a 2D array with shape (n_points, dimension), where:
        (n_points) = (n_steps) - (dimension - 1) * (delay).
    The last entry of the last window always equals the last entry in the original time series.

    :param signal: array-like, Time series signal. Shape: (n_steps).
    :param delay: float, Time delay between successive components of the embedding vector
    :param dimension:  int, Dimension of the embedding space (per variable, in the multivariate case).
    :return: array-like, Embedded time series. Shape: (n_points, dimension).
    """
    signal = np.array(signal).reshape(1, -1)
    embedder = TakensEmbedding(time_delay=delay, dimension=dimension)
    embedding = embedder.fit_transform(signal)
    return embedding

# todo: implement Taken's using Conv1D


def compute_running_average(embedding, embedding_delay):
    """
    Compute the running average of the time series from its Taken's embedding.

    The running average corresponds to the center of each embedding window,
    aligned with the original time series t_i as:
        t_i = i + (dimension - 1) * delay // 2

    The output is a 1D array padded with NaNs to ensure the same length as the original time series.

    :param embedding: array-like, Taken's embedding of the time series. Shape: (n_points, dimension).
    :param embedding_delay: int, Time delay between successive components of the embedding vector.
    :return: array-like, Running average of the time series. Shape: (n_steps).
    """
    embedding_dimension = embedding.shape[-1]
    running_average = embedding[0].mean(axis=1)

    running_average = np.pad(running_average, (embedding_dimension - 1) * embedding_delay // 2, mode='constant', constant_values=np.nan)
    return running_average


def sliding_window_embedding(signal, window_size, stride=1):
    """
    Convert a time-series of scalars or arrays
    into a sequence of windows on the original sequence.

    Each window stacks together consecutive objects,
    and consecutive windows are separated by a constant stride.
    Each sliding window of (window_size) can be interpreted
    as a point-cloud in a higher-dimensional space.
    When (window_size)=1, the embedding is equivalent to the original time series.

    The output is a 3D array with shape (n_slices, window_size, n_features), where:
        (n_features) = (signal).shape[-1],
        (n_slices) = (n_steps - window_size) // stride + 1,
    The last entry of the last window always equals the last entry in the original time series.

    :param signal: array-like, Time series signal. Shape: (n_steps, n_features).
    :param window_size: int, Size of the sliding window.
    :param stride: int, Stride between consecutive windows.
    :return: array-like, Sliding window. Shape: (n_slices, window_size, n_features).
    """
    windows = SlidingWindow(size=window_size, stride=stride)
    sequence = windows.fit_transform(signal)
    return sequence


def embedding_time_series(signal, embedding_delay, embedding_dimension, sliding_window_size, stride=1):
    """
    Compute the time series of the signal embedding using Taken's embedding and sliding window,
    padded to the original length.

    Each value corresponds with the center of the embedding window,
    aligned with the original time series t_i as:
        t_i = i * (stride) - 1 + (sliding_window_size) // 2 + (dimension - 1) * delay // 2

    The output is a 3D array padded with NaNs to ensure the same length as the original time series with
    shape: (n_slices, sliding_window_size, embedding_dimension), where
        n_slices = (n_steps - sliding_window_size - (embedding_dimension - 1) * embedding_delay) // stride + 1.

    :param signal: array-like, Time series signal. Shape: (n_steps).
    :param embedding_delay: float, Time delay for Taken's embedding.
    :param embedding_dimension: float, Dimension of the embedding space.
    :param sliding_window_size: int, Size of sliding window, or number of points in each time slice.
    :return: tuple, A tuple of the embedded time series and the corresponding time indices.
    """
    embedding = takens_embedding(signal, delay=embedding_delay, dimension=embedding_dimension)
    sequence = sliding_window_embedding(embedding[0], window_size=sliding_window_size, stride=stride)

    embedding_start_index = (embedding_dimension - 1) * embedding_delay // 2 + sliding_window_size // 2
    embedding_end_index = embedding_start_index + len(sequence) * stride

    indices = np.arange(embedding_start_index, embedding_end_index, stride)

    return sequence, indices


def tsne_embedding(embedding, n_components=2, perplexity=20, seed=None):
    """
    Compute t-SNE embedding of the given embedding.

    :param embedding: array-like, Input embedding. Shape: (n_points, n_features).
    :param n_components: int, Number of dimensions for the t-SNE projection.
    :param perplexity: float, Perplexity parameter for t-SNE.
    :param seed: int or None, Random seed for reproducibility.
    :return: array-like, t-SNE embedded data. Shape: (n_points, n_components).
    """
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=seed)
    embedding_tsne = tsne.fit_transform(embedding)
    return embedding_tsne


# todo: make it a class, and add a method to plot (based on transform), compute metrics, etc.
# todo: write docstrings so it's not duplicated with jupyer

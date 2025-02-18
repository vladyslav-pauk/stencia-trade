import pandas as pd
import ta

from src.tda.homology import compute_topological_metrics
from src.tda.embedding import embedding_time_series

def add_indicator_data(data, indicators, settings):
    data['SMA'] = ta.trend.sma_indicator(data['Close'].squeeze(), window=settings['SMA'].get('window', 20))
    data['EMA'] = ta.trend.ema_indicator(data['Close'].squeeze(), window=settings['EMA'].get('window', 20))

    embedding_delay = settings['TDA'].get('delay', 5)
    embedding_dimension = settings['TDA'].get('dimension', 5)
    sliding_window_size = settings['TDA'].get('window_size', 20)
    sliding_stride = 1
    embedding_sequence, embedding_indices = embedding_time_series(data['Close'].squeeze(), embedding_delay, embedding_dimension,
                                                                  sliding_window_size, stride=sliding_stride)

    homologies_show = [1]
    metrics_list = {"landscape": {"p": 1, "n_bins": 100}}
    metrics = pd.DataFrame(compute_topological_metrics(embedding_sequence, metrics_list, homologies_show)['Landscape Norm'])

    min_close, max_close = data['Close'].squeeze().min(), data['Close'].squeeze().max()
    min_metric, max_metric = metrics.min(), metrics.max()

    normalized_metric = (metrics - min_metric) / (max_metric - min_metric)  # Normalize to [0,1]
    normalized_metric = normalized_metric * (max_close - min_close) + min_close  # Scale to close_values range
    pd.set_option('display.max_rows', None)

    data['TDA'] = normalized_metric
    return data

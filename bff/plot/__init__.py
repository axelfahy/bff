"""Plot module of bff."""

from .plot import (
    get_n_colors,
    plot_cluster,
    plot_confusion_matrix,
    plot_correlation,
    plot_counter,
    plot_history,
    plot_kmeans,
    plot_pca_explained_variance_ratio,
    plot_pie,
    plot_predictions,
    plot_roc_curve,
    plot_series,
    plot_true_vs_pred,
    set_thousands_separator,
)

# Public object of the module.
__all__ = [
    'get_n_colors',
    'plot_cluster',
    'plot_confusion_matrix',
    'plot_correlation',
    'plot_counter',
    'plot_history',
    'plot_kmeans',
    'plot_pca_explained_variance_ratio',
    'plot_pie',
    'plot_predictions',
    'plot_roc_curve',
    'plot_series',
    'plot_true_vs_pred',
    'set_thousands_separator',
]

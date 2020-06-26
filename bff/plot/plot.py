# -*- coding: utf-8 -*-
"""Plot functions of the bff library.

This module contains fancy plot functions.
"""
import itertools
import logging
from collections import Counter
from typing import Any, List, Optional, Sequence, Tuple, Union
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import seaborn as sns

import bff.fancy

TNum = Union[int, float]

LOGGER = logging.getLogger(__name__)

register_matplotlib_converters()


def add_identity(ax: plt.axes, *args, **kwargs) -> plt.axes:
    """
    Add identity line on axis.

    The identity line (diagonal) is useful to have a better visualization
    of predicted values y against the ground truth x.

    Lower and upper limits must be retrieved in case they are not
    equal on each axes.

    Parameters
    ----------
    ax : plt.axes
        Axes from matplotlib on which to add the identity line.
    *args
        Additional positional arguments to be passed to the
        `plt.plot` function from matplotlib.
    **kwargs
        Additional keyword arguments to be passed to the
        `plt.plot` function from matplotlib.

    Returns
    -------
    plt.axes
        Modified axes with the identity line on it.
    """
    identity, = ax.plot([], [], *args, **kwargs)

    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])

    callback(ax)
    ax.callbacks.connect('xlim_changed', callback)
    ax.callbacks.connect('ylim_changed', callback)
    return ax


def get_n_colors(n: int, cmap: str = 'rainbow') -> List:
    """
    Get `n` colors from a color map.

    A color is represented by an array having 4 components (r, g, b, a).
    A list of array is return containing the `n` colors.

    Parameters
    ----------
    n : int
        Number of colors to get.
    cmap : str, default 'rainbow'
        Color map for the colors to retrieve.

    Returns
    -------
    list
        List of colors from the color map.
    """
    assert cmap in plt.colormaps(), f'Colormap {cmap} does not exist.'
    return list(cm.get_cmap(cmap)(np.linspace(0, 1, n)))


def plot_cluster(df: pd.DataFrame,
                 cluster_col_1: str = 'col_1',
                 cluster_col_2: str = 'col_2',
                 label_col: Optional[str] = None,
                 colors: Optional[Sequence[str]] = None,
                 labels: Optional[Union[str, Sequence[str]]] = None,
                 label_x: str = 'Dimension 1',
                 label_y: str = 'Dimension 2',
                 title: str = 'Clustering',
                 ax: Optional[plt.axes] = None,
                 loc: Union[str, int] = 'best',
                 s: Optional[Union[TNum, Sequence]] = mpl.rcParams['lines.markersize'] * 2,
                 figsize: Tuple[int, int] = (14, 7),
                 dpi: int = 80,
                 style: str = 'default',
                 **kwargs) -> plt.axes:
    """
    Plot 2D clustering.

    Clustering, such as T-SNE or UMAP must already be computed and
    stored inside two separate column of the DataFrame.

    See the example for more information.

    If there are some labels (and the `label_col` is given), there will be one color by label,
    if there are no label, you can specify a list of colors to be applied for each point of data.

    The `label_col` will create a plot by label. The label will be sorted and the `labels`
    argument must be given in the same order as the sorted labels.

    If no label and no color are provided, a default color will be used for all points.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the data and the classes to plot.
    cluster_col_1 : str, default 'col_1'
        First column of the DataFrame containing the cluster's values.
    cluster_col_2 : str, default 'col_2'
        Second column of the DataFrame containing the cluster's values.
    label_col : str, optional
        Column of the DataFrame containing the labels of the data.
        The labels will be sorted so the `labels` parameter must be in the same order.
        If given, there will be one color by label. Colors can be
        provided with the `colors` argument.
    colors : sequence of str, optional
        Colors for each classes to plot or for each point of data if there is no label.
    labels : str or sequence of str, optional
        Labels of the plotted classes, must be in the same order as the real labels,
        which are ordered, and the same length. If no class, can be a single value.
    label_x : str, default 'Dimension 1'
        Label for x axis.
    label_y : str, default 'Dimension 2'
        Label for y axis.
    title : str, default 't-SNE'
        Title for the plot (axis level).
    ax : plt.axes, optional
        Axes from matplotlib, if None, new figure and axes will be created.
    loc : str or int, default 'best'
        Location of the legend on the plot.
        Either the legend string or legend code are possible.
    s : number or sequence
        Size of the points on the graph. Default is the matplotlib markersize * 2.
    figsize : Tuple[int, int], default (14, 7)
        Size of the figure to plot.
    dpi : int, default 80
        Resolution of the figure.
    style : str, default 'default'
        Style to use for matplotlib.pyplot.
        The style is use only in this context and not applied globally.
    **kwargs
        Additional keyword arguments to be passed to the
        `plt.scatter` function from matplotlib.

    Returns
    -------
    plt.axes
        Axes returned by the `plt.subplots` function.

    Examples
    --------
    >>> from sklearn import datasets, manifold
    >>> X, y = datasets.make_circles(n_samples=300, factor=.5, noise=.05)
    >>> df = pd.DataFrame(X).assign(label=y)
    >>> tsne = manifold.TSNE()
    >>> tsne_results = tsne.fit_transform(df.drop('label', axis='columns'))
    >>> df_tsne = df[['label']].assign(tsne_1=tsne_results[:, 0], tsne_2=tsne_results[:, 1])
    >>> plot_tsne(df_tsne, cluster_col_1='tsne_1', cluster_col_2='tsne_2',
    ...           label_col='label', colors=['r', 'b'])
    """
    assert cluster_col_1 in df.columns, (
        f'DataFrame does not contain column: {cluster_col_1}')
    assert cluster_col_2 in df.columns, (
        f'DataFrame does not contain column: {cluster_col_2}')
    if label_col is not None:
        assert label_col in df.columns, (
            f'DataFrame does not contain column: {label_col}')

    with plt.style.context(style):
        if ax is None:
            __, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Flag to determine if the legend must be printed.
        print_legend = True

        # If there is a label, use one color by label.
        # If colors are not provided, or wrong number, create them.
        if label_col is not None:
            labels_unique = df[label_col].unique()
            if colors is None:
                colors = get_n_colors(len(labels_unique), 'rainbow')
            else:
                colors = bff.value_2_list(colors)
                if len(colors) != len(labels_unique):
                    LOGGER.warning(f'Number of colors does not match the number of labels '
                                   f'({len(colors)}/{len(labels_unique)}), '
                                   f'using last color for missing ones.')
                    colors = colors + [colors[-1]] * \
                        (len(labels_unique) - len(colors))  # type: ignore

            if labels is not None:
                labels = bff.value_2_list(labels)
                if len(labels) < len(labels_unique):
                    LOGGER.warning(f'Not enough labels ({len(labels)}/{len(labels_unique)}).')
                    labels = labels + [None] * (len(labels_unique) - len(labels))  # type: ignore
            else:
                print_legend = False
                labels = [None] * len(labels_unique)  # type: ignore

            for i, l in enumerate(sorted(df[label_col].unique())):
                data_1 = df.query(f'{label_col} == @l')[cluster_col_1].values
                data_2 = df.query(f'{label_col} == @l')[cluster_col_2].values
                ax.scatter(
                    data_1,
                    data_2,
                    s=s,
                    c=np.array([colors[i]]),  # type: ignore
                    label=labels[i],  # type: ignore
                    lw=0.1,
                    **kwargs
                )
        # If there is no label, plot all the points.
        # If there are some colors, uses the colors, else plot all with the same color.
        else:
            print_legend = False
            if colors is None or len(colors) != df.shape[0]:
                colors = None
            data_1 = df[cluster_col_1].values
            data_2 = df[cluster_col_2].values
            ax.scatter(
                data_1,
                data_2,
                s=s,
                c=colors,
                label=labels,
                lw=0.1,
                alpha=1,
                **kwargs
            )

        ax.set_xlabel(label_x, fontsize=12)
        ax.set_ylabel(label_y, fontsize=12)
        ax.set_title(title, fontsize=14)

        # Style.
        # Remove border on the top and right.
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Set alpha on remaining borders.
        ax.spines['left'].set_alpha(0.4)
        ax.spines['bottom'].set_alpha(0.4)

        # Only show ticks on the left and bottom spines
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        # Style of ticks.
        plt.xticks(fontsize=10, alpha=0.7)
        plt.yticks(fontsize=10, alpha=0.7)

        if print_legend:
            ax.legend(loc=loc)

        return ax


def plot_confusion_matrix(y_true: Union[np.array, pd.Series, Sequence],
                          y_pred: Union[np.array, pd.Series, Sequence],
                          labels_filter: Optional[Union[np.array, Sequence]] = None,
                          ticklabels: Any = 'auto',
                          sample_weight: Optional[str] = None,
                          normalize: Optional[str] = None,
                          stats: Optional[str] = None,
                          annotation_fmt: Optional[str] = None,
                          cbar_fmt: Optional[FuncFormatter] = None,
                          title: str = 'Confusion matrix',
                          ax: Optional[plt.axes] = None,
                          rotation_xticks: Union[float, None] = 90,
                          rotation_yticks: Optional[float] = None,
                          figsize: Tuple[int, int] = (13, 10),
                          dpi: int = 80,
                          style: str = 'white') -> plt.axes:
    """
    Plot the confusion matrix.

    The confusion matrix is computed in the function.

    Parameters
    ----------
    y_true : np.array, pd.Series or Sequence
        Actual values.
    y_pred : np.array, pd.Series or Sequence
        Predicted values by the model.
    labels_filter : array-like of shape (n_classes,), default None
        List of labels to index the matrix. This may be used to reorder or
        select a subset of labels. If `None` is given, those that appear at
        least once in `y_true` or `y_pred` are used in sorted order.
    ticklabels : 'auto', bool, list-like, or int, default 'auto'
        If True, plot the column names of the DataFrame. If False, don’t plot the column names.
        If list-like, plot these alternate labels as the xticklabels. If an integer,
        use the column names but plot only every n label. If “auto”,
        try to densely plot non-overlapping labels.
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights.
    normalize : str {'true', 'pred', 'all'}, optional
        Normalizes confusion matrix over the true (rows), predicted (columns)
        conditions or all the population. If None, confusion matrix will not be
        normalized.
    stats : str {'accuracy', 'precision', 'recall', 'f1-score'}, optional
        Calculate and display the wanted statistic below the figure.
    annotation_fmt : str, optional
        Format for the annotation on the confusion matrix.
        If not provided, default value is ',d' or '.2f' if normalize is given.
    cbar_fmt : FuncFormatter, optional
        Formatter for the colorbar. Default is with thousand separator and one decimal.
        If normalize and not provided, cbar format is not changed.
    title : str, default 'Confusion matrix'
        Title for the plot (axis level).
    ax : plt.axes, optional
        Axes from matplotlib, if None, new figure and axes will be created.
    rotation_xticks : float or None, default 90
        Rotation of x ticks if any.
    rotation_yticks : float, optional
        Rotation of x ticks if any.
        Set to 90 to put them vertically.
    figsize : Tuple[int, int], default (13, 10)
        Size of the figure to plot.
    dpi : int, default 80
        Resolution of the figure.
    style : str, default 'white'
        Style to use for seaborn.axes_style.
        The style is use only in this context and not applied globally.

    Returns
    -------
    plt.axes
        Axes returned by the `plt.subplots` function.

    Examples
    --------
    >>> y_true = ['dog', 'cat', 'bird', 'cat', 'dog', 'dog']
    >>> y_pred = ['cat', 'cat', 'bird', 'dog', 'bird', 'dog']
    >>> plot_confusion_matrix(y_true, y_pred, stats='accuracy')
    """
    bff.fancy._check_sklearn_support('plot_confusion_matrix')
    from sklearn.metrics import classification_report, confusion_matrix

    # Compute the confusion matrix.
    cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weight,
                          labels=labels_filter, normalize=normalize)

    with sns.axes_style(style):
        if ax is None:
            __, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

        if ticklabels in (True, 'auto') and labels_filter is None:
            ticklabels = sorted(set(list(y_true) + list(y_pred)))

        if annotation_fmt is None:
            annotation_fmt = '.2g' if normalize else ',d'

        if cbar_fmt is None:
            if np.max(cm) > 1_000:
                cbar_fmt = FuncFormatter(lambda x, p: format(int(x), ',d'))
        # Draw the heatmap with the mask and correct aspect ratio.
        sns.heatmap(cm, cmap=plt.cm.Blues, ax=ax, annot=True, fmt=annotation_fmt, square=True,
                    linewidths=0.5, cbar_kws={'shrink': 0.75, 'format': cbar_fmt},
                    xticklabels=ticklabels, yticklabels=ticklabels)

        if stats:
            report = classification_report(y_true, y_pred,
                                           labels=labels_filter,
                                           target_names=ticklabels,
                                           sample_weight=sample_weight,
                                           output_dict=True)
            if stats == 'accuracy':
                ax.text(1.05, 0.05, f'{report[stats]:.2f}', horizontalalignment='left',
                        verticalalignment='center', transform=ax.transAxes)
            else:
                # Depending on the metric, there is one value by class.
                # For each class, print the value of the metric.
                for i, label in enumerate(ticklabels):
                    if stats in report[label].keys():
                        ax.text(1.05, 0.05 - (0.03 * i),
                                f'{label}: {report[label][stats]:.2f}',
                                horizontalalignment='left',
                                verticalalignment='center', transform=ax.transAxes)
                    else:
                        LOGGER.error(f'Wrong key {stats}, possible values: '
                                     f'{list(report[label].keys())}.')
            # Print the metric used.
            if stats in report.keys() or stats in report[ticklabels[0]].keys():
                ax.text(1.05, 0.08, f'{stats.capitalize()}', fontweight='bold',
                        horizontalalignment='left',
                        verticalalignment='center', transform=ax.transAxes)

        ax.set_xlabel('Predicted label', fontsize=12)
        ax.set_ylabel('True label', fontsize=12)
        ax.set_title(title, fontsize=14)
        # Style of the ticks.
        plt.xticks(fontsize=12, alpha=1, rotation=rotation_xticks)
        plt.yticks(fontsize=12, alpha=1, rotation=rotation_yticks)

        return ax


def plot_correlation(df: pd.DataFrame,
                     already_computed: bool = False,
                     method: str = 'pearson',
                     title: str = 'Correlation between variables',
                     ax: Optional[plt.axes] = None,
                     rotation_xticks: Optional[float] = 90,
                     rotation_yticks: Optional[float] = None,
                     figsize: Tuple[int, int] = (13, 10),
                     dpi: int = 80,
                     style: str = 'white',
                     **kwargs) -> plt.axes:
    """
    Plot the correlation between variables of a pandas DataFrame.

    The computing of the correlation can be done either in the
    function or before.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the values or the correlations.
    already_computed : bool, default False
        Set to True if the DataFrame already contains the correlations.
    method : str, default 'pearson'
        Type of normalization. See pandas.DataFrame.corr for possible values.
    title : str, default 'Correlation between variables'
        Title for the plot (axis level).
    ax : plt.axes, optional
        Axes from matplotlib, if None, new figure and axes will be created.
    rotation_xticks : float or None, default 90
        Rotation of x ticks if any.
    rotation_yticks : float, optional
        Rotation of x ticks if any.
        Set to 90 to put them vertically.
    figsize : Tuple[int, int], default (13, 10)
        Size of the figure to plot.
    dpi : int, default 80
        Resolution of the figure.
    style : str, default 'white'
        Style to use for seaborn.axes_style.
        The style is use only in this context and not applied globally.
    **kwargs
        Additional keyword arguments to be passed to the
        `sns.heatmap` function from seaborn.

    Returns
    -------
    plt.axes
        Axes returned by the `plt.subplots` function.
    """
    # Compute the correlation if needed.
    if not already_computed:
        df = df.corr(method=method)

    # Generate a mask for the upper triangle.
    mask = np.zeros_like(df, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True     # pylint: disable=unsupported-assignment-operation

    # Generate a custom diverging colormap.
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    with sns.axes_style(style):
        if ax is None:
            __, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

        # Draw the heatmap with the mask and correct aspect ratio.
        sns.heatmap(df, mask=mask, cmap=cmap, ax=ax, vmin=-1, vmax=1, center=0,
                    annot=True, square=True, linewidths=0.5,
                    cbar_kws={'shrink': 0.75}, **kwargs)

        ax.set_title(title, fontsize=14)
        # Style.
        # Remove border on the top and right.
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        # Style of ticks.
        plt.xticks(fontsize=10, alpha=0.7, rotation=rotation_xticks)
        plt.yticks(fontsize=10, alpha=0.7, rotation=rotation_yticks)

        return ax


def plot_counter(counter: Union[Counter, dict],
                 label_x: str = 'x',
                 label_y: str = 'y',
                 title: str = 'Bar chart',
                 width: float = 0.9,
                 threshold: int = 0,
                 vertical: bool = True,
                 ax: Optional[plt.axes] = None,
                 rotation_xticks: Optional[float] = None,
                 grid: Union[str, None] = 'y',
                 figsize: Tuple[int, int] = (14, 5),
                 dpi: int = 80,
                 style: str = 'default',
                 **kwargs) -> plt.axes:
    """
    Plot the values of a counter as a bar plot.

    Values above the ratio are written as text on top of the bar.

    Parameters
    ----------
    counter : collections.Counter or dictionary
        Counter or dictionary to plot.
    label_x : str, default 'x'
        Label for x axis.
    label_y : str, default 'y'
        Label for y axis.
    title : str, default 'Bar chart'
        Title for the plot (axis level).
    width : float, default 0.9
        Width of the bar. If below 1.0, there will be space between them.
    threshold : int, default = 0
        Threshold above which the value is written on the plot as text.
        By default, all bar have their text.
    vertical : bool, default True
        By default, vertical bar. If set to False, will plot using `plt.barh`
        and inverse the labels.
    ax : plt.axes, optional
        Axes from matplotlib, if None, new figure and axes will be created.
    loc : str or int, default 'best'
        Location of the legend on the plot.
        Either the legend string or legend code are possible.
    rotation_xticks : float, optional
        Rotation of x ticks if any.
        Set to 90 to put them vertically.
    grid : str or None, default 'y'
        Axis where to activate the grid ('both', 'x', 'y').
        To turn off, set to None.
    figsize : Tuple[int, int], default (14, 5)
        Size of the figure to plot.
    dpi : int, default 80
        Resolution of the figure.
    style : str, default 'default'
        Style to use for matplotlib.pyplot.
        The style is use only in this context and not applied globally.
    **kwargs
        Additional keyword arguments to be passed to the
        `plt.plot` function from matplotlib.

    Returns
    -------
    plt.axes
        Axes returned by the `plt.subplots` function.

    Examples
    --------
    >>> from collections import Counter
    >>> counter = Counter({'red': 4, 'blue': 2})
    >>> plot_counter(counter, title='MyTitle', rotation_xticks=90)
    """
    with plt.style.context(style):
        if ax is None:
            __, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

        labels, values = zip(*sorted(counter.items()))

        indexes = np.arange(len(labels))

        if vertical:
            ax.bar(indexes, values, width, **kwargs)
        else:
            ax.barh(indexes, values, width, **kwargs)
            label_x, label_y = label_y, label_x

        ax.set_xlabel(label_x, fontsize=12)
        ax.set_ylabel(label_y, fontsize=12)
        ax.set_title(title, fontsize=14)

        # Write the real value on the bar if above a given threshold.
        counter_max = max(counter.values())
        space = counter_max * 1.01 - counter_max
        for i in ax.patches:
            if vertical:
                if i.get_height() > threshold:
                    ax.text(i.get_x() + i.get_width() / 2, i.get_height() + space,
                            f'{i.get_height():^{len(str(counter_max))},}',
                            ha='center', va='bottom',
                            fontsize=10, color='black', alpha=0.6)
            else:
                if i.get_width() > threshold:
                    ax.text(i.get_width() + space, i.get_y() + i.get_height() / 2,
                            f'{i.get_width():>{len(str(counter_max))},}',
                            ha='left', va='center',
                            fontsize=10, color='black', alpha=0.6)

        # Style.
        # Remove border on the top and right.
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Set alpha on remaining borders.
        ax.spines['left'].set_alpha(0.4)
        ax.spines['bottom'].set_alpha(0.4)

        # Remove ticks on y axis.
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('none')

        # Draw tick lines on wanted axes.
        if grid:
            ax.axes.grid(True, which='major', axis=grid, color='black',
                         alpha=0.3, linestyle='--', lw=0.5)

        # Style of ticks.
        set_thousands_separator(ax, which='both', nb_decimals=1)
        if vertical:
            plt.xticks(indexes, labels, fontsize=10, alpha=0.7, rotation=rotation_xticks)
            plt.yticks(fontsize=10, alpha=0.7)
        else:
            plt.xticks(fontsize=10, alpha=0.7, rotation=rotation_xticks)
            plt.yticks(indexes, labels, fontsize=10, alpha=0.7)

        return ax


def plot_history(history: dict,
                 metric: Optional[str] = None,
                 twinx: bool = False,
                 title: str = 'Model history',
                 axes: Optional[plt.axes] = None,
                 loc: Union[str, int] = 'best',
                 grid: Optional[str] = None,
                 figsize: Tuple[int, int] = (12, 7),
                 dpi: int = 80,
                 style: str = 'default',
                 **kwargs) -> Union[plt.axes, Sequence[plt.axes]]:
    """
    Plot the history of the model trained using Keras.

    Parameters
    ----------
    history : dict
        Dictionary from the history object of the training.
    metric : str, optional
        Metric to plot.
        If no metric is provided, will only print the loss.
    twinx : bool, default False
        If metric and twinx, plot the loss and the metric on the same axis.
        Four colors will be used for the plot.
    title : str, default 'Model history'
        Main title for the plot (figure level).
    axes : plt.axes, optional
        Axes from matplotlib, if None, new figure and axes will be created.
        If metric is provided, need to have at least 2 axes.
    loc : str or int, default 'best'
        Location of the legend on the plot.
        Either the legend string or legend code are possible.
    grid : str, optional
        Axis where to activate the grid ('both', 'x', 'y').
        To turn off, set to None.
    figsize : Tuple[int, int], default (12, 7)
        Size of the figure to plot.
    dpi : int, default 80
        Resolution of the figure.
    style : str, default 'default'
        Style to use for matplotlib.pyplot.
        The style is use only in this context and not applied globally.
    **kwargs
        Additional keyword arguments to be passed to the
        `plt.plot` function from matplotlib.

    Returns
    -------
    plt.axes
        Axes object or array of Axes objects returned by the `plt.subplots`
        function.

    Examples
    --------
    >>> history = model.fit(...)
    >>> plot_history(history.history, metric='acc', title='MyTitle', linestyle=':')
    """
    if metric:
        assert metric in history.keys(), (
            f'Metric {metric} does not exist in history.\n'
            f'Possible metrics: {history.keys()}')

    with plt.style.context(style):
        # Given axes are not check for now.
        # If metric is given, must have at least 2 axes.
        # If two axes, share the x.
        two_axes = bool(metric) and not twinx
        if axes is None:
            fig, axes = plt.subplots(2 if two_axes else 1, 1,
                                     sharex=two_axes,
                                     figsize=figsize, dpi=dpi)
        else:
            fig = plt.gcf()

        # The loss is always plot.
        ax_loss = axes[1] if two_axes else axes
        ax_loss.plot(history['loss'],
                     label=f"Training loss ({history['loss'][-1]:.3f})",
                     **kwargs)
        # If there is a validation, plot it as well.
        if 'val_loss' in history.keys():
            ax_loss.plot(history['val_loss'],
                         label=f"Validation loss ({history['val_loss'][-1]:.3f})",
                         **kwargs)
        ax_loss.set_xlabel('Epochs', fontsize=12)
        ax_loss.set_ylabel('Loss', fontsize=12)

        # Plot the metric, if any if provided.
        # If `twinx`, plot on same axis with another scale.
        # If not, use a subplot below the loss.
        if metric:
            ax_metric = axes.twinx() if twinx else axes[0]
            # If twinx, share the same prop cycler to have different colors.
            if twinx:
                ax_metric._get_lines.prop_cycler = ax_loss._get_lines.prop_cycler

            ax_metric.plot(history[metric],
                           label=f"Training {metric} ({history[metric][-1]:.3f})",
                           **kwargs)
            # Plot the validation if any.
            if f'val_{metric}' in history.keys():
                ax_metric.plot(history[f'val_{metric}'],
                               label=f"Validation {metric} ({history[f'val_{metric}'][-1]:.3f})",
                               **kwargs)
            ax_metric.set_ylabel(metric.capitalize(), fontsize=12)

        # Global title of the plot, if multiple subplots.
        if two_axes:
            fig.suptitle(title, fontsize=14)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        else:
            ax_loss.set_title(title, fontsize=14)

        # Style, applied on all axis.
        axes_to_style = [ax_loss, ax_metric] if bool(metric) else [ax_loss]
        for ax in axes_to_style:
            # Remove border on the top and right.
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # Set alpha on remaining borders.
            ax.spines['left'].set_alpha(0.4)
            ax.spines['bottom'].set_alpha(0.4)

            # Keep ticks on the left.
            ax.yaxis.set_ticks_position('left')
            if not two_axes:
                ax.xaxis.set_ticks_position('bottom')

            # Draw tick lines on wanted axes.
            if grid:
                ax.axes.grid(True, which='major', axis=grid, color='black',
                             alpha=0.3, linestyle='--', lw=0.5)

            # For the xticks, check if need to plot the decimal.
            if all(i.is_integer() for i in ax.get_xticks()):
                set_thousands_separator(ax, which='x', nb_decimals=0)
            else:
                set_thousands_separator(ax, which='x', nb_decimals=1)
            set_thousands_separator(ax, which='y', nb_decimals=2)

        # If `twinx`, add the right spine.
        if bool(metric) and twinx:
            ax_metric.spines['right'].set_visible(True)
            ax_metric.spines['right'].set_alpha(0.4)
            ax_metric.yaxis.set_ticks_position('right')

            # Add the legend manually since the number of entries may vary.
            handles = [ax.get_legend_handles_labels() for ax in axes_to_style]
            ax_loss.legend(itertools.chain.from_iterable([i[0] for i in handles]),
                           itertools.chain.from_iterable([i[1] for i in handles]), loc=loc)
        else:
            ax_loss.legend(loc=loc)
            if bool(metric):
                ax_metric.legend(loc=loc)

        return axes


def plot_kmeans(df: pd.DataFrame,
                kmeans_col_1: str = 'kmeans_1',
                kmeans_col_2: str = 'kmeans_2',
                label_col: Optional[str] = 'label',
                centroids: Optional[Sequence[Tuple[float, float]]] = None,
                cmap: str = 'viridis',
                label_x: str = 'Dimension 1',
                label_y: str = 'Dimension 2',
                title: str = 'K-Means',
                ax: Optional[plt.axes] = None,
                loc: Union[str, int] = 'best',
                s: TNum = mpl.rcParams['lines.markersize'] * 2,
                figsize: Tuple[int, int] = (14, 7),
                dpi: int = 80,
                style: str = 'default',
                **kwargs) -> plt.axes:
    """
    Plot K-Means clustering in two dimensions.

    K-Means must be already computed and stored inside two separate column of the DataFrame.
    See the example for more information.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the data and the classes to plot.
    kmeans_col_1 : str, default 'kmeans_1'
        First column of the DataFrame containing the kmeans values.
    kmeans_col_2 : str, default 'kmeans_2'
        Second column of the DataFrame containing the kmeans values.
    label_col : str, default 'label'
        Column of the DataFrame containing the results of kmeans.
    centroids : sequence of tuples of two floats, optional
        Centroids of the clusters to plot, if provided.
        There should be two coordinates by centroid.
    cmap : str, default 'viridis'
        Color map for the color of the kmeans' classes.
    label_x : str, default 'Dimension 1'
        Label for x axis.
    label_y : str, default 'Dimension 2'
        Label for y axis.
    title : str, default 'K-Means'
        Title for the plot (axis level).
    ax : plt.axes, optional
        Axes from matplotlib, if None, new figure and axes will be created.
    loc : str or int, default 'best'
        Location of the legend on the plot.
        Either the legend string or legend code are possible.
    s : number
        Size of the points on the graph. Default is the matplotlib markersize * 2.
    figsize : Tuple[int, int], default (14, 7)
        Size of the figure to plot.
    dpi : int, default 80
        Resolution of the figure.
    style : str, default 'default'
        Style to use for matplotlib.pyplot.
        The style is use only in this context and not applied globally.
    **kwargs
        Additional keyword arguments to be passed to the
        `plt.scatter` function from matplotlib.

    Returns
    -------
    plt.axes
        Axes returned by the `plt.subplots` function.

    Examples
    --------
    >>> from sklearn.cluster import KMeans
    >>> kmeans = KMeans(n_clusters=3, random_state=0).fit(pca)
    >>> y = kmeans.predict(pca)
    >>> centers = kmeans.cluster_centers_
    >>> df = pd.DataFrame({'kmeans_1': pca[:, 0], 'kmeans_2': pca[:, 1], 'label': y})
    >>> plot_kmeans(df, centroids=centers)
    """
    assert kmeans_col_1 in df.columns, (
        f'DataFrame does not contain column: {kmeans_col_1}')
    assert kmeans_col_2 in df.columns, (
        f'DataFrame does not contain column: {kmeans_col_2}')
    assert label_col in df.columns, (
        f'DataFrame does not contain column: {label_col}')

    with plt.style.context(style):
        if ax is None:
            __, ax = plt.subplots(figsize=figsize, dpi=dpi)

        ax.scatter(
            df[kmeans_col_1].values,
            df[kmeans_col_2].values,
            s=s,
            c=df[label_col].values,
            cmap=cmap,
            lw=0.1,
            alpha=1,
            **kwargs
        )

        if centroids is not None:
            centroids_x = [v[0] for v in centroids]
            centroids_y = [v[1] for v in centroids]
            ax.scatter(centroids_x, centroids_y, label='Centroids',
                       c='black', s=s * 10, alpha=0.6)
            ax.legend(loc=loc)

        ax.set_xlabel(label_x, fontsize=12)
        ax.set_ylabel(label_y, fontsize=12)
        ax.set_title(title, fontsize=14)

        # Style.
        # Remove border on the top and right.
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Set alpha on remaining borders.
        ax.spines['left'].set_alpha(0.4)
        ax.spines['bottom'].set_alpha(0.4)

        # Only show ticks on the left and bottom spines
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        # Style of ticks.
        plt.xticks(fontsize=10, alpha=0.7)
        plt.yticks(fontsize=10, alpha=0.7)

        set_thousands_separator(ax, which='both', nb_decimals=0)

        plt.tight_layout()

        return ax


def plot_pca_explained_variance_ratio(pca,
                                      label_x: str = 'Number of components',
                                      label_y: str = 'Cumulative explained variance',
                                      title: str = 'PCA explained variance ratio',
                                      hline: Optional[float] = None,
                                      ax: Optional[plt.axes] = None,
                                      lim_x: Optional[Tuple[TNum, TNum]] = None,
                                      lim_y: Optional[Tuple[TNum, TNum]] = None,
                                      grid: Optional[str] = None,
                                      figsize: Tuple[int, int] = (10, 4), dpi: int = 80,
                                      style: str = 'default', **kwargs) -> plt.axes:
    """
    Plot the explained variance ratio of PCA.

    PCA must be already fitted.

    Parameters
    ----------
    pca : sklearn.decomposition.PCA
        PCA object to plot.
    label_x : str, default 'Number of components'
        Label for x axis.
    label_y : str, default 'Cumulative explained variance'
        Label for y axis.
    title : str, default 'PCA explained variance ratio'
        Title for the plot (axis level).
    hline : float, optional
        Horizontal line (darkorange) to draw on the plot (e.g. at 0.8 to see
        the number of components needed to keep 80% of the variance).
    ax : plt.axes, optional
        Axes from matplotlib, if None, new figure and axes will be created.
    lim_x : Tuple[TNum, TNum], optional
        Limit for the x axis.
    lim_y : Tuple[TNum, TNum], optional
        Limit for the y axis.
    grid : str, optional
        Axis where to activate the grid ('both', 'x', 'y').
        To turn off, set to None.
    figsize : Tuple[int, int], default (10, 4)
        Size of the figure to plot.
    dpi : int, default 80
        Resolution of the figure.
    style : str, default 'default'
        Style to use for matplotlib.pyplot.
        The style is use only in this context and not applied globally.
    **kwargs
        Additional keyword arguments to be passed to the
        `plt.plot` function from matplotlib.

    Returns
    -------
    plt.axes
        Axes returned by the `plt.subplots` function.

    Examples
    --------
    >>> from sklearn.decomposition import PCA
    >>> pca = PCA(n_components=20)
    >>> pca.fit(data)
    >>> plot_pca_explained_variance_ratio(pca)
    """
    with plt.style.context(style):
        if ax is None:
            __, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

        ax.plot(np.cumsum(pca.explained_variance_ratio_))

        if hline:
            ax.axhline(hline, color='darkorange', alpha=0.5)

        if lim_x:
            ax.set_xlim(lim_x)
        if lim_y:
            ax.set_ylim(lim_y)

        ax.set_xlabel(label_x, fontsize=12)
        ax.set_ylabel(label_y, fontsize=12)
        ax.set_title(title, fontsize=14)

        # Style.
        # Remove border on the top and right.
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Set alpha on remaining borders.
        ax.spines['left'].set_alpha(0.4)
        ax.spines['bottom'].set_alpha(0.4)

        # Remove ticks on y axis.
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        # Style of ticks.
        plt.xticks(fontsize=10, alpha=0.7)
        plt.yticks(fontsize=10, alpha=0.7)

        # Draw tick lines on wanted axes.
        if grid:
            ax.axes.grid(True, which='major', axis=grid, color='black',
                         alpha=0.3, linestyle='--', lw=0.5)

        set_thousands_separator(ax, which='x', nb_decimals=0)
        set_thousands_separator(ax, which='y', nb_decimals=2)

        plt.tight_layout()

        return ax


def plot_pie(data: Union[Counter, dict],
             explode: float = 0.0,
             circle: bool = True,
             colors: Optional[Sequence[str]] = None,
             textprops: Optional[dict] = None,
             title: str = 'Pie chart',
             ax: Optional[plt.axes] = None,
             loc: Optional[Union[str, int]] = None,
             figsize: Tuple[int, int] = (14, 8),
             dpi: int = 80,
             style: str = 'default',
             **kwargs) -> plt.axes:
    """
    Plot the values of a dictionary as a pie chart or a circle.

    Parameters
    ----------
    data : collections.Counter or dictionary
        Data to plot in the pie, labels as keys and count as values.
    explode : float, default 0.0
        If provided, explode the pie with the given value.
    circle : bool, default True
        If True, empty the center of the pie, making it look like a circle.
    colors : sequence of str, optional
        If provided, use the given colors, else use a default cmap (`plt.cm.rainbow`).
    textprops : dict, optional
        Dict of arguments to pass to the text objects.
    title : str, default 'Pie chart'
        Title for the plot (axis level).
    ax : plt.axes, optional
        Axes from matplotlib, if None, new figure and axes will be created.
    loc : str or int, optional
        If provided, show the legend on the plot.
    rotation_xticks : float, optional
        Rotation of x ticks if any.
        Set to 90 to put them vertically.
    grid : str or None, default 'y'
        Axis where to activate the grid ('both', 'x', 'y').
        To turn off, set to None.
    figsize : Tuple[int, int], default (14, 8)
        Size of the figure to plot.
    dpi : int, default 80
        Resolution of the figure.
    style : str, default 'default'
        Style to use for matplotlib.pyplot.
        The style is use only in this context and not applied globally.
    **kwargs
        Additional keyword arguments to be passed to the
        `plt.pie` function from matplotlib.

    Returns
    -------
    plt.axes
        Axes returned by the `plt.subplots` function.

    Examples
    --------
    >>> from collections import Counter
    >>> counter = Counter({'red': 4, 'blue': 2, 'green': 7})
    >>> plot_pie(counter, title='MyTitle')
    """
    with plt.style.context(style):
        if ax is None:
            __, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

            # Function to format the labels on the pie chart.
            # Percent, with real value in parentheses.
            def format_label(percent, values):
                absolute = int(percent / 100. * sum(values))
                return f'{percent:.1f}%\n({absolute:,})'

            if colors is None:
                colors = get_n_colors(len(data), 'rainbow')
            else:
                assert len(colors) == len(data), (
                    'The number of colors does not match the number of labels.')

            wedges, texts, autotexts = ax.pie(data.values(), labels=data.keys(),
                                              colors=colors,
                                              autopct=lambda pct: format_label(pct, data.values()),
                                              explode=[explode] * len(data),
                                              pctdistance=0.85 if circle else 0.5,
                                              textprops=textprops, **kwargs)
            # Draw the inside circle.
            if circle:
                plt.setp(wedges, width=0.3)
                # Ensure that the pie is a circle.
                ax.set_aspect('equal')

            # Change the size of the labels.
            for text in texts:
                text.set_fontsize(14)
            plt.setp(autotexts, size=12, weight='bold')

            ax.set_title(title, fontsize=14)

            if loc:
                ax.legend(loc=loc)

            plt.tight_layout()

            return ax


def plot_predictions(y_true: Union[np.array, pd.Series, pd.DataFrame],
                     y_pred: Union[np.array, pd.Series, pd.DataFrame],
                     x_true: Optional[Union[np.array, pd.Series, pd.DataFrame]] = None,
                     x_pred: Optional[Union[np.array, pd.Series, pd.DataFrame]] = None,
                     label_true: str = 'Actual',
                     label_pred: str = 'Predicted',
                     label_x: str = 'x',
                     label_y: str = 'y',
                     title: str = 'Model predictions',
                     ax: Optional[plt.axes] = None,
                     loc: Union[str, int] = 'best',
                     rotation_xticks: Optional[float] = None,
                     grid: Union[str, None] = 'y',
                     figsize: Tuple[int, int] = (14, 5),
                     dpi: int = 80,
                     style: str = 'default',
                     **kwargs) -> plt.axes:
    """
    Plot the predictions of the model.

    If a DataFrame is provided, it must only contain one column.

    Parameters
    ----------
    y_true : np.array, pd.Series or pd.DataFrame
        Actual values.
    y_pred : np.array, pd.Series or pd.DataFrame
        Predicted values by the model.
    x_true : np.array, pd.Series, pd.DataFrame, optional
        X coordinates for actual values.
        If not given, will be integer starting from 0.
    x_pred : np.array, pd.Series, pd.DataFrame, optional
        X coordinates for predicted values.
        If not given, will be integer starting from 0.
    label_true : str, default 'Actual'
        Label for the actual values.
    label_pred : str, default 'Predicted'
        Label for the predicted values.
    label_x : str, default 'x'
        Label for x axis.
    label_y : str, default 'y'
        Label for y axis.
    title : str, default 'Model predictions'
        Title for the plot (axis level).
    ax : plt.axes, default None
        Axes from matplotlib, if None, new figure and axes will be created.
    loc : str or int, default 'best'
        Location of the legend on the plot.
        Either the legend string or legend code are possible.
    rotation_xticks : float, optional
        Rotation of x ticks if any.
        Set to 90 to put them vertically.
    grid : str or None, default 'y'
        Axis where to activate the grid ('both', 'x', 'y').
        To turn off, set to None.
    figsize : Tuple[int, int], default (14, 5)
        Size of the figure to plot.
    dpi : int, default 80
        Resolution of the figure.
    style : str, default 'default'
        Style to use for matplotlib.pyplot.
        The style is use only in this context and not applied globally.
    **kwargs
        Additional keyword arguments to be passed to the
        `plt.plot` function from matplotlib.

    Returns
    -------
    plt.axes
        Axes returned by the `plt.subplots` function.

    Examples
    --------
    >>> y_pred = model.predict(x_test, ...)
    >>> plot_predictions(y_true, y_pred, title='MyTitle', linestyle=':')
    """
    with plt.style.context(style):
        if ax is None:
            __, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

        # If x is given, it must be the same length as y.
        if x_pred is not None:
            assert len(x_pred) == len(y_pred), '`x_pred` and `y_pred` must have the same size.'
        if x_true is not None:
            assert len(x_true) == len(y_true), '`x_true` and `y_true` must have the same size.'

        # Plot predictions.
        if x_pred is not None:
            ax.plot(x_pred, np.array(y_pred).flatten(), color='r',
                    label=label_pred, **kwargs)
        else:
            ax.plot(np.array(y_pred).flatten(), color='r',
                    label=label_pred, **kwargs)
        # Plot actual values on top of the predictions.
        if x_true is not None:
            ax.plot(x_true, np.array(y_true).flatten(), color='b',
                    label=label_true, **kwargs)
        else:
            ax.plot(np.array(y_true).flatten(), color='b',
                    label=label_true, **kwargs)

        ax.set_xlabel(label_x, fontsize=12)
        ax.set_ylabel(label_y, fontsize=12)
        ax.set_title(title, fontsize=14)

        # Style.
        # Remove borders on the top and right.
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Set alpha on remaining borders.
        ax.spines['left'].set_alpha(0.4)
        ax.spines['bottom'].set_alpha(0.4)

        # Remove ticks on y axis.
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('none')
        # Style of ticks.
        plt.xticks(fontsize=10, alpha=0.7, rotation=rotation_xticks)
        plt.yticks(fontsize=10, alpha=0.7)

        # Draw tick lines on wanted axes.
        if grid:
            ax.axes.grid(True, which='major', axis=grid, color='black',
                         alpha=0.3, linestyle='--', lw=0.5)

        # If x are given, it might not be numerical.
        if x_pred is None and x_true is None:
            set_thousands_separator(ax, which='both', nb_decimals=1)

        # Sort labels and handles by labels.
        handles, labels = ax.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles),
                                      key=lambda t: t[0]))
        ax.legend(handles, labels, loc=loc)

        return ax


def plot_series(df: pd.DataFrame,
                column: str,
                groupby: Optional[str] = None,
                with_sem: bool = False,
                with_peaks: bool = False,
                with_missing_datetimes: bool = False,
                distance_scale: float = 0.04,
                label_x: str = 'Datetime',
                label_y: Optional[str] = None,
                title: str = 'Plot of series',
                ax: Optional[plt.axes] = None,
                color: str = '#3F5D7D',
                loc: Union[str, int] = 'best',
                rotation_xticks: Optional[float] = None,
                grid: Union[str, None] = 'y',
                figsize: Tuple[int, int] = (14, 6),
                dpi: int = 80,
                style: str = 'default',
                **kwargs) -> plt.axes:
    """
    Plot time series with datetime with the given resample (`groupby`).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to plot, with datetime as index.
    column : str
        Column of the DataFrame to display.
    groupby : str, optional
        Grouping for the resampling by mean of the data.
        For example, can resample from seconds ('S') to minutes ('T').
        By default, no resampling is applied.
    with_sem : bool, default False
        Display the standard error of the mean (SEM) if set to true.
        Only possible if a resampling has been done.
    with_peaks : bool, default False
        Display the peaks of the serie if set to true.
    with_missing_datetimes : bool, default False
        Display the missing datetimes with vertical red lines.
        Only possible if a resampling has been done.
    distance_scale: float, default 0.04
        Scaling for the minimal distance between peaks.
        Only used if `with_peaks` is set to true.
    label_x : str, default 'Datetime'
        Label for x axis.
    label_y : str, default None
        Label for y axis. If None, will take the column name as label.
    title : str, default 'Plot of series'
        Title for the plot (axis level).
    ax : plt.axes, optional
        Axes from matplotlib, if None, new figure and ax will be created.
    color : str, default '#3F5D7D'
        Default color for the plot.
    loc : str or int, default 'best'
        Location of the legend on the plot.
        Either the legend string or legend code are possible.
    rotation_xticks : float, optional
        Rotation of x ticks if any.
        Set to 90 to put them vertically.
    grid : str or None, default 'y'
        Axis where to activate the grid ('both', 'x', 'y').
        To turn off, set to None.
    figsize : Tuple[int, int], default (14, 6)
        Size of the figure to plot.
    dpi : int, default 80
        Resolution of the figure.
    style : str, default 'default'
        Style to use for matplotlib.pyplot.
        The style is use only in this context and not applied globally.
    **kwargs
        Additional keyword arguments to be passed to the
        `plt.plot` function from matplotlib.

    Returns
    -------
    plt.axes
        Axes object or array of Axes objects returned by the `plt.subplots`
        function.

    Examples
    --------
    >>> df_acceleration = fake.get_data_with_datetime_index(...)
    >>> _, axes = plt.subplots(nrows=3, ncols=1, figsize=(14, 20), dpi=80)
    >>> colors = {'x': 'steelblue', 'y': 'darkorange', 'z': 'darkgreen'}
    >>> for i, acc in enumerate(['x', 'y', 'z']):
    ...     plot_series(df_acceleration, acc, groupby='T',
    ...                 ax=axes[i], color=colors[acc])
    """
    assert isinstance(df.index, pd.DatetimeIndex), (
        'DataFrame must have a datetime index.')
    assert column in df.columns, (
        f'DataFrame does not contain column: {column}')

    with plt.style.context(style):
        if ax is None:
            __, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # By default, the y label is the column name.
        if label_y is None:
            label_y = column.capitalize()

        # Get the values to plot.
        df_plot = df[column]
        if groupby:
            df_plot = df_plot.resample(groupby).mean()

        x = df_plot.index

        # If the label is not provided, set the column as label.
        if 'label' not in kwargs:
            kwargs['label'] = column

        # Set default linewidth if none of the possible arguments are provided.
        if not any(k in kwargs for k in ['lw', 'linewidth']):
            kwargs['lw'] = 2

        ax.plot(x, df_plot, color=color, **kwargs)

        # With sem (standard error of the mean).
        sem_alpha = 0.3
        if groupby and with_sem:
            df_sem = df_plot.sem()

            ax.fill_between(x, df_plot - df_sem, df_plot + df_sem,
                            color='grey', alpha=sem_alpha)

        # Plot the peak as circle.
        if with_peaks:
            peak_dates, peak_values = bff.fancy.get_peaks(df_plot, distance_scale)
            ax.plot(peak_dates, peak_values, linestyle='', marker='o',
                    color='plum')

        # Plot vertical line where there is missing datetimes.
        if groupby and with_missing_datetimes:
            df_date_missing = pd.date_range(start=df.index.min(),
                                            end=df.index.max(),
                                            freq=groupby).difference(df_plot.dropna().index)
            for date in df_date_missing.tolist():
                ax.axvline(date, color='crimson')

        ax.set_xlabel(label_x, fontsize=12)
        ax.set_ylabel(label_y, fontsize=12)
        # Add the groupby if any
        if groupby:
            title += f' (mean by {groupby})'
        ax.set_title(title, fontsize=14)

        # Style.
        # Remove border on the top and right.
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Set alpha on remaining borders.
        ax.spines['left'].set_alpha(0.4)
        ax.spines['bottom'].set_alpha(0.4)

        # Remove ticks on y axis.
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('none')
        # Style of ticks.
        plt.xticks(fontsize=10, alpha=0.7, rotation=rotation_xticks)
        plt.yticks(fontsize=10, alpha=0.7)

        # Draw tick lines on wanted axes.
        if grid:
            ax.axes.grid(True, which='major', axis=grid, color='black',
                         alpha=0.3, linestyle='--', lw=0.5)

        set_thousands_separator(ax, which='y', nb_decimals=1)

        handles, labels = ax.get_legend_handles_labels()
        labels_cap = [label.capitalize() for label in labels]
        # Add the sem on the legend.
        if groupby and with_sem:
            handles.append(mpatches.Patch(color='grey', alpha=sem_alpha))
            labels_cap.append('Standard error of the mean (SEM)')

        # Add the peak symbol on the legend.
        if with_peaks:
            handles.append(mlines.Line2D([], [], linestyle='', marker='o',
                                         color='plum'))
            labels_cap.append('Peaks')

        # Add the missing date line on the legend.
        if groupby and with_missing_datetimes:
            handles.append(mlines.Line2D([], [], linestyle='-', color='crimson'))
            labels_cap.append('Missing datetimes')

        ax.legend(handles, labels_cap, loc=loc)
        plt.tight_layout()

        return ax


def plot_true_vs_pred(y_true: Union[np.array, pd.DataFrame],
                      y_pred: Union[np.array, pd.DataFrame],
                      with_correlation: bool = False,
                      with_determination: bool = True,
                      with_histograms: bool = False,
                      with_identity: bool = False,
                      label_x: str = 'Ground truth',
                      label_y: str = 'Prediction',
                      title: str = 'Predicted vs Actual',
                      ax: Optional[plt.axes] = None,
                      lim_x: Optional[Tuple[TNum, TNum]] = None,
                      lim_y: Optional[Tuple[TNum, TNum]] = None,
                      grid: Union[str, None] = 'both',
                      figsize: Tuple[int, int] = (14, 7),
                      dpi: int = 80,
                      style: str = 'default',
                      **kwargs) -> plt.axes:
    """
    Plot the ground truth against the predictions of the model.

    If a DataFrame is provided, it must only contain one column.

    Parameters
    ----------
    y_true : np.array or pd.DataFrame
        Actual values.
    y_pred : np.array or pd.DataFrame
        Predicted values by the model.
    with_correlation : bool, default False
        If true, print correlation coefficient in the top left corner.
    with_determination : bool, default True
        If true, print the determination coefficient in the top left corner.
        If both `with_correlation` and `with_determination` are set to true,
        the correlation coefficient is printed.
    with_histograms : bool, default False
        If true, plot histograms of `y_true` and `y_pred` on the sides.
        Not possible if the `ax` is provided.
    with_identity : bool, default False
        If true, plot the identity line on the scatter plot.
    label_x : str, default 'Ground truth'
        Label for x axis.
    label_y : str, default 'Prediction'
        Label for y axis.
    title : str, default 'Predicted vs Actual'
        Title for the plot (axis level).
    ax : plt.axes, optional
        Axes from matplotlib, if None, new figure and axes will be created.
    lim_x : Tuple[TNum, TNum], optional
        Limit for the x axis. If None, automatically calculated according
        to the limits of the data, with an extra 5% for readability.
    lim_y : Tuple[TNum, TNum], optional
        Limit for the y axis. If None, automatically calculated according
        to the limits of the data, with an extra 5% for readability.
    grid : str or None, default 'both'
        Axis where to activate the grid ('both', 'x', 'y').
        To turn off, set to None.
    figsize : Tuple[int, int], default (14, 7)
        Size of the figure to plot.
    dpi : int, default 80
        Resolution of the figure.
    style : str, default 'default'
        Style to use for matplotlib.pyplot.
        The style is use only in this context and not applied globally.
    **kwargs
        Additional keyword arguments to be passed to the
        `plt.plot` function from matplotlib.

    Returns
    -------
    plt.axes
        Axes returned by the `plt.subplots` function.
        If `with_histograms`, return the three axes.

    Examples
    --------
    >>> y_pred = model.predict(x_test, ...)
    >>> plot_true_vs_pred(y_true, y_pred, title='MyTitle', linestyle=':')
    """
    with plt.style.context(style):
        if ax is None:
            if with_histograms:
                fig = plt.figure(figsize=figsize, dpi=dpi)
                grid_plt = plt.GridSpec(4, 4, hspace=0.5, wspace=0.2)
                # Define the axes
                ax_main = fig.add_subplot(grid_plt[:-1, :-1])
                ax_right = fig.add_subplot(grid_plt[:-1, -1], xticklabels=[], yticklabels=[])
                ax_bottom = fig.add_subplot(grid_plt[-1, 0:-1], xticklabels=[], yticklabels=[])
            else:
                __, ax_main = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        else:
            assert not with_histograms, (
                'Option `with_histograms is not possible when providing an axis.')
            ax_main = ax

        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        # Default marker is black if not provided.
        if 'c' not in kwargs:
            kwargs['c'] = 'black'

        ax_main.scatter(y_true, y_pred, **kwargs)

        ax_main.set_xlabel(label_x, fontsize=12)
        ax_main.set_ylabel(label_y, fontsize=12)
        ax_main.set_title(title, fontsize=14)

        # Calculate the limit of the plot if not provided,
        # add and extra 5% for readability.
        def get_limit(limit, data, percent=5):
            if not limit or not isinstance(limit, tuple):
                lim_max = data.max()
                lim_min = data.min()
                margin = (lim_max - lim_min) * percent / 100
                limit = (lim_min - margin, lim_max + margin)
            return limit

        # Set x and y limits.
        ax_main.set_xlim(get_limit(lim_x, y_true))
        ax_main.set_ylim(get_limit(lim_y, y_pred))

        # Add correlation in upper left position.
        if with_correlation:
            ax_main.text(0.025, 0.925,
                         f'$R={np.round(np.corrcoef(y_true, y_pred)[0][1], 3)}$',
                         fontsize=12, transform=ax_main.transAxes)

        # Add coefficient of determination in upper left position.
        # If both `with_determination` and `with_correlation`, prints only the correlation.
        if with_determination and not with_correlation:
            ax_main.text(0.025, 0.925,
                         f'$R^2={np.round(np.corrcoef(y_true, y_pred)[0][1] ** 2, 3)}$',
                         fontsize=12, transform=ax_main.transAxes)

        if with_histograms:
            # Histogram on the right.
            ax_bottom.hist(y_pred, histtype='stepfilled', orientation='vertical',
                           color='cadetblue')
            ax_bottom.invert_yaxis()

            # Histogram in the bottom.
            ax_right.hist(y_true, histtype='stepfilled', orientation='horizontal',
                          color='cadetblue')

            # Style of histograms.
            # Remove borders.
            for spine in ['top', 'right', 'left', 'bottom']:
                ax_bottom.spines[spine].set_alpha(0.3)
                ax_right.spines[spine].set_alpha(0.3)
            # Remove ticks on axis.
            ax_bottom.xaxis.set_ticks_position('none')
            ax_bottom.yaxis.set_ticks_position('none')
            ax_right.xaxis.set_ticks_position('none')
            ax_right.yaxis.set_ticks_position('none')

        # Add identity line on main plot.
        if with_identity:
            add_identity(ax_main, color='royalblue')

        # Style.
        # Remove borders on the top and right.
        ax_main.spines['top'].set_visible(False)
        ax_main.spines['right'].set_visible(False)
        # Set alpha on remaining borders.
        ax_main.spines['left'].set_alpha(0.4)
        ax_main.spines['bottom'].set_alpha(0.4)

        # Remove ticks on axis.
        ax_main.xaxis.set_ticks_position('none')
        ax_main.yaxis.set_ticks_position('none')
        # Style of ticks.
        plt.xticks(fontsize=10, alpha=0.7)
        plt.yticks(fontsize=10, alpha=0.7)

        # Draw tick lines on wanted axes.
        if grid:
            ax_main.axes.grid(True, which='major', axis=grid, color='black',
                              alpha=0.3, linestyle='--', lw=0.5)

        set_thousands_separator(ax_main, which='both', nb_decimals=1)

        return ax_main if not with_histograms else (ax_main, ax_right, ax_bottom)


def set_thousands_separator(axes: plt.axes, which: str = 'both',
                            nb_decimals: int = 1) -> plt.axes:
    """
    Set thousands separator on the axes.

    Parameters
    ----------
    axes : plt.axes
        Axes from matplotlib, can be a single ax or an array of axes.
    which : str, default 'both'
        Which axis to format with the thousand separator ('both', 'x', 'y').
    nb_decimals: int, default 1
        Number of decimals to use for the number.

    Returns
    -------
    plt.axes
        Axis with the formatted axes.

    Examples
    --------
    >>> fig, ax = plt.subplots(1, 1)
    >>> ax.plot(...)
    >>> set_thousands_separator(ax, which='x', nb_decimals=3)
    """
    for ax in np.asarray(bff.value_2_list(axes)).flatten():
        # Set a thousand separator for axis.
        if which in ('x', 'both'):
            ax.xaxis.set_major_formatter(
                mpl.ticker.FuncFormatter(lambda x, p: f'{x:,.{nb_decimals}f}')
            )
        if which in ('y', 'both'):
            ax.yaxis.set_major_formatter(
                mpl.ticker.FuncFormatter(lambda x, p: f'{x:,.{nb_decimals}f}')
            )
    return axes

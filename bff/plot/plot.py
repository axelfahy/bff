# -*- coding: utf-8 -*-
"""Plot functions of the bff library.

This module contains fancy plot functions.
"""
import logging
from typing import Sequence, Tuple, Union
from scipy.stats import sem
import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bff.fancy

TNum = Union[int, float]

LOGGER = logging.getLogger(__name__)


def plot_history(history, metric: Union[str, None] = None, title: str = 'Model history',
                 axes: plt.axes = None, figsize: Tuple[int, int] = (12, 4),
                 grid: bool = False, style: str = 'default',
                 **kwargs) -> Union[plt.axes, Sequence[plt.axes]]:
    """
    Plot the history of the model trained using Keras.

    Parameters
    ----------
    history : tensorflow.keras.callback.History
        History of the training.
    metric : str, default None
        Metric to plot.
        If no metric is provided, will only print the loss.
    title : str, default 'Model history'
        Main title for the plot (figure level).
    axes : plt.axes, default None
        Axes from matplotlib, if None, new figure and axes will be created.
        If metric is provided, need to have at least 2 axes.
    figsize : Tuple[int, int], default (12, 4)
        Size of the figure to plot.
    grid : bool, default False
        Turn the axes grids on or off.
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
    >>> plot_history(history, metric='acc', title='MyTitle', linestyle=':')
    """
    if metric:
        assert metric in history.history.keys(), (
            f'Metric {metric} does not exist in history.\n'
            f'Possible metrics: {history.history.keys()}')

    with plt.style.context(style):
        # Given axes are not check for now.
        # If metric is given, must have at least 2 axes.
        if axes is None:
            fig, axes = plt.subplots(1, 2 if metric else 1, figsize=figsize)
        else:
            fig = plt.gcf()

        if metric:
            # Summarize history for metric, if any.
            axes[0].plot(history.history[metric],
                         label=f"Train ({history.history[metric][-1]:.4f})",
                         **kwargs)
            axes[0].plot(history.history[f'val_{metric}'],
                         label=f"Validation ({history.history[f'val_{metric}'][-1]:.4f})",
                         **kwargs)
            axes[0].set_title(f'Model {metric}')
            axes[0].set_xlabel('Epochs')
            axes[0].set_ylabel(metric.capitalize())
            axes[0].legend(loc='upper left')

        # Summarize history for loss.
        ax_loss = axes[1] if metric else axes
        ax_loss.plot(history.history['loss'],
                     label=f"Train ({history.history['loss'][-1]:.4f})",
                     **kwargs)
        ax_loss.plot(history.history['val_loss'],
                     label=f"Validation ({history.history['val_loss'][-1]:.4f})",
                     **kwargs)
        ax_loss.set_title('Model loss')
        ax_loss.set_xlabel('Epochs')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend(loc='upper left')

        # Put the grid on axes.
        if metric:
            for ax in axes.flatten():
                ax.grid(grid)
        else:
            axes.grid(grid)

        fig.suptitle(title)
        return axes


def plot_predictions(y_true: Union[np.array, pd.DataFrame],
                     y_pred: Union[np.array, pd.DataFrame],
                     label_true: str = 'Actual', label_pred: str = 'Predicted',
                     label_x: str = 'x', label_y: str = 'y',
                     title: str = 'Model predictions',
                     ax: plt.axes = None,
                     figsize: Tuple[int, int] = (12, 4), grid: bool = False,
                     style: str = 'default', **kwargs) -> plt.axes:
    """
    Plot the predictions of the model.

    If a DataFrame is provided, it must only contain one column.

    Parameters
    ----------
    y_true : np.array or pd.DataFrame
        Actual values.
    y_pred : np.array or pd.DataFrame
        Predicted values by the model.
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
    figsize : Tuple[int, int], default (12, 4)
        Size of the figure to plot.
    grid : bool, default False
        Turn the axes grids on or off.
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
            __, ax = plt.subplots(1, 1, figsize=figsize)

        # Plot predictions.
        ax.plot(np.array(y_pred).flatten(), color='r',
                label=label_pred, **kwargs)
        # Plot actual values on top of the predictions.
        ax.plot(np.array(y_true).flatten(), color='b',
                label=label_true, **kwargs)
        ax.set_xlabel(label_x)
        ax.set_ylabel(label_y)
        ax.set_title(title)
        ax.grid(grid)

        # Sort labels and handles by labels.
        handles, labels = ax.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles),
                                      key=lambda t: t[0]))
        ax.legend(handles, labels, loc='upper left')

        return ax


def plot_series(df: pd.DataFrame, column: str, groupby: str = '1S',
                with_sem: bool = True, with_peaks: bool = False,
                with_missing_datetimes: bool = False,
                distance_scale: float = 0.04, label_x: str = 'Datetime',
                label_y: Union[str, None] = None, title: str = 'Plot of series',
                ax: plt.axes = None, color: str = '#3F5D7D',
                loc: Union[str, int] = 'best',
                figsize: Tuple[int, int] = (14, 6), dpi: int = 80,
                style: str = 'default', **kwargs) -> plt.axes:
    """
    Plot time series with datetime with the given resample (`groupby`).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to plot, with datetime as index.
    column : str
        Column of the DataFrame to display.
    groupby : str, default '1S'
        Grouping for the resampling by mean of the data.
        For example, can resample from seconds ('S') to minutes ('T').
    with_sem : bool, default True
        Display the standard error of the mean (SEM) if set to true.
    with_peaks : bool, default False
        Display the peaks of the serie if set to true.
    with_missing_datetimes : bool, default False
        Display the missing datetimes with vertical red lines.
    distance_scale: float, default 0.04
        Scaling for the minimal distance between peaks.
        Only used if `with_peaks` is set to true.
    label_x : str, default 'Datetime'
        Label for x axis.
    label_y : str, default None
        Label for y axis. If None, will take the column name as label.
    title : str, default 'Plot of series'
        Title for the plot (axis level).
    ax : plt.axes, default None
        Axes from matplotlib, if None, new figure and ax will be created.
    color : str, default '#3F5D7D'
        Default color for the plot.
    loc : str or int, default 'best'
        Location of the legend on the plot.
        Either the legend string or legend code are possible.
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
    assert 'datetime' in df.index.names, (
        'DataFrame must have a datetime index.')
    assert column in df.columns, (
        f'DataFrame does not contain column: {column}')

    with plt.style.context(style):
        if ax is None:
            __, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # By default, the y label if the column name.
        if label_y is None:
            label_y = column.capitalize()

        # Get the values to plot.
        df_plot = (df[column].groupby('datetime').mean()
                   .resample(groupby).mean())
        x = df_plot.index

        ax.plot(x, df_plot, label=column, color=color, lw=2, **kwargs)

        # With sem (standard error of the mean).
        if with_sem:
            df_sem = (df[column]
                      .groupby('datetime')
                      .mean()
                      .resample(groupby)
                      .apply(sem)
                      if groupby not in ('S', '1S') else
                      df[column].groupby('datetime').apply(sem))

            ax.fill_between(x, df_plot - df_sem, df_plot + df_sem,
                            color='grey', alpha=0.2)

        # Plot the peak as circle.
        if with_peaks:
            peak_dates, peak_values = bff.fancy.get_peaks(df_plot, distance_scale)
            ax.plot(peak_dates, peak_values, linestyle='', marker='o',
                    color='plum')

        # Plot vertical line where there is missing datetimes.
        if with_missing_datetimes:
            df_date_missing = pd.date_range(start=df.index.get_level_values(0).min(),
                                            end=df.index.get_level_values(0).max(),
                                            freq=groupby).difference(df.index.get_level_values(0))
            for date in df_date_missing.tolist():
                ax.axvline(date, color='crimson')

        ax.set_xlabel(label_x, fontsize=12)
        ax.set_ylabel(label_y, fontsize=12)
        ax.set_title(f'{title} (mean by {groupby})', fontsize=14)

        # Style.
        # Remove border on the top and right.
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Remove ticks on y axis.
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('bottom')

        # Draw Horizontal Tick lines.
        ax.xaxis.grid(False)
        ax.yaxis.grid(which='major', color='black', alpha=0.5,
                      linestyle='--', lw=0.5)

        # Set thousand separator for y axis.
        ax.yaxis.set_major_formatter(
            mpl.ticker.FuncFormatter(lambda x, p: f'{x:,.1f}')
        )

        handles, labels = ax.get_legend_handles_labels()
        labels_cap = [label.capitalize() for label in labels]
        # Add the sem on the legend.
        if with_sem:
            handles.append(mpatches.Patch(color='grey', alpha=0.2))
            labels_cap.append('Standard error of the mean (SEM)')

        # Add the peak symbol on the legend.
        if with_peaks:
            handles.append(mlines.Line2D([], [], linestyle='', marker='o',
                                         color='plum'))
            labels_cap.append('Peaks')

        # Add the missing date line on the legend.
        if with_missing_datetimes:
            handles.append(mlines.Line2D([], [], linestyle='-', color='crimson'))
            labels_cap.append('Missing datetimes')

        ax.legend(handles, labels_cap, loc=loc)
        plt.tight_layout()

        return ax


def plot_true_vs_pred(y_true: Union[np.array, pd.DataFrame],
                      y_pred: Union[np.array, pd.DataFrame],
                      marker: Union[str, int] = 'k.', corr: bool = True,
                      label_x: str = 'Ground truth',
                      label_y: str = 'Prediction',
                      title: str = 'Predicted vs Actual',
                      lim_x: Union[Tuple[TNum, TNum], None] = None,
                      lim_y: Union[Tuple[TNum, TNum], None] = None,
                      ax: plt.axes = None, figsize: Tuple[int, int] = (12, 5),
                      grid: bool = False, style: str = 'default',
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
    label_x : str, default 'Ground truth'
        Label for x axis.
    label_y : str, default 'Prediction'
        Label for y axis.
    title : str, default 'Predicted vs Actual'
        Title for the plot (axis level).
    lim_x : Tuple[TNum, TNum], default None
        Limit for the x axis. If None, automatically calculated according
        to the limits of the data, with an extra 5% for readability.
    lim_y : Tuple[TNum, TNum], default None
        Limit for the y axis. If None, automatically calculated according
        to the limits of the data, with an extra 5% for readability.
    ax : plt.axes, default None
        Axes from matplotlib, if None, new figure and axes will be created.
    figsize : Tuple[int, int], default (12, 5)
        Size of the figure to plot.
    grid : bool, default False
        Turn the axes grids on or off.
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
    >>> plot_true_vs_pred(y_true, y_pred, title='MyTitle', linestyle=':')
    """
    with plt.style.context(style):
        if ax is None:
            __, ax = plt.subplots(1, 1, figsize=figsize)

        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        ax.plot(y_true, y_pred, marker, **kwargs)
        ax.set_xlabel(label_x)
        ax.set_ylabel(label_y)
        ax.set_title(title)
        ax.grid(grid)

        # Calculate the limit of the plot if not provided,
        # add and extra 5% for readability.
        def get_limit(limit, data, percent=5):
            if not limit or not isinstance(limit, tuple):
                lim_max = data.max()
                lim_min = data.min()
                margin = (lim_max - lim_min) * percent / 100
                limit = (lim_min - margin, lim_max + margin)
            return limit

        ax.set_xlim(get_limit(lim_x, y_true))
        ax.set_ylim(get_limit(lim_y, y_pred))

        # Add correlation in upper left position.
        if corr:
            ax.text(0.025, 0.925,
                    f'R={np.round(np.corrcoef(y_true, y_pred)[0][1], 3)}',
                    transform=ax.transAxes)

        return ax

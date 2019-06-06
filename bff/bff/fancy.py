# -*- coding: utf-8 -*-
"""FancyPythonThings

This module contains various useful functions.
"""
import collections
import sys
from dateutil import parser
from functools import wraps
from typing import Callable, Dict, List, Sequence, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TNum = Union[int, float]

def concat_with_categories(df_left: pd.DataFrame, df_right: pd.DataFrame,
                           **kwargs) -> pd.DataFrame:
    """
    Concatenation of Pandas DataFrame having categorical columns.

    With the `concat` function from Pandas, when merging two DataFrames
    having categorical columns, categories not present in both DataFrames
    and with the same code are lost. Columns are cast to `object`,
    which takes more memory.

    In this function, a union of categorical values from both DataFrames
    is done and both DataFrames are recategorized with the complete list of
    categorical values before the concatenation. This way, the category
    field is preserved.

    Original DataFrame are copied, hence preserved.

    Parameters
    ----------
    df_left : pd.DataFrame
        Left DataFrame to merge.
    df_right : pd.DataFrame
        Right DataFrame to merge.
    **kwargs
        Additional keyword arguments to be passed to the `pd.concat` function.

    Returns
    -------
    pd.DataFrame
        Concatenation of both DataFrames.

    Examples
    --------
    >>> import pandas as pd
    >>> column_types = {'name': 'object',
    ...                 'color': 'category',
    ...                 'country': 'category'}
    >>> columns = list(column_types.keys())
    >>> df_left = pd.DataFrame([['John', 'red', 'China'],
                                ['Jane', 'blue', 'Switzerland']],
    ...                        columns=columns).astype(column_types)
    >>> df_right = pd.DataFrame([['Mary', 'yellow', 'France'],
                                 ['Fred', 'blue', 'Italy']],
    ...                         columns=columns).astype(column_types)
    >>> df_left
       name color      country
    0  John   red        China
    1  Jane  blue  Switzerland
    >>> df_left.dtypes
    name         object
    color      category
    country    category
    dtype: object

    The following concatenation shows the issue when using the `concat`
    function from pandas:

    >>> res_fail = pd.concat([df_left, df_right], ignore_index=True)
    >>> res_fail
       name   color      country
    0  John     red        China
    1  Jane    blue  Switzerland
    2  Mary  yellow       France
    3  Fred    blue       Italy
    >>> res_fail.dtypes
    name       object
    color      object
    country    object
    dtype: object

    All types are back to `object` since not all categorical values were
    present in both DataFrames.

    With this custom implementation, the categorical type is preserved:

    >>> res_ok = concat_with_categories(df_left, df_right, ignore_index=True)
    >>> res_ok
       name   color      country
    0  John     red        China
    1  Jane    blue  Switzerland
    2  Mary  yellow       France
    3  Fred    blue       Italy
    >>> res_ok.dtypes
    name         object
    color      category
    country    category
    dtype: object
    """
    assert sorted(df_left.columns.values) == sorted(df_right.columns.values), (
        f'DataFrames must have identical columns '
        f'({df_left.columns.values} != {df_right.columns.values})')

    df_a = df_left.copy()
    df_b = df_right.copy()

    for col in df_a.columns:
        # Process only the categorical columns.
        if pd.api.types.is_categorical_dtype(df_a[col].dtype):
            # Get all possible values for the categories.
            cats = pd.api.types.union_categoricals([df_a[col], df_b[col]],
                                                   sort_categories=True)
            # Set all the possibles categories.
            df_a[col] = pd.Categorical(df_a[col], categories=cats.categories)
            df_b[col] = pd.Categorical(df_b[col], categories=cats.categories)
    return pd.concat([df_a, df_b], **kwargs)


def parse_date(func: Callable = None,
               date_fields: Sequence[str] = ('date')) -> Callable:
    """
    Cast str date into datetime format.

    This decorator casts string arguments of a function to datetime.datetime
    type. This allows to specify either string of datetime format for a
    function argument. The name of the parameters to cast must be specified in
    the `date_fields`.

    The cast is done using the `parse` function from the
    `dateutil <https://dateutil.readthedocs.io/en/stable/parser.html>`_
    package. All supported format are those from the library and may evolve.

    In order to use the decorator both with or without parenthesis when calling
    it without parameter, the `date_fields` argument is keyword only. This
    allows checking if the parameter was given or not.

    Parameters
    ----------
    func : Callable
        Function with the arguments to parse.
    date_fields : Sequence of str, default 'date'
        Sequence containing the fields with dates.

    Returns
    -------
    Callable
        Function with the date fields cast to datetime.datetime type.

    Examples
    --------
    >>> @parse_date
    ... def dummy_function(**kwargs):
    ...     print(f'Args {kwargs}')
    ...
    >>> dummy_function(date='20190325')
    Args {'date': datetime.datetime(2019, 3, 25, 0, 0)}
    >>> dummy_function(date='Mon, 21 March, 2015')
    Args {'date': datetime.datetime(2015, 3, 21, 0, 0)}
    >>> dummy_function(date='2019-03-09 08:03:00')
    Args {'date': datetime.datetime(2019, 3, 9, 8, 3)}
    >>> dummy_function(date='March 27 2019')
    Args {'date': datetime.datetime(2019, 3, 27, 0, 0)}
    >>> dummy_function(date='wrong string')
    Value `wrong string` for field `date` is not convertible to a date format.
    Args {'date': 'wrong string'}
    """
    def _parse_date(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Parse the arguments of the function, if the date field is present
            # and is str, cast to datetime format.
            for key, value in kwargs.items():
                if key in date_fields and isinstance(value, str):
                    try:
                        kwargs[key] = parser.parse(value)
                    except ValueError:
                        print(f'Value `{value}` for field `{key}` is not '
                              'convertible to a date format.', file=sys.stderr)
            return func(*args, **kwargs)
        return wrapper
    return _parse_date(func) if func else _parse_date


def plot_history(history, metric: str = None, title: str = 'Model history',
                 axes: plt.axes = None, figsize: Tuple[int, int] = (12, 4),
                 grid: bool = False, style: str = 'default',
                 **kwargs) -> Union[plt.axes, Sequence[plt.axes]]:
    """
    Plot the history of the model trained using Keras.

    Parameters
    ----------
    history : tensorflow.keras.callbask.History
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


def plot_true_vs_pred(y_true: Union[np.array, pd.DataFrame],
                      y_pred: Union[np.array, pd.DataFrame],
                      marker: Union[str, int] = 'k.', corr: bool = True,
                      label_x: str = 'Ground truth',
                      label_y: str = 'Prediction',
                      title: str = 'Predicted vs Actual',
                      lim_x: Tuple[TNum, TNum] = None,
                      lim_y: Tuple[TNum, TNum] = None,
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


def read_sql_by_chunks(sql: str, cnxn, params: Union[List, Dict] = None,
                       chunksize: int = 8_000_000, column_types: Dict = None,
                       **kwargs) -> pd.DataFrame:
    """
    Read SQL query by chunks into a DataFrame.

    This function uses the `read_sql` from Pandas with the `chunksize` option.

    The columns of the DataFrame are cast in order to be memory efficient and
    preserved when adding the several chunks of the iterator.

    Parameters
    ----------
    sql : str
        SQL query to be executed.
    cnxn : SQLAlchemy connectable (engine/connection) or database string URI
        Connection object representing a single connection to the database.
    params : list or dict, default None
        List of parameters to pass to execute method.
    chunksize : int, default 8,000,000
        Number of rows to include in each chunk.
    column_types : dict, default None
        Dictionary with the name of the column as key and the type as value.
        No cast is done if None.
    **kwargs
        Additional keyword arguments to be passed to the
        `pd.read_sql` function.

    Returns
    -------
    pd.DataFrame
        DataFrame with the concatenation of the chunks in the wanted type.
    """
    sql_it = pd.read_sql(sql, cnxn, params=params, chunksize=chunksize,
                         **kwargs)
    # Read the first chunk and cast the types.
    res = next(sql_it)
    if column_types:
        res = res.astype(column_types)
    for df in sql_it:
        # Concatenate each chunk with the preservation of the categories.
        if column_types:
            df = df.astype(column_types)
        res = concat_with_categories(res, df, ignore_index=True)
    return res


def sliding_window(sequence: Sequence, window_size: int, step: int):
    """
    Apply a sliding window over the sequence.

    Each window is yielded. If there is a remainder, the remainder is yielded
    last, and will be smaller than the other windows.

    Parameters
    ----------
    sequence : Sequence
        Sequence to apply the sliding window on
        (can be str, list, numpy.array, etc.).
    window_size : int
        Size of the window to apply on the sequence.
    step : int
        Step for each sliding window.

    Yields
    ------
    Sequence
        Sequence generated.

    Examples
    --------
    >>> list(sliding_window('abcdef', 2, 1))
    ['ab', 'bc', 'cd', 'de', 'ef']
    >>> list(sliding_window(np.array([1, 2, 3, 4, 5, 6]), 5, 5))
    [array([1, 2, 3, 4, 5]), array([6])]
    """
    # Check for types.
    try:
        __ = iter(sequence)
    except TypeError:
        raise TypeError('Sequence must by iterable.')

    if not isinstance(step, int):
        raise TypeError('Step must be an integer.')
    if not isinstance(window_size, int):
        raise TypeError('Window size must be an integer.')
    # Check for values.
    if window_size < step or window_size <= 0:
        raise ValueError('Window_size must be larger or equal '
                         'than step and higher than 0.')
    if step <= 0:
        raise ValueError('Step must be higher than 0.')
    if len(sequence) < window_size:
        raise ValueError('Length of sequence must be larger '
                         'or equal than window_size.')

    nb_chunks = int(((len(sequence) - window_size) / step) + 1)
    mod = len(sequence) % window_size
    for i in range(0, nb_chunks * step, step):
        yield sequence[i:i+window_size]
    if mod:
        start = len(sequence) - (window_size - step) - mod
        yield sequence[start:]


def value_2_list(**kwargs) -> Dict[str, Sequence]:
    """
    Convert single values into list.

    For each argument provided, if the type is not a sequence,
    convert the single value into a list.
    Strings are not considered as a sequence in this scenario.

    Parameters
    ----------
    **kwargs
        Parameters passed to the function.

    Returns
    -------
    dict
        Dictionary with the single values put into a list.

    Raises
    ------
    TypeError
        If a non-keyword argument is passed to the function.

    Examples
    --------
    >>> value_2_list(name='John Doe', age=42, children=('Jane Doe', 14))
    {'name': ['John Doe'], 'age': [42], 'children': ('Jane Doe', 14)}
    >>> value_2_list(countries=['Swiss', 'Spain'])
    {'countries': ['Swiss', 'Spain']}
    """
    for k, v in kwargs.items():
        if not isinstance(v, collections.abc.Sequence) or isinstance(v, str):
            kwargs[k] = [v]
    return kwargs

# -*- coding: utf-8 -*-
"""Functions of bff library.

This module contains various useful fancy functions.
"""
import collections
import logging
import math
import sys
from functools import wraps
from typing import Any, Callable, Dict, Hashable, List, Sequence, Set, Union
from dateutil import parser
from scipy import signal
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def cast_to_category_pd(df: pd.DataFrame, deep: bool = True) -> pd.DataFrame:
    """
    Automatically converts columns that are worth stored as ``category`` dtype.

    To be casted a column must not be numerical and must have less than 50%
    of unique values.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with the columns to cast.
    deep: bool, default True
        Whether or not to perform a deep copy of the original DataFrame.

    Returns
    -------
    pd.DataFrame
        Optimized copy of the input DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> columns = ['name', 'age', 'country']
    >>> df = pd.DataFrame([['John', 24, 'China'],
    ...                    ['Mary', 20, 'China'],
    ...                    ['Jane', 25, 'Switzerland'],
    ...                    ['Greg', 23, 'China'],
    ...                    ['James', 28, 'China']],
    ...                   columns=columns)
    >>> df
        name  age      country
    0   John   24        China
    1   Jane   25  Switzerland
    2  James   28        China
    >>> df.dtypes
    name       object
    age         int64
    country    object
    dtype: object
    >>> df_optimized = cast_to_category_pd(df)
    >>> df_optimized.dtypes
    name       object
    age         int64
    country  category
    dtype: object
    """
    return (df.copy(deep=deep)
            .astype({col: 'category' for col in df.columns
                     if (df[col].dtype == 'object'
                         and df[col].nunique() / df[col].shape[0] < 0.5)
                     }
                    )
            )


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
    ...                         ['Jane', 'blue', 'Switzerland']],
    ...                        columns=columns).astype(column_types)
    >>> df_right = pd.DataFrame([['Mary', 'yellow', 'France'],
    ...                          ['Fred', 'blue', 'Italy']],
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


def get_peaks(s: pd.Series, distance_scale: float = 0.04):
    """
    Get the peaks of a time series having datetime as index.

    Only the peaks having a height higher than 0.75 quantile are returned
    and a distance between two peaks at least ``df.shape[0]*distance_scale``.

    Return the dates and the corresponding value of the peaks.

    Parameters
    ----------
    s : pd.Series
        Series to get the peaks from, with datetime as index.
    distance_scale : str, default 0.04
        Scaling for the minimal distances between two peaks.
        Multiplication of the length of the DataFrame
        with the `distance_scale` value.

    Returns
    -------
    dates : np.ndarray
        Dates when the peaks occur.
    heights : np.ndarray
        Heights of the peaks at the corresponding dates.
    """
    assert isinstance(s.index, pd.DatetimeIndex), (
        'Serie must have a datetime index.')

    peaks = signal.find_peaks(s.values,
                              height=s.quantile(0.75),
                              distance=math.ceil(s.shape[0] * distance_scale))
    peaks_dates = s.reset_index().iloc[:, 0][peaks[0]]
    return peaks_dates.values, peaks[1]['peak_heights']


def idict(d: Dict[Any, Hashable]) -> Dict[Hashable, Any]:
    """
    Invert a dictionary.

    Keys will be become values and values will become keys.

    Parameters
    ----------
    d : dict of any to hashable
        Dictionary to invert.

    Returns
    -------
    dict of hashable to any
        Inverted dictionary.

    Raises
    ------
    TypeError
        If original values are not Hashable.

    Examples
    --------
    >>> idict({1: 4, 2: 5})
    {4: 1, 5: 2}
    >>> idict({1: 4, 2: 4, 3: 6})
    {4: 2, 6: 3}
    """
    try:
        s = set(d.values())

        if len(s) < len(d.values()):
            LOGGER.warning('[DATA LOSS] Same values for multiple keys, '
                           'inverted dict will not contain all keys')
    except TypeError:
        raise TypeError(f'TypeError: values of dict {d} are not hashable.')

    return {v: k for k, v in d.items()}


def kwargs_2_list(**kwargs) -> Dict[str, Sequence]:
    """
    Convert all single values from keyword arguments into lists.

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
    >>> kwargs_2_list(name='John Doe', age=42, children=('Jane Doe', 14))
    {'name': ['John Doe'], 'age': [42], 'children': ('Jane Doe', 14)}
    >>> kwargs_2_list(countries=['Swiss', 'Spain'])
    {'countries': ['Swiss', 'Spain']}
    """
    for k, v in kwargs.items():
        kwargs[k] = value_2_list(v)
    return kwargs


def mem_usage_pd(pd_obj: Union[pd.DataFrame, pd.Series], index: bool = True, deep: bool = True,
                 details: bool = False) -> Dict[str, Union[str, Set[Any]]]:
    """
    Calculate the memory usage of a pandas object.

    If `details`, returns a dictionary with the memory usage and type of
    each column (DataFrames only). Key=column, value=(memory, type).
    Else returns a dictionary with the total memory usage. Key=`total`, value=memory.

    Parameters
    ----------
    pd_obj : pd.DataFrame or pd.Series
        DataFrame or Series to calculate the memory usage.
    index : bool, default True
        If True, include the memory usage of the index.
    deep : bool, default True
        If True, introspect the data deeply by interrogating object dtypes for system-level
        memory consumption.
    details : bool, default False
        If True and a DataFrame is given, give the detail (memory and type) of each column.

    Returns
    -------
    dict of str to str
        Dictionary with the column or total as key and the memory usage as value (with 'MB').

    Raises
    ------
    AttributeError
        If argument is not a pandas object.

    Examples
    --------
    >>> df = pd.DataFrame({'A': [f'value{i}' for i in range(100_000)],
    ...                    'B': [i for i in range(100_000)],
    ...                    'C': [float(i) for i in range(100_000)]}).set_index('A')
    >>> mem_usage_pd(df)
    {'total': '7.90 MB'}
    >>> mem_usage_pd(df, details=True)
    {'Index': {'6.38 MB', 'Index type'},
     'B': {'0.76 MB', dtype('int64')},
     'C': {'0.76 MB', dtype('float64')},
     'total': '7.90 MB'}
    >>> serie = df.reset_index()['B']
    >>> mem_usage_pd(serie)
    {'total': '0.76 MB'}
    >>> mem_usage_pd(serie, details=True)
    2019-06-24 11:23:39,500 Details is only available for DataFrames.
    {'total': '0.76 MB'}
    """
    try:
        usage_b = pd_obj.memory_usage(index=index, deep=deep)
    except AttributeError:
        raise AttributeError(f'Object does not have a `memory_usage` function, '
                             'use only pandas objects.')

    # Convert bytes to megabytes.
    usage_mb = usage_b / 1024 ** 2

    res: Dict[str, Union[str, Set[Any]]] = {}

    if details:
        if isinstance(pd_obj, pd.DataFrame):
            res.update({idx: {f'{value:03.2f} MB',
                              pd_obj[idx].dtype if idx != 'Index' else 'Index type'}
                        for (idx, value) in usage_mb.iteritems()})
        else:
            LOGGER.warning('Details is only available for DataFrames.')
    # Sum the memory usage of the columns if this is a DataFrame.
    if isinstance(pd_obj, pd.DataFrame):
        usage_mb = usage_mb.sum()
    res['total'] = f'{usage_mb:03.2f} MB'
    return res


def parse_date(func: Union[Callable, None] = None,
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


def read_sql_by_chunks(sql: str, cnxn, params: Union[List, Dict, None] = None,
                       chunksize: int = 8_000_000, column_types: Union[Dict, None] = None,
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
        raise TypeError('Sequence must be iterable.')

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
        yield sequence[i:i + window_size]
    if mod:
        start = len(sequence) - (window_size - step) - mod
        yield sequence[start:]


def value_2_list(value: Any) -> Sequence:
    """
    Convert a single value into a list with a single value.

    If the value is alredy a sequence, it is returned without modification.
    Type `np.ndarray` is not put inside another sequence.

    Strings are not considered as a sequence in this scenario.

    Parameters
    ----------
    value
        Value to convert to a sequence.

    Returns
    -------
    sequence
        Value put into a sequence.

    Examples
    --------
    >>> value_2_list(42)
    [42]
    >>> value_2_list('Swiss')
    ['Swiss']
    >>> value_2_list('Swiss')
    ['Swiss']
    """
    if (not isinstance(value, (collections.abc.Sequence, np.ndarray)) or isinstance(value, str)):
        value = [value]
    return value

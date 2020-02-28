# -*- coding: utf-8 -*-
"""Functions of bff library.

This module contains various useful fancy functions.
"""
from collections import abc, Counter
import logging
import math
import multiprocessing
import sys
from functools import partial, wraps
from typing import Any, Callable, Dict, Hashable, List, Optional, Sequence, Set, Tuple, Union
from dateutil import parser
from scipy import signal
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def avg_dicts(*args):
    """
    Average all the values in the given dictionaries.

    Dictionaries must only have numerical values.
    If a key is not present in one of the dictionary, the value is 0.

    Parameters
    ----------
    *args
        Dictionaries to average, as positional arguments.

    Returns
    -------
    dict
        Dictionary with the average of all inputs.

    Raises
    ------
    TypeError
        If a value is not a number.
    """
    try:
        total = sum(map(Counter, args), Counter())
        n_dict = len(args)
        return {k: v / n_dict for k, v in total.items()}
    except TypeError as e:
        raise TypeError('Some values of the dictionaries are not numbers.') from e


def cast_to_category_pd(df: pd.DataFrame, deep: bool = True) -> pd.DataFrame:
    """
    Automatically converts columns of pandas DataFrame that are worth stored as ``category`` dtype.

    To be casted a column must not be numerical and must have less than 50%
    of unique values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the columns to cast.
    deep : bool, default True
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
        f'({df_left.columns.values} != {df_right.columns.values}).')

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
    except TypeError as e:
        raise TypeError(f'TypeError: values of dict {d} are not hashable.') from e

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


def log_df(df: pd.DataFrame, f: Callable[[pd.DataFrame], Any] = lambda x: x.shape,
           msg: str = '') -> pd.DataFrame:
    r"""
    Log information on a DataFrame before returning it.

    The given function is applied and the result is printed.
    The original DataFrame is returned, unmodified.

    This allows to print debug information in method chaining.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to log.
    f : Callable, default is the shape of the DataFrame
        Function to apply on the DataFrame and to log.

    Returns
    -------
    pd.DataFrame
        The DataFrame, unmodified.

    Examples
    --------
    >>> import pandas as pd
    >>> import pandas.util.testing as tm
    >>> df = tm.makeDataFrame().head()
    >>> df_res = (df.pipe(log_df)
    ...           .assign(E=2)
    ...           .pipe(log_df, f=lambda x: x.head(), msg='My df: \n')
    ...           .pipe(log_df, lambda x: x.shape, 'New shape=')
    ...          )
    2019-11-04 13:31:34,742 [INFO   ] bff.fancy: (5, 4)
    2019-11-04 13:31:34,758 [INFO   ] bff.fancy: My df:
                       A         B         C         D  E
    7t93kTGSqJ -0.104845 -1.296579 -0.487572  0.928964  2
    P8CEEHf07x -0.462075 -2.426990 -0.538038  0.487148  2
    0DlwZOOj83 -1.964108 -1.272991  0.622618 -0.562890  2
    LcrsmbFAjk -0.827403 -0.015269 -0.970148  0.683915  2
    kHfxaURF8t  0.654381  0.353666 -0.830602  1.788581  2
    2019-11-04 13:31:34,758 [INFO   ] bff.fancy: New shape=(5, 5)
    """
    LOGGER.info(f'{msg}{f(df)}')
    return df


def mem_usage_pd(pd_obj: Union[pd.DataFrame, pd.Series], index: bool = True, deep: bool = True,
                 details: bool = True) -> Dict[str, Union[str, Set[Any]]]:
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
    details : bool, default True
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
    >>> import pandas as pd
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
    except AttributeError as e:
        raise AttributeError(f'Object does not have a `memory_usage` function, '
                             'use only pandas objects.') from e

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


def normalization_pd(df: pd.DataFrame, scaler=None,
                     columns: Optional[Union[str, Sequence[str]]] = None,
                     suffix: Optional[str] = None, new_type: np.dtype = np.float32,
                     **kwargs) -> pd.DataFrame:
    """
    Normalize columns of a pandas DataFrame using the given scaler.

    If the columns are not provided, will normalize all the numerical columns.

    If the original columns are integers (`RangeIndex`), it is not possible to replace
    them. This will create new columns having the same integer, but as a string name.

    By default, if the suffix is not provided, columns are overridden.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to normalize.
    scaler : TransformerMixin, default MinMaxScaler
        Scaler of sklearn to use for the normalization.
    columns : sequence of str, default None
        Columns to normalize. If None, normalize all numerical columns.
    suffix : str, default None
        If provided, create the normalization in new columns having this suffix.
    new_type : np.dtype, default np.float32
        New type for the columns.
    **kwargs
        Additional keyword arguments to be passed to the
        scaler function from sklearn.

    Returns
    -------
    pd.DataFrame
        DataFrame with the normalized columns.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sklearn.preprocessing import StandardScaler
    >>> data = {'x': [123, 27, 38, 45, 67], 'y': [456, 45.4, 32, 34, 90]}
    >>> df = pd.DataFrame(data)
    >>> df
         x      y
    0  123  456.0
    1   27   45.4
    2   38   32.0
    3   45   34.0
    4   67   90.0
    >>> df_std = df.pipe(normalization_pd, columns=['x'], scaler=StandardScaler)
    >>> df_std
              x      y
    0  1.847198  456.0
    1 -0.967580   45.4
    2 -0.645053   32.0
    3 -0.439809   34.0
    4  0.205244   90.0
    >>> df_min_max = normalization_pd(df, suffix='_norm', feature_range=(0, 2),
    ...                               new_type=np.float64)
    >>> df_min_max
         x      y    x_norm    y_norm
    0  123  456.0  2.000000  2.000000
    1   27   45.4  0.000000  0.063208
    2   38   32.0  0.229167  0.000000
    3   45   34.0  0.375000  0.009434
    4   67   90.0  0.833333  0.273585
    """
    # If columns are not provided, select all the numerical columns of the DataFrame.
    # If provided, select only the numerical ones.
    cols_to_norm = ([col for col in value_2_list(columns) if np.issubdtype(df[col], np.number)]
                    if columns else df.select_dtypes(include=[np.number]).columns)
    return df.assign(**{f'{col}{suffix}' if suffix else str(col):
                        scaler(**kwargs).fit_transform(df[[col]].values.astype(new_type))
                        for col in cols_to_norm})


def parse_date(func: Optional[Callable] = None,
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


def pipe_multiprocessing_pd(df: pd.DataFrame, func: Callable, *,
                            nb_proc: Optional[int] = None, **kwargs) -> pd.DataFrame:
    """
    Compute function on DataFrame with `nb_proc` processes.

    The given function must return a new DataFrame.
    Rows must be independant and not depend from a value generated using the whole DataFrame.

    The function uses as many processes as cpu available on the machine.

    The DataFrame is splitted in `nb_proc` processes and then each
    splitted DataFrame is computed by a different process.
    The results are then concatenated an returned.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that must be computed by the function.
    func : function
        Function that takes the DataFrame as input.
    nb_proc : Union[int, None], default None
        Number of processor to use. If not provided,
        uses `multiprocessing.cpu_count()` number of processes.
    **kwargs
        Additional keyword arguments to be passed to `func`.

    Returns
    -------
    pd.DataFrame
        Return the DataFrame computed by `func`.
    """
    nb_proc = nb_proc or multiprocessing.cpu_count()
    chunks = np.array_split(df, nb_proc)
    with multiprocessing.Pool(processes=nb_proc) as pool:
        # Results of pool.map is in the same order as given,
        # so we can concatenate the DataFrames directly.
        results = pool.map(partial(func, **kwargs), chunks)
    return pd.concat(results, axis='index')


def read_sql_by_chunks(sql: str, cnxn, params: Optional[Union[List, Dict]] = None,
                       chunksize: int = 8_000_000, column_types: Optional[Dict] = None,
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


def size_2_square(n: int) -> Tuple[int, int]:
    """
    Return the size of the side to create a square able to contain n elements.

    This is mainly used to have the correct sizes of the sides when creating squared subplots.

    Parameters
    ----------
    n: int
        Number of elements that need to fit inside the square.

    Returns
    -------
    Tuple of int
        Tuple of int with the size of each part of the square.
        Both element of the tuple are similar.
    """
    size_float = math.sqrt(n)
    size_int = int(size_float) if size_float.is_integer() else math.ceil(size_float)
    return (size_int, size_int)


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
    except TypeError as e:
        raise TypeError('Sequence must be iterable.') from e

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
    >>> value_2_list(['Swiss'])
    ['Swiss']
    """
    if (not isinstance(value, (abc.Sequence, np.ndarray)) or isinstance(value, str)):
        value = [value]
    return value

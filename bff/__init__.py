"""All of bff' functions."""
import logging

from ._version import get_versions

# Import submodules.
from . import plot

from .fancy import (
    cast_to_category_pd,
    concat_with_categories,
    get_peaks,
    idict,
    mem_usage_pd,
    parse_date,
    read_sql_by_chunks,
    sliding_window,
    value_2_list,
)

from .config import FancyConfig

# Public object of the module.
__all__ = [
    'cast_to_category_pd',
    'concat_with_categories',
    'get_peaks',
    'idict',
    'mem_usage_pd',
    'parse_date',
    'plot',
    'read_sql_by_chunks',
    'sliding_window',
    'value_2_list',
    'FancyConfig',
]

# Logging configuration.
FORMAT = '%(asctime)s [%(levelname)-7s] %(name)s: %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)

__version__ = get_versions()['version']
del get_versions

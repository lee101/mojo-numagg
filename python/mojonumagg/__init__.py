"""Grouped and moving aggregations backed by Mojo."""

from .grouped import (
    group_nanall,
    group_nanany,
    group_nanargmax,
    group_nanargmin,
    group_nancount,
    group_nanfirst,
    group_nanlast,
    group_nanmax,
    group_nanmean,
    group_nanmin,
    group_nanprod,
    group_nanstd,
    group_nansum,
    group_nansum_of_squares,
    group_nanvar,
)
from .moving import move_corr, move_cov, move_mean, move_std, move_sum, move_var

__version__ = "0.1.0"

GROUPED_FUNCS = [
    group_nanall,
    group_nanany,
    group_nanargmax,
    group_nanargmin,
    group_nancount,
    group_nanfirst,
    group_nanlast,
    group_nanmax,
    group_nanmean,
    group_nanmin,
    group_nanprod,
    group_nanstd,
    group_nansum,
    group_nansum_of_squares,
    group_nanvar,
]

MOVE_FUNCS = [move_corr, move_cov, move_mean, move_std, move_sum, move_var]

__all__ = [
    "GROUPED_FUNCS",
    "MOVE_FUNCS",
    "__version__",
    "group_nanall",
    "group_nanany",
    "group_nanargmax",
    "group_nanargmin",
    "group_nancount",
    "group_nanfirst",
    "group_nanlast",
    "group_nanmax",
    "group_nanmean",
    "group_nanmin",
    "group_nanprod",
    "group_nanstd",
    "group_nansum",
    "group_nansum_of_squares",
    "group_nanvar",
    "move_corr",
    "move_cov",
    "move_mean",
    "move_std",
    "move_sum",
    "move_var",
]

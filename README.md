[Installation](#installation) |
[Documentation](https://bff.readthedocs.io/en/latest/)

# bff
> Best Fancy Functions, your Best Friend Forever

<p align="left">
    <a href="https://pypi.org/project/bff/">
        <img src="https://img.shields.io/pypi/v/bff.svg" alt="Latest Release" /></a>
    <a href="https://travis-ci.com/axelfahy/bff">
        <img src="https://api.travis-ci.com/axelfahy/bff.svg?branch=master" alt="Build Status" /></a>
    <a href="https://coveralls.io/github/axelfahy/bff?branch=master">
        <img src="https://coveralls.io/repos/github/axelfahy/bff/badge.svg?branch=master" alt="Coverage Status" /></a>
    <a href="https://pypi.org/project/bff/">
        <img src="https://img.shields.io/badge/python-3.7-blue.svg" alt="Python37" /></a>
</p>

This package contains some utility functions from plots to data manipulations and could become your new bff.

## Installation

```sh
pip install bff
```

## Documentation

Available [here](https://bff.readthedocs.io/en/latest/).

## Development setup

```sh
git clone https://github.com/axelfahy/bff.git
cd bff
python -m venv venv-dev
source venv-dev/bin/activate
pip install -r requirements_dev.txt
pip install -e .
```

## Tests

```sh
make all
```

To test plots, images with baseline should be placed in `tests/baseline` and can be generated using `make build-baseline`.

As of *v0.2*, plots are not yet tested in the travis build.

## Release History

* 0.2.5
    * ADD: Function ``log_df`` to print function results during method chaining.
    * ADD: Function ``avg_dicts`` to make the average of multiple similar dictionaries.
    * ADD: Function ``size_2_square`` to calculate the square needed for the given size (e.g. in subplots).
    * ADD: Option ``with_identity`` to plot an identity line in the ``plot_true_vs_pred`` function.
    * ADD: Option ``with_determination`` to plot the coefficient of determination in the ``plot_true_vs_pred`` function.
    * ADD: Function ``size_to_square`` to calculate automatically the size of the side of a square needed to store all the elements.
    * CHANGE: Default value of option ``details`` in ``mem_usage_pd`` function is now ``True``.
* 0.2.4
    * ADD: Function ``set_thousands_separator`` to add a thousand separator and set the number of decimals on x and/or y ticks.
    * ADD: Option to define x-axis in ``plot_predictions`` function.
    * FIX: Cast columns to string in ``normalization_pd`` function.
    * FIX: Add possibility to define custom label in ``plot_series`` function using the kwargs instead of an argument.
* 0.2.3
    * ADD: Function ``normalization_pd`` to normalize a DataFrame.
    * ADD: Function ``plot_correlation`` to plot the correlation of variables in a DataFrame.
* 0.2.2
    * FIX: Function ``value_2_list`` renamed to ``kwargs_2_list``.
    * ADD: Function ``value_2_list`` to cast a single value.
* 0.2.1
    * ADD: Function ``plot_counter`` to plot counter as bar plot.
* 0.2.0
    * ADD: Separation of plots in submodule ``plot``. This breaks the previous API.
    * ADD: Tests for the plot module using ``pytest-mlp``.
    * ADD: Images from plot in the documentation and notebook with examples.
    * FIX: Correction of resampling in the ``plot_series`` function.
* 0.1.9
    * ADD: Option ``loc`` in ``plot_series`` function.
    * ADD: Function ``cast_to_category_pd`` to cast columns to category ``dtype`` automatically.
* 0.1.8
    * ADD: Option ``with_missing_datetimes`` in ``plot_series`` function.
    * ADD: Mypy for type verification.
    * FIX: Tests when raising exceptions in ``sliding_window`` function.
* 0.1.7
    * ADD: ``FancyConfig`` to handle configuration files.
* 0.1.6
    * FIX: Correction of dependencies for doc.
* 0.1.5
    * ADD: Documentation of project on Read the Docs.
* 0.1.4
    * ADD: Function ``mem_usage_pd`` to calculate the memory usage of a pandas object.
    * ADD: Function ``idict`` to invert the key / values of a dictionary.
    * ADD: Add Makefile for testing code and style.
    * ADD: Add python-versioneer to handle version of package.
* 0.1.3
    * ADD: Travis, flake8, coveralls and PyUp configurations.
    * ADD: Function ``get_peaks`` to get the peaks of a time series.
    * ADD: Function ``plot_series`` to plot a time series.
    * CHANGE: Restructuration of repo.
* 0.1.2
    * ADD: Function ``plot_predictions`` function to plot the actual values and the predictions of a model.
    * CHANGE: Add axes in plot functions.
* 0.1.1
    * ADD: Readme with instructions.
    * CHANGE: Improvement of `plot_history` function.
    * FIX: Fix the imports in the test.
* 0.1.0
    * Initial release.

## Meta

Axel Fahy – axel@fahy.net

Distributed under the MIT license. See ``LICENSE`` for more information.

[https://github.com/axelfahy](https://github.com/axelfahy)

## Contributing

1. Fork it (<https://github.com/yourname/yourproject/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

## Version number

The version of the package is link to the tag pushed.

To set a new version:

```sh
git tag v0.1.4
git push --tags
```


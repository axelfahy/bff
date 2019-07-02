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
pip install requirements_dev.txt
pip install -e .
```

## Tests

```sh
make all
```

## Release History

* 0.1.7
    * ADD: ``FancyConfig`` to handle configuration files.
* 0.1.6
    * FIX: Correction of dependencies for doc.
* 0.1.5
    * ADD: Documentation of project on Read the Docs.
* 0.1.4
    * ADD: Function `mem_usage_pd` to calculate the memory usage of a pandas object.
    * ADD: Function `idict` to invert the key / values of a dictionary.
    * ADD: Add Makefile for testing code and style.
    * ADD: Add python-versioneer to handle version of package.
* 0.1.3
    * CHANGE: Restructuration of repo.
    * ADD: Travis, flake8, coveralls and PyUp configurations.
    * ADD: Function `get_peaks` to get the peaks of a time series.
    * ADD: Function `plot_series` to plot a time series.
* 0.1.2
    * CHANGE: Add axes in plot functions.
    * ADD: Function `plot_predictions` function to plot the actual values and the predictions of a model.
* 0.1.1
    * CHANGE: Improvement of `plot_history` function.
    * ADD: Readme with instructions.
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


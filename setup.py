#! /usr/bin/env python
"""bff setup file."""
import pathlib
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
import versioneer

# The directory containing this file
HERE = pathlib.Path(__file__).parent

DESCRIPTION = 'Best Fancy Functions, your Best Friend Forever'
LONG_DESCRIPTION = HERE.joinpath('README.md').read_text()

DISTNAME = 'bff'
LICENSE = 'MIT'
AUTHOR = 'Axel Fahy'
EMAIL = 'axel@fahy.net'
URL = 'https://github.com/axelfahy/bff'
DOWNLOAD_URL = ''
REQUIRES = [
    'matplotlib==3.0.3',
    'numpy==1.16.4',
    'pandas==0.24.2',
    'scipy==1.3.0',
    'typing==3.6.6'
]
CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Environment :: Console',
    'Intended Audience :: Developers',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7']


class NoopTestCommand(TestCommand):
    def __init__(self, dist):
        print('Bff does not support running tests with '
              '`python setup.py test`. Please run `make all`.')


cmdclass = versioneer.get_cmdclass()
cmdclass.update({"test": NoopTestCommand})

setup(name=DISTNAME,
      version=versioneer.get_version(),
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type='text/markdown',
      url=URL,
      download_url=DOWNLOAD_URL,
      author=AUTHOR,
      author_email=EMAIL,
      license=LICENSE,
      packages=find_packages(exclude=('tests',)),
      install_requires=REQUIRES,
      python_requires='>=3.6',
      cmdclass=cmdclass,
      zip_safe=False)

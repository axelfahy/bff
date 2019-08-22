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
URL = 'https://bff.readthedocs.io/en/latest/'
DOWNLOAD_URL = ''
PROJECT_URLS = {
    'Bug Tracker': 'https://github.com/axelfahy/bff/issues',
    'Documentation': 'https://bff.readthedocs.io/en/latest/',
    'Source Code': 'https://github.com/axelfahy/bff'
}
REQUIRES = [
    'matplotlib==3.1.1',
    'numpy==1.17.0',
    'pandas==0.25.0',
    'python-dateutil==2.8.0',
    'pyyaml==5.1.2',
    'scipy==1.3.1',
    'typing==3.7.4'
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
      maintainer=AUTHOR,
      version=versioneer.get_version(),
      packages=find_packages(exclude=('tests',)),
      maintainer_email=EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      cmdclass=cmdclass,
      url=URL,
      download_url=DOWNLOAD_URL,
      project_urls=PROJECT_URLS,
      long_description=LONG_DESCRIPTION,
      long_description_content_type='text/markdown',
      classifiers=CLASSIFIERS,
      python_requires='>=3.6',
      install_requires=REQUIRES,
      zip_safe=False)

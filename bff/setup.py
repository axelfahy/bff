import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

DESCRIPTION = 'Best Fancy Functions, your Best Friend Forever'
LONG_DESCRIPTION = HERE.joinpath('README.md').read_text()

DISTNAME = 'bff'
VERSION = '0.1.2'
LICENSE = 'MIT'
AUTHOR = 'Axel Fahy'
EMAIL = 'axel@fahy.net'
URL = 'https://github.com/axelfahy/FancyThings/tree/master/bff'
DOWNLOAD_URL = ''
REQUIRES = [
    'matplotlib==3.0.3',
    'pandas==0.24.2',
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

setup(name=DISTNAME,
      version=VERSION,
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
      zip_safe=False)

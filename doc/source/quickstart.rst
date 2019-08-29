.. _quickstart:

Quick Start
===========

Install bff
-----------

If you use ``pip``, you can install it with::

    pip install bff

Examples
--------

Here are some examples of possible plots from the `plot` module.

.. toctree::
   :maxdepth: 2

   example_plots

Development
-----------

Setup
~~~~~

The developement environment can be installed as follow:

.. code-block:: bash

   git clone https://github.com/axelfahy/bff.git
   cd bff
   python -m venv venv-dev
   source venv-dev/bin/activate
   pip install -r requirements_dev.txt
   pip install -e .

Unittest
~~~~~~~~

You can run the test using::

   make all

This will run unittests for code and code style checks.

To test plots, images with baseline should be placed in `tests/baseline` and can be generated using :code:`make build-baseline`.

As of *v0.2*, plots are not yet tested in the travis build.

Contributing
------------

Contributions are welcome!

If you want to contribute, you should proceed as follows::

    1. Fork it (<https://github.com/yourname/yourproject/fork>)
    2. Create your feature branch (`git checkout -b feature/fooBar`)
    3. Commit your changes (`git commit -am 'Add some fooBar'`)
    4. Push to the branch (`git push origin feature/fooBar`)
    5. Create a new Pull Request

If this is supposed to be a new realease, the new version must be set in the tag::

    git tag vx.y.z
    git push --tags


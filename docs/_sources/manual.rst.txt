Manual
======

Requirements
------------
NASLib has the following requirments:

* Linux operating system (for example Ubuntu, Mac OS X).
* Python (>=3.7).
* Pytorch.
* This is a bulleted list.

Setting up a virtual environment
--------------------------------
We recommend to set up a virtual environment

.. code-block:: console

    python3 -m venv naslib
    source naslib/bin/activate

.. note::
    Make sure you use the latest version of pip

    .. code-block:: console

        pip install --upgrade pip setuptools wheel
        pip install cython

Setting up NASLib
-----------------
Clone and install.
If you plan to modify naslib consider adding the -e option for pip install

.. code-block:: console

    git clone ...
    cd naslib
    pip install .

To validate the installation, you can run tests

.. code-block:: console

    cd tests
    coverage run -m unittest discover

The test coverage can be seen with coverage report.




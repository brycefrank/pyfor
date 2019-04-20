Installation
============

`miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ or Anaconda is required for your system before beginning. pyfor depends on many
packages that are otherwise tricky and difficult to install (especially gdal and its bindings),
and conda provides a quick and easy way to manage many different Python environments on your
system simultaneously.

The following bash commands will install this branch of pyfor. It requires installation of miniconda (see above). This will install all of the prerequisites in that environment, named pyfor_env. pyfor depends on a lot of heavy libraries, so expect construction of the environment to take a little time.

.. code:: bash

    git clone https://github.com/brycefrank/pyfor.git
    cd pyfor
    conda env create -f environment.yml

    # For Linux / macOS:
    source activate pyfor_env

    # For Windows:
    activate pyfor_env

    pip install .

Following these commands, pyfor should load in an activated Python shell:

.. code:: python

    import pyfor

If you see no errors, you are ready to process.
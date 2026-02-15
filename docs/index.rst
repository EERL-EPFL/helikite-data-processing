.. helikite-data-processing documentation master file, created by
   sphinx-quickstart.
   This file serves as the primary landing page for the Helikite Data Processing documentation.

Welcome to Helikite Data Processing's Documentation!
=====================================================

Overview
--------
This library supports Helikite campaigns by unifying field-collected data, generating quicklooks,
and performing quality control on instrument recordings.
It is available on PyPI and is designed to be used as a Python package within Jupyter notebooks.

Installation & Environment Setup
----------------------------------

There are several ways to install and run Helikite Data Processing:

1. Pip Installation

Helikite is published on PyPI: https://pypi.org/project/helikite-data-processing/. To install via pip, run:

.. code-block:: bash

    pip install helikite-data-processing

2. Setting Up a Poetry Environment

For an isolated development environment or if you prefer Poetry for dependency management:

- **Clone the repository:**

  .. code-block:: bash

      git clone https://github.com/EERL-EPFL/helikite-data-processing.git
      cd helikite-data-processing

- **Install dependencies with Poetry:**

  .. code-block:: bash

      poetry install

3. Using Jupyter Notebooks

Helikite includes several Jupyter notebooks demonstrating various processing workflows. To work with these notebooks:

- **Start Jupyter Lab within your Poetry environment:**

  .. code-block:: bash

      poetry run jupyter lab

- **Open the notebooks** from the ``notebooks/`` folder.
  - ``level0_DataProcessing``: Level 0 processing tutorial. Level 0 synchronizes timestamps across instruments and merges their data into a unified structure.
  - ``level1_DataProcessing``: Level 1 processing tutorial. Level 1 performs quality control, averages humidity and temperature measurements, calculates flight altitude
using the barometric equation, and applies instrument-specific processing.
  - ``level1_5_DataProcessing``: Level 1.5 processing tutorial. Level 1.5 detects flags that indicate environmental or flight conditions, such as hovering, pollution exposure,
or cloud immersion.
  - ``level2_DataProcessing``: Level 2 processing tutorial. Level 2 averages data to 10-second intervals and can merge flights into a final campaign dataset.

Using the Library
-----------------

Once installed, you can import the library into your own Python scripts. The library is designed so that most functions are accessible by simply importing it with:

.. code-block:: python

    import helikite

For example, you can access core processing functions, instrument classes, and data cleaning utilities.
API Reference
-------------
Below is the auto-generated API reference documentation that covers all modules, classes, and functions available in Helikite Data Processing.

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   autoapi/index

Notebooks & Tutorials
----------------------
A collection of Jupyter notebooks in the ``notebooks/`` folder provides practical,
step-by-step examples of common workflows.

.. toctree::
   :maxdepth: 2
   :caption: Notebooks & Tutorials

   notebooks/level0_DataProcessing
   notebooks/level1_DataProcessing
   notebooks/level1_5_DataProcessing
   notebooks/level2_DataProcessing

Additional Resources
--------------------
- **Auto-Published Documentation:** Visit the [Helikite Data Processing Documentation Site](https://eerl-epfl.github.io/helikite-data-processing/) for in-depth API details.
- **GitHub Repository:** https://github.com/EERL-EPFL/helikite-data-processing
- **Community Support:** If you have questions or run into issues, please open an issue on GitHub.

Indices and Tables
------------------
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. helikite-data-processing documentation master file, created by
   sphinx-quickstart.
   This file serves as the primary landing page for the Helikite Data Processing documentation.

Welcome to Helikite Data Processing's Documentation!
=====================================================

Overview
--------
Helikite Data Processing is a Python library designed to support Helikite campaigns by unifying field-collected data, generating quicklooks, and performing quality control on instrument data. Whether you’re a non-programmer who simply needs to run the provided command-line interface (CLI) or a developer looking to integrate its powerful API into your own workflows, this documentation will guide you every step of the way.

Installation & Environment Setup
----------------------------------

There are several ways to install and run Helikite Data Processing:

1. Pip Installation

Helikite is published on PyPI: https://pypi.org/project/helikite-data-processing/. To install via pip, run:

.. code-block:: bash

    pip install helikite-data-processing

After installation, the CLI is available as a system command:

.. code-block:: bash

    helikite --help

2. Setting Up a Poetry Environment

For an isolated development environment or if you prefer Poetry for dependency management:

- **Clone the repository:**

  .. code-block:: bash

      git clone https://github.com/EERL-EPFL/helikite-data-processing.git
      cd helikite-data-processing

- **Install dependencies with Poetry:**

  .. code-block:: bash

      poetry install

- **Run the CLI within Poetry:**

  .. code-block:: bash

      poetry run helikite --help

3. Using Jupyter Notebooks

Helikite includes several Jupyter notebooks demonstrating various processing workflows. To work with these notebooks:

- **Start Jupyter Lab within your Poetry environment:**

  .. code-block:: bash

      poetry run jupyter lab

- **Open the notebooks** from the ``notebooks/`` folder. Notable examples include:
  - ``level0.ipynb`` or ``level0_tutorial.ipynb``: An introductory tutorial covering basic processing.
  - ``OutlierRemoval.ipynb``: Demonstrates techniques for identifying and removing outliers.
  - ``FeatureFlagging.ipynb``: Shows how to apply feature flags to control processing features.
  - ``metadata.ipynb``: Provides examples for handling metadata.

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
A collection of Jupyter notebooks in the ``notebooks/`` folder provides practical, step-by-step examples of common workflows. These include:

- **Level 0 Tutorial:** An introductory guide covering basic data processing steps.
- **Outlier Removal:** Detailed techniques for outlier detection and removal.
- **Feature Flagging:** How to enable and apply feature flags within your processing pipeline.
- **Metadata Handling:** Examples for processing and utilizing metadata.

.. toctree::
   :maxdepth: 2
   :caption: Notebooks & Tutorials

   notebooks/level0_tutorial
   notebooks/OutlierRemoval
   notebooks/FeatureFlagging
   notebooks/metadata

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

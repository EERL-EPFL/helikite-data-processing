# EASE (hElikite dAta proceSsing codE)

This library supports Helikite campaigns by unifying field-collected data, generating quicklooks,
and performing quality control on instrument recordings.
It is available on PyPI and is designed to be used as a Python package within Jupyter notebooks.

## Table of Contents

1. [Installation](#installation)
2. [Using the Library](#using-the-library)

   1. [Level 0 (Cleaner)](#level-0-cleaner)
   2. [Level 1](#level-1)
   3. [Level 1.5](#level-15)
   4. [Level 2](#level-2)
   5. [Configuration](#configuration)
3. [Documentation & Examples](#documentation--examples)
4. [Development](#development)

   1. [The Instrument class](#the-instrument-class)
   2. [Adding more instruments](#adding-more-instruments)


# Installation

## Installation for standard users

Helikite is published on PyPI. To install it with `pip`, run:

```bash
pip install helikite-data-processing
```

## Installation for contributors

For an isolated development environment, or if you prefer Poetry for dependency management:

**Clone the repository**

```
git clone https://github.com/EERL-EPFL/helikite-data-processing.git
cd helikite-data-processing
```

**Install dependencies with Poetry**

```
poetry install
```


# Using the Library

Helikite is intended to be used as an importable Python package. The standard workflow is organized into multiple
processing levels, typically executed through Jupyter notebooks. These notebooks allow interactive control during
processing, such as selecting flight takeoff and landing times or marking outliers and flags
(for example, hovering periods).

The library also supports automatic detection of outliers and flags, enabling fully non-interactive processing.
Automatic runs may produce results that differ from manual review, so they should be used with caution.

An example script that processes all flights from a campaign in non-interactive mode is available
[here](./notebooks/execute_all.py).

If the library is installed from source, run the following from the project root to view usage instructions:

```
poetry run python ./notebooks/execute_all.py --help
```


## Level 0 (Cleaner)

Level 0 synchronizes timestamps across instruments and merges their data into a unified structure.

See the [Level 0 notebook](notebooks/level0_DataProcessing.ipynb) for a detailed example, or the `execute_level0`
function in the [script](./notebooks/execute_all.py).


## Level 1

Level 1 performs quality control, averages humidity and temperature measurements, calculates flight altitude
using the barometric equation, and applies instrument-specific processing.

See the [Level 1 notebook](notebooks/level1_DataProcessing.ipynb) for a detailed example, or the `execute_level1`
function in the [script](./notebooks/execute_all.py).


## Level 1.5

Level 1.5 detects flags that indicate environmental or flight conditions, such as hovering, pollution exposure,
or cloud immersion.

See the [Level 1.5 notebook](notebooks/level1_5_DataProcessing.ipynb) for a detailed example, or the `execute_level1_5` 
function in the [script](./notebooks/execute_all.py).


## Level 2

Level 2 averages data to 10-second intervals and can merge flights into a final campaign dataset.

See the [Level 2 notebook](notebooks/level2_DataProcessing.ipynb) for a detailed example, or the `execute_level2`
function in the [script](./notebooks/execute_all.py).


## Configuration

Each notebook uses a configuration file, and the same file is applied to all processing levels for a given flight.
An example configuration:

```
flight: "1"
flight_date: 2025-11-26
flight_suffix: "A"

output_schema: "ORACLES_25_26"
campaign_data_dirpath: /home/EERL/data/ORACLES/Helikite/2025-2026/Data/
processing_dir: "./outputs/2025-2026"
```

Where:

* `campaign_data_dirpath` contains folder `2025-11-26_A` corresponding to an individual campaign flight
* `output_schema` defines plot and output formatting (see available schemas [here](./helikite/classes/output_schemas.py))

Custom schemas can be registered using `OutputSchemas.register(name, SCHEMA)` 
if default configurations do not match campaign needs.

A complete list of available modules and functions is documented on the auto-published documentation site.


# Documentation & Examples

Full API documentation is available on the
[Helikite Data Processing Documentation](https://eerl-epfl.github.io/helikite-data-processing/) site.


# Development

## The Instrument class

All instruments implement a shared interface that allows instrument-specific behavior to override default processing. 
Data processing components call these methods during workflow execution.


## Adding more instruments

New instrument classes should inherit from `Instrument` and define a unique name. Example:

```python
def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.name = 'mcpc'
```

Required methods include:

### `file_identifier()`

Determines whether a CSV file belongs to the instrument by inspecting header lines.

```python
def file_identifier(self, first_lines_of_csv) -> bool:
    if ("win0Fit0,win0Fit1,win0Fit2,win0Fit3,win0Fit4,win0Fit5,win0Fit6,"
        "win0Fit7,win0Fit8,win0Fit9,win1Fit0,win1Fit1,win1Fit2") in first_lines_of_csv[0]:
        return True
    return False
```

### `read_data()`

Parses raw instrument data.

### `data_corrections()`

Applies instrument-specific corrections.

### `set_time_as_index()`

Converts the instrument's timestamp information into a common pandas `DateTimeIndex`.

```python
def set_time_as_index(self, df: pd.DataFrame) -> pd.DataFrame:
    df['DateTime'] = pd.to_datetime(
        df['#YY/MM/DD'].str.strip() + ' ' + df['HR:MN:SC'].str.strip(),
        format='%y/%m/%d %H:%M:%S'
    )
    df.drop(columns=["#YY/MM/DD", "HR:MN:SC"], inplace=True)
    df.set_index('DateTime', inplace=True)
    return df
```

### `__repr__()`

Returns a short instrument label used in certain plots (for example, `"FC"` for the flight computer).

Additional implementation details and examples are available in the auto-published documentation.


# Command-line Interface (Outdated)

The CLI is not up to date with the main processing workflow.
If CLI usage is still required, refer to the legacy documentation: `./cli.md`.


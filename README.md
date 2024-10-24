# helikite

This library aims to support Helikite campaigns by unifying data collected from
fieldwork and producing quicklooks on recorded data from instruments to assist
with instrument housekeeping and quality control.

## Table of Contents
1. [Getting started](#getting-started)
   1. [Docker](#docker)
   2. [Makefile](#makefile)
2. [Development](#development)
   1. [The instrument class](#the-instrument-class)
   2. [Adding more instruments](#adding-more-instruments)
3. [Configuration](#configuration)
   1. [Application constants](#application-constants)
   1. [Runtime configuration](#runtime)
4. [Using the library](#using-the-library)

# Getting started

There are three stages to the application. These are:

1. (**Optional**: This will happen in the next step if the config file has not
   been created not exist)

   Generate a config file in the `input` folder. This writes out a
   configuration file that the application will use to refer to instrument
   files and their individual configurations that have been defined in this
   library. This only needs to be run if there is no config file that exists in
   the input folder already, or if more instruments need to be added.

   **Note**: It will overwrite the config in the `input` folder.

   Once generated, the user may remove instruments that should not be
   considered in the preprocessing or processing stages (stage 2 and 3).

   This stage's command-line argument is: `generate_config`.

2. Scanning the input folder of raw instrument files and assigning them to the
   instrument configurations in the config file that were generated in stage 1.
   The configuration file is updated with the file location, and other
   instrument-specific metadata such as scan date (should it be located in the
   file header and not in a field -- see _Smart Tether_ configuration)

   This stage's command-line argument is: `preprocess`.

   After this stage is complete, the user may edit the generated `config.yaml`
   by removing instruments or setting variables that will be used by the next
   stage.

3. Processing the input data files based on the config file in the same
   `input` folder, normalising the timestamps of each record, saving their
   outputs and generating plots in the `output` folder.

   Each time the application runs, it will create a UTC timestamped folder in
   the `output` folder with the processed input files, plots and a copy of the
   configuration file used. Therefore, the configuration file can be edited
   many times and running this step will not affect any previous output.

   This is the default behaviour of the application (no command-line
   arguments).

## Docker

### Building from code

1. Build the docker image

   This only needs to be built once unless there is a change in code.

   ```bash
   docker build -t helikite .
   ```

2. Generate project folders and config file

   Define two folders: `inputs` and `outputs`. These are mapped to the
   docker container with the left section of the `-v` argument
   (in the below example the relative paths are used
   `./inputs` and `./outputs`).

   Generate the configuration file in the `input` folder:

   ```bash
   docker run \
       -v ./inputs:/app/inputs \
       -v ./outputs:/app/outputs \
       helikite:latest generate_config
   ```

   Note: `helikite:latest` refers to the `latest` version of the docker
   image that was built in step 1. This is always the most recent image
   that was built. If If you would like to use an older build, find the image
   ID with the command `docker images helikite` and replace `latest` with it.

   ```bash
   $ docker images helikite

   REPOSITORY   TAG       IMAGE ID       CREATED      SIZE
   helikite     latest    4a04184a12aa   9 days ago   497MB
   ```

3. Preprocess the configuration file

   Fill the metadata in the config.yaml by associating the files in the input
   directory to the known configurations in the library (such as determining
   instrument, date, etc)

   ```bash
   docker run \
       -v ./inputs:/app/inputs \
       -v ./outputs:/app/outputs \
       helikite:latest preprocess
   ```

4. Generate the plots and processed input files. These will be placed in the
   `output` folder that is defined here, in a directory based on the UTC time
   of when this command is run.

   ```bash
   docker run \
       -v ./inputs:/app/inputs \
       -v ./outputs:/app/outputs \
       helikite:latest
   ```

### Downloading Github package

The image is built and served already on Github. All the above steps can be
issued by replacing the `helikite:latest` image with the Github one
(`ghcr.io/eerl-epfl/helikite-data-processing:latest`):

Therefore,

```bash
docker run \
   -v ./inputs:/app/inputs \
   -v ./outputs:/app/outputs \
   ghcr.io/eerl-epfl/helikite-data-processing:latest generate_config
```

## Using the Makefile

Run the following commands as needed:

- To build the Docker image:

  ```
  make build
  ```

- To generate the configuration file in the `inputs` folder:

  ```
  make generate_config
  ```

- To preprocess the configuration file by filling the metadata:

  ```
  make preprocess
  ```

- To generate plots and processed input files, which will be placed in a
timestamped directory within the `outputs` folder:

  ```
  make process
  ```

# Development

## The `Instrument` class
The structure of the Instrument class allows specific data cleaning activities
to be overriden for each instrument which inherits it. The main application
in `helikite.py` will call these class functions.

## Adding more instruments
The config file is generated in the `generate_config`/`preprocess` steps by
iterating through the instantiated classes that are imported in
`helikite/instruments/__init__.py`. Therefore, creating a new class of parent
`Instrument`, and importing it into `__init__.py` will allow it to be
included in the project.

Firstly, the class should inherit the `Instrument` class and provide a name
for the instrument in `self.name`. This name is the prefix given to the
instrument in column outputs, and how the application will handle it. It should
be unique to all other instruments. Below is the example for the `MCPC`
instrument.

``` python
def __init__(
   self,
   *args,
   **kwargs
) -> None:
   super().__init__(*args, **kwargs)
   self.name = 'mcpc'
```

The minimum functions that should be defined in each instrument are the
`file_identifier()` and `set_time_as_index()` functions. The
`file_identifier()` function will accept the first 50
(defined in `constants.QTY_LINES_TO_IDENTIFY_INSTRUMENT`) lines of a csv file
and it is up to the function to report True if it matches some criteria. This
is generally the header line. For example, this is the case in the `pico`
instrument at the time of writing:

``` python
# helikite/instruments/pico.py

def file_identifier(
   self,
   first_lines_of_csv
) -> bool:
   if (
      "win0Fit0,win0Fit1,win0Fit2,win0Fit3,win0Fit4,win0Fit5,win0Fit6,"
      "win0Fit7,win0Fit8,win0Fit9,win1Fit0,win1Fit1,win1Fit2"
   ) in first_lines_of_csv[0]:
      return True

   return False
```

Secondly, the `set_time_as_index()` function will define what is needed to
convert the instrument's timing schema to match a `DateTime` format to build
a common pandas `DateTimeIndex` for all instruments. The example of the
`filter` instrument below combines the two columns `#YY/MM/DD` and `HR:MN:SC`,
strips the whitespace from the end of their values and then parses their format
into datetime, before setting the index as `DateTime`. Some instruments may
require more work than others, but the goal here is to end up with a DateTime
index named `DateTime` and removing the redundant time columns.

```python
# helikite/instruments/filter.py

def set_time_as_index(
   self,
   df: pd.DataFrame
) -> pd.DataFrame:
   ''' Set the DateTime as index of the dataframe

   Filter instrument contains date and time separately and appears to
   include an extra whitespace in the field of each of those two columns
   '''

   # Combine both date and time columns into one, strip extra whitespace
   df['DateTime'] = pd.to_datetime(
      df['#YY/MM/DD'].str.strip() + ' ' + df['HR:MN:SC'].str.strip(),
      format='%y/%m/%d %H:%M:%S'
   )
   df.drop(columns=["#YY/MM/DD", "HR:MN:SC"], inplace=True)

   # Define the datetime column as the index
   df.set_index('DateTime', inplace=True)

   return df
```

Each instrument manufacturer will define its own data structure, so it is
up to these functions to define the best way to handle them. One example here
is the `smart_tether` instrument which only includes timestamps in the data,
and a date in the metadata provided in the header. This introduces two
problems; How do we define a function as illustrated above if the date is not
given to us in each data record, and secondly, what happens if the time rolls
over midnight?

For the first problem in this case, a function `date_extractor()` (see below)
has been included, which is run during the `preprocess` step during the time
when the first 50 lines are read and passed to the `file_identifier()`
function. This function will output the date, and then write this date into the
config file for that instrument under the `date` attribute (where in most
cases this is just `null`).

``` python
# helikite/instruments/smart_tether.py

def date_extractor(
   self,
   first_lines_of_csv
) -> datetime.datetime:
   date_line = first_lines_of_csv[1]
   date_string = date_line.split(' ')[-1].strip()

   return datetime.datetime.strptime(date_string, "%m/%d/%Y")
```
Then when the application runs, processing all the data, this date value is
retrieved from the configuration and used during the `set_time_as_index()`
function. In this function, a method to identify a midnight rollover is also
used:

```python
# helikite/instruments/smart_tether.py
# Note the self.date variable, which is already available to the class after
# importing the configuration from the YAML

def set_time_as_index(
   self,
   df: pd.DataFrame
) -> pd.DataFrame:
   ''' Set the DateTime as index of the dataframe and correct if needed

   Using values in the time_offset variable, correct DateTime index

   As the rows store only a time variable, a rollover at midnight is
   possible. This function checks for this and corrects the date if needed
   '''

   # Date from header (stored in self.date), then add time
   df['DateTime'] = pd.to_datetime(
      self.date + pd.to_timedelta(df['Time'])
   )

   # Check for midnight rollover. Can assume that the data will never be
   # longer than a day, so just check once for a midnight rollover
   start_time = pd.Timestamp(df.iloc[0]['Time'])
   for i, row in df.iterrows():
      # check if the timestamp is earlier than the start time (i.e. it's
      # the next day)
      if pd.Timestamp(row['Time']) < start_time:
            # add a day to the date column
            logger.info("SmartTether date passes midnight. Correcting...")
            logger.info(F"Adding a day at: {df.at[i, 'DateTime']}")
            df.at[i, 'DateTime'] += pd.Timedelta(days=1)

   df.drop(columns=["Time"], inplace=True)

   # Define the datetime column as the index
   df.set_index('DateTime', inplace=True)

   return df
```
## Instantiating the instrument

After defining the class functions, the instrument should be instantiated
with variables defining the datatype of each column, and specific for reading
the data.

The below example is the instantiated object for the `STAP` instrument.

```python
# helikite/instruments/stap.py

stap = STAP(
    dtype={
        "datetimes": "Int64",
        "sample_press_mbar": "Float64",
        "sample_temp_C": "Float64",
        "sigmab": "Float64",
        "sigmag": "Float64",
        "sigmar": "Float64",
        "sigmab_smth": "Float64",
        "sigmag_smth": "Float64",
        "sigmar_smth": "Float64",
    },
    na_values=["NAN"],
    export_order=500,
    cols_export=["sample_press_mbar", "sample_temp_C", "sigmab",
                 "sigmag", "sigmar", "sigmab_smth", "sigmag_smth",
                 "sigmar_smth"],
    cols_housekeeping=["sample_press_mbar", "sample_temp_C", "sigmab",
                       "sigmag", "sigmar", "sigmab_smth", "sigmag_smth",
                       "sigmar_smth"],
    pressure_variable='sample_press_mbar')
```

These class variables define specific representations of the data for the
helikite application. The list of these can be found in the base `Instrument`
class in `helikite/instruments/base.py`. As the data is read in with the
pandas `read_csv()` function, many are provided to pass to this function.
If a more complicated situation is required, the `read_data()` function can be
overridden much like what was done in the previous steps.

Some noteworthy variables here are:
- `export_order`: This defined in a hierarchy where the data should be placed
in the export files. The value will be used in a sort function which will sort
from `0` to `inf.` where 0 is the first set of data to be placed in the CSV
export. If no number is defined it it placed at the end.
- `cols_export` and `cols_housekeeping`: This list of variables will the ones
included in the final aggregated data exports for the data and housekeeping
files respectively. If none are added here, the data will not be considered to
be added to these files.
- `pressure_variable`: Adding the column name of the instrument's pressure
reading will add it to the pressure plots in the qualitycheck plots.

Finally, add this instantiated instrument to the `__init__.py` file in
`helikite/instruments`.
## Configuration
There are three locations where the application defines the parameters to
execute. There are the instrument configuration parameters defined in the
[Instrument class](#the-instrument-class) described above, there are
application constants and the runtime configuration.
### Application constants
These are constants that aren't changed frequently, such as the filename,
the path of the input/output folders, the structure of the log outputs and
some default plotting parameters. These are located in `helikite/constants.py`.
### Runtime
The runtime configuration `config.yaml` should sit in the `input` folder where
the input data resides. This file is generated in the `generate_config` or
`preprocess` stages of the application and is designed to specify runtime
arguments for each instrument, including adjustments for time and location
of the data. It also holds parameters for plotting and trimming the data. The
generation of this file is explained in the [getting started](#getting-started)
section above.

Below is an example of a `config.yaml` file generated and adjusted:
```yaml
global:                             # Global parameters
  time_trim:                        # The start/end times to trim the data to
    end: 2022-09-29 12:34:36        # These can both be null, to not trim
    start: 2022-09-29 10:21:58
ground_station:     # Provides values for altitude calculation
  altitude: null    # Altitude at start (if null, this will be 0)
  pressure: null    # Pressure at start (if null, averages from first 10s)
  temperature: 7.8  # Pressure at start (if null, averages from first 10s)
instruments:
  filter:
    config: filter
    date: null
    file: /app/inputs/220209A3.TXT  # Location of the data
    pressure_offset: null
    time_offset:                    # Modifies the timestamp in the data
      hour: 5555
      minute: 0
      second: 0

  ...  ## All the other instruments

  smart_tether:
    config: smart_tether
    date: 2022-09-29 00:00:00       # Date provided by preprocessing
    file: /app/inputs/LOG_20220929_A.csv
    pressure_offset: 2.5
    time_offset:
      hour: 0
      minute: 0
      second: -26
plots:
  altitude_ground_level: false      # True: Plots from ground, false: sea level
  grid:
    resample_seconds: 60            # Resamples the plot data to n seconds
  heatmap:                          # Alters the colour scale for the heatmaps
    msems_inverted:
      zmax: null                    # Max value of colour scale
      zmid: null                    # Midpoint of colour scale
      zmin: null                    # Min value of the colour scale
    msems_scan:
      zmax: null
      zmid: null
      zmin: null
  msems_readings_averaged:  # Periods to generate averaged MSEMS plots
    Period1:                           # Name here becomes title given to plot
      log_y: false                     # Log scale enabled/disabled on y-axis
      time_end: 2022-09-29 11:13:00    # Timestamp of the end of the period
      time_start: 2022-09-29 11:07:00  # Timestamp of the start of the period
    Period2:
      log_y: false
      time_end: 2022-09-29 11:44:30
      time_start: 2022-09-29 11:40:00
```

In general, this does not need to be altered to process the data.

## Using the library
The library is designed to be run in a docker container or via its command line
utility.

### Installing the command line utility

Note: As of writing this README, the repository is not a published package on
PyPi. Therefore, the following instructions are for local installation.

The library uses poetry to manage dependencies. If this is not installed,
follow the instructions [here](https://python-poetry.org/docs/).

Afterwards, clone the repository, navigate to the root of the project and run
the following command:

```bash
poetry install
```

The library will now be installed locally and can be accessed with
`poetry run helikite`.

You can navigate the command line utility by running
`poetry run helikite --help`. An example follows below:

```bash
[ejthomas@cadiz helikite-data-processing]$ poetry run helikite

 Usage: helikite [OPTIONS] COMMAND [ARGS]...

 helikite-data-processing - Library to generate quicklooks and data quality
 checks on Helikite campaigns
 Use the --help option on a subcommand to see more information about it.

╭─ Options ───────────────────────────────────────────────────────────────────╮
│ --version          Prints the version number of helikite-data-processing    │
│ --help             Show this message and exit.                              │
╰─────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ──────────────────────────────────────────────────────────────────╮
│ execute           Execute the main processing and plotting of the data      │
│ generate-config   Generate a configuration file in the input folder         │
│ preprocess        Preprocess the data and generate the configuration file   │
╰─────────────────────────────────────────────────────────────────────────────╯

```

The commands follow the same structure as the docker commands above. For futher
help on each option, run `poetry run helikite <command> --help`.

For example:
```bash
[ejthomas@cadiz helikite-data-processing]$ poetry run helikite preprocess --help

 Usage: helikite preprocess [OPTIONS]

 Preprocess the data and generate the configuration file

╭─ Options ───────────────────────────────────────────────────────────────────╮
│ --overwrite       --no-overwrite          Overwrite the existing            │
│                                           configuration file                │
│                                           [default: no-overwrite]           │
│ --config-file                       TEXT  The configuration file to use     │
│                                           [default: config.yaml]            │
│ --input-folder                      TEXT  [default:                         │
│                                           /home/ejthomas/projects/EERL/Hel… │
│ --help                                    Show this message and exit.       │
╰─────────────────────────────────────────────────────────────────────────────╯
```

### Installing as a dependency
Similarly as above, as this library is not on PyPi, it can be installed as a
dependency in a `pyproject.toml` file via a git repository. An example is
below:

```bash
poetry add git+https://github.com/EERL/helikite-data-processing.git
```

Otherwise, you may follow the instructions from the previous step and use the
jupyter lab dependency to run the library. In the repository folder, run
`poetry run jupyter lab`, and the library can be imported and used in the
notebook.

## Cleaner

The `cleaner` module is designed to clean up the output folders of the
application. Follow the [Level 0 notebook](./notebooks/level0.ipynb) for instructions on how to use.
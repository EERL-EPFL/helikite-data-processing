# Command-line Interface (Outdated)

> **Note:** CLI is not up-to-date with the main processing functionality. 
> Please, refer to the [README.md](./README.md) to see the currently recommended way of using the library. 

Helikite can be used both as a standalone CLI tool and as an importable Python package.
For non-programmers, the CLI is the simplest way to use the library.
For programmers, the library can be imported and used in your own scripts:

```python
import helikite
from helikite.processing import preprocess, sorting
from helikite.constants import constants

# For example, to generate a configuration file programmatically:
preprocess.generate_config()
```
## Table of Contents

1. [Docker](#docker)
2. [Makefile](#makefile)
3. [Usage](#usage)
4. [Configuration](#configuration)
    1. [Application constants](#application-constants)
    2. [Runtime configuration](#runtime)

# Docker

> **Note:** Docker usage is now optional. For most users, installing via pip is the recommended approach.

## Building and Running with Docker

1. **Build the Docker image:**

   ```bash
   docker build -t helikite .
   ```

2. **Generate project folders and create the configuration file:**

   ```bash
   docker run \
       -v ./inputs:/app/inputs \
       -v ./outputs:/app/outputs \
       helikite:latest generate_config
   ```

3. **Preprocess the configuration file:**

   ```bash
   docker run \
       -v ./inputs:/app/inputs \
       -v ./outputs:/app/outputs \
       helikite:latest preprocess
   ```

4. **Process data and generate plots:**

   ```bash
   docker run \
       -v ./inputs:/app/inputs \
       -v ./outputs:/app/outputs \
       helikite:latest
   ```

You can also use the pre-built image from GitHub Packages:

```bash
docker run \
   -v ./inputs:/app/inputs \
   -v ./outputs:/app/outputs \
   ghcr.io/eerl-epfl/helikite-data-processing:latest generate_config
```

# Makefile

The Makefile provides simple commands for common tasks:

```bash
make build             # Build the Docker image
make generate_config   # Generate the configuration file in the inputs folder
make preprocess        # Preprocess data and update the configuration file
make process           # Process data and generate plots (output goes into a timestamped folder)
```

# Usage
After installation, the CLI is available as a system command:

```bash
helikite --help
```

1. **Generate a configuration file:**
   This creates a config file in your `inputs` folder.
   ```bash
   helikite generate-config
   ```

2. **Preprocess:**
   Scans the input folder, associates raw instrument files to configurations, and updates the config file.
   ```bash
   helikite preprocess
   ```

3. **Process:**
   Processes the input data based on the configuration, normalizes timestamps, and generates plots.
   (Running without any command runs this stage.)
   ```bash
   helikite
   ```

For detailed help on any command, append `--help` (e.g., `helikite preprocess --help`).

# Configuration

There are three sources of configuration parameters:

## Application constants

These are defined in `helikite/constants.py` and include settings such as filenames, folder paths for inputs/outputs, logging formats, and default plotting parameters.

## Runtime configuration

The runtime configuration is stored in `config.yaml` (located in your `inputs` folder). This file is generated during the `generate_config` or `preprocess` steps. It holds runtime arguments for each instrument (e.g., file locations, time adjustments, and plotting settings).

Below is an example snippet from a generated `config.yaml`:

```yaml
global:
  time_trim:
    start: 2022-09-29 10:21:58
    end: 2022-09-29 12:34:36
ground_station:
  altitude: null
  pressure: null
  temperature: 7.8
instruments:
  filter:
    config: filter
    date: null
    file: /app/inputs/220209A3.TXT
    pressure_offset: null
    time_offset:
      hour: 5555
      minute: 0
      second: 0
plots:
  altitude_ground_level: false
  grid:
    resample_seconds: 60
```
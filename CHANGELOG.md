# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.1] - 2025-03-17

### Added
- Sphinxdocs auto-generated documentation to github pages on push to the `main` branch.
- Automated PyPi publishing to the `helikite-data-processing` package on version bump.

### Changed
- The version property to be conventionally accessible with the __version__
  attribute in __init__.py instead of parsed from `pyproject.toml`.

## [1.1.0] - 2025-01-14

### Changed
- Transformed the project from a container-only application to a library.

### Added
- Introduced command-line functionality while maintaining the quicklook legacy functionality.
- Migrated functions from notebooks into the library:
    - CO2, cross-correlation, altitude
    - Established a structure for unit testing these functions.
- Developed a Level 0 cleaning utility:
    - Documented the functions within the cleaning utility.
    - Provided a how-to notebook example.
- Created custom interactive tools for tagging and identifying outliers:
    - Included notebook examples.
- Added preliminary support for Flight Computer v2:
    - Implemented cleaning functions based on initial data samples.

## [Unreleased]


## [1.1.0] - 2025-01-14
## [v1.0.3] - 2024-02-27
## [v1.0.2] - 2023-05-18
## [v1.0.1] - 2023-05-18
## [v1.0.0] - 2023-05-02

[unreleased]: https://github.com/EERL-EPFL/helikite-data-processing/compare/1.1.1...HEAD
[1.1.1]: https://github.com/EERL-EPFL/helikite-data-processing/compare/v1.1.0...1.1.1
[1.1.0]: https://github.com/EERL-EPFL/helikite-data-processing/compare/v1.0.3...1.1.0
[v1.0.3]: https://github.com/EERL-EPFL/helikite-data-processing/compare/v1.0.2...v1.0.3
[v1.0.2]: https://github.com/EERL-EPFL/helikite-data-processing/compare/v1.0.1...v1.0.2
[v1.0.1]: https://github.com/EERL-EPFL/helikite-data-processing/compare/v1.0.0...v1.0.1
[v1.0.0]: https://github.com/EERL-EPFL/helikite-data-processing/releases/tag/v1.0.0
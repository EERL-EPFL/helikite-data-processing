import inspect
import logging
import pathlib
from abc import abstractmethod, ABC
from collections import defaultdict
from functools import wraps
from graphlib import TopologicalSorter
from typing import Any

import pandas as pd
from pydantic import BaseModel

from helikite.classes.output_schemas import OutputSchema, Level
from helikite.constants import constants
from helikite.instruments import Instrument, flight_computer_v2, flight_computer_v1
from helikite.instruments.flight_computer import FlightComputer
from helikite.processing.helpers import suppress_plots

logger = logging.getLogger(__name__)
logger.setLevel(constants.LOGLEVEL_CONSOLE)


def function_dependencies(required_operations: list[str | tuple[str, ...]], changes_df: bool, use_once: bool,
                          complete_with_arg: Any | None = None, complete_req: bool = False):
    """A decorator to enforce that a method can only run if the required
    operations have been completed and not rerun.
    """

    def decorator(func):
        @wraps(func)  # This will preserve the original docstring and signature
        def wrapper(self, *args, **kwargs):
            # Obtained the complete operation name
            operation_name = func.__name__
            dependencies = required_operations.copy()

            if complete_with_arg is not None:
                signature = inspect.signature(func)
                bound_args = signature.bind(self, *args, **kwargs)
                bound_args.apply_defaults()

                arg = bound_args.arguments[str(complete_with_arg)]
                operation_name += "_" + str(arg)

                if complete_req:
                    dependencies = [f"{operation}_{arg}" for operation in dependencies]

            # Check if the function has already been completed
            if use_once and operation_name in self._completed_operations:
                print(
                    f"The operation '{operation_name}' has already been "
                    "completed and cannot be run again."
                )
                return

            functions_required = []
            # Ensure all required operations have been completed
            for operation in dependencies:
                # for operations tuple any of the operations in the tuple can be completed to satisfy the dependency
                if isinstance(operation, tuple):
                    if all(op not in self._completed_operations for op in operation):
                        functions_required.append(f"{' | '.join(operation)}")
                else:
                    if operation not in self._completed_operations:
                        functions_required.append(operation)

            if functions_required:
                print(
                    f"This function '{operation_name}' requires the "
                    "following operations first: "
                    f"{', '.join(functions_required)}."
                )
                return  # Halt execution of the function if dependency missing

            # Run the function
            result = func(self, *args, **kwargs)

            # Mark the function as completed
            self._completed_operations.append(operation_name)

            return result

        # Store dependencies and use_once information in the wrapper function
        wrapper.__dependencies__ = required_operations
        wrapper.__use_once__ = use_once
        wrapper.__changes_df__ = changes_df
        wrapper.__arg__ = str(complete_with_arg)

        return wrapper

    return decorator


class BaseProcessor(ABC):
    def __init__(
        self,
        output_schema: OutputSchema,
        instruments: list[Instrument],
        reference_instrument: Instrument,
    ) -> None:
        self._output_schema = output_schema
        self._instruments: list[Instrument] = instruments
        self._reference_instrument: Instrument = reference_instrument
        self._completed_operations: list[str] = []

    @property
    @abstractmethod
    def level(self) -> Level:
        ...

    @property
    def output_schema(self):
        return self._output_schema

    @property
    def instruments(self):
        return self._instruments

    @property
    def reference_instrument(self):
        return self._reference_instrument

    @property
    @abstractmethod
    def df(self) -> pd.DataFrame | None:
        ...

    def _check_schema_contains_instrument(self, instrument: Instrument) -> bool:
        if instrument not in self._output_schema.instruments:
            logger.warning(f"{self._output_schema.__class__.__name__} does not contain {instrument.name}. Skipping.")
            return False
        return True

    @property
    def _flight_computer(self) -> FlightComputer | None:
        for instrument in self._output_schema.instruments:
            if isinstance(instrument, FlightComputer):
                return instrument
        return None

    @abstractmethod
    def _data_state_info(self) -> list[str]:
        ...

    def _operations_state_info(self) -> list[str]:
        operations_state_info = []

        # Add the functions that have been called and completed
        operations_state_info.append("\nCompleted operations")
        operations_state_info.append("-" * 30)

        if len(self._completed_operations) == 0:
            operations_state_info.append("No operations have been completed.")

        for operation in self._completed_operations:
            operations_state_info.append(f"{operation:<25}")

        return operations_state_info

    def state(self):
        """Prints the current state of the class in a tabular format"""

        state_info = []

        state_info += self._data_state_info()
        state_info += self._operations_state_info()

        # Print all the collected info in a nicely formatted layout
        print("\n".join(state_info))
        print()

    def help(self):
        """Prints available methods in a clean format"""

        print("\nThere are several methods available to process the data:")

        methods = inspect.getmembers(self, predicate=inspect.ismethod)
        for name, method in methods:
            if not name.startswith("_"):
                # Get method signature (arguments)
                signature = inspect.signature(method)
                func_wrapper = getattr(self.__class__, name)

                # Extract func dependencies and use_once details from decorator
                dependencies = getattr(func_wrapper, "__dependencies__", [])
                use_once = getattr(func_wrapper, "__use_once__", False)

                # Print method name and signature
                print(f"- {name}{signature}")

                # Get the first line of the method docstring
                docstring = inspect.getdoc(method)
                if docstring:
                    first_line = docstring.splitlines()[
                        0
                    ]  # Get only the first line
                    print(f"\t{first_line}")
                else:
                    print("\tNo docstring available.")

                # Print function dependencies and use_once details
                if dependencies:
                    dependencies = [d if isinstance(d, str) else " | ".join(d) for d in dependencies]
                    print(f"\tDependencies: {', '.join(dependencies)}")
                if use_once:
                    print("\tNote: Can only be run once")

    def _outliers_file_state_info(self, outliers_file: str, add_all: bool = False) -> list[str]:
        state_info = []

        outliers = pd.read_csv(outliers_file, index_col=0, parse_dates=True)
        columns_with_outliers = outliers.columns if add_all else outliers.columns[outliers.any()]

        state_info.append(f"{'Outliers file':<40}{outliers_file}")
        state_info.append(f"{'Variable':<40}{'Number of outliers':<20}")
        state_info.append("-" * 60)

        for column in columns_with_outliers:
            state_info.append(
                f"{column:<40}{outliers[column].sum():<20}"
            )

        return state_info

    def _print_success_errors(
        self,
        operation: str,
        success: list[str],
        errors: list[tuple[str, Any]],
    ) -> None:
        print(
            f"Set {operation} for "
            f"({len(success)}/{len(self._instruments)}): {', '.join(success)}"
        )
        print(f"Errors ({len(errors)}/{len(self._instruments)}):")
        for error in errors:
            print(f"Error ({error[0]}): {error[1]}")

    @abstractmethod
    def export_data(self, filepath: str | pathlib.Path | None = None):
        ...


def get_instruments_from_cleaned_data(
    df: pd.DataFrame,
    metadata: BaseModel,
    flight_computer_version: str | None,
) -> tuple[list[Instrument], Instrument]:
    flight_computer_name = "flight_computer"

    instruments = []
    reference_instrument = None

    for name in metadata.instruments:
        is_reference = name == metadata.reference_instrument

        if name == flight_computer_name:
            if flight_computer_version is None:
                if f"{flight_computer_name}_{flight_computer_v1.pressure_variable}" in df.columns:
                    flight_computer_version = "v1"
                elif f"{flight_computer_name}_{flight_computer_v2.pressure_variable}" in df.columns:
                    flight_computer_version = "v2"
                else:
                    raise ValueError("Could not determine flight computer version. "
                                     "Please specify `flight_computer_version` manually.")
            name += f"_{flight_computer_version}"

        instrument = Instrument.REGISTRY[name]
        if is_reference:
            reference_instrument = instrument
        instruments.append(instrument)

    return instruments, reference_instrument

def launch_operations_changing_df(data_processor: BaseProcessor):
    cls = data_processor.__class__
    operations = {}
    operations_kwargs = defaultdict(list)
    for attr_name, attr in cls.__dict__.items():
        if callable(attr) and getattr(attr, "__changes_df__", False):
            operations[attr_name] = getattr(attr, "__dependencies__")
            arg = getattr(attr, "__arg__")

            match arg:
                case "flag":
                    for flag in data_processor.output_schema.flags:
                        operations_kwargs[attr_name].append({"flag": flag})
                case "instrument":
                    for instrument in data_processor.instruments:
                        operations_kwargs[attr_name].append({"instrument": instrument})
                case _:
                    continue

    operations_sorted = list(TopologicalSorter(operations).static_order())

    with suppress_plots():
        for operation in operations_sorted:
            if operation not in operations_kwargs:
                operations_kwargs[operation] = [{}]
            for kwargs in operations_kwargs[operation]:
                getattr(data_processor, operation)(**kwargs)

    return data_processor

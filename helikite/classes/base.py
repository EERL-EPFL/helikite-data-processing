import inspect
import logging
from abc import abstractmethod, ABC
from collections import defaultdict
from dataclasses import dataclass
from functools import wraps
from itertools import cycle
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from pydantic import BaseModel

from helikite.constants import constants
from helikite.instruments import Instrument, flight_computer_v2, smart_tether, pops, msems_readings, msems_inverted, \
    msems_scan, mcda, filter, tapir, cpc, flight_computer_v1, stap, stap_raw, co2
from helikite.instruments.flight_computer import FlightComputer

logger = logging.getLogger(__name__)
logger.setLevel(constants.LOGLEVEL_CONSOLE)

def _build_colors_defaultdict():
    cmap = plt.get_cmap("tab10")
    colors = cycle(cmap.colors)
    color_dict = defaultdict(lambda: next(colors))

    return color_dict


@dataclass(frozen=True)
class OutputSchema:
    instruments: list[Instrument]
    """List of instruments whose columns should be present in the output dataframe."""
    colors: dict[Instrument, str]
    """Instrument-to-color dictionary for the consistent across a campaign plotting"""
    reference_instrument_candidates: list[Instrument]
    """Reference instrument candidates for the automatic instruments detection"""


class OutputSchemas:
    ORACLES = OutputSchema(
        instruments=[
            flight_computer_v2,
            smart_tether,
            pops,
            msems_readings,
            msems_inverted,
            msems_scan,
            mcda,
            filter,
            tapir,
            cpc,
        ],
        colors={
            flight_computer_v2: "C0",
            smart_tether: "C1",
            pops: "C2",
            msems_readings: "C3",
            msems_inverted: "C6",
            msems_scan: "C5",
            mcda: "C4",
            filter: "C7",
            tapir: "C8",
            cpc: "C9",
        },
        reference_instrument_candidates=[flight_computer_v2, smart_tether, pops],
    )

    TURTMANN = OutputSchema(
        instruments=[
            flight_computer_v1,
            smart_tether,
            pops,
            msems_readings,
            msems_inverted,
            msems_scan,
            stap,
            co2,
            filter,
        ],
        colors={
            flight_computer_v1: "C0",
            smart_tether: "C1",
            pops: "C2",
            msems_readings: "C3",
            msems_inverted: "C6",
            msems_scan: "C5",
            stap: "C4",
            stap_raw: "C8",
            co2: "C9",
            filter: "C7",
        },
        reference_instrument_candidates=[flight_computer_v2, smart_tether, pops],
    )

    ALL = OutputSchema(
        instruments=list(Instrument.REGISTRY.values()),
        colors=_build_colors_defaultdict(),
        reference_instrument_candidates=[flight_computer_v2, flight_computer_v1, smart_tether, pops],
    )

    @classmethod
    def from_name(cls, name: str):
        return getattr(cls, name.upper())


def function_dependencies(required_operations: list[str], changes_df: bool, use_once: bool):
    """A decorator to enforce that a method can only run if the required
    operations have been completed and not rerun.
    """

    def decorator(func):
        @wraps(func)  # This will preserve the original docstring and signature
        def wrapper(self, *args, **kwargs):
            # Check if the function has already been completed
            if use_once and func.__name__ in self._completed_operations:
                print(
                    f"The operation '{func.__name__}' has already been "
                    "completed and cannot be run again."
                )
                return

            functions_required = []
            # Ensure all required operations have been completed
            for operation in required_operations:
                if operation not in self._completed_operations:
                    functions_required.append(operation)

            if functions_required:
                print(
                    f"This function '{func.__name__}()' requires the "
                    "following operations first: "
                    f"{'(), '.join(functions_required)}()."
                )
                return  # Halt execution of the function if dependency missing

            # Run the function
            result = func(self, *args, **kwargs)

            # Mark the function as completed
            self._completed_operations.append(func.__name__)

            return result

        # Store dependencies and use_once information in the wrapper function
        wrapper.__dependencies__ = required_operations
        wrapper.__use_once__ = use_once
        wrapper.__changes_df__ = changes_df

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
    def instruments(self):
        return self._instruments

    @property
    def reference_instrument(self):
        return self._reference_instrument

    def _check_schema_contains_instrument(self, instrument: Instrument) -> bool:
        if instrument not in self._output_schema.instruments:
            logger.warning(f"{self._output_schema.__class__.__name__} does not contain {instrument.name}. Skipping.")
            return False
        return True

    @property
    def _flight_computer(self) -> FlightComputer:
        flight_computer = next(instrument for instrument in self._instruments if isinstance(instrument, FlightComputer))
        return flight_computer

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
                    print(f"\tDependencies: {', '.join(dependencies)}")
                if use_once:
                    print("\tNote: Can only be run once")

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


def get_instruments_from_cleaned_data(df: pd.DataFrame, metadata: BaseModel) -> tuple[list[Instrument], Instrument]:
    flight_computer_version = None
    flight_computer_outdated_name = "flight_computer"

    instruments = []
    reference_instrument = None

    for name in metadata.instruments:
        is_reference = name == metadata.reference_instrument

        if name == flight_computer_outdated_name:
            if flight_computer_version is None:
                if f"{flight_computer_outdated_name}_{flight_computer_v1.pressure_variable}" in df.columns:
                    flight_computer_version = "v1"
                elif f"{flight_computer_outdated_name}_{flight_computer_v2.pressure_variable}" in df.columns:
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

from dataclasses import dataclass
from enum import Enum

from helikite.instruments import Instrument, flight_computer_v2, smart_tether, pops, msems_readings, \
    msems_inverted, msems_scan, mcda, filter, tapir, cpc


@dataclass(frozen=True)
class OutputSchema:
    instruments: list[Instrument]
    """List of instruments whose columns should be present in the output dataframe."""


# TODO: implement other schemas used
class OutputSchemas(OutputSchema, Enum):
    ORACLES = [
        flight_computer_v2,
        smart_tether,
        pops,
        msems_readings,
        msems_inverted,
        msems_scan,
        mcda,
        filter,
        tapir,
        cpc
    ]

import logging
from dataclasses import dataclass
from enum import Enum

from helikite.constants import constants
from helikite.instruments import Instrument, flight_computer_v1, flight_computer_v2, smart_tether, pops, msems_readings, \
    msems_inverted, msems_scan, mcda, filter, tapir, cpc, stap, stap_raw, co2, mcpc

logger = logging.getLogger(__name__)
logger.setLevel(constants.LOGLEVEL_CONSOLE)


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
    TURTMANN = [
        flight_computer_v1,
        smart_tether,
        pops,
        msems_readings,
        msems_inverted,
        msems_scan,
        stap,
        stap_raw,
        co2,
        filter,
        mcpc,
    ]

from collections import defaultdict
from dataclasses import dataclass
from itertools import cycle

from matplotlib import pyplot as plt

from helikite.instruments import Instrument, flight_computer_v2, smart_tether, msems_readings, msems_inverted, \
    msems_scan, pops, mcda, filter, tapir, cpc, flight_computer_v1, stap, co2, stap_raw


def _build_colors_defaultdict():
    cmap = plt.get_cmap("tab10")
    colors = cycle(cmap.colors)
    color_dict = defaultdict(lambda: next(colors))

    return color_dict


@dataclass(frozen=True)
class OutputSchema:
    campaign: str | None
    """Campaign name"""
    instruments: list[Instrument]
    """List of instruments whose columns should be present in the output dataframe."""
    colors: dict[Instrument, str]
    """Instrument-to-color dictionary for the consistent across a campaign plotting"""
    reference_instrument_candidates: list[Instrument]
    """Reference instrument candidates for the automatic instruments detection"""
    flags: tuple[str] = ("flag_pollution", "flag_hovering", "flag_cloud")
    """List of flags which should be present in the output dataframe."""


class OutputSchemas:
    ORACLES = OutputSchema(
        campaign="ORACLES",
        instruments=[
            flight_computer_v2,
            smart_tether,
            msems_readings,
            msems_inverted,
            msems_scan,
            pops,
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
        campaign="TURTMANN",
        instruments=[
            flight_computer_v1,
            smart_tether,
            msems_readings,
            msems_inverted,
            msems_scan,
            pops,
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
        campaign=None,
        instruments=list(Instrument.REGISTRY.values()),
        colors=_build_colors_defaultdict(),
        reference_instrument_candidates=[flight_computer_v2, flight_computer_v1, smart_tether, pops],
    )

    @classmethod
    def from_name(cls, name: str):
        return getattr(cls, name.upper())

import dataclasses
from collections import defaultdict
from dataclasses import dataclass
from itertools import cycle
from numbers import Number

from matplotlib import pyplot as plt

from helikite.instruments import Instrument, flight_computer_v2, smart_tether, msems_readings, msems_inverted, \
    msems_scan, pops, mcda, filter, tapir, cpc, flight_computer_v1, stap, co2, stap_raw


def _build_colors_defaultdict():
    cmap = plt.get_cmap("tab10")
    colors = cycle(cmap.colors)
    color_dict = defaultdict(lambda: next(colors))

    return color_dict


@dataclass
class FlightProfileVariable:
    column_name: str
    shade_flags: list[str] = dataclasses.field(default_factory=list)
    plot_kwargs: dict = dataclasses.field(default_factory=dict)
    alpha_descent: float = 0.5
    x_bounds: tuple[float, float] | None = None
    x_divider: Number | None = None
    x_label: str | None = None


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
    flight_profile_variables: list[FlightProfileVariable] = dataclasses.field(default_factory=list)
    flags: tuple[str] = ("flag_pollution", "flag_hovering", "flag_cloud")
    """List of flags which should be present in the output dataframe."""


class OutputSchemas:
    _REGISTRY: dict[str, OutputSchema] = {}

    ORACLES_24_25 = OutputSchema(
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
        flight_profile_variables=[
            FlightProfileVariable(
                column_name="Average_Temperature",
                plot_kwargs=dict(color="brown", linewidth=3.0),
                x_divider=2,  # divider hint, bounds calculated
                x_label="Temp (°C)",
            ),
            FlightProfileVariable(
                column_name="Average_RH",
                plot_kwargs=dict(color="orange", linewidth=3.0),
                x_bounds=(60, 100),
                x_divider=10,
                x_label="RH (%)",
            ),
            FlightProfileVariable(
                column_name="msems_inverted_dN_totalconc_stp",
                plot_kwargs=dict(color="indigo", marker="."),
                alpha_descent=0.3,
                x_bounds=(0, 1200),
                x_divider=200,
                x_label="mSEMS conc. (cm$^{-3}$) [8-250 nm]",
            ),
            FlightProfileVariable(
                column_name="cpc_totalconc_stp",
                plot_kwargs=dict(color="orchid", linewidth=3.0),
                x_bounds=(0, 1200),
                x_divider=200,
                x_label="CPC conc. (cm$^{-3}$) [7–2000 nm]",
            ),
            FlightProfileVariable(
                column_name="pops_total_conc_stp",
                plot_kwargs=dict(color="teal", linewidth=3.0),
                x_bounds=(0, 60),
                x_divider=10,
                x_label="POPS conc. (cm$^{-3}$) [186-3370 nm]",
            ),
            FlightProfileVariable(
                column_name="mcda_dN_totalconc_stp",
                plot_kwargs=dict(color="salmon", linewidth=3.0),
                x_bounds=(0, 60),
                x_divider=10,
                x_label="mCDA conc. (cm$^{-3}$) [0.66–33 um]",
            ),
            FlightProfileVariable(
                column_name="smart_tether_Wind (m/s)",
                plot_kwargs=dict(color="palevioletred", marker=".", linestyle="none"),
                alpha_descent=0.2,
                x_divider=2,  # divider hint, bounds calculated
                x_label="WS (m/s)",
            ),
            FlightProfileVariable(
                column_name="smart_tether_Wind (degrees)",
                plot_kwargs=dict(color="olivedrab", marker=".", linestyle="none"),
                alpha_descent=0.3,
                x_bounds=(0, 360),
                x_divider=90,
                x_label="WD (deg)",
            ),
        ]
    )

    # the difference from 24-25 is that there is no tapir
    ORACLES_25_26 = dataclasses.replace(
        ORACLES_24_25,
        instruments=[
            flight_computer_v2,
            smart_tether,
            msems_readings,
            msems_inverted,
            msems_scan,
            pops,
            mcda,
            filter,
            cpc,
        ],
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
    def from_name(cls, name: str) -> OutputSchema:
        try:
            return cls._REGISTRY[name.upper()]
        except KeyError:
            raise KeyError(f"Unknown OutputSchema '{name}'")

    @classmethod
    def _register_builtin(cls):
        for name, value in vars(cls).items():
            if isinstance(value, OutputSchema):
                cls._REGISTRY[name.upper()] = value

    @classmethod
    def register(cls, name: str, schema: OutputSchema, *, overwrite: bool = False):
        key = name.upper()
        if not overwrite and key in cls._REGISTRY:
            raise ValueError(f"OutputSchema '{name}' is already registered")
        cls._REGISTRY[key] = schema


OutputSchemas._register_builtin()

import dataclasses
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from itertools import cycle
from numbers import Number
from typing import Callable

import pandas as pd
from matplotlib import pyplot as plt

from helikite.instruments import Instrument, flight_computer_v2, smart_tether, msems_readings, msems_inverted, \
    msems_scan, pops, mcda, filter, tapir, cpc, flight_computer_v1, stap, co2, stap_raw
from helikite.processing.post.fda import FDAParameters, FDA_PARAMS_POLLUTION, FDA_PARAMS_HOVERING, FDA_PARAMS_CLOUD


def _build_colors_defaultdict():
    cmap = plt.get_cmap("tab10")
    colors = cycle(cmap.colors)
    color_dict = defaultdict(lambda: next(colors))

    return color_dict


class Level(Enum):
    LEVEL0 = 0
    LEVEL1 = 1
    LEVEL1_5 = 1.5
    LEVEL2 = 2


@dataclass(frozen=True)
class Flag:
    flag_name: str
    column_name: str
    params: FDAParameters
    y_scale: str = "log"

    def __str__(self):
        return self.flag_name


@dataclass(frozen=True)
class FlightProfileVariable:
    column_name: str
    shade_flags: list[str] = dataclasses.field(default_factory=list)
    plot_kwargs: dict = dataclasses.field(default_factory=dict)
    alpha_ascent: float = 1.0
    alpha_descent: float = 0.5
    x_min: Number | None = None
    x_max: Number | None = None
    x_divider: Number | None = None
    x_label: str | None = None


@dataclass(frozen=True)
class FlightProfileVariableShade:
    name: str
    condition: Callable[[Level, pd.Series], pd.Series]
    """
    Predicate function that receives a column value and returns True if the corresponding row should be shaded,
    False otherwise.
    """
    label: str
    span_kwargs: dict
    line_name: str | None = None
    line_kwargs: dict | None = None


flag_pollution_cpc = Flag(flag_name="flag_pollution", column_name="CPC_total_N", params=FDA_PARAMS_POLLUTION)
flag_hovering = Flag(flag_name="flag_hovering", column_name="Altitude", params=FDA_PARAMS_HOVERING, y_scale="linear")
flag_cloud_mcda = Flag(flag_name="flag_cloud", column_name="mCDA_total_N", params=FDA_PARAMS_CLOUD)

shade_pollution_cpc = FlightProfileVariableShade(
    name=flag_pollution_cpc.flag_name,
    condition=lambda l, v: v,
    label="Pollution",
    span_kwargs=dict(color="lightcoral", alpha=0.5),
    line_name=flag_pollution_cpc.column_name,
    line_kwargs=dict(color="rosybrown", label="CPC (cm⁻³)", linewidth=1),
)
shade_pollution_cpc_ground = dataclasses.replace(
    shade_pollution_cpc,
    line_kwargs=dict(color="rosybrown", label="ground CPC (cm⁻³)", linewidth=1),
)
shade_hovering = FlightProfileVariableShade(
    name=flag_hovering.flag_name,
    condition=lambda l, v: v,
    label="Hovering",
    span_kwargs=dict(color="beige", alpha=1.0),
)
shade_cloud_mcda = FlightProfileVariableShade(
    name=flag_cloud_mcda.flag_name,
    condition=lambda l, v: v,
    label="Cloud",
    span_kwargs=dict(color="lightblue", alpha=0.5),
)


def filter_shade_condition(level: Level, values: pd.Series) -> pd.Series:
    match level:
        case Level.LEVEL2:
            return values != 0.0
        case _:
            return values != 1.0


shade_filter = FlightProfileVariableShade(
    name="Filter_position",
    condition=filter_shade_condition,
    label="Filter",
    span_kwargs=dict(facecolor='none', edgecolor='gray', hatch='////', alpha=0.5),
)


@dataclass(frozen=True)
class OutputSchema:
    campaign: str | None
    """Campaign name"""
    instruments: list[Instrument]
    """List of instruments whose columns should be present in the output dataframe."""
    colors: dict[Instrument | str, str]
    """[Instrument|column]-to-color dictionary for the consistent across a campaign plotting."""
    reference_instrument_candidates: list[Instrument]
    """Reference instrument candidates for the automatic instruments detection"""
    flight_profile_variables: list[FlightProfileVariable] = dataclasses.field(default_factory=list)
    flight_profile_shades: list[FlightProfileVariableShade] = (
        shade_hovering, shade_pollution_cpc, shade_cloud_mcda, shade_filter
    )
    flags: list[Flag] = (flag_pollution_cpc, flag_hovering, flag_cloud_mcda)
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
        colors=_build_colors_defaultdict() | {
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
                x_divider=1,  # divider hint, bounds calculated
                x_label="Temp (°C)",
            ),
            FlightProfileVariable(
                column_name="Average_RH",
                plot_kwargs=dict(color="orange", linewidth=3.0),
                x_max = 100,
                x_divider=10,
                x_label="RH (%)",
            ),
            FlightProfileVariable(
                column_name="cpc_totalconc_stp",
                plot_kwargs=dict(color="orchid", linewidth=3.0),
                x_min = 0,
                x_divider=100,
                x_label="CPC conc. (cm$^{-3}$) [7–2000 nm]",
            ),
            FlightProfileVariable(
                column_name="msems_inverted_dN_totalconc_stp",
                shade_flags=["flag_pollution", "flag_cloud"],
                plot_kwargs=dict(color="indigo", marker=".", linestyle="none"),
                alpha_descent=0.3,
                x_min = 0,
                x_divider=200,
                x_label="mSEMS conc. (cm$^{-3}$) [8-250 nm]",
            ),
            FlightProfileVariable(
                column_name="pops_total_conc_stp",
                shade_flags=["flag_cloud"],
                plot_kwargs=dict(color="teal", linewidth=3.0),
                x_min = 0,
                x_divider=10,
                x_label="POPS conc. (cm$^{-3}$) [186-3370 nm]",
            ),
            FlightProfileVariable(
                column_name="mcda_dN_totalconc_stp",
                plot_kwargs=dict(color="salmon", linewidth=3.0),
                alpha_ascent=0.6,
                alpha_descent=0.3,
                x_min = 0,
                x_divider=10,
                x_label="mCDA conc. (cm$^{-3}$) [0.66–33 um]",
            ),
            FlightProfileVariable(
                column_name="smart_tether_Wind (m/s)",
                plot_kwargs=dict(color="palevioletred", marker=".", linestyle="none"),
                alpha_descent=0.2,
                x_min = 0,
                x_divider=1,  # divider hint, bounds calculated
                x_label="WS (m/s)",
            ),
            FlightProfileVariable(
                column_name="smart_tether_Wind (degrees)",
                plot_kwargs=dict(color="olivedrab", marker=".", linestyle="none"),
                alpha_descent=0.3,
                x_min = 0,
                x_max = 360,
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
        flight_profile_shades=[
            shade_hovering,
            shade_pollution_cpc_ground,
            shade_cloud_mcda,
            shade_filter
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
        colors=_build_colors_defaultdict() | {
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

    @classmethod
    def keys(cls):
        return cls._REGISTRY.keys()


OutputSchemas._register_builtin()

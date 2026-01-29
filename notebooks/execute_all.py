import argparse
import datetime
import os
import re
import traceback
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from helikite import Cleaner
from helikite.classes.data_processing_level1 import DataProcessorLevel1
from helikite.classes.data_processing_level1_5 import DataProcessorLevel1_5
from helikite.classes.data_processing_level2 import DataProcessorLevel2
from helikite.classes.output_schemas import OutputSchemas, flag_pollution
from helikite.config import Config
from helikite.instruments import flight_computer_v2, msems_scan
from helikite.metadata.utils import load_parquet


def execute_level0(config: Config):
    input_dir = config.campaign_data_dirpath / config.flight_basename
    output_level0_dir = config.processing_dir / "Level0"
    output_level0_dir.mkdir(parents=True, exist_ok=True)

    cleaner = Cleaner(
        output_schema=OutputSchemas.from_name(config.output_schema),
        input_folder=input_dir,
        flight_date=config.flight_date,
        flight=config.flight,
        interactive=False,
    )
    reference_instrument = cleaner.reference_instrument

    if flight_computer_v2 not in cleaner.instruments:
        return

    cleaner.set_time_as_index()
    cleaner.fill_missing_timestamps(reference_instrument, freq="1s", fill_method="ffill")
    cleaner.data_corrections()
    cleaner.set_pressure_column()

    cleaner.pressure_based_time_synchronization()

    cleaner.define_flight_times()
    if cleaner.time_takeoff is None:
        cleaner.time_takeoff = reference_instrument.df.index[0].to_pydatetime() + datetime.timedelta(seconds=120)
    if cleaner.time_landing is None:
        cleaner.time_landing = reference_instrument.df.index[-1].to_pydatetime() - datetime.timedelta(seconds=120)

    cleaner.remove_duplicates()
    cleaner.merge_instruments()

    save_path = output_level0_dir / f"Level0_{config.flight_basename}_Flight_{config.flight}_TimeSync.png"
    cleaner.plot_time_sync(save_path, skip=[msems_scan])
    cleaner.shift_msems_columns_by_90s()

    cleaner.export_data(filepath=output_level0_dir / f"level0_{config.flight_basename}")


def execute_level1(config: Config):
    input_level0_dir = config.processing_dir / "Level0"
    output_level1_dir = config.processing_dir / "Level1"
    output_level1_dir.mkdir(parents=True, exist_ok=True)

    df_level0, metadata = load_parquet(input_level0_dir / f"level0_{config.flight_basename}.parquet")

    data_processor = DataProcessorLevel1(getattr(OutputSchemas, config.output_schema), df_level0, metadata)
    data_processor.add_missing_columns()

    outliers_file = output_level1_dir / f"level1_{config.flight_basename}_outliers.csv"
    data_processor.detect_outliers(outliers_file=outliers_file)
    data_processor.choose_outliers(outliers_file=outliers_file)
    data_processor.set_outliers_to_nan()

    data_processor.fillna_if_all_missing({"flight_computer_Lat": 7039.724, "flight_computer_Long": 817.1591})
    data_processor.plot_outliers_check()
    data_processor.convert_gps_coordinates(lat_col='flight_computer_Lat', lon_col='flight_computer_Long',
                                           lat_dir='S', lon_dir='W')
    data_processor.plot_gps_on_map(center_coords=(-70.6587, -8.2850), zoom_start=14)
    data_processor.T_RH_averaging(columns_t=None, columns_rh=None, nan_threshold=400)
    data_processor.plot_T_RH(save_path=output_level1_dir / f"Level1_{config.flight_basename}_T_RH_averaging.png")
    data_processor.altitude_calculation_barometric(offset_to_add=0)
    data_processor.plot_altitude()

    for instrument in data_processor.instruments:
        data_processor.calculate_derived(instrument)
        data_processor.normalize(instrument)
        data_processor.plot_raw_and_normalized_data(instrument)
        data_processor.plot_distribution(instrument)
        data_processor.plot_vertical_distribution(instrument)

    save_path = output_level1_dir / f'Level1_{config.flight_basename}_Flight_{config.flight}.png'
    data_processor.plot_flight_profiles(config.flight_basename, save_path, variables=None)

    save_path = output_level1_dir / f'Level1_{config.flight_basename}_SizeDistr_Flight_{config.flight}.png'
    data_processor.plot_size_distr(config.flight_basename, save_path, time_start=None, time_end=None)

    data_processor.export_data(filepath=output_level1_dir / f'level1_{config.flight_basename}.csv')


def execute_level1_5(config: Config):
    input_dir = config.processing_dir
    output_level1_5_dir = config.processing_dir / "Level1.5"
    output_level1_5_dir.mkdir(parents=True, exist_ok=True)

    df_level1 = DataProcessorLevel1.read_data(input_dir / "Level1" / f"level1_{config.flight_basename}.csv")

    _, metadata = load_parquet(input_dir / "Level0" / f"level0_{config.flight_basename}.parquet")

    data_processor = DataProcessorLevel1_5(getattr(OutputSchemas, config.output_schema), df_level1, metadata)
    data_processor.fill_msems_takeoff_landing(time_window_seconds=90)
    data_processor.remove_before_takeoff_and_after_landing()
    data_processor.filter_columns()
    data_processor.rename_columns()
    data_processor.round_flightnbr_campaign(decimals=2)

    output_flags_dir = output_level1_5_dir / "flags"

    cpc_on_ground = data_processor.output_schema == OutputSchemas.ORACLES_25_26
    for flag in data_processor.output_schema.flags:
        auto_file = output_flags_dir / f"level1.5_{config.flight_basename}_{flag.flag_name}_auto.csv"
        data_processor.detect_flag(flag, auto_file, plot_detection=True)
        data_processor.choose_flag(flag, auto_file, auto_file)

        if flag.flag_name == flag_pollution.flag_name and cpc_on_ground:
            close_to_ground = data_processor.df["Altitude"] < 70
            data_processor.set_flag(flag, auto_file, mask=close_to_ground)
        else:
            data_processor.set_flag(flag, auto_file)

    save_path = output_level1_5_dir / f'Level1.5_{config.flight_basename}_Flight_{config.flight}.png'
    data_processor.plot_flight_profiles(config.flight_basename, save_path, variables=None)

    save_path = output_level1_5_dir / f'Level1.5_{config.flight_basename}_SizeDistr_Flight_{config.flight}.png'
    data_processor.plot_size_distr(config.flight_basename, save_path, time_start=None, time_end=None)

    data_processor.export_data(output_level1_5_dir / f"level1.5_{config.flight_basename}.csv")


def execute_level2(config: Config):
    input_dir = config.processing_dir
    output_level2_dir = config.processing_dir / "Level2"
    output_level2_dir.mkdir(parents=True, exist_ok=True)

    df_level1_5 = DataProcessorLevel1_5.read_data(input_dir / "Level1.5" / f"level1.5_{config.flight_basename}.csv")

    _, metadata = load_parquet(input_dir / "Level0" / f"level0_{config.flight_basename}.parquet")

    data_processor = DataProcessorLevel2(getattr(OutputSchemas, config.output_schema), df_level1_5, metadata)
    data_processor.average(rule="10s")
    save_path = output_level2_dir / f'Level2_{config.flight_basename}_Flight_{config.flight}.png'

    data_processor.plot_flight_profiles(config.flight_basename, save_path, variables=None)

    save_path = output_level2_dir / f'Level2_{config.flight_basename}_SizeDistr_Flight_{config.flight}.png'
    data_processor.plot_size_distr(config.flight_basename, save_path, time_start=None, time_end=None)

    data_processor.export_data(output_level2_dir / f"level2_{config.flight_basename}.csv")


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("output_schema", type=str, choices=OutputSchemas.keys())
    arg_parser.add_argument("campaign_dir", type=Path)
    arg_parser.add_argument("processing_dir", type=Path)
    arg_parser.add_argument("--overview-path", type=Path)
    args = arg_parser.parse_args()

    basename_to_flight_nr = _get_basename_to_flight_nr_mapping(args.campaign_dir.parent)

    pattern = re.compile(
        r"^(?P<date>\d{4}-\d{2}-\d{2})_(?P<suffix>[A-Z])$"
    )

    processed_count = 0
    for flight_dirname in sorted(os.listdir(args.campaign_dir)):
        match = pattern.match(flight_dirname)
        if match is None:
            continue

        date = datetime.datetime.strptime(match.group("date"), '%Y-%m-%d').date()
        suffix = match.group("suffix")
        basename = f"{date}_{suffix}"

        config = Config(
            flight=basename_to_flight_nr.get(basename, str(processed_count + 1)),
            flight_date=date,
            flight_suffix=suffix,
            output_schema=args.output_schema,
            campaign_data_dirpath=args.campaign_dir,
            processing_dir=args.processing_dir,
        )

        try:
            # defaults get changed at some point, so set parameters back to default values before processing a new flight
            # TODO: remove this temporary fix once the reason why defaults change is understood
            plt.rcdefaults()

            execute_level0(config)
            execute_level1(config)
            execute_level1_5(config)
            execute_level2(config)
        except Exception as e:
            print(f"Error in processing {basename}")
            traceback.print_exc()

        processed_count += 1


def _get_basename_to_flight_nr_mapping(dirpath: Path):
    for filename in os.listdir(dirpath):
        if filename.endswith(".xlsx"):
            df = pd.read_excel(dirpath / filename)
            for col in ["Flight", "Date", "Code"]:
                if col not in df.columns:
                    continue
            df = df[["Flight", "Date", "Code"]].dropna()
            df["Flight"] = df["Flight"].astype(int)
            mapping = {
                f"{row['Date'].date()}_{row['Code']}": str(row["Flight"])
                for _, row in df.iterrows()
            }
            return mapping
    return {}


if __name__ == '__main__':
    main()

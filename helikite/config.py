import datetime
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel


class Config(BaseModel):
    flight: str
    flight_date: datetime.date
    flight_suffix: str

    output_schema: str
    campaign_data_dirpath: Path

    @property
    def flight_basename(self) -> str:
        return f"{self.flight_date}_{self.flight_suffix}"


def load_config(config_path: Path) -> Config:
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    return Config(**data)

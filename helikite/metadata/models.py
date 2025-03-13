from pydantic import BaseModel
import datetime


class Level0(BaseModel):
    flight: str | None = None
    flight_date: datetime.date
    takeoff_time: datetime.datetime
    landing_time: datetime.datetime
    reference_instrument: str
    instruments: list[str] = []  # The str repr of each Instrument

    def __repr__(self) -> str:
        return (
            f"Flight {self.flight} from {self.takeoff_time} to "
            f"{self.landing_time}"
        )

    def __str__(self) -> str:
        return "\n".join(
            f"{field}: {value}" for field, value in self.__dict__.items()
        )

    def _repr_html_(self) -> str:
        return self.__str__().replace("\n", "<br>")

import os
import fcntl
import pandas as pd
from pydantic import BaseModel
import datetime


class Flight(BaseModel):
    flight: str
    takeoff_time: datetime.datetime
    landing_time: datetime.datetime

    def __repr__(self) -> str:
        return (
            f"Flight {self.flight} from {self.takeoff_time} to "
            f"{self.landing_time}"
        )


class FlightState:
    """Store flight times

    Currently, flight times are stored in a CSV file. This class simulates
    a REST API and can be adapted to an actual API when needed.
    """

    def __init__(self, filename) -> None:
        self.filename = filename
        self.df = pd.read_csv(filename)
        self.df["takeoff_time"] = pd.to_datetime(self.df["takeoff_time"])
        self.df["landing_time"] = pd.to_datetime(self.df["landing_time"])

    def __repr__(self) -> str:
        return f"Filename: {self.filename}\nFlights in file: {len(self.df)}"

    def _convert_to_flight_model(self, row):
        """Convert a DataFrame row to a Pydantic Flight model."""
        return Flight(
            flight=row["flight"],
            takeoff_time=row["takeoff_time"],
            landing_time=row["landing_time"],
        )

    def get_all_flights(self):
        """Simulates a GET request to retrieve all flight data as Pydantic models."""
        return [
            self._convert_to_flight_model(row) for _, row in self.df.iterrows()
        ]

    def get_flight(self, flight_number):
        """Simulates a GET request to retrieve data for a specific flight."""
        flight = self.df[self.df["flight"] == flight_number]
        if flight.empty:
            return None
        return self._convert_to_flight_model(flight.iloc[0])

    def add_flight(self, flight_number, takeoff_time, landing_time):
        """Simulates a POST request to add a new flight."""
        if flight_number in self.df["flight"].values:
            raise ValueError(f"Flight {flight_number} already exists.")

        new_flight = pd.DataFrame(
            [
                {
                    "flight": flight_number,
                    "takeoff_time": pd.to_datetime(takeoff_time),
                    "landing_time": pd.to_datetime(landing_time),
                }
            ]
        )

        # Use pd.concat instead of append
        self.df = pd.concat([self.df, new_flight], ignore_index=True)
        self.save()

    def update_flight(
        self, flight_number, takeoff_time=None, landing_time=None
    ):
        """Simulates a PUT/PATCH request to update an existing flight."""
        flight = self.df[self.df["flight"] == flight_number]
        if flight.empty:
            raise ValueError(f"Flight {flight_number} does not exist.")

        if takeoff_time:
            self.df.loc[self.df["flight"] == flight_number, "takeoff_time"] = (
                pd.to_datetime(takeoff_time)
            )
        if landing_time:
            self.df.loc[self.df["flight"] == flight_number, "landing_time"] = (
                pd.to_datetime(landing_time)
            )
        self.save()

    def delete_flight(self, flight_number):
        """Simulates a DELETE request to remove a flight."""
        self.df = self.df[self.df["flight"] != flight_number]
        self.save()

    def save(self):
        """Locks the file, writes to temp, then replaces original"""
        temp_filename = self.filename + ".tmp"
        with open(temp_filename, "w") as temp_file:
            fcntl.flock(temp_file, fcntl.LOCK_EX)  # Lock the temp file
            self.df.to_csv(temp_file, index=False)
            fcntl.flock(temp_file, fcntl.LOCK_UN)
        os.replace(temp_filename, self.filename)  # Atomic replace

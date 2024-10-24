import os
import tempfile
import pytest
from datetime import datetime
from helikite.metadata.flights import (
    FlightState,
    Flight,
)


@pytest.fixture
def temp_csv_file():
    """Create a temporary CSV file for testing."""
    temp_file = tempfile.NamedTemporaryFile(
        delete=False, suffix=".csv", mode="w+"
    )
    temp_file.write("flight,takeoff_time,landing_time\n")
    temp_file.write("20241020A,2024-10-20 08:00:00,2024-10-20 10:00:00\n")
    temp_file.write("20241020B,2024-10-20 12:30:00,2024-10-20 14:30:00\n")
    temp_file.write("20241020C,2024-10-20 15:00:00,2024-10-20 17:00:00\n")
    temp_file.write("20241020D,2024-10-20 18:45:00,2024-10-20 21:00:00\n")
    temp_file.write("20241021A,2024-10-21 07:45:00,2024-10-21 09:45:00\n")
    temp_file.write("20241021B,2024-10-21 10:15:00,2024-10-21 12:15:00\n")
    temp_file.write("20241021C,2024-10-21 13:30:00,2024-10-21 15:30:00\n")
    temp_file.write("20241021D,2024-10-21 16:00:00,2024-10-21 18:00:00\n")
    temp_file.seek(0)
    yield temp_file.name
    os.remove(temp_file.name)


def test_get_all_flights(temp_csv_file):
    """Test retrieving all flights."""
    flight_state = FlightState(temp_csv_file)
    all_flights = flight_state.get_all_flights()
    assert len(all_flights) == 8  # Ensure there are 8 flights
    assert all_flights[0].flight == "20241020A"
    assert all_flights[1].flight == "20241020B"
    assert all_flights[2].flight == "20241020C"


def test_get_flight(temp_csv_file):
    """Test retrieving a specific flight."""
    flight_state = FlightState(temp_csv_file)
    flight = flight_state.get_flight("20241020A")
    assert flight is not None
    assert flight.flight == "20241020A"
    assert flight.takeoff_time == datetime(2024, 10, 20, 8, 0, 0)
    assert flight.landing_time == datetime(2024, 10, 20, 10, 0, 0)

    # Test flight that does not exist
    flight = flight_state.get_flight("non_existent_flight")
    assert flight is None


def test_add_flight(temp_csv_file):
    """Test adding a new flight."""
    flight_state = FlightState(temp_csv_file)
    flight_state.add_flight(
        "20240503A", "2024-05-03 10:00:00", "2024-05-03 12:00:00"
    )

    # Verify that the flight was added
    flight = flight_state.get_flight("20240503A")
    assert flight is not None
    assert flight.flight == "20240503A"
    assert flight.takeoff_time == datetime(2024, 5, 3, 10, 0, 0)
    assert flight.landing_time == datetime(2024, 5, 3, 12, 0, 0)


def test_update_flight(temp_csv_file):
    """Test updating an existing flight."""
    flight_state = FlightState(temp_csv_file)
    flight_state.update_flight("20241020A", takeoff_time="2024-10-20 09:00:00")

    # Verify the takeoff time was updated
    flight = flight_state.get_flight("20241020A")
    assert flight is not None
    assert flight.takeoff_time == datetime(2024, 10, 20, 9, 0, 0)


def test_delete_flight(temp_csv_file):
    """Test deleting a flight."""
    flight_state = FlightState(temp_csv_file)
    flight_state.delete_flight("20241020A")

    # Verify that the flight was deleted
    flight = flight_state.get_flight("20241020A")
    assert flight is None

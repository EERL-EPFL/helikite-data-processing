import numpy as np


def Air_density(T, P, RH=0):
    """
    Calculate dry and wet air density using the ideal gas law.

    This function computes the dry air density from the ideal gas law based on
    a reference state and adjusts for moisture content using the water vapor
    pressure to yield the wet air density.

    Parameters
    ----------
    T : float
        Temperature in Kelvin.
    P : float
        Pressure in hPa.
    RH : float, optional
        Relative humidity in percent (default is 0).

    Returns
    -------
    tuple of float
        A tuple containing:
            - dry air density in kg/m³
            - wet air density in kg/m³

    Notes
    -----
    The calculation uses standard reference conditions (rho0 = 1.29 kg/m³,
    P0 = 1013.25 hPa, T0 = 273.15 K) and applies an adjustment based on water
    vapor pressure.
    """

    rho0 = 1.29
    P0 = 1013.25
    T0 = 273.15
    rho = rho0 * (P / P0) * (T0 / T)
    rhow = rho * (1 - 0.378 * waterpressure(RH, T, P) / P)

    return rho, rhow


def waterpressure(RH, T, P):
    """
    Calculate the partial pressure of water vapor.

    This function determines the water vapor pressure given the relative
    humidity, temperature, and pressure. It computes the saturation vapor
    pressure and then scales it by the relative humidity.

    Parameters
    ----------
    RH : float
        Relative humidity in percent.
    T : float
        Temperature in Kelvin.
    P : float
        Pressure in hPa.

    Returns
    -------
    float
        Partial pressure of water vapor in hPa.
    """

    Pw = Watersatpress(P, T) * RH / 100.0
    return Pw


def Watersatpress(press, temp):
    """
    Calculate water saturation vapor pressure for moist air.

    This function computes the saturation vapor pressure for water based on the
    WMO CIMO guide, valid for temperatures between -45°C and 60°C. The
    temperature is first converted from Kelvin to Celsius before applying the
    empirical formula.

    Parameters
    ----------
    press : float
        Pressure in hPa.
    temp : float
        Temperature in Kelvin.

    Returns
    -------
    float
        Saturation vapor pressure of water (H₂O) in hPa.

    References
    ----------
    https://www.wmo.int/pages/prog/www/IMOP/CIMO-Guide.html
    """

    temp = temp - 273.16  # conversion to centigrade temperature
    ew = 6.112 * np.exp(
        17.62 * temp / (243.12 + temp)
    )  # calculate saturation pressure for pure water vapour
    f = 1.0016 + 3.15 * 10 ** (-6) * press - 0.0074 / press

    WsatP = ew * f

    return WsatP


def EstimateAltitude(P0, Pb, T0):
    """
    Estimate the altitude difference using a barometric formula.

    This function calculates the altitude based on the reference pressure, the
    observed barometric pressure, and the reference temperature. A standard
    relative humidity of 50% is assumed to compute the wet air density used in
    the calculation.

    Parameters
    ----------
    P0 : float
        Reference pressure in hPa.
    Pb : float
        Observed barometric pressure in hPa.
    T0 : float
        Reference temperature in Kelvin.

    Returns
    -------
    float
        Estimated altitude in meters.
    """

    Rho0 = Air_density(T0, P0, RH=50)[1]
    g = 9.8
    H = 100 * P0 / (Rho0 * g)
    Elevation = -H * np.log(Pb / P0)

    return Elevation


def calculate_altitude_hypsometric_simple(p0, p, t):
    """
    Calculate altitude using a simplified hypsometric equation.

    This function estimates altitude based on the hypsometric formula. It uses
    the reference pressure (p0), observed pressure (p), and temperature in
    Celsius (t) to compute the altitude.

    Parameters
    ----------
    p0 : float
        Reference pressure in hPa.
    p : float
        Observed pressure in hPa.
    t : float
        Temperature in Celsius.

    Returns
    -------
    float
        Estimated altitude in meters.
    """

    altitude = ((((p0 / p) * (1 / 5.257)) - 1)(t + 273.15)) / 0.0065

    return altitude


def calculate_altitude_for_row(row):
    """
    Calculate altitude from a data row of meteorological measurements.

    This function extracts pressure and temperature data from a dictionary-like
    object and computes the altitude using a simplified hypsometric formula.

    Parameters
    ----------
    row : dict-like
        Data structure containing the following keys:
            - "Pref": reference pressure in hPa.
            - "P_baro": observed barometric pressure in hPa.
            - "TEMP1": temperature in Celsius.

    Returns
    -------
    float
        Estimated altitude in meters.
    """

    p0 = row["Pref"]
    p = row["P_baro"]
    t = row["TEMP1"]

    # t_kelvin = t + 273.15  # Convert Celsius to Kelvin (not used?)

    altitude = calculate_altitude_hypsometric_simple(p0, p, t)

    return altitude

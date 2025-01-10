import numpy as np


def Air_density(T, P, RH=0):
    """Returns air density based on ideal gas law.
    Input:
        T in Kelvin
        P in hPa
        RH in %
    Output:
        dry air density in kg/m3
        wet air density in kg/m3
    """

    rho0 = 1.29
    P0 = 1013.25
    T0 = 273.15
    rho = rho0 * (P / P0) * (T0 / T)
    rhow = rho * (1 - 0.378 * waterpressure(RH, T, P) / P)

    return rho, rhow


def waterpressure(RH, T, P):
    """Returns water vapour pressure at given P and T.
    Input:
        RH in %
        T in Kelvin
        P in hPa
    Output:
        Partial pressure of water vapour in hPa
    """

    Pw = Watersatpress(P, T) * RH / 100.0
    return Pw


def Watersatpress(press, temp):
    """Calculates water saturation vapir pressure for moist air.
    The equation is based on WMO CIMO guide and should be valid from -45 to 60C
    ref link: https://www.wmo.int/pages/prog/www/IMOP/CIMO-Guide.html.
    input:
        P in hPa
        T in kelvin
    output:
        H2O saturation vapour pressure in hPa
    """

    temp = temp - 273.16  # conversion to centigrade temperature
    ew = 6.112 * np.exp(
        17.62 * temp / (243.12 + temp)
    )  # calculate saturation pressure for pure water vapour
    f = 1.0016 + 3.15 * 10 ** (-6) * press - 0.0074 / press

    WsatP = ew * f

    return WsatP


def EstimateAltitude(P0, Pb, T0):

    Rho0 = Air_density(T0, P0, RH=50)[1]
    g = 9.8
    H = 100 * P0 / (Rho0 * g)
    Elevation = -H * np.log(Pb / P0)

    return Elevation


def calculate_altitude_hypsometric_simple(p0, p, t):

    altitude = ((((p0 / p) * (1 / 5.257)) - 1)(t + 273.15)) / 0.0065

    return altitude


def calculate_altitude_for_row(row):

    p0 = row["Pref"]
    p = row["P_baro"]
    t = row["TEMP1"]

    # t_kelvin = t + 273.15  # Convert Celsius to Kelvin (not used?)

    altitude = calculate_altitude_hypsometric_simple(p0, p, t)

    return altitude

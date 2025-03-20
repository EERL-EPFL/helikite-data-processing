"""These functions are used for CO2 monitor corrections"""

import numpy as np


def stp_convert_dry(x, t, p1):
    """
    Convert a measurement to STP conditions for dry air.

    This function adjusts a measurement from actual conditions to standard
    temperature and pressure (STP) conditions for dry air. It follows the
    approach of `stp.convert.dry` by Roman Pohorsky.

    Parameters
    ----------
    x : float or array-like
        Measured value to be converted.
    t : float or array-like
        Temperature in °C.
    p1 : float or array-like
        Pressure in hPa.

    Returns
    -------
    float or array-like
        Measurement converted to STP conditions.

    Notes
    -----
    The conversion process involves:
      1. Converting the pressure from hPa to Pa.
      2. Converting the temperature from °C to Kelvin.
      3. Calculating the STP conversion factor and applying it to measurement.

    Examples
    --------
    >>> df_level0['FC_CO2_STP'] = stp_convert_dry(
    ...     df_level0['FC_CO2'],
    ...     df_level0['T ref'],
    ...     df_level0['FC_P corr']
    ... )
    """

    p = p1 * 100
    t = t + 273.15
    v_stp = (273.15 / t) * (p / 101315)
    x_stp = x / v_stp

    return x_stp


# def stp_convert_moist(x, t, p1, rh):
#     """
#     Pressure in hPa
#     Temperature in °C

#     TODO: Check function
#     """

#     p = p1 * 100
#     t = t + 273.15

#     if t > 273.15:
#         e_s = (
#             np.exp(34.494 - (4924.9 / ((t - 273.15) + 237.1)))
#             / ((t - 273.15) + 105) ** 1.57
#         )
#     else:
#         e_s = (
#             np.exp(43.494 - (6545.8 / ((t - 273.15) + 278)))
#             / ((t - 273.15) + 868) ** 2
#         )

#     e = (rh * e_s) / 100
#     t_v = t / (1 - (e / p) * (1 - 0.622))
#     v_stp = (273.15 / t_v) * (p / 1013.15)
#     x_stp = x / v_stp

#     return x_stp


def stp_moist_test(x, t, p1, rh):

    p = p1 * 100  # in Pa
    t = t + 273.15  # in K

    for i in range(len(t)):
        if t[i] > 273.15:
            e_s = (
                np.exp(34.494 - (4924.9 / ((t[i] - 273.15) + 237.1)))
                / ((t[i] - 273.15) + 105) ** 1.57
            )
        else:
            e_s = (
                np.exp(43.494 - (6545.8 / ((t[i] - 273.15) + 278)))
                / ((t[i] - 273.15) + 868) ** 2
            )

        e = (rh * e_s) / 100
        t_v = t / (1 - (e / p) * (1 - 0.622))
        v_stp = (273.15 / t_v) * (p / 101315)
        x_stp = x / v_stp

    return x_stp

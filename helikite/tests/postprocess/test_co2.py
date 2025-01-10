import numpy as np
from helikite.processing.post import co2

X = np.array([400.0, 410.0, 420.0])  # CO2 ppm values
T = np.array([15.0, 16.0, 17.0])  # Temperature in °C
P1 = np.array([1010.0, 1005.0, 1000.0])  # Pressure in hPa
RH = np.array([50.0, 60.0, 70.0])  # Relative humidity in %


def test_stp_convert_dry():
    expected_output = [423.28, 437.53, 452.01]
    results = []

    for x, t, p in zip(X, T, P1):
        result = co2.stp_convert_dry(x, t, p)
        results.append(result)

    np.testing.assert_allclose(results, expected_output, rtol=1e-2)


# def test_stp_convert_moist():
#     # TODO: Check function
#     expected_output = [4.25, 4.39, 4.54]
#     results = []

#     for x, t, p, rh in zip(X, T, P1, RH):
#         result = co2.stp_convert_moist(x, t, p, rh)
#         results.append(result)

#     np.testing.assert_allclose(results, expected_output, rtol=1e-2)


def test_stp_moist_test():
    expected_output = np.array([424.82, 439.46, 454.34])

    result = co2.stp_moist_test(X, T, P1, RH)
    np.testing.assert_allclose(result, expected_output, rtol=1e-2)

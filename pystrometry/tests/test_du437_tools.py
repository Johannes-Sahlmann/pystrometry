import numpy as np
from numpy.testing import assert_allclose

from ..utils.du437_tools import geometric_elements_with_uncertainties

def test_geometric_elements_with_uncertainties():
    """Test computation of uncertainties in geometric elements from ABFG+covariance matrix."""
    absolute_tolerance = 3e-5  # this relatively high tolerance was necessary because of rounding errors in the test inputs

    # first test case
    thiele_innes_parameters = np.array([0.5887957934067639, -0.783542139419248, -0.586541897649227, -0.4380942241437222])
    thiele_innes_parameters_errors  = np.array([0.14288834024020794, 0.10684726126075933, 0.13904500248228854, 0.1772629684907557])
    correlation_matrix_ti = np.array([[1.0, 0.977777963152516, 0.9843118342310587, -0.9705972306140258],
                                      [0.977777963152516, 1.0, 0.9854463019218139, -0.9847018411767419],
                                      [0.9843118342310587, 0.9854463019218139, 1.0, -0.984254484163135],
                                      [-0.9705972306140258, -0.9847018411767419, -0.984254484163135, 1.0]])

    ge1, ge1_err = geometric_elements_with_uncertainties(thiele_innes_parameters,
                                                         thiele_innes_parameters_errors,
                                                         correlation_matrix_ti, post_process=True)

    ge1_reference = np.array([0.980117, 3.146510, 2.218900, 2.414238])
    ge1_reference[1:] = np.rad2deg(ge1_reference[1:])
    ge1_err_reference = np.array([0.018120, 0.220922, 0.066129, 0.042308])
    ge1_err_reference[1:] = np.rad2deg(ge1_err_reference[1:])

    assert_allclose(ge1, ge1_reference, atol=absolute_tolerance)
    assert_allclose(ge1_err, ge1_err_reference, atol=absolute_tolerance)

    # second test case
    thiele_innes_parameters = np.array([-2.961570196500256, -2.422183719010767, 1.211375364690141, -1.4678757108790645])
    thiele_innes_parameters_errors  = np.array([.07764212692792022, 0.09451361935572963, 0.1961760927350047, 0.15440786748261082])
    correlation_matrix_ti = np.array([[1.0, -0.9135449715394732, 0.9211753621269316, 0.9399225380071006],
                                      [-0.9135449715394732, 1.0, -0.9033380261771987, -0.9188459402029718],
                                      [0.9211753621269316, -0.9033380261771987, 1.0, 0.9868011084831556],
                                      [0.9399225380071006, -0.9188459402029718, 0.9868011084831556, 1.0]])

    ge1, ge1_err = geometric_elements_with_uncertainties(thiele_innes_parameters,
                                                         thiele_innes_parameters_errors,
                                                         correlation_matrix_ti, post_process=True)

    ge1_reference = np.array([3.825959, 3.144507, 0.684094, 1.050160])
    ge1_reference[1:] = np.rad2deg(ge1_reference[1:])
    ge1_err_reference = np.array([0.024898, 0.067359, 0.014764, 0.007025])
    ge1_err_reference[1:] = np.rad2deg(ge1_err_reference[1:])

    assert_allclose(ge1, ge1_reference, atol=absolute_tolerance)
    assert_allclose(ge1_err, ge1_err_reference, atol=absolute_tolerance)


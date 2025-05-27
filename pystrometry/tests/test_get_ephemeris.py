from astropy.table import Table
from astropy.time import Time
from ..pystrometry import get_ephemeris

def test_get_ephemeris_valid_input():
    # Define valid test inputs
    center = 'g@399'
    target = '0'
    start_time = Time('2023-01-01T00:00:00')
    stop_time = Time('2024-01-10T00:00:00')
    step_size = '1d'

    # Call the function
    result = get_ephemeris(center=center, target=target, start_time=start_time, stop_time=stop_time, step_size=step_size)

    # Validate the result
    assert isinstance(result, Table), "Result should be an Astropy Table"
    assert len(result) > 0, "Result table should not be empty"
    assert 'X' in result.colnames, "Result table should contain 'X' column"
    assert 'Y' in result.colnames, "Result table should contain 'Y' column"
    assert 'Z' in result.colnames, "Result table should contain 'Z' column"

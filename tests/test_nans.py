import pytest
import numpy as np
from mahalanobis import Mahalanobis


@pytest.fixture(scope='module')
def nD_array():
    n_dim_array = np.arange(60, dtype=float)
    n_dim_array[[3, 5, 10, 11, 12, 25, 28, 38, 41, 44, 51, 55]] = np.nan
    n_dim_array = n_dim_array.reshape(-1, 3)

    yield n_dim_array

    del n_dim_array

def test_swap_nans_median(nD_array):
    test_instance = Mahalanobis(nD_array, 5)
    test_instance._Mahalanobis__swap_nans()
    assert test_instance.calibration_chunk.tolist(), [[0.0, 1.0, 2.0], [6.0, 4.0, 8.0], [6.0, 7.0, 8.0],
                                                      [9.0, 5.5, 8.0], [6.0, 13.0, 14.0]]
    del test_instance


def test_swap_nans_mean(nD_array):
    test_instance = Mahalanobis(nD_array, 5, nan_method='mean')
    test_instance._Mahalanobis__swap_nans()
    assert test_instance.calibration_chunk.tolist(), [[0.0, 1.0, 2.0], [5.0, 4.0, 8.0], [6.0, 7.0, 8.0],
                                                      [9.0, 6.25, 8.0], [5.0, 13.0, 14.0]]
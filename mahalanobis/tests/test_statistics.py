import pytest
import numpy as np
from mahalanobis import Mahalanobis
from mahalanobis import ShapeError


@pytest.fixture
def n_dim_array():
    nD_array = np.arange(60).reshape(-1, 3)
    yield nD_array
    del nD_array

@pytest.fixture
def test_instance_nD(n_dim_array):
    test_instance = Mahalanobis(n_dim_array, 5)
    yield test_instance
    del test_instance

def test_mean_multidimensional(test_instance_nD):
    assert test_instance_nD.calc_mean().tolist() == [6, 7, 8]

def test_set_new_valid_multidimensional_mean(test_instance_nD):
    test_instance_nD.set_mean(np.array([9, 10, 11]))
    assert test_instance_nD.get_mean().tolist() == [9, 10, 11]

def test_set_new_multidimensional_mean_invalid_nan(test_instance_nD):
    with pytest.raises(ValueError):
        test_instance_nD.set_mean(np.array([2, np.nan, 4]))

def test_set_new_multidimensional_mean_invalid_length(test_instance_nD):
    with pytest.raises(ShapeError):
        test_instance_nD.set_mean(np.array([2, 3, 4, 5]))

def test_set_new_multidimensional_mean_invalid_format(test_instance_nD):
    with pytest.raises(TypeError):
        test_instance_nD.set_mean([2, 3, 4, 5])


def test_cov_matrix_multidimensional(test_instance_nD):
    test_instance_nD.calc_mean()
    assert test_instance_nD.calc_cov_matrix().tolist() == [[90]*3]*3

def test_set_new_valid_multidimensional_cov_matrix(test_instance_nD):
    test_instance_nD.set_cov_matrix(np.array([[40]*3]*3))
    assert test_instance_nD.get_cov_matrix().tolist() == [[40]*3]*3

def test_set_new_multidimensional_cov_matrix_invalid_nan(test_instance_nD):
    with pytest.raises(ValueError):
        test_instance_nD.set_cov_matrix(np.array([[1, 2, 3], [4, 5, np.nan], [7, 8, 9]]))

def test_set_new_multidimensional_cov_matrix_invalid_squared(test_instance_nD):
    with pytest.raises(ShapeError):
        test_instance_nD.set_cov_matrix(np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]))

def test_set_new_multidimensional_cov_matrix_invalid_size(test_instance_nD):
    with pytest.raises(ShapeError):
        test_instance_nD.set_cov_matrix(np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]))

def test_set_new_multidimensional_cov_matrix_invalid_format(test_instance_nD):
    with pytest.raises(TypeError):
        test_instance_nD.set_cov_matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


@pytest.fixture
def one_dim_array():
    oneD_array = np.arange(30)
    yield oneD_array
    del oneD_array

@pytest.fixture
def test_instance_1D(one_dim_array):
    test_instance = Mahalanobis(one_dim_array, 9)
    yield test_instance
    del test_instance

def test_mean_onedimensional(test_instance_1D):
    assert test_instance_1D.calc_mean().tolist() == [4]

def test_set_new_valid_onedimensional_mean_float(test_instance_1D):
    test_instance_1D.set_mean(11)
    assert test_instance_1D.get_mean().tolist() == [11]

def test_set_new_valid_onedimensional_mean_array(test_instance_1D):
    test_instance_1D.set_mean(np.array([11]))
    assert test_instance_1D.get_mean().tolist() == [11]

def test_set_new_onedimensional_mean_invalid_nan(test_instance_1D):
    with pytest.raises(TypeError):
        test_instance_1D.set_mean(np.nan)

def test_set_new_onedimensional_mean_invalid_length(test_instance_1D):
    with pytest.raises(ValueError):
        test_instance_1D.set_mean(np.array([3, 4]))

def test_set_new_onedimensional_mean_invalid_format(test_instance_1D):
    with pytest.raises(TypeError):
        test_instance_1D.set_mean([2])


def test_cov_matrix_onedimensional(test_instance_1D):
    # must equal the regular variance
    test_instance_1D.calc_mean()
    assert test_instance_1D.calc_cov_matrix().tolist() == [[60]]

def test_set_new_valid_onedimensional_cov_matrix(test_instance_1D):
    test_instance_1D.set_cov_matrix(np.array([[2]]))
    assert test_instance_1D.get_cov_matrix().tolist() == [[2]]

def test_set_new_onedimensional_cov_matrix_invalid_nan(test_instance_1D):
    with pytest.raises(ValueError):
        test_instance_1D.set_cov_matrix(np.array([[np.nan]]))

def test_set_new_onedimensional_cov_matrix_invalid_size(test_instance_1D):
    with pytest.raises(ValueError):
        test_instance_1D.set_cov_matrix(np.array([[1],[2]]))


def test_set_new_onedimensional_cov_matrix_invalid_format(test_instance_1D):
    with pytest.raises(TypeError):
        test_instance_1D.set_cov_matrix([[1],[2]])

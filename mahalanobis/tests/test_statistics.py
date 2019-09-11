import pytest
import numpy as np

from mahalanobis import Mahalanobis, ShapeError


class oneD_environment:
    @pytest.fixture
    def one_dim_array(self):
        oneD_array = np.arange(30)
        yield oneD_array
        del oneD_array

    @pytest.fixture
    def test_instance_1D(self, one_dim_array):
        test_instance = Mahalanobis(one_dim_array, 9)
        yield test_instance
        del test_instance


class Test_1D_mean(oneD_environment):
    def test_mean_onedimensional(self, test_instance_1D):
        assert test_instance_1D.mean.tolist() == [4]

    def test_set_new_valid_onedimensional_mean_float(self, test_instance_1D):
        test_instance_1D.mean = 11
        assert test_instance_1D.mean.tolist() == [11]

    def test_set_new_valid_onedimensional_mean_array(self, test_instance_1D):
        test_instance_1D.mean = np.array([11])
        assert test_instance_1D.mean.tolist() == [11]

    def test_set_new_onedimensional_mean_invalid_nan(self, test_instance_1D):
        with pytest.raises(TypeError):
            test_instance_1D.mean = np.nan

    def test_set_new_onedimensional_mean_invalid_length(self, test_instance_1D):
        with pytest.raises(ValueError):
            test_instance_1D.mean = np.array([3, 4])

    def test_set_new_onedimensional_mean_invalid_format(self, test_instance_1D):
        with pytest.raises(TypeError):
            test_instance_1D.mean = [2]


class Test_1D_cov_matrix(oneD_environment):
    def test_cov_matrix_onedimensional(self, test_instance_1D):
        # must equal the regular variance
        assert test_instance_1D.cov_matrix.tolist() == [[60]]

    def test_set_new_valid_onedimensional_cov_matrix(self, test_instance_1D):
        test_instance_1D.cov_matrix = np.array([[2]])
        assert test_instance_1D.cov_matrix.tolist() == [[2]]

    def test_set_new_onedimensional_cov_matrix_invalid_nan(self, test_instance_1D):
        with pytest.raises(ValueError):
            test_instance_1D.cov_matrix = np.array([[np.nan]])

    def test_set_new_onedimensional_cov_matrix_invalid_size(self, test_instance_1D):
        with pytest.raises(ValueError):
            test_instance_1D.cov_matrix = np.array([[1], [2]])

    def test_set_new_onedimensional_cov_matrix_invalid_format(self, test_instance_1D):
        with pytest.raises(TypeError):
            test_instance_1D.cov_matrix = [[1], [2]]

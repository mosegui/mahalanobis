# -*- coding: utf-8 -*-
"""
@author: mosegui

The inbound array must be structured in a way the array rows are the different observations of the phenomenon
to process, whilst the columns represent the different dimensions of the process, very much like in input
arrays used in SciKit-Learn. Similarly, for one-dimensional processes, the input array must be a column vector.

Upon class instantiation, potential NaNs have to be removed from the segment of the input array that will
be used for the calibration of the Mahalanobis object (since the covariance matrix cannot be inverted if it
contains a NaN). For this reason:
    - One-dimensional arrays:
        - If NaNs are present, they are substituted with the chosen statistical indicator (mean and median supported)
        - If the array consists only of NaNs, the class raises an Exception
    - Multi-dimensionl arrays:
        - If NaNs are present, they are substituted with the chosen statistical indicator (mean and median supported)
          of the column (process feature) in which they are located.
        - Array columns consisting only of NaNs are removed prior to the calibration, thereby reducing the
          dimensionality of the problem.

Once the input (sub)array used for calibration is free on NaNs, the mean vector (the mean value of each feature) and
the covariances matrix are calculated. This process of mean and covariances calculation is referred to as 'calibration'
throughout this script. Subsequently, the Mahalanobis distances are automatically calculated for each feature of the
inbound array, stored in the instance variable 'distances'.

The Mahalanobis object has two properties 'mean' and 'cov_matrix' that allow the user to adjust their values for model
behavior exploration, provided the new feature arrays have the same dimensions as those used in the original calibration
of the Mahalanobis object. For exploring an object with different dimensions, a new class instance must be created.

Given a Mahalanobis object instance with a successful calibration, it is also possible to calculate the Mahalanobis
distances of external arrays benchmarked to that calibration, provided they match the calibration dimensions.
"""
import logging

import numpy as np

from .better_abc import abstract_attribute


class MahalanobisBenchmark:
    """This parent class attributes commonly necessary in Mahalanobis calculations for both one- and multi-dimensional
    data sets (arrays of order 1 or 2).
    """
    @abstract_attribute
    def array(self):
        pass

    def __init__(self, logger):
        self._logger = logger

    def __call__(self):
        self.distances = self._calculate_dists(self.array)

    def _select_calibration_subarray(self):
        """Sets the array that will be used for the Mahalanobis object calibration in an
        instance variable and determines the dimensionality of the problem.

        Raises
        ------
        ValueError : when the passed argument is not a list, an array, or convertible to an integer
        """
        if isinstance(self.calib_entries, int):
            self.calib_entries = int(self.calib_entries)
            self.calibration_chunk = self.array[:self.calib_entries]
        elif isinstance(self.calib_entries, list) or isinstance(self.calib_entries, np.ndarray):
            self.calib_entries = np.array(self.calib_entries)
            self.calibration_chunk = self.array[self.calib_entries]
        else:
            msg = 'Wrong format in calib_entries argument. Must be convertible to integer, list or np.array'
            self._logger.error(msg)
            raise ValueError(msg)

        self.dimensionality = self.array.shape[1]

    def _calc_nan_ratio(self):
        """Calculates the ration of NaN is the calibration segment of the input array.
        Either for the entire array for 1D problems or per column for multi-dimensional
        problems.
        """
        number_of_nans = np.count_nonzero(np.isnan(self.calibration_chunk), axis=0)

        if isinstance(self.calib_entries, int):
            self.nans_ratio = number_of_nans / self.calib_entries
        elif isinstance(self.calib_entries, np.ndarray):
            self.nans_ratio = number_of_nans / len(self.calib_entries)

        try:
            self._logger.info(f'calibration set contains {self.nans_ratio}% NaNs')
        except:
            self._logger.info('calibration set does not contain NaNs')

    def _calc_cov_matrix(self):
        """Computes the covariance matrix from the calibration set of the feature array"""
        means_array = np.tile(self._calibration_mean, (self.calibration_chunk.shape[0], 1))

        variations_array = self.calibration_chunk - means_array

        self._cov_matrix = (np.dot(variations_array.T, variations_array)) / variations_array.shape[0]

    def _invert_cov_matrix(self, cov_matrix):
        """Returns the inverse of the covariance matrix"""
        try:
            inv_cov_matrix = np.linalg.inv(cov_matrix)
            # TODO: look into possible pseudo-inverses for the case of non-invertible covariance matrices.
            # https://en.wikipedia.org/wiki/Generalized_inverse#Reflexive_generalized_inverse
        except np.linalg.LinAlgError as e:
            msg = 'Mahalanobis distances cannot be calculated with singular covariance matrix'
            self._logger.error(msg)
            self._logger.error(e, exc_info=True)
            raise e
        return inv_cov_matrix

    def _calculate_dists(self, input_array):  # TODO: Write unit tests
        """Uses the calculated mean and covariance matrix for calculating the Mahalanobis distances
        for each observation in the inbound array.

        Parameters
        ----------
        input_array : np.array
            array with the observations to be used in the calculation of the Mahalanobis distances

        Returns
        -------
        distances_array : np.array
            one-dimensional array with the Mahalanobis distances corresponding to each input observation

        Raises
        ------
        SingularError : if the covariance matrix is not invertible
        ShapeError : if the inbound array does not match the dimensionality of the problem
        """
        self._inv_cov_matrix = self._invert_cov_matrix(self._cov_matrix)

        if input_array.shape[1] != self.dimensionality:
            msg = 'Dimensions of passed array do not match calibration dimensions of Mahalanobis object'
            self._logger.error(msg)
            raise ShapeError(msg)

        diff_array = input_array - np.tile(self._calibration_mean, (input_array.shape[0], 1))

        distances_array = np.array(
            list(map(lambda difference: np.sqrt(np.dot(np.dot(difference, self._inv_cov_matrix), difference.reshape(-1, 1))), diff_array)))

        return distances_array

    def _check_shape(self, new_array, current_array):
        """Makes a shape check upon setting of a new mean vector or a new covariance matrix

        Parameters
        ----------
        new_array : np.array
            new mean vector or covariance matrix that the user wants to replace the current with
        current_array : np.array
            current array to be replaced

        Returns
        -------
        new_array or np.array([new_array]) : np.array
            inbound array si returned is it passes the shape check

        Raises
        ------
        ShapeError : if the inbound and current arrays do not match in shape
        ValueError : if the inbound array contains non-numerical characters
        TypeError : if the input mean or covariance matrix is not an array or numerical
        """
        if isinstance(new_array, np.ndarray):
            try:
                assert new_array.shape == current_array.shape
            except AssertionError:
                msg = 'new array and current array shapes do not match'
                self._logger.exception(msg, exc_info=True)
                raise SingularError(msg)

            if any([item in str(new_array.dtype) for item in ['float', 'int']]) and not np.isnan(new_array).any():
                return new_array
            else:
                msg = 'array contains non-numeric characters'
                self._logger.error(msg)
                raise ValueError(msg)

        elif isinstance(float(new_array), float) and not np.isnan(new_array):
            if isinstance(float(current_array), float):
                return np.array([new_array])
            else:
                msg = f'array has incorrect shape. Must have shape ({current_array.shape})'
                self._logger.error(msg)
                raise ShapeError(msg)

        else:
            msg = 'array must be float (not NaN) or numpy array'
            self._logger.error(msg)
            raise TypeError(msg)

    @property
    def mean(self):
        """Returns the value currently stored in self._calibration_mean

        Returns
        -------
        self._calibration_mean : float or np.array
            uni- or multivariate vector with feature means of reference set
        """
        return self._calibration_mean

    @mean.setter
    def mean(self, new_mean_array):
        """Sets the passed array as the new mean for the calculations

        Parameters
        ----------
        new_mean_array : np.array or float
            array to substitute the current mean
        """
        self._calibration_mean = self._check_shape(new_mean_array, self._calibration_mean)

    @property
    def cov_matrix(self):
        """Returns the value currently stored in self._cov_matrix

        Returns
        -------
        self._cov_matrix : np.array
            covariance matrix of reference set
        """
        return self._cov_matrix

    @cov_matrix.setter
    def cov_matrix(self, new_cov_matrix):
        """Sets the passed array as the new covariance matrix for the calculations

        Parameters
        ----------
        new_cov_matrix : np.array or float
            array to substitute the current covariance matrix
        """
        try:
            # this clause fails if 'new_cov_matrix' is neither a scalar nor convertible to a scalar (one-item array/list/tuple)
            self._cov_matrix = self._check_shape(new_cov_matrix, self._cov_matrix).reshape(1, 1)
        except:
            self._cov_matrix = self._check_shape(new_cov_matrix, self._cov_matrix)


class Mahalanobis1D(MahalanobisBenchmark):
    """Extends MahalanobisBenchmark by providing methods used for replacing NaNs in one-dimensional
    input arrays and flow control for Mahalanobis distance calculations.
    """
    def __init__(self, array, calib_entries, nan_method='median'):
        """Replaces the NaNs in the array considered for calibration of the Mahalanobis object

        Parameters
        ----------
        array : np.array
            the uni/multivariate array containing the full data set
        calib_entries : int, float or np.array
            array index up to which the data is used to calculate the Mahalanobis object calibration
        nan_method : str
            aggregation method to replace nans in array calibration chkunk. must be {'median', 'mean'}
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        super().__init__(self._logger)

        if nan_method.lower() == 'median':
            self.nan_filler = np.median
        elif nan_method.lower() == 'mean':
            self.nan_filler = np.mean
        else:
            raise NotImplementedError('NaN bypassing method needs to be mean or median')

        self._logger = logging.getLogger(self.__class__.__name__)

        self.array = array
        self.calib_entries = calib_entries

        self._select_calibration_subarray()
        self._replace_nans()
        self._calibration_mean = self.calibration_chunk.mean(axis=0)  # mean to which array entries will be compared
        self._calc_cov_matrix()

    def _get_nan_substitutes(self, array_with_nans):
        """According to the passed method it retrieves the value to substitute the array NaNs.

        Parameters
        ---------
        array_with_nans : np.array
            array containing NaNs.

        Returns
        -------
        substitute : float
            float (for 1D arrays) being the substituting value
        """
        return self.nan_filler(array_with_nans[~np.isnan(array_with_nans)])

    def _substitute_nans(self, array_to_clean, nan_substitutes):
        """Substitutes the NaNs in the input array with the value in nan_substitutes.

        Parameters
        ----------
        array_to_clean : np.array
            input array containing NaNs
        nan_substitutes : float or list of floats
            value to substitute in the NaNs of the array_to_clean

        Returns
        -------
        array_to_clean : np.array
            clean array with substituted NaNs
        """
        nan_positions = np.where(np.isnan(array_to_clean))[0]
        array_to_clean[nan_positions] = nan_substitutes

        return array_to_clean

    def _replace_nans(self):
        """ Takes the calibration array and substitutes the NaNs by the array's mean or median value."""
        self._calc_nan_ratio()

        self.nan_columns = list(np.where(self.nans_ratio == 1)[0])

        if self.nan_columns == 1:
            raise ValueError('One-dimensional input array is exclusively populated with NaNs')

        nan_substitutes = self._get_nan_substitutes(self.calibration_chunk)

        self.calibration_chunk = self._substitute_nans(self.calibration_chunk, nan_substitutes)

        self._logger.info('feature NaNs were substituted with feature {}'.format(self.nan_filler.__name__))

    def calc_distances(self, new_input):
        """Checks that the input array on which Mahalanobis distances must be calculated is not empty. If the
         new input is only a scalar, it casts it to a zero-dimensional array. Subsequently calls the
         Mahalanobis distances calculation.

        Parameters
        ----------
        new_input : np.array or float
            array with the observations to be used in the calculation of the Mahalanobis distances

        Returns
        -------
        np.array : one-dimensional array with the Mahalanobis distances corresponding to each input observation

        Raises
        ------
        ValueError : if the inbound array is empty
        """
        if new_input.size == 0.:
            msg = 'empty inbound array'
            self._logger.error(msg)
            raise ValueError(msg)
        else:
            if isinstance(float(new_input), float) and not np.isnan(new_input):
                new_input = np.array([new_input])

        return self._calculate_dists(new_input)


class MahalanobisND(MahalanobisBenchmark):
    """Extends MahalanobisBenchmark by providing methods used for replacing NaNs in multi-dimensional
    input arrays and flow control for Mahalanobis distance calculations.
    """
    def __init__(self, array, calib_entries, nan_method='median'):
        """Replaces the NaNs in the array considered for calibration of the Mahalanobis object

        Parameters
        ----------
        array : np.array
            the multivariate array containing the full data set
        calib_entries : int, float or np.array
            array index up to which the data is used to calculate the Mahalanobis object calibration
        nan_method : str
            aggregation method to replace nans in array calibration chkunk. must be {'median', 'mean'}
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        super().__init__(self._logger)

        if nan_method.lower() == 'median':
            self.nan_filler = np.median
        elif nan_method.lower() == 'mean':
            self.nan_filler = np.mean
        else:
            raise NotImplementedError('NaN bypassing method needs to be mean or median')

        self.array = array
        self.calib_entries = calib_entries

        self._select_calibration_subarray()
        self._replace_nans()
        self._calibration_mean = self.calibration_chunk.mean(axis=0)  # mean to which array entries will be compared
        self._calc_cov_matrix()

    def _reduce_multidimensional_array_dimension(self):
        """For a multi-dimensional array (multi-column), it removes all the array columns
        the calibration of which are exclusively filled with NaNs
        """
        self.array = np.delete(self.array, self.nan_columns, axis=1)

        self._select_calibration_subarray()

    def _get_nan_substitutes(self, array_with_nans):
        """According to the passed method it retrieves the value or list of values
        to substitute the array NaNs.

        Parameters
        ---------
        array_with_nans : np.array
            array containing NaNs.

        Returns
        -------
        substitute : list
            list of floats (nD arrays) with the substituting value (one element per column)
        """
        return [self.nan_filler(column[~np.isnan(column)]) for column in array_with_nans.T]

    def _substitute_nans(self, array_to_clean, nan_substitutes):
        """Substitutes the NaNs in the input array with the value in nan_substitutes.
        nan_substitutes is a list of floats and the NaNs are substituted column-wise.

        Parameters
        ----------
        array_to_clean : np.array
            input array containing NaNs
        nan_substitutes : list of floats
            value or list of values to substitute in the NaNs of the corresponding features

        Returns
        -------
        array_to_clean : np.array
            clean array with substituted NaNs
        """
        for column, column_mean in zip(array_to_clean.T, nan_substitutes):
            nan_positions = np.where(np.isnan(column))[0]
            column[nan_positions] = column_mean

        return array_to_clean

    def _replace_nans(self):
        """ Takes the calibration array and substitutes the NaNs by the mean
        or median value of the corresponding column (feature).
        """
        self._calc_nan_ratio()

        self.nan_columns = list(np.where(self.nans_ratio == 1)[0])

        self._reduce_multidimensional_array_dimension()

        nan_substitutes = self._get_nan_substitutes(self.calibration_chunk)

        self.calibration_chunk = self._substitute_nans(self.calibration_chunk, nan_substitutes)

        self._logger.info(f'feature NaNs were substituted with feature {self.nan_filler.__name__}')

    def calc_distances(self, new_input):
        """ Checks that the input array on which Mahalanobis distances must be calculated is not empty, and
         removes the array columns that in the calibration were disregarded due to sole presence of NaNs. Subsequently
         calls the Mahalanobis distances calculation.

        Parameters
        ----------
        new_input : np.array or float
            array with the observations to be used in the calculation of the Mahalanobis distances

        Returns
        -------
        np.array : one-dimensional array with the Mahalanobis distances corresponding to each input observation

        Raises
        ------
        ValueError : if the inbound array is empty
        """

        if new_input.size == 0.:
            msg = 'empty inbound array'
            self._logger.error(msg)
            raise ValueError(msg)
        else:
            new_input = np.delete(new_input, self.nan_columns, axis=1)

        return self._calculate_dists(new_input)


def Mahalanobis(input_array, calib_rows, nan_subst_method='median'):
    """This is a wrapper function that provides dynamic Mahalanobis dynamic object instantiation.
    It returns either a 'Mahalanobis1D' or 'MahalanobisND' instance, depending on the amount of
    dimensions that the input problem has. The main difference between the two cases is how the
    program deals with NaNs depending on whether the arrays has one or more columns.

    Parameters
    ----------
    input_array : np.array
        the uni/multivariate array containing the full data set
    calib_rows : int, float or np.array
        array index up to which the data is used to calculate the base of the calculation
    nan_subst_method : str
        Chooses the column 'mean' or 'median' as value replacing the corresponding NaNs

    Returns
    -------
    'Mahalanobis1D' or 'MahalanobisND' object instance
    """
    if input_array.shape[1] == 1:
        calculator = Mahalanobis1D
    elif input_array.shape[1] > 1:
        calculator = MahalanobisND
    else:
        raise ShapeError('Empty input array')

    return calculator(input_array, calib_rows, nan_method=nan_subst_method)


class ShapeError(ValueError):
    """Used for input arrays with wrong dimensions or for non-squared covariance matrices"""
    pass

class SingularError(ValueError):
    """Raises when covariance matrix is not invertible"""
    pass

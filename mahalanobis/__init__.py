# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 17:30:56 2018

@author: mosegui
"""

import numpy as np
import logging


class Mahalanobis:
    def __init__(self, array, calib_entries, nan_method='median'):
        """Instances a Mahalanobis calculator to act over a one or multidimensional numpy array

        Parameters
        ----------
        array : np.array
            the uni/multivariate array containing the full data set
        calib_entries : int, float or np.array
            array index up to which the data is used to calculate the base of the calculation
        nan_method : str
            Chooses the column 'mean' or 'median' as value replacing the corresponding NaNs
        """
        self._logger = logging.getLogger(self.__class__.__name__)

        self.array = array
        try:
            self.calib_entries = int(calib_entries)
        except:
            self.calib_entries = calib_entries

        if len(self.array.shape) > 1:
            self.dimensionality = self.array.shape[1]
        else:
            self.array = self.array.reshape(-1,1)
            self.dimensionality = 1

        if isinstance(self.calib_entries, int):
            self.calibration_chunk = self.array[:self.calib_entries]
        elif isinstance(self.calib_entries, list) or isinstance(self.calib_entries, np.ndarray):
            self.calibration_chunk = self.array[self.calib_entries]
        else:
            self._logger.error('Wrong format in calib_entries argument. Must be {float, int, list, np.array}')
            raise ValueError('Wrong format in calib_entries argument. Must be {float, int, list, np.array}')

        self.__calibration_mean = None
        self.__cov_matrix = None
        self.nan_method = nan_method
        self.distances = None

    def __reduce_multidimensional_array_dimension(self):
        """For a multi-dimensional array (multi-column), it removes all the array columns
        the calibration of which are exclusively filled with NaNs
        """
        self.array = np.delete(self.array, self.nan_columns, axis=1)

        if isinstance(self.calib_entries, int):
            self.calibration_chunk = self.array[:self.calib_entries]
        elif isinstance(self.calib_entries, list) or isinstance(self.calib_entries, np.ndarray):
            self.calibration_chunk = self.array[self.calib_entries]

        self.dimensionality = self.array.shape[1]


    def __swap_nans(self):
        """ takes the calibration array and substitutes the NaNs by the mean
        or median value of the corresponding column (feature)

        Returns
        -------
        substitute : float or list
            float (for 1D arrays) or list of floats (nD arrays) with the length of the data features
        """

        def _get_nan_substitutes(array_with_nans, substitution_method):
            """According to the passed method it retrieves the value or list of values
            to subtitue the array NaNs
            """
            if substitution_method == 'median':
                if len(self.array.shape) == 1:
                    substitute = np.median(array_with_nans[~np.isnan(array_with_nans)])
                else:
                    substitute = [np.median(column[~np.isnan(column)]) for column in array_with_nans.T]
            elif substitution_method == 'mean':
                if len(self.array.shape) == 1:
                    substitute = np.mean(array_with_nans[~np.isnan(array_with_nans)])
                else:
                    substitute = [np.mean(column[~np.isnan(column)]) for column in array_with_nans.T]
            else:
                raise ValueError('NaN bypassing method needs to be MEAN or MEDIAN')

            return substitute

        def _substitute_nans(array_to_clean, nan_substitutes):
            """Substitutes the NaNs in the input array with the value in nan_subtitutes.
            In the case of nD-array, nan_substitutes is a list of floats and the NaNs are
            substituted column-wise

            Parameters
            ----------
            array_to_clean : np.array
                input array containing NaNs
            nan_substitutes : float or list of floats
                value or list of values to substitute in the NaNs of the correspondign features

            Returns
            -------
            array_to_clean : np.array
                clean array with substituted NaNs
            """
            if len(self.array.shape) == 1:
                nan_positions = np.where(np.isnan(array_to_clean))[0]
                array_to_clean[nan_positions] = nan_substitutes
            else:
                for column, column_mean in zip(array_to_clean.T, nan_substitutes):
                    nan_positions = np.where(np.isnan(column))[0]
                    column[nan_positions] = column_mean

            return array_to_clean

        number_of_nans = np.count_nonzero(np.isnan(self.calibration_chunk), axis=0)

        if isinstance(self.calib_entries, int):
            nans_ratio = number_of_nans/self.calib_entries
        elif isinstance(self.calib_entries, list) or isinstance(self.calib_entries, np.ndarray):
            nans_ratio = number_of_nans/len(self.calib_entries)

        try:
            if len(self.array.shape) > 1:
                nans_min = 100 * np.min(nans_ratio[np.nonzero(nans_ratio)])
                nans_max = 100 * np.max(nans_ratio[np.nonzero(nans_ratio)])
                self.logger.info(f'calibration set contains {nans_min}% to {nans_max}% NaNs')
            else:
                self.logger.info(f'calibration set contains {nans_ratio}% NaNs')
        except:
            self._logger.info('calibration set does not contain NaNs')

        self.nan_columns = list(np.where(nans_ratio == 1)[0])

        if len(self.array.shape) == 1:
            if self.nan_columns == 1:
                raise ValueError('One-dimensional input array is fully with NaNs')
        else:
            self.__reduce_multidimensional_array_dimension()

        nan_substitutes = _get_nan_substitutes(self.calibration_chunk, substitution_method=self.nan_method)

        self.calibration_chunk = _substitute_nans(self.calibration_chunk, nan_substitutes)

        self._logger.info('feature NaNs were substituted with feature {}'.format(self.nan_method))

    def calc_mean(self):
        """Returns the mean of the calibration range of the feature vector

        Returns
        -------
        __calibration_mean : float or np.array
            uni- or multivariate vector with feature means of reference set
        """
        self.__swap_nans()

        self.__calibration_mean = self.calibration_chunk.mean(axis=0)
        return self.__calibration_mean


    def get_mean(self):
        """Returns the value currently stored in self.__calibration_mean

        Returns
        -------
        __calibration_mean : float or np.array
            uni- or multivariate vector with feature means of reference set
        """
        return self.__calibration_mean


    def set_mean(self, new_mean_array):
        """Sets the passed array as the new mean for the calculations

        Parameters
        ----------
        new_mean_array : np.array or float
            array to substitute the current mean
        """
        if isinstance(new_mean_array, np.ndarray):
            try:
                assert len(new_mean_array.shape) == 1
            except AssertionError:
                self._logger.exception('mean array can be at most of order one')
                raise ShapeError('mean array can be at most of order one')

            if (all(isinstance(float(x), float) for x in new_mean_array)) and not any(np.isnan(new_mean_array)):
                if len(self.array.shape) > 1 and self.array.shape[1] == len(new_mean_array):
                    self.__calibration_mean = new_mean_array
                elif len(self.array.shape) == 1 and len(new_mean_array) == 1:
                    self.__calibration_mean = new_mean_array
                else:
                    try:
                        self._logger.error(f'array has incorrect shape. Must have shape ({self.array.shape[1]},)')
                        raise ShapeError(f'array has incorrect shape. Must have ({self.array.shape[1]},)')
                    except IndexError:
                        self._logger.error(f'array has incorrect shape. Must contain 1 element')
                        raise ShapeError(f'array has incorrect shape. Must contain 1 element')
            else:
                self._logger.error(f'array contains non-numeric characters')
                raise ValueError(f'array contains non-numeric characters')

        elif isinstance(float(new_mean_array), float) and not np.isnan(new_mean_array):
            if self.array.shape[1] == 1:
                self.__calibration_mean = np.array([new_mean_array])
            else:
                self._logger.error(f'array has incorrect shape. Must have shape ({self.array.shape[1]},)')
                raise ShapeError(f'array has incorrect shape. Must have ({self.array.shape[1]},)')

        else:
            self._logger.error(f'array must be float (not NaN) or numpy array')
            raise TypeError(f'array must be float (not NaN) or numpy array')


    def calc_cov_matrix(self):
        """Computes the covariance matrix from the calibration set of the feature array

        Returns
        -------
        __cov_matrix : np.array
            covariance matrix of reference set
        """
        self.__swap_nans()

        if self.__calibration_mean is not None:
            means_array = np.tile(self.__calibration_mean, (self.calibration_chunk.shape[0], 1))
        else:
            self._logger.error('Cannot calculate covariance matrix. Must calculate mean first')
            raise ValueError('Cannot calculate covariance matrix. Must calculate mean first')

        variations_array = self.calibration_chunk - means_array

        self.__cov_matrix = np.dot(variations_array.T, variations_array)

        return self.__cov_matrix


    def get_cov_matrix(self):
        """Returns the value currently stored in self.__cov_matrix

        Returns
        -------
        __cov_matrix : np.array
            covariance matrix of reference set
        """
        return self.__cov_matrix


    def set_cov_matrix(self, new_cov_matrix):
        """Sets the passed array as the new covariance matrix for the calculations

        Parameters
        ----------
        new_cov_matrix : np.array or float
            array to substitute the current covariance matrix
        """
        if isinstance(new_cov_matrix, np.ndarray):
            try:
                assert new_cov_matrix.shape[0] == new_cov_matrix.shape[1]
            except AssertionError:
                self._logger.exception('passed covariance matrix is not squared')
                raise ShapeError('passed covariance matrix is not squared')

            if all(isinstance(float(x), float) for x in new_cov_matrix.flatten()) and not any(np.isnan(new_cov_matrix.flatten())):
                if len(self.array.shape) > 1 and self.array.shape[1] == new_cov_matrix.shape[1]:
                    self.__cov_matrix = new_cov_matrix
                elif len(self.array.shape) == 1 and new_cov_matrix.shape[1] == 1:
                    self.__cov_matrix = new_cov_matrix
                else:
                    try:
                        self._logger.error(f'array has incorrect shape. Must have shape ({self.array.shape[1]},{self.array.shape[1]})')
                        raise ShapeError(f'array has incorrect shape. Must have shape ({self.array.shape[1]},{self.array.shape[1]})')
                    except IndexError:
                        self._logger.error(f'array has incorrect shape. Must contain 1 element')
                        raise ShapeError(f'array has incorrect shape. Must contain 1 element')
            else:
                self._logger.error(f'array contains non-numeric characters')
                raise ValueError(f'array contains non-numeric characters')

        elif isinstance(float(new_cov_matrix), float) and not np.isnan(new_cov_matrix):
            if self.array.shape[1] == 1:
                self.__cov_matrix = np.array([[new_cov_matrix]])
            else:
                self._logger.error(f'array has incorrect shape. Must have shape ({self.array.shape[1]},{self.array.shape[1]})')
                raise ShapeError(f'array has incorrect shape. Must have shape ({self.array.shape[1]},{self.array.shape[1]})')

        else:
            self._logger.error(f'wrong array data format. Must be float or numpy array')
            raise TypeError(f'wrong array data format. Must be float or numpy array')


    def __calculate_dists(self, input_array):   # TODO: Write unit tests
        """Uses the calculated mean and covariance matrix for calculating the Mahalanobis distances
        for each observation in the inbound array

        Parameters
        ----------
        input_array : np.array
            array with the observations to be used in the calculation of the Mahalanobis distances

        Returns
        -------
        distances_array : np.array
            one-dimensional array with the Mahalanobis distances correspondign to each input observation
        """
        mahalanobis_list = []

        if np.linalg.det(self.__cov_matrix) != 0.:
            self.__inv_cov_matrix = np.linalg.inv(self.__cov_matrix)
        else:
            self._logger.error('Singular covariance matrix not invertible')
            raise SingularError('Mahalanobis distances cannot be calculated with singular covariance matrix')

        # TODO: improve/avoid loop function performance
        for observation in input_array:
            # is this for the case that an externally passed array has more columns
            # than the one used for calibration
            datapoint = observation[:self.__calibration_mean.shape[0]]

            diff_array = datapoint - self.__calibration_mean

            observation_distances = np.dot(np.dot(diff_array, self.__inv_cov_matrix), diff_array.T)
            mahalanobis_list.append(observation_distances)
            distances_array = np.array(mahalanobis_list)

        return distances_array


    def calc_dists_set(self):
        """Returns the list of Mahalanobis distances corresponding to the input array

        Returns
        -------
        mahalanobis_distances : np.array
            one-dimensional array with the Mahalanobis distances corresponding to each input observation
        """
        if self.__calibration_mean is None:
            self.calc_mean()

        if self.__cov_matrix is None:
            self.calc_cov_matrix()

        self.distances = self.__calculate_dists(self.array)

        return self.distances


    def get_distances(self):
        """Returns the value currently stored in self.distances

        Returns
        -------
        distances : np.array
            Mahalanobis distances corresponding to the input data set
        """
        return self.distances


    def calc_dists_array(self, new_input):
        """Calculates the Mahalanobis distances for the passed feature array, provided that a
        covariance matrix and a mean array is already calculated

        Parameters
        ----------
        new_input : np.array or float
            array with the observations to be used in the calculation of the Mahalanobis distances

        Returns
        -------
        mahalanobis_distances : np.array
            one-dimensional array with the Mahalanobis distances corresponding to each input observation
        """
        try:
            if isinstance(float(new_input), float) and not np.isnan(new_input):
                new_input = np.array([new_input])

        except:
            if self.__cov_matrix is None or self.__calibration_mean is None:
                self._logger.error('Mahalanobis distance cannot be calculated. Reference mean ' +
                                   'and/or covariance matrix are not yet defined. Call functions ' +
                                   '"calc_mean" and "calc_cov_matrix" after feeding with a ' +
                                   'reference data set')
                raise ValueError('Mahalanobis distances cannot be calculated. Fingerprinting mean' +
                                 'or cavariance matrix are not yet defined')
            else:
                if len(new_input.shape) == 1:
                    new_input = np.array([np.delete(new_input, self.nan_columns)])
                elif len(new_input.shape) == 0:
                    self._logger.error('empty inbound array')
                else:
                    new_input = np.delete(new_input, self.nan_columns, axis=1)

                mahalanobis_distances = self.__calculate_dists(new_input)

                return mahalanobis_distances


class ShapeError(ValueError):
    pass

class SingularError(ValueError):
    pass
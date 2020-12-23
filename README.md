
![CircleCI](https://circleci.com/gh/mosegui/mahalanobis.svg?style=shield)
https://img.shields.io/github/stars/mosegui/mahalanobis?style=plastic

# Mahalanobis

This package can be used for calculating distances between data points and a reference distribution according to the Mahalanobis distance algorithm. The algorithm can be seen as a generalization of the euclidean distance, but normalizing the calculated distance with the variance of the points distribution used as fingerprint.

## Getting Started

### Description

The Mahalanobis object allows for calculation of distances (using the Mahalanobis distance algorithm) between observations for any arbitrary array of orders 1 or 2.

The inbound array must be structured in a way the array rows are the different observations of the phenomenon to process, whilst the columns represent the different dimensions of the process, very much like in input arrays used in the Python scikit-learn package. Similarly, for one-dimensional processes, the input array must be a column vector.

The user introduces in the object instantiation a parameter stating the number of (leading) array rows that are to be considered for tha Mahalanobis object calibration. Hereby, it is referred to calibration the process of calculating the mean and the covariance matrix of the system. Thus, the calibration rows correspond to the observations of the system in its reference state. Alternatively, the user can pass for calibration a list or NumPy array with the indices of the rows to be considered.

Upon instance creation, potential NaNs have to be removed from the calibration subset of the input array (since the covariance matrix cannot be inverted if it has a NaN). For this reason:
   - One-dimensional arrays:
      - If NaNs are present in the calibration subset, they are substituted with the chosen statistical indicator (mean and median supported)
      - If the array consists only of NaNs, an ```Exception``` is raised
   - Multi-dimensionl arrays:
      - If NaNs are present in the calibration subset, they are substituted with the chosen statistical indicator (mean and median supported) of the column (process feature) in which they are located.
      - Array columns consisting only of NaNs are removed prior to the calibration, thereby reducing the dimensionality of the problem.

Once the calibration subset of the input array is free of NaNs, the mean vector (the mean value of each feature) and the covariances matrix are calculated. Subsequently, the Mahalanobis distances are automatically calculated for each feature of the whole inbound array, stored in the instance variable 'distances'.

NaN entries present in the input array not in the calibration subset are not a problem for the good functioning of the package, yet the resulting Mahalanobis distance for that observation will always be NaN irrespective of the values of the other dimensions.

The Mahalanobis object has two properties 'mean' and 'cov_matrix' that allow the user to adjust their values for model behavior exploration, provided the new feature arrays have the same dimensions as those used in the original calibration of the Mahalanobis object. For exploring an object with different dimensions, a brand new instance must be created.

Given a Mahalanobis object instance with a successful calibration, it is also possible to calculate the Mahalanobis distances of external arrays benchmarked to the initial calibration, provided they match the original calibration dimensions.

### Prerequisites

This package works with Python 3 onwards as it uses f-strings

### Installing

```
pip install mahalanobis
```

End with an example of getting some data out of the system or using it for a little demo

### Basic Usage

Creation of Mahalanobis object and exploration of attributes

```
>>> import numpy as np
>>>
>>> input_1D = np.arange(10).reshape(-1,1)
>>> 
>>> mah1D = Mahalanobis(input_1D, 4)  # input1D[:4] is the calibration subset
>>> 
>>> mah1D.mean
array([1.5])
>>> 
>>> mah1D.cov_matrix  # in 1D coincides with the variance
array([[1.25]])
>>> 
>>> mah1D.distances
array([[1.34164079],
       [0.4472136 ],
       [0.4472136 ],
       [1.34164079],
       [2.23606798],
       [3.13049517],
       [4.02492236],
       [4.91934955],
       [5.81377674],
       [6.70820393]])
```
The process is equal for multi-dimensional experiments. In the example below, noise from a normal distribution has been added to the input vector to avoid having a singular covariance matrix, which would be non-invertible:

```
>>> import numpy as np
>>>
>>> input_2D = (np.arange(20) + np.random.normal(0, 0.1, 20)).reshape(-1,2)
>>>
>>>  mahND = Mahalanobis(input_2D, 4)
>>>
>>> mahND.mean
>>>
array([2.96393242, 3.95787459])
>>>
>>> mahND.cov_matrix
array([[5.12109096, 4.97459864],
       [4.97459864, 4.83325815]])
>>>
>>> mahND.distances
array([[ 1.65174621],
       [ 1.73204832],
       [ 0.65837418],
       [ 1.35583425],
       [ 7.55700759],
       [ 8.35523956],
       [ 6.17897903],
       [18.48528348],
       [ 9.3534068 ],
       [27.7453261 ]])
```
an already calibrated Mahalanobis instance can be used for calculating distances on observations of a new array:

```
>>> new_2D_array = (np.arange(14) + np.random.normal(0, 0.1, 14)).reshape(-1,2)
>>> calc.calc_distances(new_2D_array)
array([[3.33499219],
       [8.19217577],
       [1.62084771],
       [3.51305577],
       [2.22934191],
       [8.23662638],
       [7.02491688]])
```
The ```mean``` ```cov_matrix``` attributes can be set by the user for custom Mahalanobis object response, provided the have the same dimensions as the arrays used in the original calibration.

## Authors

* **Daniel Moseguí González** - [GitHub](https://github.com/mosegui) - [LinkedIn](https://www.linkedin.com/in/daniel-mosegu%C3%AD-gonz%C3%A1lez-5aa02849/)

## License

This project is licensed under the GNU GPL License - see the [LICENSE](LICENSE) file for details

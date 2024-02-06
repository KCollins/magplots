# functions for visualization of magnetometer data. 

# Importing packages:
# For fill_nan:
from scipy import interpolate
import numpy as np

############################################################################################################################### 

# #  FILL_NAN: Function to eliminate NaN values from a 1D numpy array.

def fill_nan(y):
    """
        Fit a linear regression to the non-nan y values

        Arguments:
            y      : 1D numpy array with NaNs in it

        Returns:
            Same thing; no NaNs.
    """
    
    # Fit a linear regression to the non-nan y values

    # Create X matrix for linreg with an intercept and an index
    X = np.vstack((np.ones(len(y)), np.arange(len(y))))

    # Get the non-NaN values of X and y
    X_fit = X[:, ~np.isnan(y)]
    y_fit = y[~np.isnan(y)].reshape(-1, 1)

    # Estimate the coefficients of the linear regression
    # beta = np.linalg.lstsq(X_fit.T, y_fit)[0]
    beta = np.linalg.lstsq(X_fit.T, y_fit, rcond=-1)[0]


    # Fill in all the nan values using the predicted coefficients
    y.flat[np.isnan(y)] = np.dot(X[:, np.isnan(y)].T, beta)
    return y

############################################################################################################################### 

# Function to reject outliers. We'll need this to eliminate power cycling artifacts in the magnetometer plots.
def reject_outliers(y):   # y is the data in a 1D numpy array
    """
        Function to reject outliers from a 1D dataset.

        Arguments:
            y      : 1D numpy array

        Returns:
            array with outliers replaced with NaN
    """
    mean = np.mean(y)
    sd = np.std(y)
    final_list = np.copy(y)
    for n in range(len(y)):
        final_list[n] = y[n] if y[n] > mean - 3 * sd else np.nan
        final_list[n] = final_list[n] if final_list[n] < mean + 5 * sd else np.nan
    return final_list

############################################################################################################################### 
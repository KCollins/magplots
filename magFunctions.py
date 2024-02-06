# functions for visualization of magnetometer data. 

# Importing packages:
# For fill_nan:
from scipy import interpolate
import numpy as np


# For pulling data from CDAweb:
from ai import cdas
import datetime
from matplotlib import pyplot as plt
import matplotlib as mpl

import pandas as pd

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

def magfetchtgo(start, end, magname, tgopw = '', resolution = '1sec'):
    """
    Pulls data from a RESTful API with a link based on the date.

    Args:
        start (datetime.datetime): The start date of the data to be fetched.
        end (datetime.datetime): The end date of the data to be fetched.
        magname (str): The name of the magnetometer station.
        tgopw (str): Password for Tromsø Geophysical Observatory.
        resolution (str): String for data resolution; e.g., '10sec'; default '1sec'

    Returns:
        pandas.DataFrame: A pandas DataFrame containing the fetched data.
    """
    if(tgopw == ''):
        print("No password given; cannot pull data from Tromsø Geophysical Observatory. Save a password locally in tgopw.txt.")
    
    df = pd.DataFrame()

    # Loop over each day from start to end
    for day in range(start.day, end.day + 1):
        # Generate the URL for the current day
        url = f'https://flux.phys.uit.no/cgi-bin/mkascii.cgi?site={magname}4d&year={start.year}&month={start.month}&day={day}&res={resolution}&pwd='+ tgopw + '&format=XYZhtml&comps=DHZ&getdata=+Get+Data'

        # Fetch the data for the current day
        foo = pd.read_csv(url, skiprows = 6, delim_whitespace=True, usecols=range(5), index_col=False)
        # Convert the 'DD/MM/YYYY HH:MM:SS' column to datetime format
        foo['DD/MM/YYYY HH:MM:SS'] = foo['DD/MM/YYYY'] + ' ' + foo['HH:MM:SS']
        foo['UT'] = pd.to_datetime(foo['DD/MM/YYYY HH:MM:SS'], format='%d/%m/%Y %H:%M:%S')
        foo = foo[(foo['UT'] >= start) & (foo['UT'] <= end)] # remove values before start, after end
        # foo['UT'] = foo['UT'].to_pydatetime()
        # Rename the columns
        foo.rename(columns={'X': 'MAGNETIC_NORTH_-_H', 'Y': 'MAGNETIC_EAST_-_E', 'Z': 'VERTICAL_DOWN_-_Z'}, inplace=True)
        df = pd.concat([df, foo])

    # # Convert the dataframe to a dictionary
    data = {
        'UT': df['UT'].to_numpy(),
        'MAGNETIC_NORTH_-_H': df['MAGNETIC_NORTH_-_H'].to_numpy(),
        'MAGNETIC_EAST_-_E': df['MAGNETIC_EAST_-_E'].to_numpy(),
        'VERTICAL_DOWN_-_Z': df['VERTICAL_DOWN_-_Z'].to_numpy()
    }
    
    # Convert 'UT' column to datetime64[ns] array
    data['UT'] = pd.to_datetime(data['UT'], format='%Y-%m-%dT%H:%M:%S.%f')

    # Round 'UT' column to microsecond precision
    data['UT'] = data['UT'].round('us')

    # Convert 'UT' column to datetime objects
    data['UT'] = data['UT'].to_pydatetime()
    # print(type(df))
    # return df
    return data

############################################################################################################################### 


def magfetch(
    start = datetime.datetime(2016, 1, 24, 0, 0, 0), 
    end = datetime.datetime(2016, 1, 25, 0, 0, 0), 
    magname = 'atu', 
    is_verbose = False, 
    tgopw = '',
    resolution = '1sec'
):
    """
    MAGFETCH 
        Function to fetch data for a given magnetometer. Pulls from ai.cdas or DTU.

        Arguments:
            start, end   : datetimes of the start and end of sampled data range.
            magname      : IAGA ID for magnetometer being sampled. e.g.: 'upn'
            is_verbose   : Boolean for whether debugging text is printed.
            tgopw        : Password for Tromsø Geophysical Observatory
            resolution   : Data resolution for TGO data.

        Returns:
            df           : pandas dataframe with columns ['UT', 'MAGNETIC_NORTH_-_H', 'MAGNETIC_EAST_-_E', 'VERTICAL_DOWN_-_Z']
    """
    # Pull password for TGO from local .txt file:
    file = open("tgopw.txt", "r")
    tgopw = file.read()
    if(is_verbose): print('Found Tromsø Geophysical Observatory password.')
    if(magname in ['upn', 'umq', 'gdh', 'atu', 'skt', 'ghb']): 
        if(is_verbose): print('Collecting data for ' + magname + ' from TGO.')
        if(end<start): print('End is after start... check inputs.')
        data = magfetchtgo(start, end, magname, tgopw = tgopw, resolution = resolution)
    else:
        data = cdas.get_data(
            'sp_phys',
            'THG_L2_MAG_'+ magname.upper(),
            start,
            end,
            ['thg_mag_'+ magname]
        )
    if(is_verbose): print('Data for ' + magname + ' collected: ' + str(len(data['UT'])) + ' samples.')
    return data

############################################################################################################################### 

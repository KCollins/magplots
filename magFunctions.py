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

# For saving files:
import os
import os.path
from os import path

# For power plots:
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from scipy import signal
from scipy.fft import fft
from scipy.signal import butter, filtfilt, stft, spectrogram
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, hann
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    if(magname in ['upn', 'umq', 'gdh', 'atu', 'skt', 'ghb']): # northern mags of interest for which we want to pull higher cadence data from TGO.
        # Pull password for TGO from local .txt file:
        file = open("tgopw.txt", "r")
        tgopw = file.read()
        if(is_verbose): print('Found Tromsø Geophysical Observatory password.')
        if(is_verbose): print('Collecting data for ' + magname + ' from TGO.')
        if(end<start): print('End is after start... check inputs.')
        data = magfetchtgo(start, end, magname, tgopw = tgopw, resolution = resolution)
    else:
        if(is_verbose): print('Collecting data for ' + magname + ' from CDAWeb.')
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

# MAGDF Function to create multi-indexable dataframe of all mag parameters for a given period of time. 

def magdf(
    start = datetime.datetime(2016, 1, 24, 0, 0, 0), 
    end = datetime.datetime(2016, 1, 25, 0, 0, 0), 
    maglist_a = ['upn', 'umq', 'gdh', 'atu', 'skt', 'ghb'],  # Arctic magnetometers
    maglist_b = ['pg0', 'pg1', 'pg2', 'pg3', 'pg4', 'pg5'],  # Antarctic magnetometers
    is_saved = False, 
    is_verbose = False
    ):
    """
       Function to create power plots for conjugate magnetometers.

        Arguments:
            start, end   : datetimes of the start and end of plots
            maglist_a     : List of Arctic magnetometers. Default: ['upn', 'umq', 'gdh', 'atu', 'skt', 'ghb']
            maglist_b     : Corresponding list of Antarctic magnetometers. Default: ['pg0', 'pg1', 'pg2', 'pg3', 'pg4', 'pg5']
            is_saved       : Boolean for whether resulting dataframe is saved to /output directory.
            is_verbose    : Boolean for whether debugging text is printed. 

        Returns:
            Dataframe of Bx, By, Bz for each magnetometer in list.
    """
    # Magnetometer parameter dict so that we don't have to type the full string:
    d = {'Bx': 'MAGNETIC_NORTH_-_H', 'By': 'MAGNETIC_EAST_-_E', 'Bz': 'VERTICAL_DOWN_-_Z'}

    d_i = dict((v, k) for k, v in d.items()) # inverted mapping for col renaming later
    if is_saved:
        fname = 'output/' +str(start) + '_' + '.csv'
        if os.path.exists(fname):
            if(is_verbose): print('Looks like ' + fname + ' has already been generated. Pulling data...')
            return pd.read_csv(fname)
    UT = pd.date_range(start, end, freq ='S')   # preallocate time range
    full_df = pd.DataFrame(UT, columns=['UT'])   # preallocate dataframe
    full_df['UT'] = full_df['UT'].astype('datetime64[s]') # enforce 1s precision
    full_df['Magnetometer'] = ""
    for mags in [maglist_a, maglist_b]:
        for idx, magname in enumerate(mags):   # For each magnetometer, pull data and merge into full_df:
            if(is_verbose): print('Pulling data for magnetometer: ' + magname.upper())
            try:                
                df = magfetch(start, end, magname)
                df = pd.DataFrame.from_dict(df)
                df.rename(columns=d_i, inplace=True)    # mnemonic column names

                df['Magnetometer'] = magname.upper()
                full_df = pd.concat([full_df, df])

                # print(df)
            except Exception as e:
                print(e)
                continue
    full_df['UT'] = full_df['UT'].astype('datetime64[s]') # enforce 1s precision
    full_df.drop(columns = ['UT_1']) # discard superfluous column
    full_df = full_df[full_df['Magnetometer'] != ''] # drop empty rows
    full_df = full_df.drop(['UT_1'], axis=1) # drop extraneous columns
    if is_saved:
        if(is_verbose): print('Saving as a CSV.')
        full_df.to_csv(fname)
    # print(full_df)
    return full_df 

############################################################################################################################### 

def magfig(
    parameter = 'Bx',
    start = datetime.datetime(2016, 1, 24, 0, 0, 0), 
    end = datetime.datetime(2016, 1, 25, 0, 0, 0), 
    maglist_a = ['upn', 'umq', 'gdh', 'atu', 'skt', 'ghb'],
    maglist_b = ['pg0', 'pg1', 'pg2', 'pg3', 'pg4', 'pg5'],
    is_displayed = False,
    is_saved = False, 
    is_verbose = False,
    events=None, event_fontdict = {'size':20,'weight':'bold'}
):
    """
    MAGFIG
        Function to create a stackplot for a given set of conjugate magnetometers over a given length of time. 

        Arguments:
            parameter    : The parameter of interest - Bx, By, or Bz. North/South, East/West, and vertical, respectively.
            start, end   : datetimes of the start and end of plots
            maglist_a    : List of Arctic magnetometers. Default: ['upn', 'umq', 'gdh', 'atu', 'skt', 'ghb']
            maglist_b    : Corresponding list of Antarctic magnetometers. Default: ['pg0', 'pg1', 'pg2', 'pg3', 'pg4', 'pg5']
            is_displayed : Boolean for whether resulting figure is displayed inline. False by default.
            is_saved     : Boolean for whether resulting figure is saved to /output directory.
            events       : List of datetimes for events marked on figure. Empty by default.

        Returns:
            
    """
    
    
    # Magnetometer parameter dict so that we don't have to type the full string: 
    d = {'Bx':'MAGNETIC_NORTH_-_H', 'By':'MAGNETIC_EAST_-_E','Bz':'VERTICAL_DOWN_-_Z'}
    if is_saved:
        fname = 'output/' +str(start) + '_' +  str(parameter) + '.png'
        if os.path.exists(fname):
            print('Looks like ' + fname + ' has already been generated.')
            return 
            # raise Exception('This file has already been generated.')
    fig, axs = plt.subplots(len(maglist_a), figsize=(25, 25), constrained_layout=True)
    print('Plotting data for ' + str(len(maglist_a)) + ' magnetometers: ' + str(start))
    for idx, magname in enumerate(maglist_a):   # Plot Arctic mags:
        print('Plotting data for Arctic magnetometer #' + str(idx+1) + ': ' + magname.upper())
        try:             
            data = magfetch(start = start, end = end, magname = magname, is_verbose=is_verbose) 
            x =data['UT']
            y =data[d[parameter]]
            y = reject_outliers(y) # Remove power cycling artifacts on, e.g., PG2.
            axs[idx].plot(x,y)#x, y)
            axs[idx].set(xlabel='Time', ylabel=magname.upper())

            if events is not None:
                # print('Plotting events...')
                trans       = mpl.transforms.blended_transform_factory(axs[idx].transData,axs[idx].transAxes)
                for event in events:
                    evt_dtime   = event.get('datetime')
                    evt_label   = event.get('label')
                    evt_color   = event.get('color','0.4')

                    axs[idx].axvline(evt_dtime,lw=1,ls='--',color=evt_color)
                    if evt_label is not None:
                        axs[idx].text(evt_dtime,0.01,evt_label,transform=trans,
                                rotation=90,fontdict=event_fontdict,color=evt_color,
                                va='bottom',ha='right')


            try: 
                magname = maglist_b[idx]
                ax2 = axs[idx].twinx()
                print('Plotting data for Antarctic magnetometer #' + str(idx+1) + ': ' + magname.upper())
                data = magfetch(start = start, end = end, magname = magname, is_verbose=is_verbose) 
                data['UT'] = pd.to_datetime(data['UT'])#, unit='s')
                x =data['UT']
                y =data[d[parameter]]

                color = 'tab:red'
                # ax2.set_ylabel('Y2-axis', color = color)
                # ax2.plot(y, dataset_2, color = color)
                # ax2.tick_params(axis ='y', labelcolor = color)
                y = reject_outliers(y) # Remove power cycling artifacts on, e.g., PG2.
                ax2.plot(x,-y, color=color)#x, y)
                ax2.set_ylabel(magname.upper(), color = color)
                ax2.tick_params(axis ='y', labelcolor = color)
            except Exception as e:
                print(e)
                continue
        except Exception as e:
            print(e)
            continue
    fig.suptitle(str(start) + ' ' +  str(parameter), fontsize=30)    # Title the plot...
    if is_saved:
        print("Saving figure. " + fname)
        # fname = 'output/' +str(start) + '_' +  str(parameter) + '.png'
        fig.savefig(fname, dpi='figure', pad_inches=0.3)
    if is_displayed:
        return fig # TODO: Figure out how to suppress output here
        

###############################################################################################################################  

def magspect(
    parameter='Bx',
    start=datetime.datetime(2016, 1, 24, 0, 0, 0),
    end=datetime.datetime(2016, 1, 25, 0, 0, 0),
    maglist_a=['upn', 'umq', 'gdh', 'atu', 'skt', 'ghb'],
    maglist_b=['pg0', 'pg1', 'pg2', 'pg3', 'pg4', 'pg5'],
    is_displayed=False,
    is_saved=True,
    is_verbose=False,
    events=None,
    event_fontdict={'size': 20, 'weight': 'bold'},
    myFmt=mdates.DateFormatter('%H:%M')
):
    """
    Function to create power plots for conjugate magnetometers.

    Arguments:
        parameter: The parameter of interest - Bx, By, or Bz. North/South, East/West, and vertical, respectively.
        start, end: datetimes of the start and end of plots
        maglist_a: List of Arctic magnetometers. Default: ['upn', 'umq', 'gdh', 'atu', 'skt', 'ghb']
        maglist_b: Corresponding list of Antarctic magnetometers. Default: ['pg0', 'pg1', 'pg2', 'pg3', 'pg4', 'pg5']
        is_displayed: Boolean for whether resulting figure is displayed inline. False by default.
        is_saved: Boolean for whether resulting figure is saved to /output directory.
        events: List of datetimes for events marked on figure. Empty by default.
        event_fontdict: Font dict for formatting of event labels. Default: {'size': 20, 'weight': 'bold'}
        myFmt: Date formatter. By default: mdates.DateFormatter('%H:%M')

    Returns:
        Figure of stacked plots for date in question, with events marked.
    """
    d = {'Bx': 'MAGNETIC_NORTH_-_H', 'By': 'MAGNETIC_EAST_-_E', 'Bz': 'VERTICAL_DOWN_-_Z'}
    if is_saved:
        fname = 'output/' + str(start) + '_' + str(parameter) + '.png'
        if os.path.exists(fname):
            print('Looks like ' + fname + ' has already been generated.')
            return

    fig, axs = plt.subplots(len(maglist_a), 2, figsize=(25, 25), constrained_layout=True)
    print('Plotting data for ' + str(len(maglist_a)) + ' magnetometers: ' + str(start))

    for maglist, side, sideidx in zip([maglist_a, maglist_b], ['Arctic', 'Antarctic'], [0, 1]):
        for idx, magname in enumerate(maglist):
            print('Plotting data for ' + side + ' magnetometer #' + str(idx + 1) + ': ' + magname.upper())

            try:
                data = magfetch(start, end, magname, is_verbose=is_verbose)
                x = data['UT']
                y = data[d[parameter]]
                y = reject_outliers(y)
                df = pd.DataFrame(y, x)
                df = df.interpolate('linear')
                y = df[0].values

                xlim = [start, end]

                f, t, Zxx = stft(y - np.mean(y), fs=1, nperseg=1800, noverlap=1200)
                dt_list = [start + datetime.timedelta(seconds=ii) for ii in t]

                axs[idx, sideidx].grid(False)
                cmap = axs[idx, sideidx].pcolormesh(dt_list, f * 1000., np.abs(Zxx) * np.abs(Zxx), vmin=0, vmax=0.5)
                axs[idx, sideidx].set_ylim([1, 20])  # Set y-axis limits

                axs[idx, sideidx].set_title('STFT Power Spectrum: ' + magname.upper())

                if events is not None:
                    trans = mpl.transforms.blended_transform_factory(axs[idx, sideidx].transData,
                                                                     axs[idx, sideidx].transAxes)
                    for event in events:
                        evt_dtime = event.get('datetime')
                        evt_label = event.get('label')
                        evt_color = event.get('color', '0.4')

                        axs[idx, sideidx].axvline(evt_dtime, lw=1, ls='--', color=evt_color)
                        if evt_label is not None:
                            axs[idx, sideidx].text(evt_dtime, 0.01, evt_label, transform=trans,
                                                   rotation=90, fontdict=event_fontdict, color=evt_color,
                                                   va='bottom', ha='right')

            except Exception as e:
                print(e)
                continue

    fig.suptitle(str(start) + ' ' + str(parameter), fontsize=30)  # Title the plot...
    if is_saved:
        fname = 'output/PowerSpectrum_' + str(start) + '_' + str(parameter) + '.png'
        print("Saving figure. " + fname)
        fig.savefig(fname, dpi='figure', pad_inches=0.3)
    if is_displayed:
        return fig
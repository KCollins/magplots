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

# For wave power plots:
import plotly.express as px

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

def magfetchtgo(start, end, magname, tgopw = '', resolution = '10sec', is_verbose=False):
    """
    Pulls data from a RESTful API with a link based on the date.

    Args:
        start (datetime.datetime): The start date of the data to be fetched.
        end (datetime.datetime): The end date of the data to be fetched.
        magname (str): The name of the magnetometer station.
        tgopw (str): Password for Tromsø Geophysical Observatory.
        resolution (str): String for data resolution; e.g., '10sec'; default '1sec'
        is_verbose   : Boolean for whether debugging text is printed.

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
        if(is_verbose): print(url)
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
    
    if(data['MAGNETIC_NORTH_-_H'][1] == 999.9999):
        print("WARNING: Data for " + magname.upper() + " on " + str(start) + " may not be available.\n  Check your parameters and verify magnetometer coverage at https://flux.phys.uit.no/coverage/indexDTU.html.")
    # print(type(df))
    # return df
    return data

############################################################################################################################### 
def magfetch(
    start=datetime.datetime(2016, 1, 24, 0, 0, 0),
    end=datetime.datetime(2016, 1, 25, 0, 0, 0),
    magname="atu",
    is_verbose=False,
    tgopw="",
    resolution="10sec",
):
    """
    MAGFETCH

    Function to fetch data for a given magnetometer. Pulls from ai.cdas or Tromsø Geophysical Observatory.

    Arguments:
        start, end  : datetimes of the start and end of sampled data range.
        magname     : IAGA ID for magnetometer being sampled. e.g.: 'upn'
        is_verbose  : Boolean for whether debugging text is printed.
        tgopw       : Password for Tromsø Geophysical Observatory
        resolution  : Data resolution for TGO data.

    Returns:
        df      : pandas dataframe with columns ['UT', 'MAGNETIC_NORTH_-_H', 'MAGNETIC_EAST_-_E', 'VERTICAL_DOWN_-_Z']
    """

    if magname in ["upn", "umq", "gdh", "atu", "skt", "ghb"]:  # Northern mags for TGO data
        try:
            with open("tgopw.txt", "r") as file:
                tgopw = file.read().strip()
            if is_verbose:
                print("Found Tromsø Geophysical Observatory password.")
        except FileNotFoundError:
            if is_verbose:
                print("tgopw.txt not found. Checking CDAWeb...")
            tgopw = ""  # Set to empty string for CDAWeb

    if tgopw:  # Use TGO data if password found or provided
        if is_verbose:
            print("Collecting data for", magname.upper(), "from TGO.")
        data = magfetchtgo(start, end, magname, tgopw=tgopw, resolution=resolution, is_verbose=is_verbose)
    else:  # Use CDAWeb
        if is_verbose:
            print("Collecting data for", magname.upper(), "from CDAWeb.")
        data = cdas.get_data(
            "sp_phys",
            "THG_L2_MAG_" + magname.upper(),
            start,
            end,
            ["thg_mag_" + magname],
        )

    if is_verbose:
        print("Data for", magname.upper(), "collected:", len(data["UT"]), "samples.")
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

############################################################################################################################### 
def wavepwr(station_id, 
            parameter,         # Bx, By or Bz
            start, 
            end, 
            f_lower = 2.5,        # frequency threshold in mHz 
            f_upper = 3,     # frequency threshold in mHz
            is_verbose = False
           ):
    """
         Function to determine Pc5 (by default) wave power for a given magnetometer, parameter and time frame.

        Arguments: 
               station_id      : Station ID in lowercase, e.g., 'atu', 'pg4'
               parameter        : 'Bx', 'By' or 'Bz'
               start, end      : datetimes of interval
               f_lower, f_upper : Range of frequencies of interest in mHz.
               is_verbose      : Print details of calculation. False by default. 

        Returns:
               pwr        : Calculated wave power in range of interest. 
    """
    magname = station_id.lower()
    d = {'Bx':'MAGNETIC_NORTH_-_H', 'By':'MAGNETIC_EAST_-_E','Bz':'VERTICAL_DOWN_-_Z'}
    # print(magname)
    try:
        if(is_verbose): print('Checking wave power for magnetometer ' + magname.upper() + ' between ' + str(start) + ' and ' + str(end) + '.')
        data = magfetch(start, end, magname, is_verbose = is_verbose)
        x =data['UT']
        y =data[d[parameter]]


        y = reject_outliers(y) # Remove power cycling artifacts on, e.g., PG2.
        y = fill_nan(y)
        y = y - np.nanmean(y)  # Detrend

        dt = (x[1] - x[0]).seconds
        fs = 1 / dt

        datos = y

        # nblock = 1024
        # overlap = 128
        nblock = 60
        overlap = 30
        win = hann(nblock, True)

        # f, Pxxf = welch(datos, fs, window=win, noverlap=overlap, nfft=nblock, return_onesided=True, detrend=False)
        f, Pxxf = welch(datos, fs, window=win, return_onesided=True, detrend=False)
        pwr = Pxxf[3]
        if(is_verbose): print(Pxxf[((f>=f_lower/1000) & (f_upper<=3/1000))])
        if(is_verbose): print(magname.upper() + ': The estimated power from ' + str(f_lower) + ' mHz to '+ str(f_upper) + ' mHz is ' + str(pwr) + ' nT/Hz^(1/2)')
        return pwr
    except Exception as e:
        print(e)
        if(is_verbose): print('Window length: ' + str(len(win)) +'\n Signal length: ' + str(len(y))) # usually this is the issue.
        return 'Error'
    
    
############################################################################################################################### 
def wavefig(
    stations="",  # dataframe
    parameter="Bx",
    start=datetime.datetime(2016, 1, 24, 0, 0, 0),
    end=datetime.datetime(2016, 1, 25, 0, 0, 0),
    maglist_a=["upn", "umq", "gdh", "atu", "skt", "ghb"],
    maglist_b=["pg0", "pg1", "pg2", "pg3", "pg4", "pg5"],
    f_lower=2.5,  # frequency threshold in mHz
    f_upper=3,  # frequency threshold in mHz
    is_maglist_only=True,
    is_displayed=True,
    is_saved=False,
    is_data_saved=False,
    is_verbose=False,
):
    """
    WAVEFIG

    Function to create wave power plot for a given set of magnetometers.

    Arguments:
        stations     : Dataframe of stations with columns IAGA, AACGMLAT, AACGMLON.
                       If left empty, will pull from local file stations.csv.
        parameter     : The parameter of interest - Bx, By, or Bz. North/South,
                       East/West, and vertical, respectively.
        start, end    : datetimes of the start and end of plots
        maglist_a     : List of Arctic magnetometers. Default:
                       ['upn', 'umq', 'gdh', 'atu', 'skt', 'ghb']
        maglist_b     : Corresponding list of Antarctic magnetometers. Default:
                       ['pg0', 'pg1', 'pg2', 'pg3', 'pg4', 'pg5']
        f_lower, f_upper : Range of frequencies of interest in mHz.
        is_maglist_only  : Boolean for whether only maglist_a and maglist_b stations
                           are included from the complete station list.
        is_displayed   : Boolean for whether resulting figure is displayed inline.
                       False by default.
        is_saved     : Boolean for whether resulting figure is saved to /output
                       directory.
        is_data_saved   : Boolean for whether dataframe of wave power calculation
                           resusts is saved to /output directory.
        is_verbose    : Boolean for whether debugging text is printed.

    Returns:
        Figure of stacked plots for date in question, with events marked.
    """

    if stations == "":
        if is_verbose:
            print("Loading station list from local file stations.csv...")
        stations = pd.read_csv("stations.csv")

    if is_maglist_only:
        if is_verbose:
            print("Culling to only stations listed in maglist_a and maglist_b.")
        stations = stations[
            stations.IAGA.isin([item.upper() for item in maglist_a + maglist_b])
        ]  # Plot only the polar stations
        if is_verbose:
            print(stations.IAGA)

    stations["WAVEPWR"] = stations.apply(
        lambda row: wavepwr(
            row["IAGA"],
            parameter=parameter,
            start=start,
            end=end,
            f_lower=f_lower,
            f_upper=f_upper,
            is_verbose=is_verbose,
        ),
        axis=1,
    )
    stations["HEMISPHERE"] = np.sign(stations.AACGMLAT)
    stations.HEMISPHERE = stations["HEMISPHERE"].map(
        {1: "Arctic", -1: "Antarctic", 0: "Error"}
    )
    stations["ABSLAT"] = abs(stations.AACGMLAT)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot both Arctic and Antarctic stations on the same plot
    for hemisphere, color in zip(["Arctic", "Antarctic"], ["red", "blue"]):
        stations_filtered = stations[stations["HEMISPHERE"] == hemisphere]
        stations_filtered = stations_filtered.sort_values('ABSLAT')

        for i in range(len(stations_filtered)):
            x = stations_filtered.iloc[i]["ABSLAT"]
            y = stations_filtered.iloc[i]["WAVEPWR"]
            label = stations_filtered.iloc[i]["IAGA"]

            ax.plot(
                x,
                y,
                label=hemisphere,
                color=color,
                marker="o",
            )

            ax.annotate(
                label,
                (x, y),
                xytext=(0, 5),  # Adjust vertical offset as needed
                ha="center",
                va="bottom",
                fontsize=8,
                textcoords="offset points",
                color=color,  # Match label color to line color
            )

        
        x = stations_filtered["ABSLAT"].to_list()
        y = stations_filtered["WAVEPWR"].to_list()
        # Plot lines and markers
        ax.plot(x, y, label=hemisphere, color=color, marker='o')

    # Set figure title and labels
    fig.suptitle(f"{parameter} Wave Power: {start} to {end}")
    ax.set_xlabel("Latitude (Absolute)")
    ax.set_ylabel("Wave Power")

    # Add legend
    # ax.legend()

    # Configure plot layout
    fig.tight_layout()

    # Optional: display or save the figure
    if is_displayed:
        plt.show()

    if is_saved:
        fname = f"output/WavePower_{start}_to_{end}_{parameter}.png"
        if is_verbose:
            print(f"Saving figure: {fname}")
        plt.savefig(fname)

    return fig
    
# ############################################################################################################################### 

def magall(
    start=datetime.datetime(2016, 1, 24, 0, 0, 0),
    end=datetime.datetime(2016, 1, 25, 0, 0, 0),
    maglist_a=['upn', 'umq', 'gdh', 'atu', 'skt', 'ghb'],
    maglist_b=['pg0', 'pg1', 'pg2', 'pg3', 'pg4', 'pg5'],
    f_lower = 2.5,        # frequency threshold in mHz 
    f_upper = 3,     # frequency threshold in mHz
    is_displayed=False,
    is_saved=True,
    is_verbose=False,
    events=None,
    event_fontdict={'size': 20, 'weight': 'bold'},
    myFmt=mdates.DateFormatter('%H:%M'), 
    stations = "", 
    is_maglist_only = True
):
    """
    Function to create all plots for conjugate magnetometers in a given timespan. Generates plots for all parameters: 
    Bx, By, and Bz: North/South, East/West, and vertical, respectively.

    Arguments:
        start, end: datetimes of the start and end of plots
        maglist_a: List of Arctic magnetometers. Default: ['upn', 'umq', 'gdh', 'atu', 'skt', 'ghb']
        maglist_b: Corresponding list of Antarctic magnetometers. Default: ['pg0', 'pg1', 'pg2', 'pg3', 'pg4', 'pg5']
        f_lower, f_upper : Range of frequencies of interest in mHz.
        is_displayed: Boolean for whether resulting figure is displayed inline. False by default.
        is_saved: Boolean for whether resulting figure is saved to /output directory.
        events: List of datetimes for events marked on figure. Empty by default.
        event_fontdict: Font dict for formatting of event labels. Default: {'size': 20, 'weight': 'bold'}
        myFmt: Date formatter. By default: mdates.DateFormatter('%H:%M')
        stations: Table of station coordinates. (Type `help(wavefig)` for more information.)
        is_maglist_only  : Boolean for whether only maglist_a and maglist_b stations
                           are included from the complete station list.

    Returns:
        Saves all files to \output directory.
    """
    for parameter in ['Bx', 'By', 'Bz']:
        if(is_verbose): print('Computing plots for parameter ' + parameter + '.')
        if(is_verbose): print('Saving dataframe.')
        magdf(start = start, end = end, maglist_a = maglist_a, maglist_b = maglist_b, is_saved = is_saved, is_verbose = is_verbose)
        if(is_verbose): print('Saving time-domain plot.')
        magfig(parameter=parameter, start=start, end=end, maglist_a = maglist_a, maglist_b = maglist_b, is_displayed = is_displayed, is_saved = is_saved, events = events)
        if(is_verbose): print('Saving spectrogram plot.')
        magspect(parameter = parameter, start = start, end = end, maglist_a = maglist_a, maglist_b = maglist_b, is_displayed = is_displayed, is_saved = is_saved, events = events, event_fontdict = event_fontdict, myFmt = myFmt)
        if(is_verbose): print('Generating wave power plot.')
        wavefig(stations = stations, parameter = parameter, start = start, end = end, maglist_a = maglist_a, maglist_b = maglist_b, f_lower = f_lower, f_upper = f_upper, is_maglist_only = is_maglist_only,  is_displayed = is_displayed, is_saved = is_saved, is_verbose = is_verbose)
                      
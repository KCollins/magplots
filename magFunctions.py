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
from scipy.signal import welch
from scipy.signal.windows import hann
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
    return np.where((mean - 3 * sd < y) & (y < mean + 5 * sd), y, np.nan)

############################################################################################################################### 

def magfetchtgo(start, end, magname, tgopw = '', resolution = '10sec', is_verbose=False, is_url_printed=False):
    """
    Pulls data from a RESTful API with a link based on the date.

    Args:
        start (datetime.datetime): The start date of the data to be fetched.
        end (datetime.datetime): The end date of the data to be fetched.
        magname (str): The name of the magnetometer station.
        tgopw (str): Password for Tromsø Geophysical Observatory.
        resolution (str): String for data resolution; e.g., '10sec'; default '1sec'
        is_verbose   : Boolean for whether debugging text is printed.
        is_url_printed: Boolean for whether URL link to TGO is printed in debugging text.

    Returns:
        pandas.DataFrame: A pandas DataFrame containing the fetched data.
    """
    if(tgopw == ''):
        print("No password given; cannot pull data from Tromsø Geophysical Observatory. Save a password locally in tgopw.txt.")
    
    df = pd.DataFrame()
    
    # Magnetometer parameter dict so that we don't have to type the full string:
    tgo_dict = {'bfe':'bfe6d', 'roe':'roe1d', 'nrd':'nrd1d', 'thl':'thl6d', 'svs':'svs1d', 'kuv':'kuv1d', 'upn':'upn1d', 'dmh':'dmh1d', 'umq':'umq1d', 'atu':'atu1d', 'gdh': 'gdh4d', 'stf': 'stf1d', 'skt': 'skt1d', 'ghb':'ghb1d', 'fhb':'fhb1d', 'naq':'naq4d', 'tdc':'tdc4d', 'hov':'hov1d', 'sum':'sum1d'}
    
    # Loop over each day from start to end
    for day in range(start.day, end.day + 1):
        # Generate the URL for the current day
        url = f'https://flux.phys.uit.no/cgi-bin/mkascii.cgi?site={tgo_dict.get(magname) if magname in tgo_dict else magname}&year={start.year}&month={start.month}&day={day}&res={resolution}&pwd='+ tgopw + '&format=XYZhtml&comps=DHZ&getdata=+Get+Data'
        if(is_url_printed): print(url)
        # Fetch the data for the current day
        foo = pd.read_csv(url, skiprows = 6, sep=r"\s+", usecols=range(5), index_col=False)
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
    start=datetime.datetime(2016, 1, 25, 0, 0, 0),
    end=datetime.datetime(2016, 1, 26, 0, 0, 0),
    magname="atu",
    is_detrended = True,
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
        is_detrended: Boolean for whether the median is subtracted out.
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
    if is_detrended:
        if is_verbose: print('Detrending data - subtracting the median.')
        data['MAGNETIC_NORTH_-_H'] - np.median(data['MAGNETIC_NORTH_-_H'])
        data['MAGNETIC_EAST_-_E'] - np.median(data['MAGNETIC_EAST_-_E'])
        data['VERTICAL_DOWN_-_Z'] - np.median(data['VERTICAL_DOWN_-_Z'])
    return data

############################################################################################################################### 

# MAGDF Function to create multi-indexable dataframe of all mag parameters for a given period of time. 

def magdf(
    start = datetime.datetime(2016, 1, 25, 0, 0, 0), 
    end = datetime.datetime(2016, 1, 26, 0, 0, 0), 
    maglist_a = ['upn', 'umq', 'gdh', 'atu', 'skt', 'ghb'],  # Arctic magnetometers
    maglist_b = ['pg0', 'pg1', 'pg2', 'pg3', 'pg4', 'pg5'],  # Antarctic magnetometers
    is_detrended = True, 
    is_pivoted   = False, 
    is_uniform = False, 
    is_saved = False, 
    is_verbose = False,
    ):
    """
       Function to create power plots for conjugate magnetometers.

        Arguments:
            start, end   : datetimes of the start and end of plots
            maglist_a     : List of Arctic magnetometers. Default: ['upn', 'umq', 'gdh', 'atu', 'skt', 'ghb']
            maglist_b     : Corresponding list of Antarctic magnetometers. Default: ['pg0', 'pg1', 'pg2', 'pg3', 'pg4', 'pg5']
            is_detrended  : Boolean for whether median is subtracted from data. True by default.
            is_pivoted    : Boolean for whether returned dataframe is organized by timestamp. False by default.
            is_uniform    : Boolean for whether the resolution is made uniform (i.e., downsampled to slowest cadence.) True by default.
                          (Time series plots can be native resolution for both sets, but spectrogram plots should be uniform.)
            is_saved       : Boolean for whether resulting dataframe is saved to /output directory.
            is_verbose    : Boolean for whether debugging text is printed. 

        Returns:
            Dataframe of Bx, By, Bz for each magnetometer in list.
    """
    # Magnetometer parameter dict so that we don't have to type the full string:
    d = {'Bx': 'MAGNETIC_NORTH_-_H', 'By': 'MAGNETIC_EAST_-_E', 'Bz': 'VERTICAL_DOWN_-_Z'}

    d_i = dict((v, k) for k, v in d.items()) # inverted mapping for col renaming later
    if is_saved:
        fname = 'output/' +str(start) + '_to_' + str(end) + '_'  
        if(is_pivoted): fname = fname + 'pivoted_'
        if(is_uniform): fname = fname + 'uniform'
        fname = fname + '.csv'
        if os.path.exists(fname):
            if(is_verbose): print('Looks like ' + fname + ' has already been generated. Pulling data...')
            return pd.read_csv(fname, parse_dates=[0])
    UT = pd.date_range(start, end, freq ='s')   # preallocate time range
    full_df = pd.DataFrame(UT, columns=['UT'])   # preallocate dataframe
    full_df['UT'] = full_df['UT'].astype('datetime64[s]') # enforce 1s precision
    full_df['Magnetometer'] = ""
    for mags in [maglist_a, maglist_b]:
        for idx, magname in enumerate(mags):   # For each magnetometer, pull data and merge into full_df:
            if(is_verbose): print('Pulling data for magnetometer: ' + magname.upper())
            try:                
                df = magfetch(start, end, magname, is_detrended = is_detrended)
                df = pd.DataFrame.from_dict(df)
                df.rename(columns=d_i, inplace=True)    # mnemonic column names

                df['Magnetometer'] = magname.upper()
                full_df = pd.concat([full_df, df])

                # print(df)
            except Exception as e:
                print(e)
                continue
    full_df['UT'] = full_df['UT'].astype('datetime64[s]') # enforce 1s precision
    full_df = full_df[full_df['Magnetometer'] != ''] # drop empty rows
    full_df = full_df.drop(['UT_1'], # drop extraneous columns
                           axis=1,
                           errors='ignore' # some stations don't seem to have this columm # TODO why is this
                           )
    df_pivoted = full_df.pivot(index='UT', columns='Magnetometer', values=['Bx', 'By', 'Bz'])
    if is_pivoted:
        if(is_verbose): print("Pivoting to index by time.")
        full_df = df_pivoted
    if is_uniform:
        if(is_verbose): print('Discarding NaN rows.')
        df_pivoted = df_pivoted.dropna()
        if(is_pivoted == False):
            print('Returning to original format.')
            full_df = df_pivoted.unstack(level=1).unstack().unstack().transpose()
            full_df = full_df.reset_index()
        if(is_verbose): print('Dataframe shape: ' + str(full_df.shape))
    if is_saved:
        if(is_verbose): print('Saving as a CSV.')
        full_df.to_csv(fname, index=False)
    # print(full_df)
    return full_df 

############################################################################################################################### 

def magfig(
    parameter = 'Bx',
    start = datetime.datetime(2016, 1, 25, 0, 0, 0), 
    end = datetime.datetime(2016, 1, 26, 0, 0, 0), 
    maglist_a = ['upn', 'umq', 'gdh', 'atu', 'skt', 'ghb'],
    maglist_b = ['pg0', 'pg1', 'pg2', 'pg3', 'pg4', 'pg5'],
    is_detrended = True, 
    is_displayed = False,
    is_titled = True, 
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
            is_detrended  : Boolean for whether median is subtracted from data. True by default.
            is_displayed : Boolean for whether resulting figure is displayed inline. False by default.
            is_titled    : Boolean for overall plot title. True by default. 
            is_saved     : Boolean for whether resulting figure is saved to /output directory.
            events       : List of datetimes for events marked on figure. Empty by default.

        Returns:
            
    """
    
    if is_saved:
        fname = 'output/' +str(start) + '_' +  str(parameter) + '.png'
        if os.path.exists(fname):
            print('Looks like ' + fname + ' has already been generated.')
            return 
            # raise Exception('This file has already been generated.')
    fig, axs = plt.subplots(len(maglist_a), figsize=(25, 25), constrained_layout=True)
    print('Plotting data for ' + str(len(maglist_a)) + ' magnetometers: ' + str(start))

    all_the_data = magdf(
        start=start,
        end=end,
        maglist_a=maglist_a,
        maglist_b=maglist_b,
        is_detrended=is_detrended,
        is_saved=is_saved,
        is_verbose=is_verbose
    )
    

    for idx, magname in enumerate(maglist_a):   # Plot Arctic mags:
        print('Plotting data for Arctic magnetometer #' + str(idx+1) + ': ' + magname.upper())
        try:             
            data = all_the_data[all_the_data['Magnetometer'] == magname.upper()]
            x =data['UT']
            y =data[parameter]
            color = 'tab:blue'
            y = reject_outliers(y) # Remove power cycling artifacts on, e.g., PG2.
            axs[idx].plot(x,y, color=color)#x, y)
            axs[idx].set(xlabel='Time', ylabel=magname.upper())
            axs[idx].set_ylabel(magname.upper() + ' — ' + parameter, color = color)
            axs[idx].tick_params(axis ='y', labelcolor = color)

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


            #  Corresponding Antarctic mag data on same plot...
            magname = maglist_b[idx]
            ax2 = axs[idx].twinx()
            print('Plotting data for Antarctic magnetometer #' + str(idx+1) + ': ' + magname.upper())
            data = all_the_data[all_the_data['Magnetometer'] == magname.upper()]
            x =data['UT']
            y =data[parameter]

            color = 'tab:red'
            y = reject_outliers(y) # Remove power cycling artifacts on, e.g., PG2.
            ax2.plot(x,-y, color=color)
            ax2.set_ylabel(magname.upper()+ ' — ' + parameter, color = color)
            ax2.tick_params(axis ='y', labelcolor = color)
        except Exception as e:
            print(e)
            continue
    if(is_titled == True): fig.suptitle(str(start) + ' to ' + str(end) + ' — '+ str(parameter), fontsize=30)    # Title the plot...
    if is_saved:
        print("Saving figure. " + fname)
        # fname = 'output/' +str(start) + '_' +  str(parameter) + '.png'
        fig.savefig(fname, dpi='figure', pad_inches=0.3)
    if is_displayed:
        return fig # TODO: Figure out how to suppress output here
        

###############################################################################################################################  
   
def magspect(
    parameter='Bx',
    start=datetime.datetime(2016, 1, 25, 0, 0, 0),
    end=datetime.datetime(2016, 1, 26, 0, 0, 0),
    maglist_a=['upn', 'umq', 'gdh', 'atu', 'skt', 'ghb'],
    maglist_b=['pg0', 'pg1', 'pg2', 'pg3', 'pg4', 'pg5'],
    is_detrended = True,
    is_displayed=False,
    is_saved=True,
    is_verbose=False,
    is_uniform = True, 
    is_overplotted = True, 
    color = "white", # default color for overplotting time domain data
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
        is_verbose: Boolean for displaying debugging text. 
        is_uniform: Boolean to pass to magdf() so that both sets of plots are the same resolution. True by default. 
        is_overplotted: Time domain plot is overlaid on spectrogram plot. True by default.
        color: Color for overplotting time domain data. White by default.
        events: List of datetimes for events marked on figure. Empty by default.
        event_fontdict: Font dict for formatting of event labels. Default: {'size': 20, 'weight': 'bold'}
        myFmt: Date formatter. By default: mdates.DateFormatter('%H:%M')

    Returns:
        Figure of stacked plots for date in question, with events marked.
    """
    if is_uniform == False:
        print("Warning: Scaling will not work correctly without uniform sampling.")
    if is_saved:
        fname = 'output/PowerSpectrum_' + str(start) + '_' + str(parameter) + '.png'
        if os.path.exists(fname):
            print('Looks like ' + fname + ' has already been generated.')
            return

    fig, axs = plt.subplots(len(maglist_a), 2, figsize=(25, 25), constrained_layout=True)
    print('Plotting data for ' + str(len(maglist_a)) + ' magnetometers: ' + str(start))

    all_the_data = magdf(start=start,
                 end=end,
                 maglist_a=maglist_a,
                 maglist_b=maglist_b,
                 is_detrended=is_detrended,
                 is_pivoted = False,
                 is_uniform = is_uniform,
                 is_saved=is_saved,
                 is_verbose=is_verbose)
    if(is_verbose): print(all_the_data.head(10))
    # assert all_the_data.shape[1] == 5
    
    for maglist, side, sideidx in zip([maglist_a, maglist_b], ['Arctic', 'Antarctic'], [0, 1]):
        for idx, magname in enumerate(maglist):
            print('Plotting data for ' + side + ' magnetometer #' + str(idx + 1) + ': ' + magname.upper())

            try:
                data = all_the_data[all_the_data["Magnetometer"] == magname.upper()]
                assert data.shape[0] > 0
                assert data.shape[1] == 5
                x = data['UT']
                y = data[parameter]
                assert len(x) == len(y)
                y = reject_outliers(y)
                df = pd.DataFrame(y, x)
                df = df.interpolate('linear')
                y = df[0].values

                xlim = [start, end]

                # sampling rate in units of [s]
                rate = 10
                
                # sample frequency in units of [1/s]
                fs = 1/rate #if side == 'Arctic' else 1
            
                nperseg = 1800//rate #if side == 'Arctic' else 1800
                noverlap = 1200//rate #if side == 'Arctic' else 1200

                f, t, Zxx = stft(y - np.mean(y), fs=fs, nperseg=nperseg, noverlap=noverlap)
                dt_list = [start + datetime.timedelta(seconds=ii) for ii in t] # TODO

                axs[idx, sideidx].grid(False)
                cmap = axs[idx, sideidx].pcolormesh(dt_list, f * 1000., np.abs(Zxx) * np.abs(Zxx), vmin=0, vmax=0.5)
                axs[idx, sideidx].set_ylim([1, 20])  # Set y-axis limits
                axs[idx, sideidx].set_xlabel('Time') 
                axs[idx, sideidx].set_ylabel('Frequency (Hz)') 

                axs[idx, sideidx].set_title('STFT Power Spectrum: ' + magname.upper() + ' — ' + parameter)

                if(is_overplotted == True):# overplot time domain data
                        # Create a new twin axis for the time domain plot
                        ax2 = axs[idx, sideidx].twinx()
                        
                        # Plot the time domain data on the twin axis
                        ax2.plot(x, y, color=color, alpha=0.7, label=parameter)  # Adjust color and label as needed
                        
                        # Adjust x-axis limits to match spectrogram
                        ax2.set_xlim(xlim)
                        
                        # Hide the second y-axis labels and ticks (optional)
                        ax2.set_ylabel('')
                        ax2.tick_params(axis='y', which='both', labelleft=False)

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
                                                   va='bottom', ha='left')

            except Exception as e:
                print(e)
                continue

    fig.suptitle(str(start) + ' to ' + str(end) + ' — ' + str(parameter), fontsize=30)  # Title the plot...
    if is_saved:
        fname = 'output/PowerSpectrum_' + str(start) + ' to ' + str(end) + '_' + str(parameter) + '.png'
        print("Saving figure. " + fname)
        fig.savefig(fname, dpi='figure', pad_inches=0.3)
    if is_displayed:
        return fig

############################################################################################################################### 
def wavepwr(station_id, 
            parameter,         # Bx, By or Bz
            start, 
            end, 
            f_lower = 1.667,        # frequency threshold in mHz (600 secs => 1.667 mHz)
            f_upper = 6.667,        # frequency threshold in mHz (150 secs => 6.667 mHz)
            is_saved = False,
            is_verbose = False,
            is_detrended = True
           ):
    """
         Function to determine Pc5 (by default) wave power for a given magnetometer, parameter and time frame.

        Arguments: 
               station_id      : Station ID in lowercase, e.g., 'atu', 'pg4'
               parameter        : 'Bx', 'By' or 'Bz'
               start, end      : datetimes of interval
               f_lower, f_upper : Range of frequencies of interest in mHz.
               is_saved       : Boolean for whether loaded data is saved to /output directory.
               is_verbose      : Print details of calculation. False by default. 
               is_detrended  : Boolean for whether median is subtracted from data. True by default.

        Returns:
               pwr        : Calculated wave power in range of interest. 
    """
    magname = station_id.lower()

    all_the_data = magdf(
        start=start,
        end=end,
        maglist_a=[magname], # it does not matter whether this is actually an Arctic magnetometer
        maglist_b=[],
        is_detrended=is_detrended,
        is_saved=is_saved,
        is_verbose=is_verbose
    )
    
    win = 0 # preallocate
    # print(magname)
    try:
        if(is_verbose): print('Checking wave power for magnetometer ' + magname.upper() + ' between ' + str(start) + ' and ' + str(end) + '.')
        data = all_the_data[all_the_data['Magnetometer'] == magname.upper()]
        x =data['UT']
        y =data[parameter]


        y = reject_outliers(y) # Remove power cycling artifacts on, e.g., PG2.
        y = fill_nan(y)
        y = y - np.nanmean(y)  # Detrend

        dt = (x.iloc[1] - x.iloc[0]).seconds
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
    start=datetime.datetime(2016, 1, 25, 0, 0, 0),
    end=datetime.datetime(2016, 1, 26, 0, 0, 0),
    maglist_a=["upn", "umq", "gdh", "atu", "skt", "ghb"],
    maglist_b=["pg0", "pg1", "pg2", "pg3", "pg4", "pg5"],
    f_lower=1.667,  # frequency threshold in mHz
    f_upper=6.667,  # frequency threshold in mHz
    is_maglist_only=True,
    is_detrended = True,
    is_displayed=True,
    is_saved=False,
    is_data_saved=False,
    is_verbose=False,
):
    """
    WAVEFIG

    Function to create wave power plot for a given set of magnetometers.

    Arguments:
        stations  : Dataframe of stations with columns IAGA, AACGMLAT, AACGMLON.
                    If left empty, will pull from local file stations.csv.
        parameter  : The parameter of interest - Bx, By, or Bz. North/South,
                    East/West, and vertical, respectively.
        start, end  : datetimes of the start and end of plots
        maglist_a  : List of Arctic magnetometers. Default:
                    ['upn', 'umq', 'gdh', 'atu', 'skt', 'ghb']
        maglist_b  : Corresponding list of Antarctic magnetometers. Default:
                    ['pg0', 'pg1', 'pg2', 'pg3', 'pg4', 'pg5']
        f_lower, f_upper : Range of frequencies of interest in mHz.
        is_maglist_only : Boolean for whether only maglist_a and maglist_b stations
                    are included from the complete station list.
        is_detrended  : Boolean for whether median is subtracted from data. 
                    True by default.
        is_displayed  : Boolean for whether resulting figure is displayed inline.
                    False by default.
        is_saved  : Boolean for whether resulting figure is saved to /output
                    directory.
        is_data_saved  : Boolean for whether dataframe of wave power calculation
                    resusts is saved to /output directory.
        is_verbose  : Boolean for whether debugging text is printed.

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
            is_saved=is_saved, # TODO is_saved and is_data_saved could be confusing to users here
            is_verbose=is_verbose,
            is_detrended = is_detrended
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
        stations_filtered = stations_filtered.sort_values("ABSLAT")

        # Use primary y-axis for Arctic stations
        if hemisphere == "Arctic":
            ax.plot(
                stations_filtered["ABSLAT"],
                stations_filtered["WAVEPWR"],
                label=hemisphere,
                color=color,
                marker="o",
                linestyle="-",
            )
            ax.set_ylabel("Arctic Wave Power (nT/Hz^(1/2))", color=color)
            ax.tick_params(axis="y", labelcolor=color)
        # Create secondary y-axis for Antarctic stations
        else:
            ax2 = ax.twinx()
            ax2.plot(
                stations_filtered["ABSLAT"],
                stations_filtered["WAVEPWR"],
                label=hemisphere,
                color=color,
                marker="o",
                linestyle="-",
            )

            ax2.set_ylabel("Antarctic Wave Power (nT/Hz^(1/2))", color=color)
            ax2.tick_params(axis="y", labelcolor=color)

        # Annotate each point with station label
        for i in range(len(stations_filtered)):
            x = stations_filtered.iloc[i]["ABSLAT"]
            y = stations_filtered.iloc[i]["WAVEPWR"]
            label = stations_filtered.iloc[i]["IAGA"]
            if (stations_filtered.iloc[i]["HEMISPHERE"] == "Arctic"):
                my_axis = ax
            else:
                my_axis = ax2
            my_axis.annotate(
                label,
                (x, y),
                xytext=(0, 5),  # Adjust vertical offset as needed
                ha="center",
                va="bottom",
                fontsize=8,
                textcoords="offset points",
                color=color,  # Match label color to line color
            )


    # Configure plot layout
    fig.tight_layout()

    # Optional: display or save the figure
    if is_displayed:
        plt.show()

    if is_saved:
        fname = f"output/WavePower_{start}_to_{end}_{f_lower}mHz to {f_upper}mHz_{parameter}.png"
        if is_verbose:
            print(f"Saving figure: {fname}")
        plt.savefig(fname)

    return fig

    
# ############################################################################################################################### 

def magall(
    start=datetime.datetime(2016, 1, 25, 0, 0, 0),
    end=datetime.datetime(2016, 1, 26, 0, 0, 0),
    maglist_a=['upn', 'umq', 'gdh', 'atu', 'skt', 'ghb'],
    maglist_b=['pg0', 'pg1', 'pg2', 'pg3', 'pg4', 'pg5'],
    f_lower = 1.667,        # frequency threshold in mHz 
    f_upper = 6.667,     # frequency threshold in mHz
    is_detrended = True, 
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
        is_detrended  : Boolean for whether median is subtracted from data. True by default.
        is_displayed: Boolean for whether resulting figure is displayed inline. False by default.
        is_saved: Boolean for whether resulting figure is saved to /output directory.
        events: List of datetimes for events marked on figure. Empty by default.
        event_fontdict: Font dict for formatting of event labels. Default: {'size': 20, 'weight': 'bold'}
        myFmt: Date formatter. By default: mdates.DateFormatter('%H:%M')
        stations: Table of station coordinates. (Type `help(wavefig)` for more information.)
        is_maglist_only  : Boolean for whether only maglist_a and maglist_b stations
                           are included from the complete station list.

    Returns:
        Saves all files to output/ directory.
    """
    for parameter in ['Bx', 'By', 'Bz']:
        if(is_verbose): print('Computing plots for parameter ' + parameter + '.')
        if(is_verbose): print('Saving dataframe.')
        magdf(start = start, end = end, maglist_a = maglist_a, maglist_b = maglist_b, is_saved = is_saved, is_verbose = is_verbose)
        if(is_verbose): print('Saving time-domain plot.')
        magfig(parameter=parameter, start=start, end=end, maglist_a = maglist_a, maglist_b = maglist_b, is_displayed = is_displayed, is_saved = is_saved, events = events)
        if(is_verbose): print('Saving spectrogram plot.')
        magspect(parameter = parameter, start = start, end = end, maglist_a = maglist_a, maglist_b = maglist_b, is_displayed = is_displayed, is_verbose = is_verbose, is_saved = is_saved, 
                 # events = events, 
                 event_fontdict = event_fontdict, myFmt = myFmt)
        if(is_verbose): print('Generating wave power plot.')
        wavefig(stations = stations, parameter = parameter, start = start, end = end, maglist_a = maglist_a, maglist_b = maglist_b, f_lower = f_lower, f_upper = f_upper, is_maglist_only = is_maglist_only,  is_displayed = is_displayed, is_saved = is_saved, is_verbose = is_verbose)
                      
import os
import obspy
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cProfile import label
from cmath import pi
from obspy.io.segy.segy import _read_segy
from obspy import UTCDateTime
from obspy.core.stream import Stream
from obspy import Trace
from obspy.signal.cross_correlation import correlate

os.system('cls')
path_all = 'C:\HM\Chulin_Processing\Data' + "\\"
source_path_all = path_all + 'Pilot_segy' + "\\"
sac_path = path_all + 'sac' + "\\"
location_file = 'C:\HM\Chulin_Processing' + "\\"
seis_location_file = 'C:\HM\Chulin_Processing' + "\\"
list_source = os.listdir(source_path_all)
for i in range(len(list_source)):
    num_source_shock = list_source[i]
    path_shock = source_path_all + num_source_shock
    list_shock = os.listdir(path_shock)
    for j in range(len(list_shock)): 
        num_shock = j+1
        # ===============
        # Importing Data
        # ===============
        sta_list = os.listdir(sac_path)
        source_shock = num_source_shock + "\\"
        source_locations = pd.read_csv(location_file + 'shot_location.txt', sep="\t", error_bad_lines=False, skiprows=3)
        source_location = source_locations.Position[int(num_source_shock)-1]
        lat_source = float(source_location[1:9])
        lon_source = float(source_location[11:20])
        source_location_lonlat = (lat_source, lon_source)
        # ========================================
        # Transform the format of the source file
        # ========================================
        list_shock.sort()
        file_name_source = list_shock[int(num_shock)-1]
        segy = _read_segy(source_path_all + source_shock + file_name_source)
        y_segy = segy.traces[0].data
        x_segy = range(0,len(y_segy))
        tr = Trace(np.array(y_segy))
        tr.write("hypof.mseed", format="MSEED")
        # ========================================
        # Read the source file and do resampling
        # ========================================
        st_hypof = obspy.read('hypof.mseed')
        st_hypof.decimate(10)
        tr_hypof = st_hypof[0]
        starttime_for_transver = '20' + file_name_source[5] + file_name_source[6] + '-' + file_name_source[7] + file_name_source[8] + '-' + file_name_source[9] + file_name_source[10] + 'T' + file_name_source[12] + file_name_source[13] + ':' + file_name_source[14] + file_name_source[15] + ':' + file_name_source[16] + file_name_source[17]
        tr_hypof.stats.starttime = UTCDateTime(starttime_for_transver)
        tr_hypof_y = tr_hypof.data
        # ===================================
        # Trim and filter the seismic signal
        # ===================================
        Dist_all = np.arange(len(sta_list))
        fig = plt.figure(figsize=(15, 18))
        fig.patch.set_facecolor('white')
        for k in range(len(sta_list)):
            st_trim = obspy.read(sac_path + sta_list[k])
            st_trim_filter = st_trim.filter('bandpass', freqmin=5.0, freqmax=50.0)
            hour = float(file_name_source[12] + file_name_source[13])
            mins = float(file_name_source[14] + file_name_source[15])
            sec = float(file_name_source[16] + file_name_source[17])
            starttime = st_trim_filter[0].stats.starttime + hour*60*60 + mins*60 + sec
            endtime = st_trim_filter[0].stats.starttime + hour*60*60 + mins*60 + sec + 8 - 0.01
            st_trim_filter.trim(starttime=starttime, endtime=endtime)
            # ===================
            # Cross Correlation
            # ===================
            tr = st_trim_filter[0]
            cc = correlate(tr, tr_hypof, 800)
            x_cc = range(0, len(cc))
            y_cc = cc
            # ==========================================
            # Distance between the source and instrument
            # ==========================================
            seis_locations = pd.read_csv(seis_location_file + 'seis_sta.txt', sep = "\t")
            Thistime_sta = sta_list[k][0:4]
            this_sta_inall = seis_locations.sta.str.find(Thistime_sta)
            num_this_sta = this_sta_inall[this_sta_inall == 0].index[0]
            lon_seis = float(seis_locations.Lon[num_this_sta])
            lat_seis = float(seis_locations.Lat[num_this_sta])
            seis_location_lonlat = (lat_seis, lon_seis)
            dist_deg = math.dist(source_location_lonlat, seis_location_lonlat)
            dist = 2*math.pi*6378137*dist_deg/360
            if seis_locations.Inst[num_this_sta] == "GP":
                plot_gp = plt.plot(np.linspace(0, len(x_cc)/100, num=1601)-8, (y_cc/max(y_cc)*100)+dist, 'k', linewidth=0.8)
                plt.text(8.1, np.mean((y_cc/max(y_cc))*100+dist), str(sta_list[k][:4])+"(GP)", color='k', fontsize=8)
            else:
                plot_bb = plt.plot(np.linspace(0, len(x_cc)/100, num=1601)-8, (y_cc/max(y_cc)*100)+dist, 'r', linewidth=0.8)
                plt.text(8.5, np.mean((y_cc/max(y_cc))*100+dist), str(sta_list[k][:4])+"(BB)", color='r', fontsize=8)
            # ==========================
            # Parameter for making plot
            # ==========================
            plt.xlabel('Sec.')
            plt.ylabel('Dist.')
            plt.xlim(0,8)
            plt.title("source: " + source_shock[0:3] + " - " + str(num_shock), fontweight='bold', fontsize=20)
            plt.rcParams['font.sans-serif'] = 'Times New Roman'
            #plt.show()
            saveimage_path = path_all + "Output" + "\\" + source_shock[0:3] 
            os.makedirs(saveimage_path, exist_ok=True)
            plt.savefig(saveimage_path + "\\" + source_shock[0:3] + "_" + str(num_shock), dpi=120)
os.remove('hypof.mseed')
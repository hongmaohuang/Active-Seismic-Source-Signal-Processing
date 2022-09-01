# %%
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
seis_locations = pd.read_csv(seis_location_file + 'seis_sta.txt', sep = "\t")
time_source = pd.read_table(seis_location_file + 'shot_time table.txt', skiprows=3, header=None)
New_time_table = np.zeros((len(time_source),3))
All_New_time_table = np.zeros((len(time_source),3))
for i in range(len(time_source)):
    if len(time_source[0][i][21:36]) != 0:
        New_time_table[i,0] = int(time_source[0][i][21:23])
        New_time_table[i,1] = int(time_source[0][i][24:26])
        New_time_table[i,2] = float(time_source[0][i][27:36])       
sum_alltime = np.zeros((len(New_time_table),1))
for i in range(len(New_time_table)):
    sum_alltime[i] = sum(New_time_table[i,:])
ind_zero = np.where(sum_alltime!=0)
New_time_table = New_time_table[ind_zero[0]]
len_lasttime = 0
for i in range(len(list_source)):
    num_source_shock = list_source[i]
    source_shock = num_source_shock + "\\"
    path_shock_last = source_path_all + list_source[i-1]
    path_shock = source_path_all + list_source[i]
    list_shock = os.listdir(path_shock)
    sta_list = os.listdir(sac_path)
    startline = len_lasttime
    for j in range(len(list_shock)): 
        num_shock = j+1
        # ===============
        # Importing Data
        # ===============
        source_locations = pd.read_csv(location_file + 'shot_location.txt', sep="\t", error_bad_lines=False, skiprows=3)
        source_location = source_locations.Position[int(num_source_shock)-1]
        lat_source = float(source_location[1:9])
        lon_source = float(source_location[11:20])
        source_location_lonlat = (lat_source, lon_source)
        # ========================================
        # Transform the format of the source file
        # ========================================
        list_shock.sort()
        file_name_source = list_shock[j]
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
        # %%
        starttime_for_transver = '20' + file_name_source[5] + file_name_source[6] + '-' + file_name_source[7] + file_name_source[8] + '-' + file_name_source[9] + file_name_source[10] + 'T' + file_name_source[12] + file_name_source[13] + ':' + file_name_source[14] + file_name_source[15] + ':' + file_name_source[16] + file_name_source[17]
        if len(str(New_time_table[j+startline][0:3][1]))==3:
            starttime_for_transver_table = '20' + file_name_source[5] + file_name_source[6] + '-' + file_name_source[7] + file_name_source[8] + '-' + file_name_source[9] + file_name_source[10] + 'T' + "0" + str(int(New_time_table[j+startline][0:3][0])) + ':0' + str(int(New_time_table[j+startline][0:3][1])) + ':' + str(float(New_time_table[j+startline][0:3][2])) 
        else:
            starttime_for_transver_table = '20' + file_name_source[5] + file_name_source[6] + '-' + file_name_source[7] + file_name_source[8] + '-' + file_name_source[9] + file_name_source[10] + 'T' + "0" + str(int(New_time_table[j+startline][0:3][0])) + ':' + str(int(New_time_table[j+startline][0:3][1])) + ':' + str(float(New_time_table[j+startline][0:3][2])) 
        print(source_shock[0:3] + "_" + str(num_shock))       
        print(starttime_for_transver)
        print(starttime_for_transver_table)
        tr_hypof.stats.starttime = UTCDateTime(starttime_for_transver_table)
        tr_hypof_y = tr_hypof.data
        # %%
        # ===================================
        # Trim and filter the seismic signal
        # ===================================
        Dist_all = np.arange(len(sta_list))
        fig = plt.figure(figsize=(22, 15))
        fig.patch.set_facecolor('white')
        for k in range(len(sta_list)):
            st_trim = obspy.read(sac_path + sta_list[k])
            st_trim_filter = st_trim.filter('bandpass', freqmin=5.0, freqmax=15)
            hour = float(starttime_for_transver_table[11:13])
            # %%
            mins = float(starttime_for_transver_table[14:16])
            sec = float(starttime_for_transver_table[17:36])
            starttime = st_trim_filter[0].stats.starttime + hour*60*60 + mins*60 + sec
            endtime = st_trim_filter[0].stats.starttime + hour*60*60 + mins*60 + sec + 8 - 0.01
            st_trim_filter.trim(starttime=starttime, endtime=endtime)
            st_trim_filter.detrend('demean')
            st_trim_filter.detrend('linear')
            st_trim_filter.detrend('linear')
            st_trim_filter.detrend('linear')
            st_trim_filter.taper(max_percentage=0.05, type='cosine', max_length=len(st_trim_filter[0].data), side='both')
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
            Thistime_sta = sta_list[k][0:4]
            this_sta_inall = seis_locations.sta.str.find(Thistime_sta)
            num_this_sta = this_sta_inall[this_sta_inall == 0].index[0]
            lon_seis = float(seis_locations.Lon[num_this_sta])
            lat_seis = float(seis_locations.Lat[num_this_sta])
            seis_location_lonlat = (lat_seis, lon_seis)
            dist_deg = math.dist(source_location_lonlat, seis_location_lonlat)
            dist = 2*math.pi*6378137*dist_deg/360
            # =================
            # Save as a  trace
            # =================
            tr_cc = Trace(cc)
            tr_cc.stats.starttime = UTCDateTime(starttime_for_transver_table)
            st_cc = Stream(traces=[tr_cc])
            savesac_path = path_all + "Output_SAC" + "\\" + source_shock[0:3] 
            os.makedirs(savesac_path, exist_ok=True)
            st_cc.write(savesac_path + "\\" + source_shock[0:3] + "_Shock-" + str(num_shock) + "." + sta_list[k][:4] + "." + "HHZ_Dist-" + str(format(dist, '0.2f')) + "m" + '.sac', format='SAC')
            # =====
            # Plot
            # =====
            if seis_locations.Inst[num_this_sta] == "GP":
                plot_gp = plt.plot(np.linspace(0, len(x_cc)/100, num=1601)-8, (y_cc/max(y_cc)*100)+dist, 'k', linewidth=0.8)
                plt.text(8.1, np.mean((y_cc/max(y_cc))*100+dist), str(sta_list[k][:4])+"(GP)", color='k', fontsize=15)
            else:
                plot_bb = plt.plot(np.linspace(0, len(x_cc)/100, num=1601)-8, (y_cc/max(y_cc)*100)+dist, 'r', linewidth=0.8)
                plt.text(8.8, np.mean((y_cc/max(y_cc))*100+dist), str(sta_list[k][:4])+"(BB)", color='r', fontsize=15)
            # ==========================
            # Parameter for making plot
            # ==========================
            plt.xlabel('Time [sec]', fontsize = 20)
            plt.ylabel('Distance [m]', fontsize = 20)
            plt.xlim(-4,8)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.title("source: " + source_shock[0:3] + " - " + str(num_shock), fontweight='bold', fontsize=35)
            plt.rcParams['font.sans-serif'] = 'Times New Roman'
            #plt.show()
            saveimage_path = path_all + "Output_Images" + "\\" + source_shock[0:3] 
            os.makedirs(saveimage_path, exist_ok=True)
            plt.savefig(saveimage_path + "\\" + source_shock[0:3] + "_" + str(num_shock), dpi=120)
    len_lasttime = len_lasttime + len(os.listdir(path_shock))
os.remove('hypof.mseed')
# %%
# %% 
import os
import obspy
import numpy as np
import matplotlib.pyplot as plt
from obspy.io.segy.segy import _read_segy
from obspy import UTCDateTime
from obspy.core.stream import Stream
from obspy import Trace
from obspy.signal.cross_correlation import correlate
# 
# 本程式將畫出使用者自訂某一震源的某一次震動在每個測站接收到的訊號。
# This code is for demostrating the signal in each station from the shock designated by user.


# 1. 將震源函數轉成mseed並再次讀取（為了畫圖跟計算方面）
# 2. 震源函數降取樣（原1000Hz）
# 3. 切出原波型（震動發生前後5分鐘，共10分鐘）查看震動波型
# 4. 將波型zoom in到與震源函數相同的時間段（震動開始後8秒）
# 5. 做correlation
# 6. 畫圖

os.system('cls')
# Input

'''
file_path = 'C:\\HM\\Chulin_Processing\\Data\\Pilot_segy\\' + "015" + "\\"
file_name = "8056_220324_051221.sgy"
num_shock = "10"
sac_path = "C:\HM\Chulin_Processing\Data\sac" + "\\"
Station_name = "CL07" + ".HHZ" + ".SAC"
'''

file_path = input("Please enter the path of the source function file:") + "\\"
file_name = input("Please enter the name of the source function file:")
num_shock = input("Please enter the number of the shock:")
sac_path = input("Please enter the path of the SAC file:") + "\\"
Station_name = input("Please enter the station name:") + ".HHZ" + ".SAC"
   

# 震源函數轉檔
segy = _read_segy(file_path + file_name)
y_segy = segy.traces[0].data
x_segy = range(0,len(y_segy))
tr = Trace(np.array(y_segy))
tr.write("hypof.mseed", format="MSEED")

# 讀取震源函數並重新取樣
st_hypof = obspy.read('hypof.mseed')
st_hypof.decimate(10)
tr_hypof = st_hypof[0]
starttime_for_transver = '20' + file_name[5] + file_name[6] + '-' + file_name[7] + file_name[8] + '-' + file_name[9] + file_name[10] + 'T' + file_name[12] + file_name[13] + ':' + file_name[14] + file_name[15] + ':' + file_name[16] + file_name[17]
tr_hypof.stats.starttime = UTCDateTime(starttime_for_transver)
tr_hypof_y = tr_hypof.data

# 切SAC前後五分鐘看看波型
st = obspy.read(sac_path + Station_name)
st_trim_for_check = st.copy()
st_trim_for_check_filter = st_trim_for_check.filter('bandpass', freqmin=5.0, freqmax=50.0)
hour = float(file_name[12] + file_name[13])
mins = float(file_name[14] + file_name[15])
sec = float(file_name[16] + file_name[17])
starttime = st[0].stats.starttime + hour*60*60 + mins*60 + sec - 5*60
endtime = st[0].stats.starttime + hour*60*60 + mins*60 + sec + 5*60
st_trim_for_check_filter.trim(starttime=starttime, endtime=endtime)

# 切SAC分析
st_trim = st.copy()
st_trim_filter = st_trim.filter('bandpass', freqmin=5.0, freqmax=50.0)
starttime = st[0].stats.starttime + hour*60*60 + mins*60 + sec
endtime = st[0].stats.starttime + hour*60*60 + mins*60 + sec + 8 - 0.01
st_trim_filter.trim(starttime=starttime, endtime=endtime)
st_trim_filter_y = st_trim_filter[0].data

# correlation
cc = correlate(st_trim_filter[0], tr_hypof, 800)
x_cc = range(0, len(cc))
y_cc = cc
tr_y_cc = Trace(np.array(y_cc))
tr_y_cc.stats.starttime = UTCDateTime(starttime_for_transver)

# %% Plot
fig = plt.figure(figsize=(18,20))
nfft = 64
noverlap = 60
per_lap = noverlap/nfft
hspace = 0.9
#fig = plt.figure()
fig.patch.set_facecolor('white')
ax1 = plt.subplot2grid((7,1), (0,0), rowspan=1)
ax1.plot(st_trim_for_check_filter[0].times(), st_trim_for_check_filter[0].data, 'k')
min_ylabel = min(st_trim_for_check_filter[0].data)
max_ylabel = max(st_trim_for_check_filter[0].data)
ax1.plot([300, 300], [min_ylabel, max_ylabel], color="red")
ax1.plot([300, 308], [min_ylabel, min_ylabel], color="red")
ax1.plot([300, 308], [max_ylabel, max_ylabel], color="red")
ax1.plot([308, 308], [min_ylabel, max_ylabel], color="red")
ax1.set_xlim(0, 600)
#ax1.legend()
ax1.title.set_text('Original Waveform for 10 min')
#看震動發生前後五分鐘的狀況

ax2 = plt.subplot2grid((7,1), (1,0), rowspan=1)
ax2.plot(np.arange(0, len(st_trim_filter_y)/100, 0.01), st_trim_filter_y, 'k')
ax2.title.set_text('Original Waveform for 8 sec')
min_ylabel = min(st_trim_filter_y)
max_ylabel = max(st_trim_filter_y)
ax2.set_xlim(0,8)
#地震訊號

ax3 = plt.subplot2grid((7,1), (2,0), rowspan=1)
ax3.plot(np.arange(0, len(tr_hypof_y)/100, 0.01), tr_hypof_y, 'k')
ax3.title.set_text('Source Function')
min_ylabel = min(tr_hypof_y)
max_ylabel = max(tr_hypof_y)
ax3.set_xlim(0,8)
#震源函數

ax4 = plt.subplot2grid((7,1), (3,0), rowspan=1)
ax4.plot((np.linspace(0, len(x_cc)/100, num=1601))-8, y_cc, 'k')
ax4.title.set_text('Cross-Correlation')
min_ylabel = min(y_cc)
max_ylabel = max(y_cc)
ax4.set_xlim(0,8)
#Correlation後

plt.subplots_adjust(hspace=hspace)
ax5 = plt.subplot2grid((7,1),(4,0), rowspan=3, colspan=1)
ax5.specgram(tr_y_cc, Fs=100, cmap='jet', NFFT=nfft, noverlap=noverlap, scale='dB', xextent=(-8,8))
ax5.set_xlim(0,8)
cbar_x = ax5.get_position().x1 + 0.01 
cbar_y = ax5.get_position().y0
cbar_h = ax5.get_position().height
cbar = fig.add_axes([cbar_x, cbar_y, 0.02, cbar_h])
plt.colorbar(ax5.images[0], label='(dB)', cax=cbar) 
ax5.title.set_text('Spectrogram')
#時頻圖

plt.subplots_adjust(hspace=hspace)

suptitle_plot = Station_name[:8] + "_" + file_path[40] + file_path[41] + file_path[42] + "(" + num_shock + ")"

fig.suptitle(suptitle_plot, fontweight='bold', fontsize=20)
plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.show()
#os.remove('hypof.mseed')

# %%

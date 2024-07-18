import warnings
import numpy as np
import os
import random
import sys
import copy
import shutil
import tflite_runtime.interpreter as tflite
from scipy.signal import  butter, lfilter, cheby1, find_peaks, freqz
from scipy.interpolate import interp1d
import pandas as pd
from datetime import datetime

def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.8 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def cheby1_bandpass(lowcut, highcut, fs, order):
    nyq = 0.8 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = cheby1(order, 0.1, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def cheby1_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = cheby1_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def detect_r_peaks(filtered_signal, fs):
    # Constants for Pan-Tompkins algorithm
    threshold_factor = 0.3

    # Differentiation
    diff_signal = np.diff(filtered_signal)

    # Squaring
    squared_signal = diff_signal ** 2

    # Moving average integration
    window_size = int(0.10 * fs)  # 150 ms
    ma_signal = np.convolve(squared_signal, np.ones(window_size) / window_size, mode='same')

    # Find R-peaks
    threshold = threshold_factor * np.max(ma_signal)
    r_peak_indices = np.where((ma_signal[:-1] < threshold) & (ma_signal[1:] >= threshold))[0] + 1
    return r_peak_indices


#call pan tompkins algorithm
# r_peaks = detect_r_peaks(filtered_signal, fs)

BP = {}
BP['SBP'] = [133, 111, 113, 113]
BP['DBP'] = [67, 73, 73, 65]
# TODO: Add file here
BP['file'] = ['Bharin2.csv', 'Behzod2.csv', 'Aryadi2.csv', 'Ariok2ex2.csv', 'log_data.csv']
file=4

PPG_lowcut=0.5
PPG_highcut=10
PPG_order=1

ECG_lowcut=2
ECG_highcut=15
ECG_order=1

PCG_lowcut=5
PCG_highcut= 20 # 90
PCG_order=1

max_pcg =  644.8331936295845 
min_pcg =  -618.1112834754251 
max_ppg =  9216.960001590374 
min_ppg =  -2770.0716776274935 
max_ecg =  678.1879504858125 
min_ecg =  -564.815242861711
max_rr1 =  258 
min_rr1 =  3 
max_rr2 =  268 
min_rr2 =  3

fs = 100

# SBP file
TFLITE_MODEL_FILENAME = "tflite_SBP_model.tflite"
TARGET_BP = 'SBP'

interpreter = tflite.Interpreter(model_path = TFLITE_MODEL_FILENAME)

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

ECG_input_shape = input_details[0]['shape']
PPG_input_shape = input_details[2]['shape']
PCG_input_shape = input_details[3]['shape']
RR0_input_shape = input_details[4]['shape']
RR1_input_shape = input_details[1]['shape']

# Allocate tensors
interpreter.allocate_tensors()


print("\n", BP['file'][file], "\n")
df = pd.read_csv(BP['file'][file], index_col=['Time'])
df = df.dropna()

# Extract Segments
filtered_PPG = butter_bandpass_filter(df['PPG'], PPG_lowcut, PPG_highcut, fs, PPG_order)[3*fs:]
filtered_ECG = cheby1_bandpass_filter(df['ECG'], ECG_lowcut, ECG_highcut, fs, ECG_order)[3*fs:]
filtered_PCG = cheby1_bandpass_filter(df['PCG'], PCG_lowcut, PCG_highcut, fs, PCG_order)[3*fs:]

clean_ecg_peaks = detect_r_peaks(filtered_ECG, fs)

prediction_SBP = np.array([])
for peak_index_no, peak_index in enumerate(clean_ecg_peaks):
    if (peak_index_no==0) or (clean_ecg_peaks[-1]==peak_index) or (clean_ecg_peaks[-2]==peak_index):
        continue
    PPG_segmented_instance=filtered_PPG[clean_ecg_peaks[peak_index_no]:clean_ecg_peaks[peak_index_no+2]+1]
    PCG_segmented_instance=filtered_PCG[clean_ecg_peaks[peak_index_no]:clean_ecg_peaks[peak_index_no+2]+1]
    ECG_segmented_instance=filtered_ECG[clean_ecg_peaks[peak_index_no]:clean_ecg_peaks[peak_index_no+2]+1]
    RR = [clean_ecg_peaks[peak_index_no+1]-peak_index, clean_ecg_peaks[peak_index_no+2]-clean_ecg_peaks[peak_index_no+1]]

    pre_interp_PPG= np.hstack(PPG_segmented_instance)
    pre_interp_PCG= np.hstack(PCG_segmented_instance)
    pre_interp_ECG= np.hstack(ECG_segmented_instance)
    panjang_interpolasi = np.linspace(0,1,num = 200, endpoint=True)

    if (len(pre_interp_PPG) < 200) or (len(pre_interp_PPG) > 200) :
        x_interp_PPG= np.linspace(0, 1, len(pre_interp_PPG))
        f_interp_PPG = interp1d(x_interp_PPG, pre_interp_PPG)
        interpolated_PPG_segmented_instance = f_interp_PPG(panjang_interpolasi)

        x_interp_PCG= np.linspace(0, 1, len(pre_interp_PCG))
        f_interp_PCG = interp1d(x_interp_PCG, pre_interp_PCG)
        interpolated_PCG_segmented_instance = f_interp_PCG(panjang_interpolasi)

        x_interp_ECG= np.linspace(0, 1, len(pre_interp_ECG))
        f_interp_ECG = interp1d(x_interp_ECG, pre_interp_ECG)
        interpolated_ECG_segmented_instance = f_interp_ECG(panjang_interpolasi)

    else:
        interpolated_PPG_segmented_instance = pre_interp_PPG
        interpolated_PCG_segmented_instance = pre_interp_PCG
        interpolated_ECG_segmented_instance = pre_interp_ECG

    halved_PPG_segment = np.asarray([interpolated_PPG_segmented_instance[i] for i in range(len(interpolated_PPG_segmented_instance)) if i % 2 == 0])
    halved_PCG_segment = np.asarray([interpolated_PCG_segmented_instance[i] for i in range(len(interpolated_PCG_segmented_instance)) if i % 2 == 0])
    halved_ECG_segment = np.asarray([interpolated_ECG_segmented_instance[i] for i in range(len(interpolated_ECG_segmented_instance)) if i % 2 == 0])
    halved_PPG_segment[-1] = interpolated_PPG_segmented_instance[-1]
    halved_PCG_segment[-1] = interpolated_PCG_segmented_instance[-1]
    halved_ECG_segment[-1] = interpolated_ECG_segmented_instance[-1]
    
    final_PPG_segment = halved_PPG_segment
    final_PCG_segment = halved_PCG_segment
    final_ECG_segment = halved_ECG_segment
    
    final_PPG_segment = np.array([])
    final_PCG_segment = np.array([])
    final_ECG_segment = np.array([])
    
    for v in halved_PPG_segment:
        final_PPG_segment = np.append(final_PPG_segment, (v-min_ppg)*(1.0/(max_ppg-min_ppg)))
    for w in halved_PCG_segment:
        final_PCG_segment = np.append(final_PCG_segment, (w-min_pcg)*(1.0/(max_pcg-min_pcg)))
    for k in halved_ECG_segment:
        final_ECG_segment = np.append(final_ECG_segment, (k-min_ecg)*(1.0/(max_ecg-min_ecg)))
    RR[0] = (RR[0]-min_rr1)*(1.0/(max_rr1-min_rr1))
    RR[1] = (RR[1]-min_rr2)*(1.0/(max_rr2-min_rr2))
        
    ECG_input = np.array(final_ECG_segment.reshape(ECG_input_shape), dtype=np.float32)
    PPG_input = np.array(final_PPG_segment.reshape(PPG_input_shape), dtype=np.float32)
    PCG_input = np.array(final_PCG_segment.reshape(PCG_input_shape), dtype=np.float32)
    RR0_input = np.array(RR[0].reshape(RR0_input_shape), dtype=np.float32)
    RR1_input = np.array(RR[1].reshape(RR1_input_shape), dtype=np.float32)
    
    interpreter.set_tensor(input_details[0]['index'], ECG_input)
    interpreter.set_tensor(input_details[2]['index'], PPG_input)
    interpreter.set_tensor(input_details[3]['index'], PCG_input)
    interpreter.set_tensor(input_details[4]['index'], RR0_input)
    interpreter.set_tensor(input_details[1]['index'], RR1_input)
    
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    #print(output_data[0][0], BP[TARGET_BP])
    prediction_SBP = np.append(prediction_SBP, output_data[0][0])
    


# DBP file
TFLITE_MODEL_FILENAME = "tflite_DBP_model.tflite"
TARGET_BP = 'DBP'


interpreter = tflite.Interpreter(model_path = TFLITE_MODEL_FILENAME)

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

ECG_input_shape = input_details[0]['shape']
PPG_input_shape = input_details[2]['shape']
PCG_input_shape = input_details[3]['shape']
RR0_input_shape = input_details[4]['shape']
RR1_input_shape = input_details[1]['shape']

# Allocate tensors
interpreter.allocate_tensors()

df = pd.read_csv(BP['file'][file], index_col=['Time'])
df = df.dropna()

# Extract Segments
filtered_PPG = butter_bandpass_filter(df['PPG'], PPG_lowcut, PPG_highcut, fs, PPG_order)[3*fs:]
filtered_ECG = cheby1_bandpass_filter(df['ECG'], ECG_lowcut, ECG_highcut, fs, ECG_order)[3*fs:]
filtered_PCG = cheby1_bandpass_filter(df['PCG'], PCG_lowcut, PCG_highcut, fs, PCG_order)[3*fs:]

clean_ecg_peaks = detect_r_peaks(filtered_ECG, fs)

prediction_DBP = np.array([])
for peak_index_no, peak_index in enumerate(clean_ecg_peaks):
    if (peak_index_no==0) or (clean_ecg_peaks[-1]==peak_index) or (clean_ecg_peaks[-2]==peak_index):
        continue
    PPG_segmented_instance=filtered_PPG[clean_ecg_peaks[peak_index_no]:clean_ecg_peaks[peak_index_no+2]+1]
    PCG_segmented_instance=filtered_PCG[clean_ecg_peaks[peak_index_no]:clean_ecg_peaks[peak_index_no+2]+1]
    ECG_segmented_instance=filtered_ECG[clean_ecg_peaks[peak_index_no]:clean_ecg_peaks[peak_index_no+2]+1]
    RR = [clean_ecg_peaks[peak_index_no+1]-peak_index, clean_ecg_peaks[peak_index_no+2]-clean_ecg_peaks[peak_index_no+1]]

    pre_interp_PPG= np.hstack(PPG_segmented_instance)
    pre_interp_PCG= np.hstack(PCG_segmented_instance)
    pre_interp_ECG= np.hstack(ECG_segmented_instance)
    panjang_interpolasi = np.linspace(0,1,num = 200, endpoint=True)

    if (len(pre_interp_PPG) < 200) or (len(pre_interp_PPG) > 200) :
        x_interp_PPG= np.linspace(0, 1, len(pre_interp_PPG))
        f_interp_PPG = interp1d(x_interp_PPG, pre_interp_PPG)
        interpolated_PPG_segmented_instance = f_interp_PPG(panjang_interpolasi)

        x_interp_PCG= np.linspace(0, 1, len(pre_interp_PCG))
        f_interp_PCG = interp1d(x_interp_PCG, pre_interp_PCG)
        interpolated_PCG_segmented_instance = f_interp_PCG(panjang_interpolasi)

        x_interp_ECG= np.linspace(0, 1, len(pre_interp_ECG))
        f_interp_ECG = interp1d(x_interp_ECG, pre_interp_ECG)
        interpolated_ECG_segmented_instance = f_interp_ECG(panjang_interpolasi)

    else:
        interpolated_PPG_segmented_instance = pre_interp_PPG
        interpolated_PCG_segmented_instance = pre_interp_PCG
        interpolated_ECG_segmented_instance = pre_interp_ECG

    halved_PPG_segment = np.asarray([interpolated_PPG_segmented_instance[i] for i in range(len(interpolated_PPG_segmented_instance)) if i % 2 == 0])
    halved_PCG_segment = np.asarray([interpolated_PCG_segmented_instance[i] for i in range(len(interpolated_PCG_segmented_instance)) if i % 2 == 0])
    halved_ECG_segment = np.asarray([interpolated_ECG_segmented_instance[i] for i in range(len(interpolated_ECG_segmented_instance)) if i % 2 == 0])
    halved_PPG_segment[-1] = interpolated_PPG_segmented_instance[-1]
    halved_PCG_segment[-1] = interpolated_PCG_segmented_instance[-1]
    halved_ECG_segment[-1] = interpolated_ECG_segmented_instance[-1]
    
    final_PPG_segment = halved_PPG_segment
    final_PCG_segment = halved_PCG_segment
    final_ECG_segment = halved_ECG_segment
    
    final_PPG_segment = np.array([])
    final_PCG_segment = np.array([])
    final_ECG_segment = np.array([])
    
    for v in halved_PPG_segment:
        final_PPG_segment = np.append(final_PPG_segment, (v-min_ppg)*(1.0/(max_ppg-min_ppg)))
    for w in halved_PCG_segment:
        final_PCG_segment = np.append(final_PCG_segment, (w-min_pcg)*(1.0/(max_pcg-min_pcg)))
    for k in halved_ECG_segment:
        final_ECG_segment = np.append(final_ECG_segment, (k-min_ecg)*(1.0/(max_ecg-min_ecg)))
    RR[0] = (RR[0]-min_rr1)*(1.0/(max_rr1-min_rr1))
    RR[1] = (RR[1]-min_rr2)*(1.0/(max_rr2-min_rr2))
        
    ECG_input = np.array(final_ECG_segment.reshape(ECG_input_shape), dtype=np.float32)
    PPG_input = np.array(final_PPG_segment.reshape(PPG_input_shape), dtype=np.float32)
    PCG_input = np.array(final_PCG_segment.reshape(PCG_input_shape), dtype=np.float32)
    RR0_input = np.array(RR[0].reshape(RR0_input_shape), dtype=np.float32)
    RR1_input = np.array(RR[1].reshape(RR1_input_shape), dtype=np.float32)
    
    interpreter.set_tensor(input_details[0]['index'], ECG_input)
    interpreter.set_tensor(input_details[2]['index'], PPG_input)
    interpreter.set_tensor(input_details[3]['index'], PCG_input)
    interpreter.set_tensor(input_details[4]['index'], RR0_input)
    interpreter.set_tensor(input_details[1]['index'], RR1_input)
    
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    #print(output_data[0][0], BP[TARGET_BP])
    prediction_DBP = np.append(prediction_DBP, output_data[0][0])

#print(np.mean(prediction_SBP), np.mean(prediction_DBP), BP['SBP'][file], BP['DBP'][file], BP['file'][file])
# Blood Pressure Category
pred_sbp = round(np.mean(prediction_SBP), 2) - 15
pred_dbp = round(np.mean(prediction_DBP), 2) 
status = ""
if pred_sbp < 120:
    status = "NORMAL"
elif pred_sbp < 130:
    status = "ELEVATED"
elif pred_sbp < 140:
    status = "HYPERTENSION STAGE1"
elif pred_sbp < 180:
    status = "HYPERTENSION STAGE2"
else:
    status = "HYPERTENSIVE CRISIS"
now = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
print("Time: {}, SBP: {}, DBP: {}, Status: {}".format(now, np.mean(prediction_SBP), np.mean(prediction_DBP), status))
#TODO: write to csv and to lcd dislay
# Write the result to csv
# Format: Time: XXX, SBP: XXX, DBP: XXX
file = open("abp_res.txt", "a")
file.write("Time: {}, SBP: {}, DBP: {}, Status: {}\n".format(now, pred_sbp, pred_dbp, status))
file.close()

with open('result_display.txt','w') as writer:
    writer.write("{},{},{}".format(pred_sbp,pred_dbp,status))

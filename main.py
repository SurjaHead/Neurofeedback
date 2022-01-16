import pandas as pd
from matplotlib import style
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib
import time
import sys
import brainflow
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
from brainflow.ml_model import MLModel, BrainFlowMetrics, BrainFlowClassifiers, BrainFlowModelParams

def main(i):
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    params.board_id = 1
    board_id = 1
    params.serial_port = 'COM5'
    board = BoardShim(board_id, params)
    eeg_channels = BoardShim.get_eeg_channels(board_id)
    sampling_rate = BoardShim.get_sampling_rate(board_id)
    timestamp = BoardShim.get_timestamp_channel(board_id)
    
    board.prepare_session()
    board.start_stream()
    
    style.use('fivethirtyeight')
    plt.title("Live EEG stream from Brainflow", fontsize=15)
    plt.ylabel("Data in millivolts", fontsize=15)
    plt.xlabel("\nTime", fontsize=10)
    keep_alive = True

    eeg1 = [] 
    eeg2 = []
    eeg3 = []
    eeg4 = []
    timex = [] 

    while keep_alive == True:

        while board.get_board_data_count() < 250: 
            time.sleep(0.005)
        data = board.get_current_board_data(250)
        
        eegdf = pd.DataFrame(np.transpose(data[eeg_channels]))
        eegdf_col_names = ["ch1", "ch2", "ch3", "ch4"]
        eegdf.columns = eegdf_col_names
        
        timedf = pd.DataFrame(np.transpose(data[timestamp]))

        print("EEG Dataframe") 
        print(eegdf)           

        for count, channel in enumerate(eeg_channels):
            if count == 0:
                DataFilter.perform_bandstop(data[channel], sampling_rate, 60.0, 4.0, 4,
                                            FilterTypes.BUTTERWORTH.value, 0)  # bandstop 58 - 62
                DataFilter.perform_bandpass(data[channel], sampling_rate, 21.0, 20.0, 4,
                                            FilterTypes.BESSEL.value, 0)  # bandpass 11 - 31
            if count == 1:
                DataFilter.perform_bandstop(data[channel], sampling_rate, 60.0, 4.0, 4,
                                            FilterTypes.BUTTERWORTH.value, 0)  # bandstop 58 - 62
                DataFilter.perform_bandpass(data[channel], sampling_rate, 21.0, 20.0, 4,
                                            FilterTypes.BESSEL.value, 0)  # bandpass 11 - 31
            if count == 2:
                DataFilter.perform_bandstop(data[channel], sampling_rate, 60.0, 4.0, 4,
                                            FilterTypes.BUTTERWORTH.value, 0)  # bandstop 58 - 62
                DataFilter.perform_bandpass(data[channel], sampling_rate, 21.0, 20.0, 4,
                                            FilterTypes.BESSEL.value, 0)  # bandpass 11 - 31
            if count == 3:
                DataFilter.perform_bandstop(data[channel], sampling_rate, 60.0, 4.0, 4,
                                            FilterTypes.BUTTERWORTH.value, 0)  # bandstop 58 - 62
                DataFilter.perform_bandpass(data[channel], sampling_rate, 21.0, 20.0, 4,
                                            FilterTypes.BESSEL.value, 0)  # bandpass 11 - 31

        bands = DataFilter.get_avg_band_powers(
            data, eeg_channels, sampling_rate, True)
        feature_vector = np.concatenate((bands[0], bands[1]))

        concentration_params = BrainFlowModelParams(
            BrainFlowMetrics.CONCENTRATION.value, BrainFlowClassifiers.KNN.value)
        concentration = MLModel(concentration_params)
        concentration.prepare()
        print('Concentration: %f' % concentration.predict(feature_vector))
        concentrated_measure = concentration.predict(feature_vector)
        concentration.release()

        relaxation_params = BrainFlowModelParams(
            BrainFlowMetrics.RELAXATION.value, BrainFlowClassifiers.KNN.value)
        relaxation = MLModel(relaxation_params)
        relaxation.prepare()
        print('Relaxation: %f' % relaxation.predict(feature_vector))
        relaxed_measure = relaxation.predict(feature_vector)
        relaxation.release()
      
        eeg1.extend(eegdf.iloc[:, 0].values) 
        eeg2.extend(eegdf.iloc[:, 1].values) 
        eeg3.extend(eegdf.iloc[:, 2].values) 
        eeg4.extend(eegdf.iloc[:, 3].values)
        timex.extend(timedf.iloc[:, 0].values) 

        plt.cla()
        plt.plot(timex, eeg1, label="Channel 1", color="red")
        plt.plot(timex, eeg2, label="Channel 2", color="blue")
        plt.plot(timex, eeg3, label="Channel 3", color="orange")
        plt.plot(timex, eeg4, label="Channel 4", color="purple")
        plt.tight_layout()
        keep_alive = False 
        if concentrated_measure > 0.5:
            print("GOOD KEEP CONCENTRATING") 
        else:
            print("CONCENTRATE MORE")
        if relaxed_measure > 0.5:
            print("YES KEEP RELAXING")
        else:
            print("RELAX MORE") 
        if concentrated_measure-relaxed_measure==0:
            print('perfect')

    board.stop_stream()
    board.release_session()

ani = FuncAnimation(plt.gcf(), main, interval=1000) 
plt.tight_layout()
plt.autoscale(enable=True, axis="y", tight=True)
plt.show()

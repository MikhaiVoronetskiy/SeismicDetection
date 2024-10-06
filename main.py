import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy.signal.trigger import classic_sta_lta, plot_trigger, trigger_onset
import glob
import os
from datetime import datetime, timedelta








def detect_anomalies(csv_file_path, global_min=-1.4962862873198858e-07, global_max=1.653729495509616e-07, timeframe=1000, sta_len=120, lta_len=1000,  threshold_on=4, threshold_off=1.5, y_step=1e-10):
    """
    Load seismic data from a CSV file.
    Expects columns: 'time_abs(%Y-%m-%dT%H:%M:%S.%f)', 'time_rel(sec)', and 'velocity(m/s)'.
    """
    data = pd.read_csv(csv_file_path)
    data['time_abs'] = pd.to_datetime(data['time_abs(%Y-%m-%dT%H:%M:%S.%f)'], format='%Y-%m-%dT%H:%M:%S.%f')

    times, velocities = data['time_rel(sec)'].values, data['velocity(m/s)'].values

    # Calculate sampling rate (assuming uniform sampling)
    sampling_rate = 1 / (times[1] - times[0])  # in Hz


    """
    Compute the STA/LTA ratio using the classic method.
    :param trace_data: Seismic trace data (e.g., velocity values).
    :param sampling_rate: Sampling rate of the data.
    :param sta_len: Short time window length in seconds.
    :param lta_len: Long time window length in seconds.
    :return: STA/LTA ratio.
    """
    sta_samples = int(sta_len * sampling_rate)
    lta_samples = int(lta_len * sampling_rate)
    cft = classic_sta_lta(velocities, sta_samples, lta_samples)


    # Detect triggers
    on_off = np.array(trigger_onset(cft, threshold_on, threshold_off))

    '''
        Plot the seismic waveform, STA/LTA characteristic function, and detections.
        :param times: Time array.
        :param data: Seismic data array.
        :param detections: Detected on and off times (start and end points of events).
        :param sta_lta_cft: STA/LTA characteristic function.
        :param threshold_on: On trigger threshold.
        :param threshold_off: Off trigger threshold.
        
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # Plot seismic trace
    ax1.plot(times, velocities, label='Seismic Waveform')
    for det in on_off:
        ax1.axvline(times[det[0]], color='red', linestyle='--',
                    label='Event Start' if det[0] == on_off[0][0] else "")
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Velocity (m/s)')
    ax1.set_title('Seismic Waveform with Detected Events')
    ax1.legend()
    ax1.grid()

    # Plot STA/LTA characteristic function
    ax2.plot(times[:len(cft)], cft, label='STA/LTA Ratio')
    ax2.axhline(threshold_on, color='green', linestyle='--', label='Trigger On Threshold')
    ax2.axhline(threshold_off, color='orange', linestyle='--', label='Trigger Off Threshold')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('STA/LTA Ratio')
    ax2.set_title('STA/LTA Characteristic Function')
    ax2.legend()
    ax2.grid()

    plt.tight_layout()
    plt.show()'''

    dictionary_matrices = {}

    # Create and plot data windows for each detected anomaly
    for start, end in on_off:
        """
        Create a 2D binary array representing a data window starting from a given index.
        :param start_index: Index in the array from where to start the data window.
        :param velocities: Array of velocity values.
        :param sampling_rate: Sampling rate of the data.
        :param global_min: Minimum velocity value for the Y range.
        :param global_max: Maximum velocity value for the Y range.
        :param y_step: Step size for velocity values.
        :param timeframe: Total number of seconds for the time window.
        :return: 2D binary array representing the data window.
        """
        # Determine the number of time steps and velocity steps for the window
        time_steps = int(timeframe * sampling_rate)
        velocity_steps = int((global_max - global_min) / y_step)

        # Create an empty 2D array of zeros
        data_window = np.zeros((time_steps, velocity_steps))

        # Loop through each time step in the window
        for i in range(time_steps):
            # Calculate the corresponding index in the velocity array
            index = start + i
            if index < len(velocities):
                # Determine the velocity at this time step
                velocity = velocities[index]

                # Determine the corresponding index in the Y (velocity) range
                if global_min <= velocity <= global_max:
                    velocity_index = int((velocity - global_min) / y_step)
                    # Set the corresponding element in the array to 1
                    data_window[i][velocity_index] = 1

        dictionary_matrices[(csv_file_path.split("/")[-1].strip(".csv"), times[start])] = data_window
    return dictionary_matrices
# Example usage with a CSV file path (replace with your file path)
csv_file_path = 'space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA/xa.s12.00.mhz.1970-03-25HR00_evid00003.csv'  # Replace this with your CSV file path
detection_df = detect_anomalies(csv_file_path)
print(detection_df)



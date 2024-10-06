import numpy
import os

import find_evidence
import main
import matrix_convolution

def getXandY():
    x = []
    y = []
    directory = 'space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA'
    filenames = os.listdir(directory)
    for filename_index in range(1):
        filename = filenames[filename_index]
        print(filename_index / len(filenames) * 100, '%' )
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            anomalies = main.detect_anomalies(file_path)
            print("Anomalies: ")
            for anomaly in anomalies:
                y_element = find_evidence.find_evidence(filename, anomaly)
                print("find_evidence: ")
                x_element = matrix_convolution.matrix_to_vector(matrix_convolution.matrix_convolution(anomalies[anomaly]))
                print("matrix_convolution: ")
                x.append(x_element)
                y.append(y_element)

    return x, y

if __name__ == "__main__":
    x, y = getXandY()
    print(x)
    print(y)
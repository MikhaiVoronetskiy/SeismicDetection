import numpy
import os
import pickle
import find_evidence
import main
import matrix_convolution

def getXandY(size=2, step=1):
    x = []
    y = []
    directory = 'space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA'
    filenames = os.listdir(directory)
    for filename_index in range(len(filenames)):
        filename = filenames[filename_index]
        print(filename_index / len(filenames) * 100, '%' )
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            anomalies = main.detect_anomalies(file_path)
            for anomaly in anomalies:
                y_element = find_evidence.find_evidence(anomaly[0], anomaly[1])
                matrix = anomalies[anomaly]
                new_matrix = []
                for row in matrix:
                    new_row = []
                    for element in row:
                        new_row.append(int(element))
                    new_matrix.append(new_row)
                x_element = matrix_convolution.matrix_to_vector(matrix_convolution.matrix_convolution(new_matrix, step=step, convolution_size=size))
                x.append(x_element)
                y.append(y_element)

    return x, y



def compile_x_y(size=2, step=1):
    x, y = getXandY(size, step)
    dict = {'x': x, 'y': y}
    with open(f'size_{size}_step_{step}.pkl', 'wb') as f:
        pickle.dump(dict, f)



import numpy
import os
import pickle
import find_evidence
import main
import matrix_convolution
import gc

def getXandY(filenames, size=2, step=1):
    x = []
    y = []
    directory = 'space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA'
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
                del matrix
                gc.collect()
                x_element = matrix_convolution.matrix_to_vector(matrix_convolution.matrix_convolution(new_matrix, step=step, convolution_size=size))
                x.append(x_element)
                y.append(y_element)

    return x, y


def get_x(size=2, step=1, directory='space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA'):
    filenames = os.listdir(directory)
    x = []
    for filename_index in range(len(filenames)):
        filename = filenames[filename_index]
        print(filename_index / len(filenames) * 100, '%' )
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            anomalies = main.detect_anomalies(file_path)
            for anomaly in anomalies:
                matrix = anomalies[anomaly]
                new_matrix = []
                for row in matrix:
                    new_row = []
                    for element in row:
                        new_row.append(int(element))
                    new_matrix.append(new_row)
                del matrix
                gc.collect()
                x_element = matrix_convolution.matrix_to_vector(matrix_convolution.matrix_convolution(new_matrix, step=step, convolution_size=size))
                x.append(x_element)

def compile_x(size=5, step=4, compilation_size=2):
    directory = 'space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA'
    compilation_size *=2 # because we have two types of files
    filenames = os.listdir(directory)
    part = 0
    while len(filenames) > 0:
        x = 0
        y = 0
        filenames_part = []
        part += 1
        #if part <= 5:
         #   continue
        #if part > 6:
       #     break
        for i in range(compilation_size):
            if len(filenames) > 0:
                filenames_part.append(filenames.pop())
        x, y = getXandY(filenames_part, size, step)
        dict = {'x': x, 'y': y}
        with open(f'test_data_size_{size}_step_{step}_part_{part}.pkl', 'wb') as f:
            pickle.dump(dict, f)

def compile_x_y(directory='space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA' ,size=5, step=4, compilation_size=4):
    compilation_size *=2 # because we have two types of files
    filenames = os.listdir(directory)
    part = 0
    while len(filenames) > 0:
        x = 0
        y = 0
        filenames_part = []
        #part += 1
        #if part <= 5:
         #   continue
        #if part > 6:
       #     break
        for i in range(compilation_size):
            if len(filenames) > 0:
                filenames_part.append(filenames.pop())
        x, y = getXandY(filenames_part, size, step)
        dict = {'x': x, 'y': y}
        with open(f'size_{size}_step_{step}_part_{part}.pkl', 'wb') as f:
            pickle.dump(dict, f)


compile_x(size=5, step=4, compilation_size=2)
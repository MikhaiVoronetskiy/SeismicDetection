import joblib
import pandas as pd
import os

import ai
import getXandY
import csv

def find_seismic_events(path_to_csv_file):
    x = {}
    replacement_dict = {
        "shallow_mq": [1, 0, 0, 0],
        "impact_mq": [0, 1, 0, 0],
        "deep_mq": [0, 0, 1, 0],
        "noise": [0, 0, 0, 1]
    }
    dict_of_time_and_vectors =getXandY.get_x(directory=path_to_csv_file)

    pipline = joblib.load('model.joblib') # after running teach_model_and_save() there will be a few models created, in this directory please choose one of them
    for time in dict_of_time_and_vectors:
        x = dict_of_time_and_vectors[time]
        y_res = pipline.predict(x)
        for i in replacement_dict:
            if y_res[i] == y_res:
                if i == "noise":
                    continue
                print(i, time)







def teach_model_and_save():
    #x, y = getXandY.getXandY()
    directory = 'space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA' #please replace if directory is different
    getXandY.compile_x_y(directory=directory) #this can take a lot of time
    filenames = os.listdir(os.getcwd())
    xs = []
    ys = []
    for filename_index in range(len(filenames)):
        filename = filenames[filename_index]
        if filename_index > 20:
            break
        if filename.endswith('.pkl') and ('size_5' in filename) and ('step_4' in filename):
            x, y = getXandY.get_x_and_y_from_pickle(filename)
            xs.extend(x)
            ys.extend(y)
    ai.main_ai(xs, ys)


find_seismic_events("space_apps_2024_seismic_detection/data/lunar/test/data/S12_GradeB/xa.s12.00.mhz.1969-12-16HR00_evid00006.csv")
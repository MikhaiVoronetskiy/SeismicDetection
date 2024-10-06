import joblib
import pandas as pd
import os

import ai
import getXandY
import csv

def find_seismic_events(path_to_csv_file):
    x = {}
    type_dict = {
        1: "shallow_mq",
        2: "impact_mq",
        3: "deep_mq",
        0: "noise"

    }
    res_file = "result_catalog.csv"
    filenames = os.listdir(os.getcwd())
    for i in filenames:
        if (i.split(".")[-1] =="csv") and (i[:5] == "test"):
            x.update(getXandY.get_x(i))
    pipline = joblib.load('model.joblib') #insert model name
    y_res = pipline.predict(x.values)
    for j in range(len(y_res)):
        if y_res[j] == 0:
            continue
        typename = type_dict[y_res[j]] # name of type
        time = list(x.keys())[j]
        
    #if not noise print type and time





def teach_model_and_save():
    x, y = getXandY.getXandY()
    directory = 'space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA' #please replace if directory is different
    getXandY.compile_x_y(directory=directory) #this can take a lot of time
    ai.main_ai(x, y)
    #save model

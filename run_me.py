import pandas as pd
import os
import getXandY


def find_seismic_events(path_to_csv_file):
    x = []
    filenames = os.listdir(os.getcwd())
    for i in filenames:
        if (i.split(".")[-1] =="csv") and i[:5] == "test"):
            x.extend(getXandY.get_x_from_pickle(i))
    pipline = joblib.load('model.joblib') #insert model name
    y_res = pipline.predict(x)
    
    #if not noise print type and time





def teach_model_and_save():
    x, y = getXandY.getXandY()
    directory = 'space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA' #please replace if directory is different
    getXandY.compile_x_y(directory=directory) #this can take a lot of time
    #teach model
    #save model

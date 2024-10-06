import pandas as pd
import os
import getXandY


def find_seismic_events(path_to_csv_file):
    x = getXandY.get_x()
    #load model
    #run xs through model
    #if not noise print type and time





def teach_model_and_save():
    x, y = getXandY.getXandY()
    directory = 'space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA' #please replace if directory is different
    getXandY.compile_x_y(directory=directory) #this can take a lot of time
    #teach model
    #save model
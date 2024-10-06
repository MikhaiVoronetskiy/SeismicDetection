import pandas as pd

import main


def find_evidence(filename, time, range_detection=2000):
    csv_file_path = 'space_apps_2024_seismic_detection/data/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv'  # Replace this with your CSV file path
    test = pd.read_csv(csv_file_path)
    filenames = test['filename'].tolist()
    times = test['time_rel(sec)'].tolist()
    types = test['mq_type'].tolist()
    for i in range(len(filenames)):
        if filenames[i] == filename:
            catalog_time = times[i]
            if (catalog_time - time) < range_detection:
                return types[i]
    return "no evidence"


csv_file_path1 = 'space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA/xa.s12.00.mhz.1971-10-18HR00_evid00043.csv'
a = find_evidence("xa.s12.00.mhz.1971-10-18HR00_evid00043",  10740)
print(a)

"""Get statistics after running benchmark.py and analysis.py.
"""
import pandas as pd
import numpy as np

dataset_fns = ['contrast_results.csv', 'resolution_results.csv']
params = ['best', 'min', 'max', 'thresh']
measurements = ['diam', 'count']



# for each dataset:
for fn in dataset_fns:
    s = fn.replace(".csv", "").replace("_", " ")
    print(f"{s.title()}")
    data = pd.read_csv(fn)
    # for the whole dataset:
    for measurement in measurements:
        for param in params:
            col = f"{measurement}_{param}"
            vals = data[col]
            m, std = vals.mean(), vals.std()
            print(f"{measurement} {param}:\t{np.round(m, 4)} +/- {np.round(std, 4)}")
    print()
    # for each folder in the set of folders:
    folder_set = list(set(data.folder))
    for folder in folder_set:
        s = folder.replace("_", " ")
        print(f"{s.title()}")
        # make a dataset with just files in that folder
        folder_inds = data.folder.values == folder
        folder_data = data[folder_inds]
        # get the mean +/- standard deviation of each param per measurements
        for measurement in measurements:
            for param in params:
                col = f"{measurement}_{param}"
                vals = folder_data[col]
                m, std = vals.mean(), vals.std()
                print(f"{measurement} {param}:\t{np.round(m, 3)}\t+/-{np.round(std, 3)}")
        print()
    print()

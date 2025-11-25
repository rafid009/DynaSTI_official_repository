import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

folder = "./topology_plots"
if not os.path.isdir(folder):
    os.makedirs(folder)
train_locations_file = "./data/nacse/X_OR_temps_train_loc.npy" # "./data/pems_bay/X_train_locs.npy" #

train_locations = np.load(train_locations_file)

df_train = pd.DataFrame(train_locations, columns=['longitude', 'latitude', 'elevation'])
df_train.to_csv(f"{folder}/nacse_train_locations.csv", index=False)

test_locations_file = "./data/nacse/X_OR_temps_test_loc.npy"
test_locations = np.load(test_locations_file)

df_test = pd.DataFrame(test_locations, columns=['longitude', 'latitude', 'elevation'])
df_test.to_csv(f"{folder}/nacse_test_locations.csv", index=False)
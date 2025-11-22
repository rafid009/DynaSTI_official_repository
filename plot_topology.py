import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

folder = "./topology_plots"
if not os.path.isdir(folder):
    os.makedirs(folder)
train_locations_file = "./data/awn/zone_8_train_locs.npy" #"./data/nacse/X_OR_temps_train_loc.npy"

train_locations = np.load(train_locations_file)

df_train = pd.DataFrame(train_locations, columns=['longitude', 'latitude', 'elevation'])
df_train.to_csv(f"{folder}/nacse_train_locations.csv", index=False)

test_locations_file = "./data/awn/zone_8_test_locs.npy"
test_locations = np.load(test_locations_file)

df_test = pd.DataFrame(test_locations, columns=['longitude', 'latitude', 'elevation'])
df_test.to_csv(f"{folder}/nacse_test_locations.csv", index=False)
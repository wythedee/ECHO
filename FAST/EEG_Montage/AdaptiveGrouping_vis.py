import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

loc_1005 = pd.read_csv('EEG_Montage/standard_1005_dedup.csv')
scale = 11
loc_128 = pd.read_csv('EEG_Montage/GSN-HydroGel-128.csv')
loc_128['y'] = loc_128['y'] - 1.3
loc_128['x'], loc_128['y'], loc_128['z'] = loc_128['x']*scale, loc_128['y']*scale, loc_128['z']*scale


fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(111, projection='3d')
scatter = ax1.scatter(loc_1005['x'], loc_1005['y'], loc_1005['z'], cmap='hsv', marker='o')
scatter = ax1.scatter(loc_128['x'], loc_128['y'], loc_128['z'],cmap='hsv', marker='x')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
# add name
for i in range(len(loc_1005[:10])):
    ax1.text(loc_1005['x'][i], loc_1005['y'][i], loc_1005['z'][i], loc_1005['name'][i])
for i in range(len(loc_128[:10])):
    ax1.text(loc_128['x'][i], loc_128['y'][i], loc_128['z'][i], loc_128['name'][i])
plt.tight_layout()
plt.show()
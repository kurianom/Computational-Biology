from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

src = Path.cwd()
data_path = src / 'Lesson_18_Project_2_visual_stimulus_parameters.npz'
data_path1 = src / 'Lesson_18_Project_2_visual_responses.npy'

# Upload of data
with np.load(data_path) as par:
    spatial = par['spatial_frequency_by_degree']
    direction = par['direction_in_degrees']
    temporal = par['temporal_frequency_in_Hz']

resp = np.load(data_path1)

# Idea, just for visualization
df_resp = pd.DataFrame(resp)

df_par = pd.DataFrame({
    'spatial_frequency_by_degree': spatial,
    'direction_in_degrees': direction,
    'temporal_frequency_in_Hz': temporal
})

print(df_resp)
print(df_par)

spatial_freqs = np.unique(spatial)
print(spatial_freqs)

# Dictionary to store the matrices
matrici = {}

# Find unique directions and sort them
directions_sorted = np.sort(df_par.loc[:, 'direction_in_degrees'].unique())
print("Directions are", directions_sorted)

# Loop over each spatial frequency
for sf in spatial_freqs:
    # Rows corresponding to this spatial frequency
    rows_sf = df_par.index[df_par['spatial_frequency_by_degree'] == sf].tolist()

    # Create the matrix (384 neurons x 12 directions)
    mat = np.zeros((resp.shape[0], len(directions_sorted)))
    
    # Loop over the directions
    for i, dir_ in enumerate(directions_sorted):
        cols_dir = df_par.index[
            (df_par['spatial_frequency_by_degree'] == sf) & 
            (df_par['direction_in_degrees'] == dir_)
        ].tolist()
        # Average over the repetitions of this stimulus
        mat[:, i] = df_resp.iloc[:, cols_dir].mean(axis=1)
    
    # Save the matrix in the dictionary
    matrici[sf] = mat

# Now 'matrici' contains 3 matrices, keys = spatial frequencies

mat_004 = matrici[0.04]
mat_008 = matrici[0.08]
mat_016 = matrici[0.16]
print("Matrix 0.04 freq with dimensions: ", mat_004.shape)  # should be (384,12)
print(mat_004)
print("Matrix 0.08 freq with dimensions: ", mat_008.shape)  # should be (384,12)
print(mat_008)
print("Matrix 0.16 freq with dimensions: ", mat_016.shape)  # should be (384,12)
print(mat_016)

#example of evolution of neuron "0" freq 0.04
neuron_id = 0
tuning_curve = mat_004[neuron_id, :]
print(tuning_curve)

plt.figure(figsize=(6,4))
plt.plot(directions_sorted, tuning_curve, marker='o')
plt.xlabel("Direction (°)")
plt.ylabel("Response")
plt.title("Direction tuning – Neuron {}".format(neuron_id))
plt.grid(True)
plt.show()

exit()

    

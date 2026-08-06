#!/usr/bin/env python3
import numpy as np
import os

# Create output directory
output_dir = 'PS'
os.makedirs(output_dir, exist_ok=True)

# Load predictions and true values
true_vals = np.load('ps_result/empirical_validation/y_empirical.npy')
test_stations = np.load('ps_result/empirical_validation/test_stations.npy')

# Load predictions (using only the validation part to match true_vals shape)
preds = np.load('ps_result/gru/model_preds.npy')
# Use only the first N elements where N is the length of true_vals
preds = preds[:len(true_vals)]

# Calculate errors
errors = preds - true_vals

# Generate DOY (Day of Year) starting from 1
doy = np.arange(1, len(preds) + 1).reshape(-1, 1)

# Combine data
data = np.hstack([doy, test_stations.reshape(-1, 1), preds, true_vals, errors])

# Output to file
output_file = os.path.join(output_dir, 'ps_all_stations.txt')
with open(output_file, 'w') as f:
    f.write('DOY StationID Predicted True Error\n')
    for row in data:
        f.write(f"{int(row[0])} {row[1]} {row[2]:.6f} {row[3]:.6f} {row[4]:.6f}\n")

print(f"PS data processed, output file: {output_file}")

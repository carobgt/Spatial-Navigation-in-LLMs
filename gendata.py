import scipy.io
import numpy as np
import os
import pandas as pd

def create_synthetic_mat_file(filename, num_trials=5, grid_size=(4, 4)):
    """
    Creates a synthetic .mat file with data for testing the .mat to CSV converter.

    Args:
        filename (str): The name of the .mat file to create (without .mat extension).
        num_trials (int): The number of trials to generate data for.
        grid_size (tuple): The dimensions (rows, cols) of the grid/map.
    """

    # --- Create the output structure ---
    out_data = {}

    # 1. Create the 'map' data
    # The map is a grid of numbers.
    map_data = np.arange(1, grid_size[0] * grid_size[1] + 1).reshape(grid_size)
    map_struct = np.array([(map_data,)], dtype=[('map', 'O')])
    out_data['map'] = map_struct

    # 2. Create 'startLoc' and 'targLoc' data
    # These are single numbers for each trial, representing locations on the map
    startLoc_data = np.random.choice(np.arange(1, grid_size[0] * grid_size[1] + 1), num_trials)
    targLoc_data = np.random.choice(np.arange(1, grid_size[0] * grid_size[1] + 1), num_trials)
    
    # Ensure that start and target locations are not the same
    for i in range(num_trials):
        while startLoc_data[i] == targLoc_data[i]:
            targLoc_data[i] = np.random.choice(np.arange(1, grid_size[0] * grid_size[1] + 1), 1)[0]
    
    startLoc_struct = np.array([(startLoc_data[i],) for i in range(num_trials)], dtype=[('startLoc', 'O')])
    targLoc_struct = np.array([(targLoc_data[i],) for i in range(num_trials)], dtype=[('targLoc', 'O')])
    out_data['startLoc'] = startLoc_struct
    out_data['targLoc'] = targLoc_struct

    # 3. Create 'stepDist' and 'pathLength' data
    # For simplicity, make stepDist always 3, and pathLength either 3 or 5
    stepDist_data = np.full(num_trials, 3)
    pathLength_data = np.random.choice([3, 5], num_trials)
    stepDist_struct = np.array([(stepDist_data[i],) for i in range(num_trials)], dtype=[('stepDist', 'O')])
    pathLength_struct = np.array([(pathLength_data[i],) for i in range(num_trials)], dtype=[('pathLength', 'O')])

    out_data['stepDist'] = stepDist_struct
    out_data['pathLength'] = pathLength_struct
    
    # 4. Create the path
    paths = []
    for i in range(num_trials):
        path = []
        current_location = startLoc_data[i]
        target_location = targLoc_data[i]

        # Find the coordinates of the current and target locations
        current_row, current_col = np.where(map_data == current_location)
        target_row, target_col = np.where(map_data == target_location)
        
        current_row = current_row[0]
        current_col = current_col[0]
        target_row = target_row[0]
        target_col = target_col[0]

        # Move towards the target one step at a time
        if pathLength_data[i] == 3:
            # Move directly towards the target
            for _ in range(pathLength_data[i]):
                if current_row < target_row:
                    current_row +=1
                elif current_row > target_row:
                    current_row -=1
                elif current_col < target_col:
                    current_col +=1
                elif current_col > target_col:
                    current_col -=1
                path.append(map_data[current_row, current_col])
        else:
            # Take a longer path
            for _ in range(pathLength_data[i]):
                if current_row == 0:
                    current_row +=1
                elif current_row == grid_size[0]-1:
                    current_row -=1
                elif current_col == 0:
                    current_col += 1
                elif current_col == grid_size[1]-1:
                    current_col -=1
                else:
                    if np.random.random() > 0.5:
                        current_row +=1
                    else:
                        current_col +=1
                path.append(map_data[current_row, current_col])

        paths.append((np.array(path),))
    
    path_struct = np.array(paths, dtype=[('path', 'O')])
    out_data['path'] = path_struct

    # --- Create the final 'out' structure ---
    # This structure matches what scipy.io.loadmat expects from a .mat file
    field_names = [('map', 'O'), ('startLoc', 'O'), ('targLoc', 'O'), ('stepDist', 'O'), ('pathLength', 'O'), ('path', 'O')]
    structured_out = np.array([(
        out_data['map'], 
        out_data['startLoc'], 
        out_data['targLoc'], 
        out_data['stepDist'],
        out_data['pathLength'],
        out_data['path']
        )], dtype=[('map', 'O'), ('startLoc', 'O'), ('targLoc', 'O'), ('stepDist', 'O'), ('pathLength', 'O'), ('path', 'O')])

    # Save the .mat file
    scipy.io.savemat(filename + '.mat', {'out': structured_out})
    print(f"Created synthetic .mat file: {filename}.mat")

# --- Example usage ---
# Create the input directory if it doesn't exist
input_dir = 'BehaviouralData'
if not os.path.exists(input_dir):
    os.makedirs(input_dir)

# Generate a few synthetic .mat files
create_synthetic_mat_file(os.path.join(input_dir, 'synthetic_data1'), num_trials=7, grid_size = (3,3))
create_synthetic_mat_file(os.path.join(input_dir, 'synthetic_data2'), num_trials=5, grid_size = (4,4))
create_synthetic_mat_file(os.path.join(input_dir, 'synthetic_data3'), num_trials=9, grid_size = (5,5))

# Now you can run your .mat to CSV conversion code:
# (paste your original code here, it will now process these files)

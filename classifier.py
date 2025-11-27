import numpy as np
import os
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import re


## Hyperparameters
n_slices = 155
n_vol = 369
batch_size = 5
directory = "archive/BraTS2020_training_data/content/data"
map_path = "archive/BraTS2020_training_data/content/data/name_mapping.csv"


#######################################################################################
## Data Loader

# Create a list of all .h5 files in the directory
h5_files = [f for f in os.listdir(directory) if f.endswith('.h5')]
print(f"Found {len(h5_files)} .h5 files:\nExample file names:{h5_files[:3]}")

# Open the first .h5 file in the list to inspect its contents
if h5_files:
    file_path = os.path.join(directory, h5_files[25070])
    with h5py.File(file_path, 'r') as file:
        print("\nKeys for each file:", list(file.keys()))
        for key in file.keys():
            print(f"\nData type of {key}:", type(file[key][()]))
            print(f"Shape of {key}:", file[key].shape)
            print(f"Array dtype: {file[key].dtype}")
            print(f"Array max val: {np.max(file[key])}")
            print(f"Array min val: {np.min(file[key])}")
else:
    print("No .h5 files found in the directory.")
  

def count_grades(csv_file):
    # Read the CSV file into a pandas DataFrame
    data = pd.read_csv(csv_file)

    # Count the occurrences of HGG and LGG in the 'Grade' column
    hgg_count = data[data['Grade'] == 'HGG'].shape[0]
    lgg_count = data[data['Grade'] == 'LGG'].shape[0]

    return hgg_count, lgg_count


def map_entries(csv_file):
    # Read the CSV file into a pandas DataFrame
    data = pd.read_csv(csv_file)

    # Create a dictionary mapping BraTS_2020_subject_ID to Grade
    mapping = dict(zip(data['BraTS_2020_subject_ID'], data['Grade']))

    return mapping


def extract_num(filename):
    # Extract the first number from the filename using regular expressions
    match = re.match(r"volume_(\d+)_slice_\d+\.h5", filename)
    if match:
        first_number = int(match.group(1))
        return first_number
    else:
        return -1


def slice_name(string):
    # Find the index where 'archive/BraTS2020_training_data/content/data\\' ends
    index = string.find('archive/BraTS2020_training_data/content/data\\')
    if index != -1:
        # Slice the string to remove the specified part
        return string[index + len('archive/BraTS2020_training_data/content/data\\'):]
    else:
        return string  # Return the original string if the specified part is not found


def format_correction(string_format, number):
    # Format the number to always have three digits
    formatted_number = "{:03d}".format(number)
    # Append the formatted number to the string format
    result = string_format + formatted_number
    return result


class BrainScanDataset(Dataset):
    def __init__(self, file_paths, deterministic=False):
        self.file_paths = file_paths
        if deterministic:  # To always generate the same test images for consistency
            np.random.seed(1)
        np.random.shuffle(self.file_paths)
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # Load h5 file, get image and mask
        file_path = self.file_paths[idx]
        with h5py.File(file_path, 'r') as file:
            image = file['image'][()]
            mask = file['mask'][()]
            
            # Reshape: (H, W, C) -> (C, H, W)
            image = image.transpose((2, 0, 1))
            mask = mask.transpose((2, 0, 1))
            
            # Adjusting pixel values for each channel in the image so they are between 0 and 255
            for i in range(image.shape[0]):    # Iterate over channels
                min_val = np.min(image[i])     # Find the min value in the channel
                image[i] = image[i] - min_val  # Shift values to ensure min is 0
                max_val = np.max(image[i]) + 1e-4     # Find max value to scale max to 1 now.
                image[i] = image[i] / max_val
            
            # Convert to float and scale the whole image
            image = torch.tensor(image, dtype=torch.float32)
            mask = torch.tensor(mask, dtype=torch.float32) 
            
        return image, mask


hgg_count, lgg_count = count_grades(map_path)
id_to_grade = map_entries(map_path)

print("COUNT: ", hgg_count, " ", lgg_count)

# Build .h5 file paths from directory containing .h5 files
h5_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.h5')]

np.random.seed(42)
np.random.shuffle(h5_files)

balanced = []
cap = 0
for unit in h5_files:
    key = format_correction("BraTS20_Training_", extract_num(slice_name(unit)))

    if cap < lgg_count * n_slices:       
        if id_to_grade[key] == "HGG":
            balanced.append(unit)
            cap += 1
    if id_to_grade[key] == "LGG":
            balanced.append(unit)
    
print("LEN OF BALANCED: ", len(balanced))

# Split the dataset into train and validation sets (90:10)
split_idx = int(0.9 * len(balanced))
train_files = balanced[:split_idx]
val_files = balanced[split_idx:]


# Create the train and val datasets
train_dataset = BrainScanDataset(train_files)
val_dataset = BrainScanDataset(val_files, deterministic=True)

# Sample dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Verifying dataloaders work
for images, masks in train_dataloader:
    print("Training batch - Images shape:", images.shape, "Masks shape:", masks.shape)
    break
for images, masks in val_dataloader:
    print("Validation batch - Images shape:", images.shape, "Masks shape:", masks.shape)
    break

#######################################################################################

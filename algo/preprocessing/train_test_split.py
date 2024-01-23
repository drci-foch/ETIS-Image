import os
import random
from typing import Dict, List, Tuple

def list_files(directory: str) -> List[str]:
    """List all files in a directory."""
    return [os.path.join(directory, file) for file in os.listdir(directory)]

def extract_patient_ids(file_paths: List[str]) -> List[str]:
    """Extract patient IDs from file paths."""
    return [os.path.basename(file)[:22] for file in file_paths]

def group_files_by_patient(file_paths: List[str]) -> Dict[str, List[str]]:
    """Group files by patient ID."""
    grouped_files = {}
    for file_path in file_paths:
        patient_id = os.path.basename(file_path)[:22]
        if patient_id not in grouped_files:
            grouped_files[patient_id] = []
        grouped_files[patient_id].append(file_path)
    return grouped_files

def train_test_split(grouped_files: Dict[str, List[str]], test_size: float = 0.2) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Split the data into training and test sets."""
    patient_ids = list(grouped_files.keys())
    random.shuffle(patient_ids)

    split_index = int(len(patient_ids) * (1 - test_size))
    train_ids, test_ids = patient_ids[:split_index], patient_ids[split_index:]

    train_set = {patient_id: grouped_files[patient_id] for patient_id in train_ids}
    test_set = {patient_id: grouped_files[patient_id] for patient_id in test_ids}

    return train_set, test_set

def group_files_by_patient_and_type(file_paths: List[str]) -> Dict[str, Dict[str, List[str]]]:
    """Group files by patient ID and type (SWI, 3DTOF, Mask)."""
    grouped_files = {}
    for file_path in file_paths:
        patient_id = os.path.basename(file_path)[:22]
        file_type = "SWI" if "SWI" in file_path else "3DTOF" if "TOF3D" in file_path else "Mask"
        
        if patient_id not in grouped_files:
            grouped_files[patient_id] = {"SWI": [], "3DTOF": [], "Mask": []}
        
        grouped_files[patient_id][file_type].append(file_path)

    return grouped_files

def save_list_to_file(file_paths: List[str], filename: str):
    """Save a list of file paths to a text file."""
    with open(filename, 'w') as file:
        for path in file_paths:
            file.write(path + '\n')

def train_test_split_XY(grouped_files: Dict[str, Dict[str, List[str]]], test_size: float = 0.2):
    """Split the data into training and test sets for X and Y."""
    patient_ids = list(grouped_files.keys())
    random.shuffle(patient_ids)

    split_index = int(len(patient_ids) * (1 - test_size))
    train_ids, test_ids = patient_ids[:split_index], patient_ids[split_index:]

    X_train = {"SWI": [], "3DTOF": []}
    Y_train = []
    X_test = {"SWI": [], "3DTOF": []}
    Y_test = []

    for patient_id in train_ids:
        for file_type in ["SWI", "3DTOF"]:
            X_train[file_type].extend(grouped_files[patient_id][file_type])
        Y_train.extend(grouped_files[patient_id]["Mask"])

    for patient_id in test_ids:
        for file_type in ["SWI", "3DTOF"]:
            X_test[file_type].extend(grouped_files[patient_id][file_type])
        Y_test.extend(grouped_files[patient_id]["Mask"])

    return X_train, Y_train, X_test, Y_test


# Paths to the directories
tof3d_dir = "E:/data/TOF3D/transformed_images_resliced"
swi_dir = "E:/data/SWI/transformed_images"
mask_dir = "E:/data/MASK/transformed_images"

tof3d_files = list_files(tof3d_dir)
swi_files = list_files(swi_dir)
mask_files = list_files(mask_dir)
all_files = tof3d_files + swi_files + mask_files
grouped_files = group_files_by_patient_and_type(all_files)
X_train, Y_train, X_test, Y_test = train_test_split_XY(grouped_files)


save_list_to_file(X_train["SWI"] + X_train["3DTOF"], "E:/data/X_train_paths.txt")
save_list_to_file(Y_train, "E:/data/Y_train_paths.txt")
save_list_to_file(X_test["SWI"] + X_test["3DTOF"], "E:/data/X_test_paths.txt")
save_list_to_file(Y_test, "E:/data/Y_test_paths.txt")

print("Train and test sets have been saved to files.")
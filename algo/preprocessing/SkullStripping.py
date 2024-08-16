import os
from HD_BET.run import run_hd_bet

def apply_batch_HDBET (source_dir, target_dir, string_to_replace=None, modification_string="", **kwargs):
    if string_to_replace != None:
        file_list = [file for file in os.listdir(source_dir) if (file.endswith(".nii.gz")) & (string_to_replace in file)]
    else:
        file_list = [file for file in os.listdir(source_dir) if file.endswith(".nii.gz")]
    
    for file in file_list:
        split_filename = file.split(".")
        
        if string_to_replace != None:
            split_filename[0] = split_filename[0].replace(string_to_replace, modification_string)
        new_filename = ".".join(split_filename)

        run_hd_bet(os.path.join(source_dir, file), os.path.join(target_dir, new_filename), **kwargs)
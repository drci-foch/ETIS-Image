import os

import nibabel as nib
import numpy as np


def load_nifti_and_slice(filepath, slice_axis=2):
    """
    Load 3D images and slice them on the z axis
    """
    image = nib.load(filepath).get_fdata()
    slices = [image[:, :, z_index] for z_index in range(image.shape[slice_axis])]
    # slices = np.take(image, indices=[i for i in range(image.shape[2])], axis=slice_axis)
    return slices


def has_mask(mask_slice):
    """
    Check for slices where there is a mask
    """
    return np.any(mask_slice > 0)  # Checks for any non-zero value in the mask slice


def save_slice_as_array(slice_img, output_folder, base_filename, slice_index, prefix="image"):
    """
    Save slices as array while :
        - Normalizing pixel into [0,255] bounds (Afterthought : We will do it after)
        - Downscale the precision to 8bits
    """
    # Normalize image slices but save masks as is
    if prefix == "image":
        # normalized_slice = ((slice_img - np.min(slice_img)) / (np.max(slice_img) - np.min(slice_img)) * 255).astype(np.uint8)
        normalized_slice = slice_img.astype(np.uint8)
    else:  # Save mask directly for binary or categorical data
        normalized_slice = slice_img.astype(np.uint8)
    np.save(os.path.join(output_folder, f"{base_filename}_{prefix}_slice_{slice_index}.npy"), normalized_slice)


def process_folder(input_image_folder, output_image_folder, input_mask_folder, output_mask_folder, slice_axis=2):
    """
    Processes automatically all files in a particular folder
    """
    for filename in os.listdir(input_image_folder):
        if filename.endswith('.nii.gz'):
            base_filename = filename[:22]  # Match mask file based on the first 22 characters of the image file name
            mask_filename = None
            for m_filename in os.listdir(input_mask_folder):
                if m_filename.startswith(base_filename):
                    mask_filename = m_filename
                    break

            if mask_filename is None:
                print(f"No corresponding mask file found for {filename}, skipping.")
                continue

            image_filepath = os.path.join(input_image_folder, filename)
            mask_filepath = os.path.join(input_mask_folder, mask_filename)
            
            image_slices = load_nifti_and_slice(image_filepath, slice_axis)
            mask_slices = load_nifti_and_slice(mask_filepath, slice_axis)
            
            for i, (slice_img, mask_slice) in enumerate(zip(image_slices, mask_slices)):
                if has_mask(mask_slice):
                    save_slice_as_array(slice_img, output_image_folder, base_filename, i, prefix="image")
                    save_slice_as_array(mask_slice, output_mask_folder, base_filename, i, prefix="mask")

SWI_PATH = 'E:/data/SWI'
MASK_PATH = 'E:/data/MASK'
SWI_OUTPUT_PATH = 'E:/data/MEDSAM_finetuning/SWI_slice_Images' 
MASK_OUTPUT_PATH = 'E:/data/MEDSAM_finetuning/Masks_slice' 

os.makedirs(SWI_OUTPUT_PATH, exist_ok=True)
os.makedirs(MASK_OUTPUT_PATH, exist_ok=True)

process_folder(input_image_folder=SWI_PATH, output_image_folder=SWI_OUTPUT_PATH, input_mask_folder=MASK_PATH, output_mask_folder=MASK_OUTPUT_PATH, slice_axis=2)

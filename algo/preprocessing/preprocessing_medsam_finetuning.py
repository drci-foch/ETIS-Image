import nibabel as nib
import numpy as np
import os

def load_nifti_and_slice(filepath, slice_axis=2):
    image = nib.load(filepath).get_fdata()
    slices = np.take(image, indices=range(image.shape[slice_axis]), axis=slice_axis)
    return slices

def has_mask(mask_slice):
    return np.any(mask_slice > 0)  # Checks for any non-zero value in the mask slice

def save_slice_as_array(slice_img, output_folder, base_filename, slice_index, prefix="image"):
    # Normalize image slices but save masks as is
    if prefix == "image":
        normalized_slice = ((slice_img - np.min(slice_img)) / (np.max(slice_img) - np.min(slice_img)) * 255).astype(np.uint8)
    else:  # Save mask directly for binary or categorical data
        normalized_slice = slice_img.astype(np.uint8)
    np.save(os.path.join(output_folder, f"{base_filename}_{prefix}_slice_{slice_index}.npy"), normalized_slice)

def process_folder(input_folder, image_output_folder, mask_output_folder, mask_folder, slice_axis=2):
    for filename in os.listdir(input_folder):
        if filename.endswith('.nii.gz'):
            base_filename = filename[:22]  # Match mask file based on the first 22 characters of the image file name
            mask_filename = None
            for m_filename in os.listdir(mask_folder):
                if m_filename.startswith(base_filename):
                    mask_filename = m_filename
                    break

            if mask_filename is None:
                print(f"No corresponding mask file found for {filename}, skipping.")
                continue

            image_filepath = os.path.join(input_folder, filename)
            mask_filepath = os.path.join(mask_folder, mask_filename)
            
            image_slices = load_nifti_and_slice(image_filepath, slice_axis)
            mask_slices = load_nifti_and_slice(mask_filepath, slice_axis)
            
            for i, (slice_img, mask_slice) in enumerate(zip(image_slices, mask_slices)):
                if has_mask(mask_slice):
                    save_slice_as_array(slice_img, image_output_folder, base_filename, i, prefix="image")
                    save_slice_as_array(mask_slice, mask_output_folder, base_filename, i, prefix="mask")

input_folder = 'E:/data/SWI'
mask_folder = 'E:/data/MASK'
image_output_folder = 'E:/data/MEDSAM_fintuning/SWI_slice_Images' 
mask_output_folder = 'E:/data/MEDSAM_fintuning/Masks_slice' 

os.makedirs(image_output_folder, exist_ok=True)
os.makedirs(mask_output_folder, exist_ok=True)

process_folder(input_folder, image_output_folder, mask_output_folder, mask_folder)

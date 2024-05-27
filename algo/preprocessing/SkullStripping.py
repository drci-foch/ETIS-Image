import scipy
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn import preprocessing

def save_array_to_nifti1(array, original_img, destination_path, output_name):
    # Transform the array to a nifti image which requires the affine of the original image.
    processed_img = nib.Nifti1Image(array, original_img.affine)
    nib.save(processed_img, os.path.join(destination_path, output_name))

def Stroke_closing(img):
    # used to close stroke prediction image
    new_img = np.zeros_like(img)
    new_img = scipy.ndimage.binary_closing(img, structure=np.ones((2,2,2)))
    return new_img

def z_score_normalize(array):
    # Z_score normalizing with error handling
    if np.std(array) != 0.0:
        return (array-np.mean(array))/np.std(array)
    else:
        return array

def get_SkullStripped_Mask(model, SWI_img, TOF_img):
    # To inference brain mask from MaskNet model
    # model specifies which pre-trained DL model is used to inference
    
    # Down sampling
    swi = SWI_img[0::4,0::4,0::4,np.newaxis] # Down sample for MaskNet, dim should be [48, 56, 48, 1]

    tof = TOF_img[0::4,0::4,0::4, np.newaxis] # Down sample for MaskNet, dim should be [48, 56, 48, 1]

    # swi_background_value = -np.mean(swi)/np.std(swi)
    # tof_background_value = -np.mean(tof)/np.std(tof)
    
    swi  = z_score_normalize(swi)
    tof  = z_score_normalize(tof)

    # swi[swi == swi_background_value] = 0
    # tof[tof == tof_background_value] = 0
    
    x = np.expand_dims(np.concatenate((swi,tof),axis=3), axis=0)

    # Standardize x before input into the neural network.
    # dim_1, dim_2, dim_3, dim_4, dim_5 = x.shape
    # robust_scaler = preprocessing.RobustScaler()
    # x_rescaled = standard_scaler.fit_transform(x.reshape(dim_2,-1)).reshape(dim_1,dim_2,dim_3,dim_4,dim_5)
    
    # inference
    y_pred = model.predict(x, verbose=0)
    y_pred = (np.squeeze(y_pred)>0.5)*1.0

    
    # the following is post processing of predicted mask by 
    # 1) selecting the major non-zero voxel
    # 2) closing
    # 3) binary fill holes
    # 4) upsampling to high resolution space by (4,4,4)
    
    mask_label, num_features = scipy.ndimage.label(y_pred)
    dilate_mask = (mask_label == scipy.stats.mode(mask_label[mask_label>0].flatten(), keepdims=True)[0][0])*1
    dilate_mask = Stroke_closing(dilate_mask)
    dilate_mask = scipy.ndimage.binary_fill_holes(dilate_mask)
    upsampling_mask = np.repeat(np.repeat(np.repeat(dilate_mask, 4, axis=0), 4, axis=1), 4, axis=2)

    return upsampling_mask

def apply_skullstripping(model, SWI_folder_path, TOF_folder_path, mask_destination_path, swi_img_destination_path, tof_img_destination_path):
    swi_files = os.listdir(SWI_folder_path)
    tof_files = os.listdir(TOF_folder_path)

    # Select files to process.
    swi_nifti_files = [file for file in swi_files if file.endswith('.nii.gz')]
    tof_nifti_files = [file for file in tof_files if file.endswith('.nii.gz')]

    for swi_file, tof_file in zip(swi_nifti_files, tof_nifti_files):
        swi_file_path = os.path.join(SWI_folder_path, swi_file)
        tof_file_path = os.path.join(TOF_folder_path, tof_file)
        swi_nii_img = nib.load(swi_file_path)
        swi_nii_data = swi_nii_img.get_fdata()
        tof_nii_img = nib.load(tof_file_path)
        tof_nii_data = tof_nii_img.get_fdata()

        mask_name = os.path.splitext(os.path.splitext(swi_file)[0])[0] + "_" + "Mask" + ".nii.gz"
        new_swi_img_name = os.path.splitext(os.path.splitext(swi_file)[0])[0] + "_" + "SkullStripped" + ".nii.gz"
        new_tof_img_name = os.path.splitext(os.path.splitext(tof_file)[0])[0] + "_" + "SkullStripped" + ".nii.gz"

        mask_data_bool = get_SkullStripped_Mask(model, swi_nii_data, tof_nii_data)
        mask_data = mask_data_bool.astype("float64")
        save_array_to_nifti1(mask_data, swi_nii_img, mask_destination_path, mask_name)
        
        skullstripped_swi_data = mask_data * swi_nii_data
        skullstripped_tof_data = mask_data * tof_nii_data

        save_array_to_nifti1(skullstripped_swi_data, tof_nii_img, swi_img_destination_path, new_swi_img_name)
        save_array_to_nifti1(skullstripped_tof_data, swi_nii_img, tof_img_destination_path, new_tof_img_name)
        
        print(f"Processed SWI image {swi_file} and TOF image {tof_file}")
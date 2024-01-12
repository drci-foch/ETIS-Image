import os
import nibabel as nib
from nibabel.processing import resample_from_to

def niftireslice(fixed_nii_path: str, moving_nii_path: str, moving_resliced_nii_path: str):
    """Reslice a nifti volume into another reference nifti space

    Keyword arguments:
        fixed_nii_path -- reference nifti path
        moving_nii_path -- path of nifti to reslice
        moving_resliced_nii_path -- path of new created resliced nifti path
    """

    img_fixed = nib.load(fixed_nii_path)
    img_moving = nib.load(moving_nii_path)
    
    img_moving_resampled = resample_from_to(img_moving, (img_fixed.shape[:3], img_fixed.affine))
    data_moving_resampled = img_moving_resampled.get_fdata()
    
    img_moving_resampled2 = nib.Nifti1Image(data_moving_resampled.astype('int16'), img_fixed.affine)
    img_moving_resampled2.set_data_dtype('int16')

    nib.save(img_moving_resampled2, moving_resliced_nii_path)


if __name__ == "__main__":
    data_folder = 'C:/Users/benysar/Desktop/Github/stroke-occlusion/data/envoi-20231207'  # Path to the main data directory

    for patient_folder in os.listdir(data_folder):
        patient_folder_path = os.path.join(data_folder, patient_folder)
        
        if os.path.isdir(patient_folder_path):
            swi_folder_path = os.path.join(patient_folder_path, 'SWI')
            swi_files = [f for f in os.listdir(swi_folder_path) if f.endswith('.nii.gz')]
            
            if swi_files:
                swi_path = os.path.join(swi_folder_path, swi_files[0])  # Assuming only one SWI file in the folder
            
                tof3d_folder_path = os.path.join(patient_folder_path, 'TOF3D')
                tof3d_files = [f for f in os.listdir(tof3d_folder_path) if f.endswith('.nii.gz')]
                
                if tof3d_files:
                    tof3d_path = os.path.join(tof3d_folder_path, tof3d_files[0])  # Assuming only one TOF3D file in the folder
                
                    resliced_path = os.path.join(patient_folder_path, 'TOF3D', f'{patient_folder}_resliced.nii.gz')
                    
                    niftireslice(fixed_nii_path=swi_path, moving_nii_path=tof3d_path, moving_resliced_nii_path=resliced_path)
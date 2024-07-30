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
    swi_folder_path = 'E:/data/SWI/transformed_images'
    tof3d_folder_path = 'E:/data/TOF3D/transformed_images'
    tof3d_resliced_folder_path = 'E:/data/TOF3D/transformed_images_resliced'

    if not os.path.exists(tof3d_resliced_folder_path):
        os.makedirs(tof3d_resliced_folder_path)

    swi_files = [f for f in os.listdir(swi_folder_path) if f.endswith('.nii.gz')]
    tof3d_files = [f for f in os.listdir(tof3d_folder_path) if f.endswith('.nii.gz')]

    for swi_file, tof3d_file in zip(swi_files, tof3d_files):
        swi_path = os.path.join(swi_folder_path, swi_file)
        tof3d_path = os.path.join(tof3d_folder_path, tof3d_file)
        resliced_path = os.path.join(tof3d_resliced_folder_path, f'{tof3d_file}_resliced.nii.gz')

        niftireslice(fixed_nii_path=swi_path, moving_nii_path=tof3d_path, moving_resliced_nii_path=resliced_path)
import sys 
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np

def niftireslice(fixed_nii_path:str, moving_nii_path:str, moving_resliced_nii_path:str):
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
    
    img_moving_resampled2 = nib.Nifti1Image(data_moving_resampled.astype('int16'),  img_fixed.affine)
    img_moving_resampled2.set_data_dtype('int16')

    nib.save(img_moving_resampled2, moving_resliced_nii_path)

if __name__ == "__main__":
    if (len(sys.argv)!=4):
        print("USAGE coreg <fixed_nii_path> <moving_nii_path> <output_moving_resampled_nii_path>")
    else:
        niftireslice(fixed_nii_path=sys.argv[1], moving_nii_path=sys.argv[2], moving_resliced_nii_path=sys.argv[3])
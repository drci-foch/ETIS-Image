import os
import numpy as np
import torchio as tio
import nibabel as nib

class ImagePaths:
    """
    A class representing a collection of image paths.

    Attributes:
        folder_path (str): The path to the folder containing the images.
        image_paths (List[str]): A list of image paths in the folder.

    Methods:
        get_image_paths(): Returns a list of image paths in the folder.
    """
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.image_paths = []

    def get_image_paths(self):
        for filename in os.listdir(self.folder_path):
            if filename.endswith('.nii.gz'):
                self.image_paths.append(os.path.join(self.folder_path, filename))
        return self.image_paths

class StatImageShape:
    """
    A class representing statistics about the shape of a collection of images.

    Attributes:
        folder_path (str): The path to the folder containing the images.
        w (float): The weight given to the median shape when computing the balanced shape.

    Methods:
        get_shapes(): Returns a list of shapes from a collection of images.
        calculate_statistics(): Computes various statistical properties of the shapes.
    """

    def __init__(self, folder_path, w=0.5):
        self.folder_path = folder_path
        self.w = w

    def get_shapes(self):
        shapes = []
        for filename in os.listdir(self.folder_path):
            if filename.endswith('.nii.gz'):
                file_path = os.path.join(self.folder_path, filename)
                try:
                    image = nib.load(file_path)
                    shapes.append(image.shape)
                except Exception as e:
                    print(f"Error loading file {filename}: {e}")
        return shapes

    def calculate_statistics(self):
        shapes = self.get_shapes()
        shapes = np.array(shapes)
        if shapes.size == 0:
            return None
        min_shape = np.min(shapes, axis=0).astype(int)
        median_shape = np.median(shapes, axis=0).astype(int)
        avg_shape = np.average(shapes, axis=0).astype(int)
        max_shape = np.max(shapes, axis=0).astype(int)
        balanced_shape = (self.w * np.array(list(median_shape)) + (1 - self.w) * np.array(list(avg_shape))).astype(int)
        return min_shape, median_shape, avg_shape, max_shape, balanced_shape

class DatasetTransformer:
    """
    A class representing a set of transformations that can be applied to a collection of images.

    Attributes:
        image_paths (List[str]): A list of image paths in the folder.
        balanced_shape (Tuple[int]): The balanced shape of the images.

    Methods:
        load_subjects(): Returns a list of TIO `Subject` objects corresponding to the input images.
        apply_transforms(): Applies a sequence of TIO transforms to the input subjects.
    """

    def __init__(self, image_paths, balanced_shape):
        self.image_paths = image_paths
        self.balanced_shape = balanced_shape

    def load_subjects(self):
        subjects = []
        for image_path in self.image_paths:
            subject = tio.Subject(
                mri=tio.ScalarImage(image_path),
            )
            subjects.append(subject)
        return subjects

    def apply_transforms(self, subjects):
        transforms = tio.Compose([
            #tio.ToCanonical(),
            tio.Resample((1)),
            tio.Resize(target_shape=[512,512,60],image_interpolation="bspline"),
            tio.RescaleIntensity(out_min_max=(1)), 
            #tio.Affine(scales=1.0, degrees=5, translation=5, image_interpolation='linear'),
            tio.ZNormalization(),
        ])
        transformed_subjects = tio.SubjectsDataset(subjects, transform=transforms)
        return transformed_subjects

def save_transformed_images(transformed_dataset, output_dir):
    """
    Saves the transformed images in the input dataset to disk.

    Parameters:
        transformed_dataset (TIO SubjectsDataset): A dataset of transformed images.
        output_dir (str): The directory where the images should be saved.
    """

    os.makedirs(output_dir, exist_ok=True)
    for i, subject in enumerate(transformed_dataset):
        transformed_image = subject['mri']
        original_filename = os.path.basename(image_paths[i])
        name_prefix = original_filename[:22]
        output_filename = os.path.join(output_dir, f"{name_prefix}_transformed.nii.gz")
        transformed_image.save(output_filename)

if __name__ == "__main__":
    folder_path = 'E:/data/MASK/'
    image_paths = ImagePaths(folder_path=folder_path)
    image_paths = image_paths.get_image_paths()
    print("Number of images :",len(image_paths))

    stat_image_shape = StatImageShape(folder_path=folder_path)
    min_shape, median_shape, avg_shape, max_shape, balanced_shape = stat_image_shape.calculate_statistics()
    print("Minimum Image Shape:", list(min_shape))
    print("Median Image Shape:", list(median_shape))
    print("Average Image Shape:", list(avg_shape))
    print("Maximum Image Shape:", list(max_shape))
    print("Balanced Image Shape:", list(balanced_shape))

    dataset_transformer = DatasetTransformer(image_paths, balanced_shape)
    subjects = dataset_transformer.load_subjects()
    dataset_transformed = dataset_transformer.apply_transforms(subjects)

    output_dir = os.path.join(folder_path, 'transformed_images')
    save_transformed_images(transformed_dataset=dataset_transformed, output_dir=output_dir)
    print("Transformed images saved to:", output_dir)

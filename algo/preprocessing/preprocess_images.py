import marimo

__generated_with = "0.14.12"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from Image_Resizing import tuple_product, save_array_to_nifti1, apply_processing_to_img_folder, resize_images_to_moving_reference
    return (resize_images_to_moving_reference,)


@app.cell
def _(resize_images_to_moving_reference):
    resize_images_to_moving_reference("E:\\Data_Booster\\data_Tours_NoOcclusion99\\TOF3D", "E:\\Data_Booster\\data_Tours_NoOcclusion99\\Resized_Images\\SWI_asReference_ImagebyImage\\TOF3D", "E:\\Data_Booster\\data_Tours_NoOcclusion99\\SWI", interpolation_method="lanczos", modification_string="")
    return


@app.cell
def _():
    from SkullStripping import apply_batch_HDBET
    return (apply_batch_HDBET,)


@app.cell
def _(apply_batch_HDBET):
    apply_batch_HDBET("C:\\Users\\wijflo\\.pyenv\\pyenv-win\\versions\\3.11.9\\Scripts\\hd-bet.exe", "E:\\Data_Booster\\data_Tours_NoOcclusion99\\SWI", "E:\\Data_Booster\\data_Tours_NoOcclusion99\\SkullStripping\\SWI_Skullstripped", modification_string="")
    return


@app.cell
def _(apply_batch_HDBET):
    apply_batch_HDBET("C:\\Users\\wijflo\\.pyenv\\pyenv-win\\versions\\3.11.9\\Scripts\\hd-bet.exe", "E:\\Data_Booster\\data_Tours_NoOcclusion99\\Resized_Images\\SWI_asReference_ImagebyImage\\TOF3D", "E:\\Data_Booster\\data_Tours_NoOcclusion99\\SkullStripping\\TOF3D_Skullstripped", modification_string="")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

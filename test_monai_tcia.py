#!/usr/bin/env python3
"""
MONAI Model Zoo with TCIA Data Example

This script demonstrates:
1. Downloading data from TCIA
2. Loading and processing DICOM and DICOM-SEG files
3. Using MONAI Model Zoo for segmentation
4. Visualizing and comparing results
"""

import os
import sys
import glob
import numpy as np
import torch

# Apply the patch to fix compatibility between pydicom 3.0.1 and pydicom_seg
import pydicom_seg_patch
pydicom_seg_patch.apply_patch()

# Import TCIA utilities
from tcia_utils import nbia
import pandas as pd

# Import ITK for DICOM reading
import itk

# Import DICOM SEG handling
import pydicom
import pydicom_seg

# Import MONAI components
from monai.data import decollate_batch
from monai.bundle import ConfigParser, download

def download_tcia_data():
    """Download data from TCIA"""
    # Download a "Shared Cart" that has been previously created via the NBIA website
    cartName = "nbia-17571668146714049"
    
    # Retrieve cart metadata
    cart_data = nbia.getSharedCart(cartName)
    
    # Download the series_uids list and return dataframe of metadata
    df = nbia.downloadSeries(cart_data, format="df")
    
    print(f"Downloaded {len(df)} series from TCIA")
    return df

def load_dicom_data(df):
    """Load DICOM data from downloaded files"""
    dicom_data_dir = "tciaDownload"
    
    # The series_uid defines their directory where the MR data was stored on disk
    mr_series_uid = df.at[df.Modality.eq('MR').idxmax(), 'Series UID']
    mr_dir = os.path.join(dicom_data_dir, mr_series_uid)
    
    # Read the DICOM MR series' objects and reconstruct them into a 3D ITK image
    mr_image = itk.imread(mr_dir, itk.F)
    
    # The series_uid defines where the RTSTRUCT was stored on disk
    seg_series_uid = df.at[df.Modality.eq('SEG').idxmax(), 'Series UID']
    seg_dir = os.path.join(dicom_data_dir, seg_series_uid)
    seg_file = glob.glob(os.path.join(seg_dir, "*.dcm"))[0]
    
    # Read the DICOM SEG object using pydicom and pydicom_seg
    seg_dicom = pydicom.dcmread(seg_file)
    seg_reader = pydicom_seg.MultiClassReader()
    seg_obj = seg_reader.read(seg_dicom)
    
    # Convert the DICOM SEG object into an itk image
    seg_image = itk.GetImageFromArray(seg_obj.data.astype(np.float32))
    seg_image.CopyInformation(mr_image)
    
    return mr_image, seg_image, mr_dir, seg_dir

def prostate_crop(img):
    """Crop image for prostate model"""
    boundary = [int(crop_size*0.2) for crop_size in img.GetLargestPossibleRegion().GetSize()]
    new_image = itk.CropImageFilter(Input=img, BoundaryCropSize=boundary)
    return new_image

def prepare_monai_model():
    """Download and prepare MONAI model"""
    model_name = "prostate_mri_anatomy"
    model_version = "0.3.1"
    zoo_dir = os.path.abspath("./models")
    
    download(name=model_name, version=model_version, bundle_dir=zoo_dir)
    
    # Return the model configuration and paths
    return {
        "model_name": model_name,
        "zoo_dir": zoo_dir,
        "output_dir": os.path.abspath("./monai_results"),
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    }

def run_inference(model_info, mr_dir, seg_dir, mr_image_prep, seg_image_prep):
    """Run inference with MONAI model"""
    # Save preprocessed images to disk
    itk.imwrite(mr_image_prep, mr_dir + ".nii.gz")
    itk.imwrite(seg_image_prep, seg_dir + ".nii.gz")
    
    # Parse the model config file
    model_config_file = os.path.join(model_info["zoo_dir"], model_info["model_name"], "configs", "inference.json")
    model_config = ConfigParser()
    model_config.read_config(model_config_file)
    
    # Update the config variables
    model_config["bundle_root"] = model_info["zoo_dir"]
    model_config["output_dir"] = model_info["output_dir"]
    
    # Load the model checkpoint
    checkpoint = os.path.join(model_info["zoo_dir"], model_info["model_name"], "models", "model.pt")
    
    # Parse model components
    preprocessing = model_config.get_parsed_content("preprocessing")
    model = model_config.get_parsed_content("network").to(model_info["device"])
    inferer = model_config.get_parsed_content("inferer")
    postprocessing = model_config.get_parsed_content("postprocessing")
    
    # Set up dataloader
    datalist = [mr_dir + ".nii.gz"]
    model_config["datalist"] = datalist
    dataloader = model_config.get_parsed_content("dataloader")
    
    # Load model weights
    model.load_state_dict(torch.load(checkpoint, map_location=model_info["device"]))
    model.eval()
    
    # Run inference
    results = []
    with torch.no_grad():
        for d in dataloader:
            images = d["image"].to(model_info["device"])
            d["pred"] = inferer(images, network=model)
            results.append([postprocessing(i) for i in decollate_batch(d)])
    
    return results

def compare_results(model_info, mr_image_prep, seg_image_prep, mr_dir):
    """Compare model results with expert segmentation"""
    # Read the result image
    result_image = itk.imread(os.path.join(model_info["output_dir"], 
                                           os.path.split(mr_dir)[1], 
                                           os.path.split(mr_dir)[1] + "_trans.nii.gz"))
    
    # Resample the result image to match the input
    interpolator = itk.NearestNeighborInterpolateImageFunction.New(seg_image_prep)
    result_image_resampled = itk.resample_image_filter(
        Input=result_image,
        Interpolator=interpolator,
        reference_image=seg_image_prep,
        use_reference_image=True
    )
    
    # Create comparison array
    result_array = itk.GetArrayViewFromImage(result_image_resampled)
    expert_array = itk.GetArrayViewFromImage(seg_image_prep)
    
    # Create a comparison image
    # 1 = ideal prostate, but model called non-prostate (red)
    # 2 = model called prostate, but ideal called non-prostate (purple)
    # 3 = model and ideal agreed (green)
    compare_model_expert = np.where(result_array!=1, 0, 2) + np.where(expert_array!=2, 0, 1)
    compare_image = itk.GetImageFromArray(compare_model_expert.astype(np.float32))
    compare_image.CopyInformation(seg_image_prep)
    
    return result_image_resampled, compare_image

def main():
    print("Starting MONAI-TCIA example...")
    
    # Download data from TCIA
    print("Downloading data from TCIA...")
    df = download_tcia_data()
    
    # Load DICOM data
    print("Loading DICOM data...")
    mr_image, seg_image, mr_dir, seg_dir = load_dicom_data(df)
    
    # Preprocess images
    print("Preprocessing images...")
    mr_image_prep = prostate_crop(mr_image)
    seg_image_prep = prostate_crop(seg_image)
    
    # Prepare MONAI model
    print("Preparing MONAI model...")
    model_info = prepare_monai_model()
    
    # Run inference
    print("Running inference...")
    results = run_inference(model_info, mr_dir, seg_dir, mr_image_prep, seg_image_prep)
    
    # Compare results
    print("Comparing results...")
    result_image, compare_image = compare_results(model_info, mr_image_prep, seg_image_prep, mr_dir)
    
    print("Done! Results saved to:", model_info["output_dir"])
    print("Note: For visualization, run this in a Jupyter notebook with itkwidgets")

if __name__ == "__main__":
    main()

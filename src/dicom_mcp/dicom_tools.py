"""
DICOM Tools Module for Model Context Protocol
Provides tools for loading, processing, and extracting information from DICOM files.
"""
import os
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pydicom
import itk
from pydicom.dataset import Dataset, FileDataset
from pydantic import BaseModel, Field

try:
    # Try to import pydicom_seg which might need patching
    import pydicom_seg
except ImportError as e:
    if "_storage_sopclass_uids" in str(e):
        # Apply the necessary patch for pydicom 3.x compatibility
        from pydicom import uid
        import sys
        import importlib.util
        
        # Create a mock module for backward compatibility
        spec = importlib.util.find_spec("pydicom")
        module = importlib.util.module_from_spec(spec)
        setattr(module, "_storage_sopclass_uids", type('obj', (object,), {
            'SegmentationStorage': uid.SegmentationStorage
        }))
        sys.modules["pydicom._storage_sopclass_uids"] = module._storage_sopclass_uids
        
        # Now try importing again
        import pydicom_seg

# Models for API
class DicomSeries(BaseModel):
    """Model representing a DICOM series"""
    series_uid: str
    study_uid: Optional[str] = None
    modality: str
    description: Optional[str] = None
    path: str
    file_count: int
    patient_id: Optional[str] = None
    
class DicomSlice(BaseModel):
    """Model representing metadata for a single DICOM slice"""
    instance_uid: str
    series_uid: str
    slice_location: Optional[float] = None
    slice_thickness: Optional[float] = None
    row_pixel_spacing: Optional[float] = None
    column_pixel_spacing: Optional[float] = None
    rows: int
    columns: int
    window_center: Optional[float] = None
    window_width: Optional[float] = None
    
class DicomImage(BaseModel):
    """Model representing a 3D image constructed from DICOM slices"""
    series_uid: str
    shape: Tuple[int, int, int]
    spacing: Tuple[float, float, float]
    origin: Tuple[float, float, float]
    direction: List[float]
    intensity_min: float
    intensity_max: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

# DICOM Processing Functions
def scan_directory(directory: str) -> List[DicomSeries]:
    """Scan a directory for DICOM files and organize them into series"""
    series_dict = {}
    
    # Find all DICOM files (recursively if needed)
    dicom_files = []
    for ext in [".dcm", ".DCM", ""]:  # Empty string for no extension files
        dicom_files.extend(glob.glob(os.path.join(directory, f"**/*{ext}"), recursive=True))
    
    for file_path in dicom_files:
        try:
            ds = pydicom.dcmread(file_path, stop_before_pixels=True)
            series_uid = ds.SeriesInstanceUID
            
            if series_uid not in series_dict:
                series_dict[series_uid] = {
                    "files": [],
                    "study_uid": getattr(ds, "StudyInstanceUID", None),
                    "modality": getattr(ds, "Modality", "Unknown"),
                    "description": getattr(ds, "SeriesDescription", None),
                    "patient_id": getattr(ds, "PatientID", None)
                }
            
            series_dict[series_uid]["files"].append(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    # Convert to DicomSeries objects
    result = []
    for series_uid, data in series_dict.items():
        # Use the directory of the first file as the series path
        series_path = os.path.dirname(data["files"][0]) if data["files"] else ""
        
        result.append(DicomSeries(
            series_uid=series_uid,
            study_uid=data["study_uid"],
            modality=data["modality"],
            description=data["description"],
            path=series_path,
            file_count=len(data["files"]),
            patient_id=data["patient_id"]
        ))
    
    return result

def load_dicom_image(series_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load a DICOM series into a 3D numpy array and metadata dictionary"""
    try:
        # Use ITK to read the DICOM series
        itk_image = itk.imread(series_path, itk.F)
        
        # Convert to numpy array
        np_array = itk.GetArrayFromImage(itk_image)
        
        # Extract metadata
        size = itk_image.GetLargestPossibleRegion().GetSize()
        spacing = itk_image.GetSpacing()
        origin = itk_image.GetOrigin()
        direction = itk_image.GetDirection().GetVnlMatrix().as_matrix().flatten().tolist()
        
        metadata = {
            "size": (int(size[0]), int(size[1]), int(size[2])),
            "spacing": (float(spacing[0]), float(spacing[1]), float(spacing[2])),
            "origin": (float(origin[0]), float(origin[1]), float(origin[2])),
            "direction": direction
        }
        
        return np_array, metadata
    except Exception as e:
        raise RuntimeError(f"Failed to load DICOM image: {e}")

def load_dicom_seg(seg_file_path: str, reference_image=None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load a DICOM SEG file into a 3D numpy array and metadata dictionary"""
    try:
        # Read the DICOM SEG object
        seg_dicom = pydicom.dcmread(seg_file_path)
        seg_reader = pydicom_seg.MultiClassReader()
        seg_obj = seg_reader.read(seg_dicom)
        
        # Convert to numpy array
        np_array = seg_obj.data.astype(np.float32)
        
        # Extract metadata including segment information
        metadata = {
            "segment_info": []
        }
        
        # Add segment information if available
        if hasattr(seg_obj, "segment_infos"):
            for idx, segment in enumerate(seg_obj.segment_infos):
                segment_data = {
                    "label": idx + 1,
                    "name": getattr(segment, "SegmentDescription", f"Segment {idx+1}"),
                    "algorithm_type": getattr(segment, "SegmentAlgorithmType", "UNKNOWN")
                }
                metadata["segment_info"].append(segment_data)
        
        # If reference image is provided, copy its metadata
        if reference_image is not None:
            itk_seg_image = itk.GetImageFromArray(np_array)
            itk_seg_image.CopyInformation(reference_image)
            
            # Update numpy array with potentially transformed data
            np_array = itk.GetArrayFromImage(itk_seg_image)
            
            # Extract updated metadata
            size = itk_seg_image.GetLargestPossibleRegion().GetSize()
            spacing = itk_seg_image.GetSpacing()
            origin = itk_seg_image.GetOrigin()
            direction = itk_seg_image.GetDirection().GetVnlMatrix().as_matrix().flatten().tolist()
            
            metadata.update({
                "size": (int(size[0]), int(size[1]), int(size[2])),
                "spacing": (float(spacing[0]), float(spacing[1]), float(spacing[2])),
                "origin": (float(origin[0]), float(origin[1]), float(origin[2])),
                "direction": direction
            })
        
        return np_array, metadata
    except Exception as e:
        raise RuntimeError(f"Failed to load DICOM SEG: {e}")

def extract_dicom_metadata(dicom_file: str) -> Dict[str, Any]:
    """Extract comprehensive metadata from a DICOM file"""
    try:
        ds = pydicom.dcmread(dicom_file, stop_before_pixels=True)
        
        # Create a clean dictionary of metadata
        metadata = {}
        
        # Patient information
        metadata["patient"] = {
            "id": getattr(ds, "PatientID", None),
            "name": str(getattr(ds, "PatientName", "")),
            "birth_date": getattr(ds, "PatientBirthDate", None),
            "sex": getattr(ds, "PatientSex", None)
        }
        
        # Study information
        metadata["study"] = {
            "instance_uid": getattr(ds, "StudyInstanceUID", None),
            "id": getattr(ds, "StudyID", None),
            "date": getattr(ds, "StudyDate", None),
            "time": getattr(ds, "StudyTime", None),
            "description": getattr(ds, "StudyDescription", None)
        }
        
        # Series information
        metadata["series"] = {
            "instance_uid": getattr(ds, "SeriesInstanceUID", None),
            "number": getattr(ds, "SeriesNumber", None),
            "date": getattr(ds, "SeriesDate", None),
            "time": getattr(ds, "SeriesTime", None),
            "description": getattr(ds, "SeriesDescription", None),
            "modality": getattr(ds, "Modality", None)
        }
        
        # Image information
        metadata["image"] = {
            "sop_instance_uid": getattr(ds, "SOPInstanceUID", None),
            "sop_class_uid": getattr(ds, "SOPClassUID", None),
            "instance_number": getattr(ds, "InstanceNumber", None),
            "rows": getattr(ds, "Rows", None),
            "columns": getattr(ds, "Columns", None),
            "pixel_spacing": getattr(ds, "PixelSpacing", None),
            "slice_thickness": getattr(ds, "SliceThickness", None),
            "slice_location": getattr(ds, "SliceLocation", None),
            "window_center": getattr(ds, "WindowCenter", None),
            "window_width": getattr(ds, "WindowWidth", None)
        }
        
        # Equipment information
        metadata["equipment"] = {
            "manufacturer": getattr(ds, "Manufacturer", None),
            "model": getattr(ds, "ManufacturerModelName", None),
            "software_versions": getattr(ds, "SoftwareVersions", None)
        }
        
        return metadata
    except Exception as e:
        raise RuntimeError(f"Failed to extract DICOM metadata: {e}")

def crop_image(image: np.ndarray, boundary_percentage: float = 0.2) -> np.ndarray:
    """Crop an image by removing the specified percentage from each boundary"""
    if boundary_percentage <= 0 or boundary_percentage >= 0.5:
        raise ValueError("Boundary percentage must be between 0 and 0.5")
    
    shape = image.shape
    boundary = [int(s * boundary_percentage) for s in shape]
    
    # Calculate slice indices
    slices = tuple(slice(b, s - b) for b, s in zip(boundary, shape))
    
    # Return cropped image
    return image[slices]

def crop_itk_image(image: Any, boundary_percentage: float = 0.2) -> Any:
    """Crop an ITK image by removing the specified percentage from each boundary"""
    size = image.GetLargestPossibleRegion().GetSize()
    boundary = [int(s * boundary_percentage) for s in size]
    return itk.CropImageFilter(Input=image, BoundaryCropSize=boundary)

import os

import numpy as np
import torch

def load_constant_mask_synthetic(
    patch_size,
    folder_path="/home/hk-project-epais/ke4365/pangu-weather/constant_masks/",
):
    """
    Load the constant masks applied in the patch embedding layer.

    patch_size: Tuple(int, int, int)
        Number of pixels in (vert, lat, lon) dimensions per patch
    folder_path: String
        Path to directory containing constant masks

    Returns
    -------
    land_mask: Tensor
        of shape (n_lat, n_lon) after padding
    soil_type: Tensor
        of shape (n_lat, n_lon) after padding
    topography: Tensor
        of shape (n_lat, n_lon) after padding
    """
    
    land_mask = torch.randn((724, 1443)).to(torch.float32)
    soil_type = torch.randn((724, 1443)).to(torch.float32)
    topography = torch.randn((724, 1443)).to(torch.float32)

    return land_mask, soil_type, topography


def load_constant_mask_2d_synthetic(
    patch_size,
    folder_path="/home/hk-project-epais/ke4365/pangu-weather/constant_masks/",
):
    """
    Load the constant masks applied in the patch embedding layer.

    patch_size: Tuple(int, int, int)
        Number of pixels in (vert, lat, lon) dimensions per patch
    folder_path: String
        Path to directory containing constant masks

    Returns
    -------
    land_mask: Tensor
        of shape (n_lat, n_lon) after padding
    soil_type: Tensor
        of shape (n_lat, n_lon) after padding
    topography: Tensor
        of shape (n_lat, n_lon) after padding
    """
    
    land_mask = torch.randn((724, 1443)).to(torch.float32)
    soil_type = torch.randn((724, 1443)).to(torch.float32)
    topography = torch.randn((724, 1443)).to(torch.float32)

    return land_mask, soil_type, topography

def load_constant_mask(
    patch_size,
    folder_path="/home/hk-project-epais/ke4365/pangu-weather/constant_masks/",
):
    """
    Load the constant masks applied in the patch embedding layer.

    patch_size: Tuple(int, int, int)
        Number of pixels in (vert, lat, lon) dimensions per patch
    folder_path: String
        Path to directory containing constant masks

    Returns
    -------
    land_mask: Tensor
        of shape (n_lat, n_lon) after padding
    soil_type: Tensor
        of shape (n_lat, n_lon) after padding
    topography: Tensor
        of shape (n_lat, n_lon) after padding
    """
    # Load data from numpy files
    data_files = [f for f in os.listdir(folder_path) if f.endswith(".npy")]
    data = {}
    for file in data_files:
        file_path = os.path.join(folder_path, file)
        data[file] = np.load(file_path)

    soil_type = data["soil_type.npy"]
    topography = data["topography.npy"]

    soil_type = (soil_type - np.mean(soil_type)) / np.std(soil_type)
    topography = (topography - np.mean(topography)) / np.std(topography)
    # Torch tensors
    land_mask = torch.tensor(data["land_mask.npy"]).to(torch.float32)
    soil_type = torch.tensor(soil_type).to(torch.float32)
    topography = torch.tensor(topography).to(torch.float32)

    # Check that the shapes of all the data are the same
    assert (
        land_mask.shape == soil_type.shape == topography.shape
    ), "Shapes of the three constant masks are not equal."

    # Now that the shapes are equal, use land_mask as the actual shapes
    x1_pad = (patch_size[1] - (land_mask.shape[0] % patch_size[1])) % patch_size[1] // 2
    x2_pad = (patch_size[1] - (land_mask.shape[0] % patch_size[1])) % patch_size[
        1
    ] - x1_pad
    y1_pad = (patch_size[2] - (land_mask.shape[1] % patch_size[2])) % patch_size[2] // 2
    y2_pad = (patch_size[2] - (land_mask.shape[1] % patch_size[2])) % patch_size[
        2
    ] - y1_pad

    # Apply padding according to patch embedding size
    # Pad the same way as input shape (ensure code is cohesive)
    land_mask = torch.nn.functional.pad(
        land_mask, pad=(y1_pad, y2_pad, x1_pad, x2_pad), mode="constant", value=0
    )
    soil_type = torch.nn.functional.pad(
        soil_type, pad=(y1_pad, y2_pad, x1_pad, x2_pad), mode="constant", value=0
    )
    topography = torch.nn.functional.pad(
        topography, pad=(y1_pad, y2_pad, x1_pad, x2_pad), mode="constant", value=0
    )

    return land_mask, soil_type, topography


def load_constant_mask_2d(
    patch_size,
    folder_path="/home/hk-project-epais/ke4365/pangu-weather/constant_masks/",
):
    """
    Load the constant masks applied in the patch embedding layer.

    patch_size: Tuple(int, int)
        Number of pixels in (lat, lon) dimensions per patch
    folder_path: String
        Path to directory containing constant masks

    Returns
    -------
    land_mask: Tensor
        of shape (n_lat, n_lon) after padding
    soil_type: Tensor
        of shape (n_lat, n_lon) after padding
    topography: Tensor
        of shape (n_lat, n_lon) after padding
    """
    # Load data from numpy files
    data_files = [f for f in os.listdir(folder_path) if f.endswith(".npy")]
    data = {}
    for file in data_files:
        file_path = os.path.join(folder_path, file)
        data[file] = np.load(file_path)

    soil_type = data["soil_type.npy"]
    topography = data["topography.npy"]

    soil_type = (soil_type - np.mean(soil_type)) / np.std(soil_type)
    topography = (topography - np.mean(topography)) / np.std(topography)
    # Torch tensors
    land_mask = torch.tensor(data["land_mask.npy"]).to(torch.float32)
    soil_type = torch.tensor(soil_type).to(torch.float32)
    topography = torch.tensor(topography).to(torch.float32)

    # Check that the shapes of all the data are the same
    assert (
        land_mask.shape == soil_type.shape == topography.shape
    ), "Shapes of the three constant masks are not equal."

    # Now that the shapes are equal, use land_mask as the actual shapes
    x1_pad = (patch_size[0] - (land_mask.shape[0] % patch_size[0])) % patch_size[0] // 2
    x2_pad = (patch_size[0] - (land_mask.shape[0] % patch_size[0])) % patch_size[
        0
    ] - x1_pad
    y1_pad = (patch_size[1] - (land_mask.shape[1] % patch_size[1])) % patch_size[1] // 2
    y2_pad = (patch_size[1] - (land_mask.shape[1] % patch_size[1])) % patch_size[
        1
    ] - y1_pad

    # Apply padding according to patch embedding size
    # Pad the same way as input shape (ensure code is cohesive)
    land_mask = torch.nn.functional.pad(
        land_mask, pad=(y1_pad, y2_pad, x1_pad, x2_pad), mode="constant", value=0
    )
    soil_type = torch.nn.functional.pad(
        soil_type, pad=(y1_pad, y2_pad, x1_pad, x2_pad), mode="constant", value=0
    )
    topography = torch.nn.functional.pad(
        topography, pad=(y1_pad, y2_pad, x1_pad, x2_pad), mode="constant", value=0
    )

    return land_mask, soil_type, topography
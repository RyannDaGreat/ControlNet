# Author: Ryan Burgert 2025

"""
ControlNet Annotators

This module provides annotators for generating control maps for ControlNet models.
Each annotator creates a specialized map (edge, depth, pose, etc.) that can be used
to guide the image generation process.

Several functions are made to wrap Annotators.
You can simply call them! You only need to specify the device on the first call.
They input and output images as defined by rp.is_image

Annotators:
- CannyDetector: Creates edge maps using Canny edge detection algorithm
- HEDdetector: Creates soft edge maps using Holistically-Nested Edge Detection
- MidasDetector: Creates depth maps for 3D structure understanding
- MLSDdetector: Detects line segments for structural guidance
- OpenposeDetector: Detects human poses including body and hand positions
- UniformerDetector: Creates semantic segmentation maps

Note: All annotators except CannyDetector support specifying a custom device.
"""

import rp
import sys
import functools

# This codebase's imports assume the ControlNet repo is the root
sys.path.append(rp.get_parent_directory(__file__, 2))

# Import all annotator functions
from .canny import CannyDetector
from .hed import HEDdetector, nms
from .midas import MidasDetector
from .openpose import OpenposeDetector
from .uniformer import UniformerDetector
from .mlsd import MLSDdetector #Not used right now!

# Define all available annotators
__all__ = [
    'run_hed',
    'run_midas',
    'run_midas_normals',
    'run_openpose',
    'run_uniformer',
]

def _is_divisible_by_patch_size(image, patch_size):
    """Private helper to check if image dimensions are divisible by patch_size"""
    dims = rp.get_image_dimensions(image)
    return all(dim % patch_size == 0 for dim in dims)

def _round_to_nearest_patch_size(patch_size=32):
    """Decorator that ensures image dimensions are multiples of patch_size during processing, but restores the image size afterwards"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(image, device=None, **kwargs):
            original_dims = rp.get_image_dimensions(image)
            
            # If already divisible, just run the function
            if _is_divisible_by_patch_size(image, patch_size):
                return func(image, device, **kwargs)
            
            # Otherwise resize, process, and resize back
            target_dims = tuple(x - x % patch_size for x in original_dims)
            resized_image = rp.cv_resize_image(image, target_dims)
            result = func(resized_image, device, **kwargs)
            return rp.cv_resize_image(result, original_dims)
        
        return wrapper
    return decorator

def run_hed(image, device=None, *, threshold=None, sigma=None):
    assert (threshold is None) == (sigma is None), 'Either specify both threshold AND sigma for non-maximum-suppression, or dont specify either one'
    output = HEDdetector(device)(image)
    if threshold is not None and sigma is not None:
        output = nms(output, threshold, sigma)
    return output

@_round_to_nearest_patch_size(32)
def run_midas(image, device=None):
    depth, normals = MidasDetector(device)(image)
    return depth

@_round_to_nearest_patch_size(32)
def run_midas_normals(image, device=None):
    depth, normals = MidasDetector(device)(image)
    return normals

def run_openpose(image, device=None, *, hands=False):
    return OpenposeDetector(device)(image, hand=hands)[0]

def run_uniformer(image, device=None, *):
    return UniformerDetector(device)(image)


# Author: Ryan Burgert 2025

"""
ControlNet Annotators

This module provides annotators for generating control maps for ControlNet models.
Each annotator creates a specialized map (edge, depth, pose, etc.) that can be used
to guide the image generation process.

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

# This codebase's imports assume the ControlNet repo is the root
sys.path.append(rp.get_parent_directory(__file__, 2))

# Import all annotator functions
from .canny import CannyDetector
from .hed import HEDdetector
from .midas import MidasDetector
from .mlsd import MLSDdetector
from .openpose import OpenposeDetector
from .uniformer import UniformerDetector

# Define all available annotators
__all__ = [
    'CannyDetector',
    'HEDdetector',
    'MidasDetector',
    'MLSDdetector',
    'OpenposeDetector',
    'UniformerDetector'
]

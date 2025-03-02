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
    """
    Runs HED edges on an image as defined by rp.is_image and returns a numpy image.
    Set 0<=threshold<=1 and sigma>=0 for nonmaximum suppression.
    Set device to use a specific GPU of your choice, otherwise it will choose automatically.
    """
    assert (threshold is None) == (sigma is None), 'Either specify both threshold AND sigma for non-maximum-suppression, or dont specify either one'
    output = HEDdetector(device)(image)
    if threshold is not None and sigma is not None:
        output = nms(output, threshold*255, sigma)
    return output

@_round_to_nearest_patch_size(32)
def run_midas(image, device=None):
    """
    Runs MIDAS monocular depth estimation on an image as defined by rp.is_image and returns a numpy image.
    Set device to use a specific GPU of your choice, otherwise it will choose automatically.
    """
    depth, normals = MidasDetector(device)(image)
    return depth

@_round_to_nearest_patch_size(32)
def run_midas_normals(image, device=None):
    """
    Estimates image normals via MIDAS monocular depth estimation on an image as defined by rp.is_image and returns a numpy image.
    Set device to use a specific GPU of your choice, otherwise it will choose automatically.
    """
    depth, normals = MidasDetector(device)(image)
    return normals

def run_openpose(image, device=None, *, hands=False):
    """
    Estimates the pose of people in an image, optionally with their hands too on a given image as defined by rp.is_image. Returns an RGB numpy image.
    Set device to use a specific GPU of your choice, otherwise it will choose automatically.
    """
    return OpenposeDetector(device)(image, hand=hands)[0]

def run_uniformer(image, device=None, *):
    """
    Returns a segmentation map as an RGB numpy image from a given image as defined by rp.is_image.
    Set device to use a specific GPU of your choice, otherwise it will choose automatically.
    """
    return UniformerDetector(device)(image)

def run_annotator_demo(*images):
    """ Run this function as-is to demo the annotator """

    if not images:
        images = [
            'https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png?20111103160805',
            'https://vincentloy.wordpress.com/wp-content/uploads/2010/09/tragic-1-city-skyscrapers.jpg?w=640',
            'https://thumbs.dreamstime.com/b/amazed-surprised-young-woman-light-clothes-hold-hands-yoga-gesture-relaxing-meditating-isolated-white-wall-amazed-142818153.jpg',
        ]

    for image in images:
        if isinstance(image, str):
            image = rp.load_image(image, use_cache=True)

        display_eta = rp.eta(12, 'Creating Images')
        def append(output, label):
            display_eta(len(images)+1)
            images.append(rp.labeled_image(output, label, font='G:Roboto', size=30, background_color='dark random teal'))

        images=[]

        # Get various annotator outputs
        append(                  image                        , 'input image'                  )
        append(run_uniformer    (image                       ), 'run_uniformer'                )
        append(run_openpose     (image                       ), 'run_openpose'                 )
        append(run_openpose     (image, hand=True            ), 'run_openpose hand=True'       )
        append(run_midas_normals(image                       ), 'run_midas_normals'            )
        append(run_midas        (image                       ), 'run_midas'                    )
        append(run_hed          (image, threshold=10, sigma=5), 'run_hed threshold=.1,sigma=10')
        append(run_hed          (image, threshold= 0, sigma=5), 'run_hed threshold=.5,sigma= 0')
        append(run_hed          (image                       ), 'run_hed'                      )
        append(rp.auto_canny    (image                       ), 'rp.auto_canny'                )
        append(rp.auto_canny    (rp.cv_box_blur(image, 5)    ), 'rp.auto_canny box-sigma=5'    )
        append(rp.auto_canny    (rp.cv_box_blur(image,10)    ), 'rp.auto_canny box-sigma=10'   )

        #Save and display
        output = rp.tiled_images(images))
        fansi_print("SAVED IMAGE:" + rp.get_absolute_path(rp.save_image(output, rp.get_unique_copy_path("annotator_demo_output.jpg"))), 'green bold')
        rp.display_image(output)

if __name__ == '__main__':
    run_annotator_demo()

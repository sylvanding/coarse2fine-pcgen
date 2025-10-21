import os
from typing import List, Optional, Tuple

import imageio
import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter


def render_to_2d_image(
    points: np.ndarray,
    pixel_size: float,
    psf_sigma: float,
    image_size: Optional[Tuple[int, int]] = None,
    volume_dims: Optional[List[float]] = None,
    z_range: Optional[Tuple[float, float]] = None,
    intensity_scale: float = 1.0,
    add_background_noise: bool = False,
    background_noise_level: float = 0.05,
    output_dtype=np.uint16,
    normalize: bool = True,
) -> np.ndarray:
    """
    Render 3D point cloud to 2D image by applying Gaussian blur to simulate microscope point spread function (PSF).

    Args:
        points (np.ndarray): 3D point cloud array of shape (N, 3) (x, y, z), units in nanometers.
        pixel_size (float): Size of each pixel, units in nanometers (e.g., 100 nm/pixel).
        psf_sigma (float): Standard deviation of Gaussian PSF, units in nanometers.
        image_size (Optional[Tuple[int, int]]): Output image dimensions (width, height), units in pixels.
                                               If None, calculated from volume_dims and pixel_size.
        volume_dims (Optional[List[float]]): Dimensions of simulation volume [x, y, z], units in nanometers.
                                             Required if image_size is not provided.
        z_range (Optional[Tuple[float, float]]): A (min, max) tuple for filtering points by Z depth.
                                                 If None, includes all points.
        intensity_scale (float): Factor to scale point intensity before blurring.
        add_background_noise (bool): If True, add Gaussian noise to the final image. Defaults to False.
        background_noise_level (float): Standard deviation for Gaussian noise, relative to max intensity.
                                        Effective only when `add_background_noise` is True. Defaults to 0.05.
        output_dtype: Output image data type (e.g., np.uint8, np.uint16).
        normalize (bool): If True, translate point cloud to start at [0, 0, 0]. Defaults to True.

    Returns:
        np.ndarray: Rendered 2D image as NumPy array.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Input `points` must be an array of shape (N, 3).")
    
    # Normalize point cloud to start at origin if requested
    if normalize and points.shape[0] > 0:
        min_coords = np.min(points, axis=0)
        points = points - min_coords

    if image_size is None:
        if volume_dims is None:
            raise ValueError("Must provide either `image_size` or `volume_dims`.")
        # Calculate image dimensions from volume dimensions
        image_width = int(np.ceil(volume_dims[0] / pixel_size))
        image_height = int(np.ceil(volume_dims[1] / pixel_size))
        image_size = (image_width, image_height)

    img_height, img_width = image_size[1], image_size[0]

    # Filter points by Z depth if Z range is specified
    if z_range:
        mask = (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])
        points = points[mask]

    if points.shape[0] == 0:
        if not add_background_noise:
            return np.zeros((img_height, img_width), dtype=output_dtype)
        # If noise is requested, generate a noise-only image
        canvas = np.zeros((img_height, img_width), dtype=np.float32)

    else:
        # Convert point coordinates from nanometers to pixel coordinates
        # We assume the volume origin (0,0) corresponds to the top-left corner of the image.
        coords_px = points[:, :2] / pixel_size

        # Create a floating-point canvas for rendering
        canvas = np.zeros((img_height, img_width), dtype=np.float32)

        # Get integer pixel coordinates and check boundaries
        x_px = np.floor(coords_px[:, 0]).astype(int)
        y_px = np.floor(coords_px[:, 1]).astype(int)

        valid_indices = (x_px >= 0) & (x_px < img_width) & (y_px >= 0) & (y_px < img_height)
        x_px, y_px = x_px[valid_indices], y_px[valid_indices]

        # Add point intensities to canvas
        # Use np.add.at for efficient addition at specified indices
        np.add.at(canvas, (y_px, x_px), intensity_scale)

    # Apply Gaussian blur to simulate PSF
    # PSF sigma value must be converted from nanometers to pixels
    sigma_pixels = psf_sigma / pixel_size
    blurred_image = gaussian_filter(canvas, sigma=sigma_pixels)

    # Normalize to [0, 1]
    max_intensity = blurred_image.max()
    if max_intensity > 0:
        normalized_image = blurred_image / max_intensity
    else:
        normalized_image = blurred_image  # This is a zero image

    # Add background noise if enabled
    if add_background_noise:
        noise = np.random.normal(loc=0.0, scale=background_noise_level, size=normalized_image.shape).astype(np.float32)
        normalized_image += noise

    # Clip to [0, 1] range to handle noise that pushes values out of bounds
    np.clip(normalized_image, 0, 1, out=normalized_image)

    # Scale to output data type
    max_val = np.iinfo(output_dtype).max
    scaled_image = normalized_image * max_val

    return scaled_image.astype(output_dtype)


def save_image(image: np.ndarray, output_path: str):
    """
    Save image to specified path. Format is inferred from file extension.
    Supported formats include PNG, TIFF, JPG, etc. For 3D images, only TIFF is supported.

    Args:
        image (np.ndarray): Image data.
        output_path (str): Path to save the image, including extension (e.g., 'image.png').
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Handle 3D TIFFs separately using the tifffile library directly
    if image.ndim == 3:
        if output_path.lower().endswith((".tiff", ".tif")):
            tifffile.imwrite(output_path, image)
        else:
            raise ValueError("For 3D images, only TIFF format is supported.")
    else:
        imageio.imwrite(output_path, image)

    print(f"Image saved to {output_path}")

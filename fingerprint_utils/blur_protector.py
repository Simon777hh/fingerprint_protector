"""
Blur-based fingerprint protector - lightweight alternative to diffusion model
No network required, fast, with smooth edge blending
"""

import cv2
import numpy as np

class BlurProtector:
    def __init__(self):
        """Initialize blur protector"""
        pass

    def protect(self, image, mask, blur_strength=31, **kwargs):
        """
        Apply Gaussian blur to masked areas with smooth edge blending

        Args:
            image: BGR numpy array (original image)
            mask: uint8 mask (0-255), higher values = more blur
            blur_strength: kernel size for Gaussian blur (odd number)

        Returns:
            protected_image: BGR numpy array with blurred fingerprint regions
        """
        # Apply Gaussian blur to the entire image
        blurred = cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)

        # Normalize mask to [0, 1] range
        mask_float = mask.astype(np.float32) / 255.0

        # Ensure mask is 3-channel for RGB blending
        if len(mask_float.shape) == 2:
            mask_float = np.expand_dims(mask_float, axis=2)

        # Blend: original where mask is 0, blurred where mask is high
        # Formula: result = original * (1 - mask) + blurred * mask
        protected = (image * (1 - mask_float) + blurred * mask_float).astype(np.uint8)

        return protected

    def unload(self):
        """Clean up (nothing to do for blur protector)"""
        pass
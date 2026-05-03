import cv2
import numpy as np


class MaskGenerator:
    def __init__(self, base_size=30, blur_radius=25):
        """
        Initialize mask generator

        Args:
            base_size: Base ellipse size in pixels (0.5x smaller than before)
            blur_radius: Gaussian blur radius for edge feathering
        """
        self.base_size = base_size
        self.blur_radius = blur_radius

    def _get_finger_size(self, hand_landmarks, tip_idx, mcp_idx, image_shape):
        """
        Calculate finger size based on distance from camera (z-axis)
        Returns: scale factor (0.5 to 1.5) and ellipse axes
        """
        tip = hand_landmarks.landmark[tip_idx]
        mcp = hand_landmarks.landmark[mcp_idx]

        # Use z-depth to determine distance (lower z = closer to camera)
        z_depth = tip.z
        scale = 1.0 - z_depth * 1.0
        scale = max(0.5, min(1.5, scale))

        # Also consider finger length as additional indicator
        h, w = image_shape[:2]
        tip_x, tip_y = int(tip.x * w), int(tip.y * h)
        mcp_x, mcp_y = int(mcp.x * w), int(mcp.y * h)
        finger_length = np.sqrt((tip_x - mcp_x) ** 2 + (tip_y - mcp_y) ** 2)

        length_scale = min(1.5, max(0.7, finger_length / 80.0))

        final_scale = scale * length_scale
        final_scale = max(0.6, min(1.6, final_scale))

        axes = int(self.base_size * final_scale)
        return axes, final_scale

    def create_mask(self, image, hand_landmarks, finger_indices):
        """
        Create elliptical mask with size based on finger distance

        Args:
            image: BGR numpy array
            hand_landmarks: MediaPipe hand landmarks object
            finger_indices: List of fingertip indices (e.g., [8, 12])

        Returns:
            mask: uint8 array (0-255), smooth gradient from center to edge
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)

        # Offset ratio: move center from fingertip toward MCP by this percentage of finger length
        # Adjust between 0.3-0.5 for best results
        OFFSET_RATIO = 0.07

        # Map fingertip indices to MCP (knuckle) indices for size calculation
        tip_to_mcp = {
            4: 2,  # thumb
            8: 5,  # index
            12: 9,  # middle
            16: 13,  # ring
            20: 17  # pinky
        }

        for tip_idx in finger_indices:
            tip = hand_landmarks.landmark[tip_idx]
            cx = int(tip.x * w)
            cy = int(tip.y * h)

            # Get MCP index for size calculation and offset direction
            mcp_idx = tip_to_mcp.get(tip_idx, tip_idx - 3)

            if mcp_idx >= 0:
                mcp = hand_landmarks.landmark[mcp_idx]
                mcp_x = int(mcp.x * w)
                mcp_y = int(mcp.y * h)

                # Calculate direction vector from fingertip to MCP (toward finger base)
                dir_x = mcp_x - cx
                dir_y = mcp_y - cy

                # Move center point along the direction toward the finger base
                offset_x = int(dir_x * OFFSET_RATIO)
                offset_y = int(dir_y * OFFSET_RATIO)

                # Apply offset to center
                center_x = cx + offset_x
                center_y = cy + offset_y

                # Calculate finger length for scaling
                finger_length = np.sqrt(dir_x ** 2 + dir_y ** 2)
                length_scale = min(1.5, max(0.7, finger_length / 80.0))

                # Calculate ellipse size
                axes, scale = self._get_finger_size(hand_landmarks, tip_idx, mcp_idx, image.shape)
                axes = int(axes * length_scale)

                # Calculate finger orientation angle
                angle = np.degrees(np.arctan2(dir_y, dir_x))
            else:
                center_x = cx
                center_y = cy
                axes = self.base_size
                scale = 1.0
                angle = 0
                finger_length = 0

            # Ellipse axes: longer along finger direction to cover more fingerprint area
            major_axis = int(axes * 1.4)  # Increased to cover more area
            minor_axis = int(axes * 0.8)

            # Bound check to ensure center stays within image
            center_x = max(0, min(w, center_x))
            center_y = max(0, min(h, center_y))

            # Create elliptical gradient mask
            rad = np.radians(angle)
            cos_a, sin_a = np.cos(rad), np.sin(rad)

            # Calculate region to iterate
            y_min = max(0, center_y - major_axis)
            y_max = min(h, center_y + major_axis)
            x_min = max(0, center_x - major_axis)
            x_max = min(w, center_x + major_axis)

            for y in range(y_min, y_max):
                dy = y - center_y
                for x in range(x_min, x_max):
                    dx = x - center_x

                    # Rotate to ellipse orientation
                    rx = dx * cos_a + dy * sin_a
                    ry = -dx * sin_a + dy * cos_a

                    # Ellipse equation
                    ellipse_val = (rx * rx) / (major_axis * major_axis) + \
                                  (ry * ry) / (minor_axis * minor_axis)

                    if ellipse_val < 1.0:
                        # Gaussian falloff: center=1, edge approaches 0
                        alpha = np.exp(-ellipse_val * 2.5)
                        mask[y, x] = max(mask[y, x], alpha)

            print(f"Finger {tip_idx}: size={axes}px, scale={scale:.2f}, offset applied")

        # Apply Gaussian blur for edge feathering
        if self.blur_radius > 0:
            mask = cv2.GaussianBlur(mask, (self.blur_radius, self.blur_radius),
                                    self.blur_radius // 3)

        # Convert to uint8 for saving (0-255)
        mask_uint8 = (mask * 255).astype(np.uint8)

        return mask_uint8

    def save_debug_mask(self, mask, path='debug_mask.jpg'):
        """Save mask for debugging"""
        cv2.imwrite(path, mask)
        non_zero = np.sum(mask > 0)
        print(f"Debug mask saved to {path} (non-zero pixels: {non_zero})")
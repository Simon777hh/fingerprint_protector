"""
Fingerprint Privacy Protection System
Main entry point - uses blur protector instead of diffusion model
"""

import cv2
import os
from fingerprint_utils import HandFeatureExtractor, GestureClassifier, MaskGenerator, BlurProtector


def capture_from_camera(output_path="captured_image.jpg"):
    """
    Capture an image from camera

    Args:
        output_path: Path to save captured image

    Returns:
        image: BGR numpy array or None if failed
    """
    print("No image found. Opening camera...")

    # Open camera (0 = default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open camera")
        return None

    # Optional: Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Press SPACE to capture, ESC to quit")
    print("Make sure your hand is clearly visible...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame")
            break

        # Show instructions on frame
        display = frame.copy()
        cv2.putText(display, "Press SPACE to capture, ESC to quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, "Show your hand gesture in the frame",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Camera - Fingerprint Protector", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # SPACE key
            cv2.imwrite(output_path, frame)
            print(f"Image captured and saved to: {output_path}")
            break
        elif key == 27:  # ESC key
            print("Capture cancelled by user")
            cap.release()
            cv2.destroyAllWindows()
            return None

    cap.release()
    cv2.destroyAllWindows()

    # Load the captured image
    image = cv2.imread(output_path)
    return image


def protect_image(image_path=None, output_path=None, use_camera_fallback=True):
    """
    Main function: detect risky gestures and protect fingerprint regions

    Args:
        image_path: Path to input image (optional)
        output_path: Path to save protected image (optional)
        use_camera_fallback: If True and image_path doesn't exist, use camera

    Returns:
        protected_image: BGR numpy array or None if failed
    """
    # Initialize components
    feature_extractor = HandFeatureExtractor()
    gesture_classifier = GestureClassifier()
    mask_generator = MaskGenerator(base_size=60, blur_radius=51)
    protector = BlurProtector()  # Using blur instead of diffusion

    # Load image
    image = None

    if image_path and os.path.exists(image_path):
        image = cv2.imread(image_path)
        if image is not None:
            print(f"Loading image from: {image_path}")

    # Fallback to camera if no valid image
    if image is None:
        if use_camera_fallback:
            captured_image = capture_from_camera("captured_image.jpg")
            if captured_image is not None:
                image = captured_image
                image_path = "captured_image.jpg"  # Update for output naming
            else:
                print("Failed to capture image")
                feature_extractor.close()
                return None
        else:
            print(f"Cannot read image: {image_path}")
            feature_extractor.close()
            return None

    print(f"Processing: {image_path if image_path else 'captured_image.jpg'}")

    # Extract features and hand landmarks
    features, hand_landmarks = feature_extractor.extract_features_from_array(image)

    if features is None:
        print("No hand detected")
        feature_extractor.close()
        return image

    # Predict gesture and get exposed fingers
    finger_indices = gesture_classifier.get_exposed_fingers(features)

    if not finger_indices:
        print("No protection needed for this gesture")
        feature_extractor.close()
        if output_path:
            cv2.imwrite(output_path, image)
        else:
            # Auto generate output path
            base = os.path.splitext(image_path)[0] if image_path else "captured"
            output_path = f"{base}_protected.jpg"
            cv2.imwrite(output_path, image)
            print(f"No protection needed. Image saved to: {output_path}")
        return image

    # Generate mask for fingerprint regions
    mask = mask_generator.create_mask(image, hand_landmarks, finger_indices)
    mask_generator.save_debug_mask(mask)

    # Apply protection (blur)
    print("Applying blur protection...")
    protected = protector.protect(image, mask, blur_strength=51)

    # Save result
    if output_path is None:
        base = os.path.splitext(image_path)[0] if image_path else "captured"
        output_path = f"{base}_protected.jpg"

    cv2.imwrite(output_path, protected)
    print(f"Protected image saved to: {output_path}")

    # Cleanup
    feature_extractor.close()

    return protected


def protect_with_camera_only(output_path="protected_capture.jpg"):
    """
    Simplified function: directly capture from camera and protect

    Args:
        output_path: Path to save protected image

    Returns:
        protected_image: BGR numpy array or None if failed
    """
    return protect_image(image_path=None, output_path=output_path, use_camera_fallback=True)


if __name__ == "__main__":
    import sys

    # Default: try selfie.jpg if no argument provided
    default_image = "selfie.jpg"

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = default_image
        print(f"No argument provided, trying default: {image_path}")

    # Check if file exists
    if os.path.exists(image_path):
        protect_image(image_path)
    else:
        print(f"File '{image_path}' not found.")
        print(f"Current directory: {os.getcwd()}")
        print("Available image files:")
        for f in os.listdir('.'):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                print(f"  - {f}")

        response = input("Use camera instead? (y/n): ").lower()
        if response == 'y':
            protect_with_camera_only()
        else:
            print("Exiting.")
import cv2
import torch
import numpy as np
from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler
from PIL import Image
import os


class ImageProtector:
    def __init__(self, device=None, model_path=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Default local model path
        self.model_path = model_path or r"C:\Users\a\PycharmProjects\fingerprint\models\sd-inpainting"
        self.pipe = None

    def _load_pipeline(self):
        """Load diffusion model from local path (offline mode)"""
        if self.pipe is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"Model not found at {self.model_path}. "
                    "Please run download_model.py first."
                )

            print(f"Loading diffusion model from: {self.model_path}")

            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
            )

            # Use faster scheduler
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )

            self.pipe = self.pipe.to(self.device)

            # Memory optimization for 4GB GPU
            if self.device.type == 'cuda':
                self.pipe.enable_attention_slicing()
                self.pipe.enable_model_cpu_offload()
                self.pipe.vae.to(dtype=torch.float16)

            print("Diffusion model loaded successfully")

        return self.pipe

    def protect(self, image, mask,
                prompt="natural skin texture, smooth surface, normal lighting",
                negative_prompt="fingerprint, ridges, lines, blurry, pixelated, mosaic, different color, artificial",
                num_inference_steps=30,
                guidance_scale=5.0,
                strength=0.65):
        """
        Protect fingerprint regions using diffusion inpainting

        Args:
            image: BGR numpy array
            mask: uint8 mask (0-255), higher values = more protection
        Returns:
            protected: BGR numpy array
        """
        original_h, original_w = image.shape[:2]

        # Resize to 512x512 for diffusion model (standard input size)
        target_size = 512
        image_resized = cv2.resize(image, (target_size, target_size))

        # Process mask: resize and normalize to [0, 1]
        mask_resized = cv2.resize(mask, (target_size, target_size)).astype(np.float32)
        mask_resized = mask_resized / 255.0
        mask_resized = np.clip(mask_resized, 0, 1)

        # For diffusion model, use binary mask with slight feathering
        # Threshold at 0.3 to keep only the core protection area
        mask_binary = (mask_resized > 0.3).astype(np.float32)
        mask_binary = cv2.GaussianBlur(mask_binary, (5, 5), 2)

        # Convert to PIL
        pil_image = Image.fromarray(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))
        pil_mask = Image.fromarray((mask_binary * 255).astype(np.uint8))

        # Load pipeline and run inference
        pipe = self._load_pipeline()

        print(f"Running inpainting with strength={strength}, steps={num_inference_steps}...")

        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=pil_image,
            mask_image=pil_mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
        ).images[0]

        # Convert back to numpy (BGR)
        inpainted = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        inpainted = cv2.resize(inpainted, (original_w, original_h))

        # Smooth blending with original using gradient mask
        mask_original = cv2.resize(mask, (original_w, original_h)).astype(np.float32) / 255.0
        mask_original = cv2.GaussianBlur(mask_original, (11, 11), 5)

        if len(mask_original.shape) == 2:
            mask_original = np.expand_dims(mask_original, axis=2)

        # Blend: more original where mask is low, more inpainted where mask is high
        blended = (image * (1 - mask_original) + inpainted * mask_original).astype(np.uint8)

        return blended

    def unload(self):
        """Free GPU memory"""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                print("GPU memory cleared")
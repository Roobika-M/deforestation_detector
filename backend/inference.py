import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import os

class UNetInference:
    def __init__(self, model_path="unet_forest.pth", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path

        # Define U-Net architecture (must match training!)
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,   # no need to load imagenet weights at inference
            in_channels=3,
            classes=1,
            activation=None
        ).to(self.device)

        # Checkpoint loading
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"❌ Model checkpoint not found: {self.model_path}\n"
                f"Please place unet_forest.pth in your backend folder."
            )

        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        print(f"✅ Loaded model from {self.model_path} on {self.device}")

        # Preprocessing pipeline
        self.transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])

    def predict_mask(self, img_pil: Image.Image, threshold: float = 0.5):
        """Takes a PIL image, returns (binary mask, probabilities)"""
        img = self.transform(img_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(img)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            mask = (probs > threshold).astype(np.uint8)

        return mask, probs


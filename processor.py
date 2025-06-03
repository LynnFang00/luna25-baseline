"""
Inference script for predicting malignancy of lung nodules
"""
import numpy as np
import dataloader
import torch
import torch.nn as nn
from torchvision import models
from models.model_3d import I3D
from models.model_2d import ResNet18
import os
import math
import logging
from models.custom_model import ConvNextLSTM

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s][%(asctime)s] %(message)s",
    datefmt="%I:%M:%S",
)

# define processor
class MalignancyProcessor:
    """
    Loads a chest CT scan, and predicts the malignancy around a nodule
    """

    def __init__(self, mode="2D", suppress_logs=False, model_name="LUNA25-baseline-2D"):

        self.size_px = 64
        self.size_mm = 50

        self.model_name = model_name
        self.mode = mode
        self.suppress_logs = suppress_logs

        if not self.suppress_logs:
            logging.info("Initializing the deep learning system")

        # Create a single device variable here
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.mode == "2D":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model_2d = ResNet18(weights=None).to(device)
        elif self.mode == "3D":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model_3d = I3D(num_classes=1, pre_trained=False, input_channels=3).to(device)
        elif self.mode == "CUSTOM":
            self.model_custom = ConvNextLSTM(pretrained=False, in_chans=3, class_num=1).to(device)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        self.model_root = "/opt/app/resources/"

    def define_inputs(self, image, header, coords):
        self.image = image
        self.header = header
        self.coords = coords

    def extract_patch(self, coord, output_shape, mode):

        patch = dataloader.extract_patch(
            CTData=self.image,
            coord=coord,
            srcVoxelOrigin=self.header["origin"],
            srcWorldMatrix=self.header["transform"],
            srcVoxelSpacing=self.header["spacing"],
            output_shape=output_shape,
            voxel_spacing=(
                self.size_mm / self.size_px,
                self.size_mm / self.size_px,
                self.size_mm / self.size_px,
            ),
            coord_space_world=True,
            mode=mode,
        )

        # ensure same datatype...
        patch = patch.astype(np.float32)

        # clip and scale...
        patch = dataloader.clip_and_scale(patch)
        return patch

    def _process_model(self, mode):

        if not self.suppress_logs:
            logging.info("Processing in " + mode)
        print(f"[DEBUG] _process_model called with mode = {mode}")

        if mode == "2D":
            output_shape = [1, self.size_px, self.size_px]
            model = self.model_2d
        elif mode == "3D":
            output_shape = [self.size_px, self.size_px, self.size_px]
            model = self.model_3d
        elif mode == "CUSTOM":
            output_shape = [self.size_px, self.size_px, self.size_px]
            model = self.model_custom

        else:
            raise ValueError(f"Unsupported mode: {mode}")

        nodules = []

        for _coord in self.coords:
            # If CUSTOM, extract in 3D mode so we get a (D,H,W) patch
            if mode == "CUSTOM":
                patch = self.extract_patch(_coord, output_shape, mode="3D")
            else:
                patch = self.extract_patch(_coord, output_shape, mode=mode)
            nodules.append(patch)

        nodules = np.array(nodules)  # shape: (num_nodules, D, H, W) or (num_nodules, 1, H, W) for 2D
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nodules = torch.from_numpy(nodules).to(device)

        print(f"[DEBUG] nodules tensor shape before repeat: {nodules.shape}")

        ckpt_path = os.path.join(self.model_root, self.model_name, "best_metric_model.pth")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt)
        model = model.to(device)
        model.eval()

        if mode == "CUSTOM":
            # At this point, `nodules` has shape (N, 1, D, H, W).
            # We need (N, 3, D, H, W) for ConvNextLSTM.
            print(f"[DEBUG] nodules tensor shape before repeat: {nodules.shape}")
            nodules = nodules.repeat(1, 3, 1, 1, 1)
            print(f"[DEBUG] nodules shape after repeat: {nodules.shape}")
            logits = model(nodules)  # ConvNextLSTM expects (N, 3, D, H, W)
        else:
            logits = model(nodules)

        logits = logits.detach().cpu().numpy()
        return logits

    def predict(self):

        logits = self._process_model(self.mode)

        probability = torch.sigmoid(torch.from_numpy(logits)).numpy()
        return probability, logits

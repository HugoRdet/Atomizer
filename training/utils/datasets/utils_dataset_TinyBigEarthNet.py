import h5py
import os
import torch
import numpy as np
import einops
from torch.utils.data import Dataset
from torchvision.transforms import v2
import random
import yaml


def read_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def rotate_image_torch(image, angle):
    if angle == 0:
        return image
    elif angle == 90:
        return image.rot90(1, [-2, -1])
    elif angle == 180:
        return image.rot90(2, [-2, -1])
    elif angle == 270:
        return image.rot90(3, [-2, -1])
    else:
        raise ValueError("Angle must be 0, 90, 180, or 270 degrees.")


def flip_image_torch(image, horizontal=False, vertical=False):
    if horizontal:
        image = image.flip(-1)
    if vertical:
        image = image.flip(-2)
    return image


def change_size(img, coordinates):
    """Crop image and center-pad back to original spatial size."""
    orig_h, orig_w = img.shape[1], img.shape[2]
    x_min, x_max, y_min, y_max = coordinates
    cropped = img[:, x_min:x_max, y_min:y_max]
    return cropped


def change_resolution(img, target_size, keep_res=False):
    orig_size = img.shape[1]
    img = v2.Resize(size=target_size)(img)
    if keep_res:
        img = v2.Resize(size=orig_size)(img)
    return img


# =============================================================================
# ModalityTransform
# =============================================================================

class ModalityTransform:
    """
    Reads pre-generated modality YAMLs and applies spatial/spectral transforms.
    Works directly with optical band indices (SAR already excluded before calling).
    """

    def __init__(self, bands_info, groups, transform_path, name_config=""):
        """
        Args:
            bands_info: dict band_name -> {idx, bandwidth, central_wavelength, ...}
                        idx refers to position in the optical-only tensor
            groups: dict group_name -> [band_names]
            transform_path: root path to pre-generated transform YAMLs
            name_config: config subfolder name
        """
        self.bands_info = bands_info
        self.groups = groups
        self.transform_path = transform_path
        self.name_config = name_config

        # band_name <-> optical index
        self._name_to_idx = {}
        self._idx_to_name = {}
        for band_name  in self.bands_info["bands_sen2_info"].keys():
            band_data=self.bands_info["bands_sen2_info"][band_name]
            idx = band_data["idx"]
            self._name_to_idx[band_name] = idx
            self._idx_to_name[idx] = band_name

        # All optical indices sorted
        self._all_indices = sorted(self._idx_to_name.keys())

        # Cache: group_name -> set of optical indices
        self._group_to_indices = {}
        for group_name, band_names in groups.items():
            self._group_to_indices[group_name] = set(
                self._name_to_idx[bn] for bn in band_names
                if bn in self._name_to_idx
            )

    def _get_yaml_path(self, img_id, mode, modality_mode):

        
        if self.name_config:
            return (f"{self.transform_path}/{self.name_config}/"
                    f"{mode}/{img_id}_transfos_{modality_mode}.yaml")
        else:
            return (f"{self.transform_path}/"
                    f"{mode}/{img_id}_transfos_{modality_mode}.yaml")

    def get_channels_from_group(self, group_name):
        return self._group_to_indices[group_name]

    def get_opposite_channels(self, group_name):
        group_indices = self.get_channels_from_group(group_name)
        return set(self._all_indices) - group_indices

    def apply(self, img, img_id, mode, modality_mode):
        """
        Apply modality transforms to optical image.
        Order: crop -> resize -> center-pad to original size -> band mask

        Output is always [C, 120, 120] with an attention mask [C, 120, 120].

        Args:
            img: [C, H, W] optical bands only (SAR already stripped)
            img_id: image identifier for YAML lookup
            mode: split folder (train/validation/test)
            modality_mode: which modality YAML to read

        Returns:
            img_out: [C, 120, 120] padded image (masked bands zeroed)
            mask: [C, 120, 120] attention mask (0 = padded/masked, 1 = valid)
            kept_indices: list of kept optical band indices
            resolution_factor: float from YAML
        """
        yaml_path = self._get_yaml_path(img_id, mode, modality_mode)
        transfos = read_yaml(yaml_path)

        resolution_factor = float(transfos.get("resolution", 1.0))
        orig_size = img.shape[1]  # 120
        C = img.shape[0]          # 12

        # --- 1. Crop ---
        if "size" in transfos:
            img = change_size(img, transfos["size"])

        # --- 2. Resize ---
        if "resolution" in transfos:
            new_resolution = int(img.shape[1] * resolution_factor)
            if new_resolution > 0 and new_resolution != img.shape[1]:
                img = change_resolution(img, target_size=new_resolution, keep_res=False)

        # --- 3. Center-pad back to original size ---
        cur_h, cur_w = img.shape[1], img.shape[2]
        padded = torch.zeros(C, orig_size, orig_size, dtype=img.dtype, device=img.device)
        spatial_mask = torch.ones(orig_size, orig_size, dtype=img.dtype, device=img.device)

        pad_top = (orig_size - cur_h) // 2
        pad_left = (orig_size - cur_w) // 2

        padded[:, pad_top:pad_top + cur_h, pad_left:pad_left + cur_w] = img
        spatial_mask[pad_top:pad_top + cur_h, pad_left:pad_left + cur_w] = 0.0

        # --- 4. Band masking ---
        channels_to_remove = set()
        if "remove" in transfos and transfos["remove"]:
            channels_to_remove = self.get_channels_from_group(transfos["remove"])
        elif "keep" in transfos and transfos["keep"]:
            channels_to_remove = self.get_opposite_channels(transfos["keep"])

        kept_indices = [
            idx for idx in self._all_indices
            if idx not in channels_to_remove
        ]

        # Build full attention mask [C, H, W]
        # 0 = valid, 1 = masked/padded
        mask = spatial_mask.unsqueeze(0).expand(C, -1, -1).clone()

        # Mark removed bands as masked (1) and zero out image
        for idx in channels_to_remove:
            padded[idx] = 0.0
            mask[idx] = 1.0

        return padded, mask, kept_indices, resolution_factor


# =============================================================================
# Dataset
# =============================================================================

class Tiny_BigEarthNet(Dataset):
    """
    BigEarthNet dataset for Atomizer multilabel classification.
    Loads from H5 (already normalized), applies modality transforms.

    H5 layout:
        image_{idx}: [14, 120, 120]  (channels 0-1: SAR, 2-13: optical)
        label_{idx}: [19] multilabel vector
        id_{idx}: int original image id
        shape_{mode}_{idx}: int token count for sampler
    """

    SAR_CHANNELS = 2  # first 2 H5 channels are SAR, always skipped

    def __init__(
        self,
        root_path: str = "./data/Tiny_BigEarthNet",
        transform=None,
        model=None,
        modality_mode="train",
        mode="train",
        dataset_config=None,
        config_model=None,
        look_up=None,
    ):
        super().__init__()

        self.file_path = root_path
        self.split = mode
        self.mode = mode
        self.modality_mode = modality_mode
        self.original_mode = mode
        self.transform = transform
        self.model = model
        self.look_up = look_up
        self.config_model = config_model
        self.num_samples = None

        # Trainer config
        self.nb_tokens = config_model["trainer"]["max_tokens"]
        self.max_tokens_reconstruction = config_model["trainer"]["max_tokens_reconstruction"]

        # =====================================================================
        # Band info — optical only
        # =====================================================================
        self.bands_info = dataset_config["bands_info"]
        self.bandwidths, self.wavelengths, self.band_names = self._parse_bands_info()

        # Pre-compute spectral indices via lookup table
        self.spectral_indices = self._build_spectral_indices()

        # =====================================================================
        # Modality transform
        # =====================================================================
        NAME_CONFIG = "regular"  # hardcoded config subfolder

        groups = dataset_config["groups"]
        transform_path = "./data/Tiny_BigEarthNet/transformations/"

        self.modality_transform = ModalityTransform(
            bands_info=self.bands_info,
            groups=groups,
            transform_path=transform_path,
            name_config=NAME_CONFIG,
        )

        # =====================================================================
        # Count samples from H5
        # =====================================================================
        self._initialize_file()

        print(f"[BigEarthNet] Split: {mode}, Modality: {modality_mode}")
        print(f"[BigEarthNet] Samples: {self.num_samples}")
        print(f"[BigEarthNet] Optical bands: {len(self.band_names)} -> {self.band_names}")

    # =========================================================================
    # INIT HELPERS
    # =========================================================================

    def _initialize_file(self):
        with h5py.File(self.file_path, 'r') as f:
            self.num_samples = len(f.keys()) // 6

    def _parse_bands_info(self):
        """Parse bands_info dict sorted by idx."""
        all_bands = []

        for band_name  in self.bands_info["bands_sen2_info"].keys():
            band_data=self.bands_info["bands_sen2_info"][band_name]
            idx = band_data["idx"]

            all_bands.append({
                "idx": idx,
                "bandwidth": band_data["bandwidth"],
                "central_wavelength": band_data["central_wavelength"],
                "name": band_name,
            })

    
        all_bands = sorted(all_bands, key=lambda x: x["idx"])

        bandwidths = torch.tensor([b["bandwidth"] for b in all_bands], dtype=torch.float32)
        wavelengths = torch.tensor([b["central_wavelength"] for b in all_bands], dtype=torch.float32)
        band_names = [b["name"] for b in all_bands]

        return bandwidths, wavelengths, band_names

    def _build_spectral_indices(self):
        """Map each optical band to a global spectral index via lookup table."""
        indices = []
        for i, (bw, wl) in enumerate(zip(self.bandwidths, self.wavelengths)):
            key = (int(bw.item()), int(wl.item()))
            if key not in self.look_up.table_wave:
                raise KeyError(
                    f"Band {self.band_names[i]} with key {key} not found in lookup table. "
                    f"Available keys: {list(self.look_up.table_wave.keys())}"
                )
            indices.append(self.look_up.table_wave[key])
        return torch.tensor(indices, dtype=torch.long)

    # =========================================================================
    # PUBLIC
    # =========================================================================

    def __len__(self):
        return self.num_samples

    def set_modality_mode(self, mode):
        self.modality_mode = mode

    def reset_modality_mode(self):
        self.modality_mode = self.original_mode

    # =========================================================================
    # GETITEM
    # =========================================================================

    def __getitem__(self, idx):
        # =================================================================
        # 1. Load from H5 (already normalized)
        # =================================================================
        with h5py.File(self.file_path, 'r') as f:
            image = torch.tensor(f[f'image_{idx}'][:])   # [14, H, W]
            label = torch.tensor(f[f'label_{idx}'][:])    # [19]
            id_img = int(f[f'id_{idx}'][()])

        # =================================================================
        # 2. Drop SAR, keep optical only
        # =================================================================
        image = image[self.SAR_CHANNELS:]  # [12, H, W]

        # =================================================================
        # 3. Augment (train only)
        # =================================================================
        if self.mode == "train":
            angle = random.choice([0, 90, 180, 270])
            image = rotate_image_torch(image, angle)
            if random.random() > 0.5:
                image = flip_image_torch(image, horizontal=True)
            if random.random() > 0.5:
                image = flip_image_torch(image, vertical=True)

        # =================================================================
        # 4. Apply modality transform (crop -> resize -> pad -> band mask)
        #    Output: [12, 120, 120] image + [12, 120, 120] attention mask
        # =================================================================
        image, attention_mask, kept_indices, resolution_factor = (
            self.modality_transform.apply(
                image, id_img, self.mode, self.modality_mode
            )
        )

        # =================================================================
        # 5. Optional additional transform
        # =================================================================
        if self.transform:
            image = self.transform(image)

        return image, attention_mask, label, id_img
"""
ImageNet Dataset for Atomiser (Classification), with GPU-side tokenization
===========================================================================

Why GPU-side tokenization
-------------------------
Every ImageNet image of a given size has the SAME token layout: positions,
spectral indices, resolution and time are identical, only the pixel values
change (the same property the U-Net's shared geometry relies on). Building the
[N, 8] token tensor on the CPU for every image would move ~8x more data than
the image itself (64x64: 12,288 atoms; 224x224: 150,528 atoms per image).

So the split is:
    ImageNetDataset         CPU: decode + augment -> uint8 image [3, H, W] + label
    ImageNetGPUTokenizer    GPU: image batch -> grouped token batch, using a
                            template built ONCE with the project's TokenBuilder

Because the template comes from TokenBuilder itself, the token convention
(column meaning, x/y units, atom order) is exactly the one every other dataset
uses. The atom -> pixel mapping is recovered by tokenizing an image whose
pixel values are their own flat index, so no assumption about TokenBuilder's
atom order is made.

Usage in a LightningModule (see the trainer):

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if "image_uint8" in batch:
            batch = self.gpu_tokenizer(batch)
        return batch

Supported data layouts (auto-detected under root_path)
-------------------------------------------------------
1. ImageFolder (full-size JPEGs):
        root/train/<wnid>/*.JPEG     root/val/<wnid>/*.JPEG
   Images are resized on the fly to `image_size` (default 64).

   This is the layout to use at 256x256: decoding + RandomResizedCrop on the CPU,
   as in standard ImageNet training (give the DataLoader enough workers).
   The val/ folder must also be split into <wnid>/ class folders.

2. Pre-converted downsampled ImageNet (fast; recommended for 64x64 only):
        root/train_images.npy  [N, H, W, 3] uint8     root/train_labels.npy [N] int (0..999)
        root/val_images.npy                           root/val_labels.npy
   Create it once from the official "Imagenet64_train / Imagenet64_val"
   pickled batches with `prepare_downsampled_imagenet(src_dir, out_dir)`.

Splits: "train" -> train, "validation" and "test" -> val (ImageNet's labelled
validation set is the standard test set; hold out part of train if you need
a separate validation split).

Bands: the three channels are given the spectral identity of Sentinel-2
B04 / B03 / B02 (red / green / blue), read from dataset_config["bands_senflood"],
so that transfer to RGB-only EuroSAT is exact on the spectral side. Set
`band_section` / `rgb_codes` to use other definitions.

Output of the dataset (per sample):
    {"image_uint8": [3, H, W] uint8, "label": long, "task": "classification"}

Output of the GPU tokenizer (per batch), identical to the other datasets:
    {"groups": {gsd: {"tokens": [B, N, 8], "mask": [B, N] bool, "shape": (3, H, W)}},
     "queries": [B, 1, 8] zeros, "queries_mask": [B, 1], "label": [B],
     "task": "classification", "target_resolution": gsd}
"""

import glob
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

try:
    from .token_builder import TokenBuilder
except ImportError:                      # standalone testing
    TokenBuilder = None


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# =============================================================================
# One-time conversion of the official downsampled ImageNet batches
# =============================================================================
def prepare_downsampled_imagenet(src_dir: str, out_dir: str, size: int = 64):
    """Convert the official Imagenet{size}_train / _val pickled batches
    (train_data_batch_1..10, val_data; 'data' [N, 3*size*size] uint8 CHW-flat,
    'labels' 1..1000) into memory-mappable .npy files:
        out_dir/{train,val}_images.npy  [N, size, size, 3] uint8
        out_dir/{train,val}_labels.npy  [N] int16 (0..999)
    """
    os.makedirs(out_dir, exist_ok=True)

    def load(path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        x = np.asarray(d["data"], dtype=np.uint8).reshape(-1, 3, size, size).transpose(0, 2, 3, 1)
        y = np.asarray(d["labels"], dtype=np.int16) - 1
        return x, y

    for split, pattern in (("train", "train_data_batch_*"), ("val", "val_data*")):
        files = sorted(glob.glob(os.path.join(src_dir, "**", pattern), recursive=True),
                       key=lambda p: int(''.join(c for c in os.path.basename(p) if c.isdigit()) or 0))
        if not files:
            raise FileNotFoundError(f"no '{pattern}' files under {src_dir}")
        n = sum(load(f)[1].shape[0] for f in files)
        imgs = np.lib.format.open_memmap(os.path.join(out_dir, f"{split}_images.npy"),
                                         mode="w+", dtype=np.uint8, shape=(n, size, size, 3))
        labs = np.empty(n, dtype=np.int16)
        i = 0
        for f in files:
            x, y = load(f)
            imgs[i:i + len(y)] = x
            labs[i:i + len(y)] = y
            i += len(y)
            print(f"[prepare] {split}: {os.path.basename(f)} -> {i}/{n}")
        imgs.flush()
        np.save(os.path.join(out_dir, f"{split}_labels.npy"), labs)
        assert labs.min() >= 0 and labs.max() <= 999, (labs.min(), labs.max())
    print(f"[prepare] done -> {out_dir}")


# =============================================================================
# Dataset (CPU side): decode + augment, return uint8 images
# =============================================================================
class ImageNetDataset(Dataset):
    NUM_CLASSES = 1000
    TASK_NAME = "classification"
    SPLIT_DIRS = {"train": "train", "validation": "val", "test": "val"}

    def __init__(
        self,
        root_path: str = "./data/imagenet64",
        transform=None,
        model=None,
        modality_mode="train",
        mode: str = "train",
        dataset_config: dict = None,
        config_model: dict = None,
        look_up=None,
        image_size: int = None,
        train_scale=None,
        hflip: bool = True,
        randaugment: bool = False,
    ):
        super().__init__()
        assert mode in self.SPLIT_DIRS, f"Unknown split: {mode}"
        self.root_path = root_path
        self.split = mode
        cfg = (config_model or {}).get("imagenet", {})
        self.image_size = int(image_size or cfg.get("image_size", 64))
        self.train = mode == "train"
        split_dir = self.SPLIT_DIRS[mode]

        npy = os.path.join(root_path, f"{split_dir}_images.npy")
        if os.path.exists(npy):
            self.layout = "npy"
            self.images = np.load(npy, mmap_mode="r")                      # [N, h, w, 3]
            self.labels = np.load(os.path.join(root_path, f"{split_dir}_labels.npy")).astype(np.int64)
            self.native = self.images.shape[1]
        elif os.path.isdir(os.path.join(root_path, split_dir)):
            self.layout = "folder"
            classes = sorted(d for d in os.listdir(os.path.join(root_path, "train"))
                             if os.path.isdir(os.path.join(root_path, "train", d)))
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
            self.samples = []
            for c in classes:
                files = sorted(glob.glob(os.path.join(root_path, split_dir, c, "*")))
                self.samples += [(f, self.class_to_idx[c]) for f in files]
            if not self.samples:
                raise RuntimeError(f"[ImageNet] no images under {root_path}/{split_dir}/<class>/")
            self.native = None
        else:
            raise FileNotFoundError(
                f"[ImageNet] expected {npy} (see prepare_downsampled_imagenet) or "
                f"{os.path.join(root_path, split_dir)}/<wnid>/*.JPEG")

        # RandomResizedCrop scale: the standard ImageNet (0.08, 1) at full resolution,
        # a milder (0.35, 1) for low-resolution (<128 px) images
        if train_scale is None:
            train_scale = cfg.get("train_scale", (0.08, 1.0) if self.image_size >= 128 else (0.35, 1.0))
        self.train_scale = tuple(train_scale)
        self.transform = self._build_transform(self.train_scale, hflip,
                                               bool(cfg.get("randaugment", randaugment)))
        n = len(self.labels) if self.layout == "npy" else len(self.samples)
        print(f"[ImageNet] split={mode} ({split_dir}) layout={self.layout} samples={n} "
              f"image_size={self.image_size} augment={'ON (crop scale ' + str(self.train_scale) + ')' if self.train else 'OFF'}")

    def _build_transform(self, train_scale, hflip, randaugment):
        import torchvision.transforms.v2 as T
        s = self.image_size
        if self.train:
            t = [T.RandomResizedCrop(s, scale=tuple(train_scale), antialias=True)]
            if hflip:
                t.append(T.RandomHorizontalFlip())
            if randaugment:
                t.append(T.RandAugment())
        elif self.layout == "npy" and self.native == s:
            t = []                                                          # already at size
        else:
            t = [T.Resize(int(round(s / 0.875)), antialias=True), T.CenterCrop(s)]
        return T.Compose(t) if t else None

    def __len__(self):
        return len(self.labels) if self.layout == "npy" else len(self.samples)

    def _load(self, index):
        if self.layout == "npy":
            img = torch.from_numpy(np.ascontiguousarray(self.images[index])).permute(2, 0, 1)
            return img, int(self.labels[index])
        from torchvision.io import decode_image, ImageReadMode
        path, label = self.samples[index]
        img = decode_image(path, mode=ImageReadMode.RGB)                     # [3, h, w] uint8
        return img, label

    def __getitem__(self, index):
        img, label = self._load(index)                                     # uint8 [3, h, w]
        if self.transform is not None:
            img = self.transform(img)
        assert img.shape == (3, self.image_size, self.image_size), img.shape
        return {"image_uint8": img.contiguous(), "label": torch.tensor(label, dtype=torch.long),
                "task": self.TASK_NAME}


def collate_imagenet(samples):
    """Stack uint8 images and labels; tokens are built later, on the GPU."""
    return {"image_uint8": torch.stack([s["image_uint8"] for s in samples]),
            "label": torch.stack([s["label"] for s in samples]),
            "task": "classification"}


# =============================================================================
# GPU tokenizer: image batch -> grouped token batch
# =============================================================================
class ImageNetGPUTokenizer(nn.Module):
    """Builds the token batch on the GPU from a template made ONCE by the
    project's TokenBuilder. Buffers are non-persistent (not saved in checkpoints)."""

    def __init__(self, image_size, dataset_config, look_up, gsd=10.0,
                 band_section="bands_senflood", rgb_codes=("B04", "B03", "B02"),
                 mean=IMAGENET_MEAN, std=IMAGENET_STD, token_builder_cls=None):
        super().__init__()
        H = W = int(image_size)
        self.H, self.W, self.gsd = H, W, float(gsd)
        tb_cls = token_builder_cls or TokenBuilder
        if tb_cls is None:
            raise ImportError("TokenBuilder not available; pass token_builder_cls")
        tb = tb_cls(look_up)

        # spectral identity of R, G, B
        bands = dataset_config[band_section]
        spec = []
        for code in rgb_codes:
            if code not in bands:
                raise KeyError(f"[ImageNet] band '{code}' not in dataset_config['{band_section}']")
            key = (int(bands[code]["bandwidth"]), int(bands[code]["central_wavelength"]))
            if key not in look_up.table_wave:
                raise KeyError(f"[ImageNet] band {code} key={key} not in lookup table_wave")
            spec.append(look_up.table_wave[key])
        spectral_indices = torch.tensor(spec, dtype=torch.long)
        resolution_idx = look_up.get_resolution_idx(self.gsd)

        # template: tokenize an image whose values are their own flat index
        index_img = torch.arange(3 * H * W, dtype=torch.float32).view(3, H, W)
        dummy_label = torch.full((H, W), 255, dtype=torch.long)
        tok = tb.build_tokens(image=index_img, label=dummy_label, resolution=self.gsd,
                              spectral_indices=spectral_indices, resolution_idx=resolution_idx,
                              time_idx=-1)
        tok = torch.as_tensor(tok, dtype=torch.float32)
        src = tok[:, 0].round().long()
        if not (torch.allclose(tok[:, 0], src.float()) and src.min() >= 0
                and src.max() < 3 * H * W and torch.unique(src).numel() == src.numel() == 3 * H * W):
            raise RuntimeError("[ImageNet] TokenBuilder does not copy pixel values one-to-one into "
                               "column 0; the GPU template cannot be built this way.")
        # the channel of every atom must carry the right spectral index
        ch = src // (H * W)
        if not torch.equal(tok[:, 3].long(), spectral_indices[ch]):
            raise RuntimeError("[ImageNet] spectral column does not match the pixel's channel")

        self.register_buffer("template", tok, persistent=False)              # [N, 8]
        self.register_buffer("src", src, persistent=False)                   # atom -> flat pixel
        self.register_buffer("mean", torch.tensor(mean).view(3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(3, 1, 1), persistent=False)
        print(f"[ImageNet] GPU tokenizer: {H}x{W}x3 -> {tok.shape[0]} atoms, gsd={self.gsd}, "
              f"bands {list(rgb_codes)} -> spectral idx {spec}")

    @torch.no_grad()
    def forward(self, batch):
        img = batch["image_uint8"]                                         # [B, 3, H, W] uint8
        B = img.shape[0]
        assert img.shape[1:] == (3, self.H, self.W), img.shape
        x = (img.float() / 255.0 - self.mean) / self.std                    # normalised
        tokens = self.template.unsqueeze(0).repeat(B, 1, 1)
        tokens[..., 0] = x.reshape(B, -1)[:, self.src]
        N = tokens.shape[1]
        dev = tokens.device
        out = dict(batch)
        out.pop("image_uint8")
        out["groups"] = {self.gsd: {"tokens": tokens,
                                    "mask": torch.zeros(B, N, dtype=torch.bool, device=dev),
                                    "shape": (3, self.H, self.W)}}
        out["queries"] = torch.zeros(B, 1, 8, device=dev)
        out["queries_mask"] = torch.zeros(B, 1, device=dev)
        out["target_resolution"] = self.gsd
        out["task"] = "classification"
        return out

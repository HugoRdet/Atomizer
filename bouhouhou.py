from configilm.extra.DataSets import BENv2_DataSet

data_dirs = {
    "images_lmdb": "data/Encoded-BigEarthNet",
    "metadata_parquet": "data/Encoded-BigEarthNet/metadata.parquet",
    "metadata_snow_cloud_parquet": "data/Encoded-BigEarthNet/metadata_for_patches_with_snow_cloud_or_shadow.parquet",
}

ds = BENv2_DataSet.BENv2DataSet(
    data_dirs=data_dirs,
    split="train",
    img_size=(12, 120, 120),
)

img, lbl = ds[0]

# Per-channel stats
for c in range(12):
    ch = img[c]
    print(f"Channel {c:2d}: min={ch.min():10.2f}  max={ch.max():10.2f}  mean={ch.mean():10.2f}  std={ch.std():10.2f}")

# Also try with 14 channels (12 S2 + 2 S1)
for n_ch in [2, 4, 10, 13, 14]:
    try:
        ds2 = BENv2_DataSet.BENv2DataSet(
            data_dirs=data_dirs,
            split="train",
            img_size=(n_ch, 120, 120),
        )
        img2, _ = ds2[0]
        print(f"\n{n_ch} channels works: range=[{img2.min():.2f}, {img2.max():.2f}]")
    except Exception as e:
        print(f"\n{n_ch} channels: {e}")
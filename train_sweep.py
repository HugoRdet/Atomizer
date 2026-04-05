from training.utils.datasets_baselines.utils_dataset_MultiEarth import MultiEarthBaselineDataset

# Test L8 same-sensor
ds = MultiEarthBaselineDataset( split="train", sensor="l8",
    n_timesteps=3, temporal_mode="stack")
sample = ds[0]
print(f"L8 image: {sample['image']['l8'].shape}")    # [21, 256, 256]
print(f"Label:    {sample['target'].shape}")           # [256, 256]
print(f"Values:   [{sample['image']['l8'].min():.3f}, {sample['image']['l8'].max():.3f}]")
print(f"Label unique: {sample['target'].unique()}")
print(f"Delta days: {sample['dates']['l8']}")

# Test S2 same-sensor
ds_s2 = MultiEarthBaselineDataset( split="train", sensor="s2",
    n_timesteps=3, temporal_mode="stack")
sample_s2 = ds_s2[0]
print(f"\nS2 image: {sample_s2['image']['s2'].shape}")  # [36, 256, 256]
print(f"Values:   [{sample_s2['image']['s2'].min():.3f}, {sample_s2['image']['s2'].max():.3f}]")

# Test cross-sensor: L8 data → S2 grid (for UNet trained on S2)
ds_cross = MultiEarthBaselineDataset( split="test", sensor="l8",
    cross_sensor_target="s2", n_timesteps=3, temporal_mode="stack")
sample_cross = ds_cross[0]
print(f"\nL8→S2 interp: {sample_cross['image']['s2'].shape}")  # [36, 256, 256]
print(f"Values: [{sample_cross['image']['s2'].min():.3f}, {sample_cross['image']['s2'].max():.3f}]")
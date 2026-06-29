# ============================================================================
# TOP OF test_pastis.py — replace the broken try/except import block with the
# SAME imports your train script uses:
#
#     from training.utils import read_yaml, Lookup_encoding
#     from training.utils.datasets.token_builder import TokenBuilder
#
# (delete:  try: from training.utils.token_building.lookup import ...
#           except Exception: Lookup_encoding = None )
# ============================================================================

# Mirror train_pastis.py's resolution registration EXACTLY so reference-grid
# offsets / token coordinates match the 0.38 run.
ALL_KNOWN_RESOLUTIONS = {
    1.0: 2048, 2.5: 2048, 10.0: 2048, 20.0: 2048, 30.0: 2048,
}


def _register_all_resolutions(lookup_table, TokenBuilder):
    for res, ref_size in ALL_KNOWN_RESOLUTIONS.items():
        TokenBuilder.REFERENCE_SIZES[res] = ref_size
        lookup_table.get_or_register_modality(res, ref_size)
        lookup_table.get_resolution_idx(res)


def build_lookup_and_config(args, Lookup_encoding):
    # Import here so the function is self-contained; these match train_pastis.py
    from training.utils import read_yaml
    from training.utils.datasets.token_builder import TokenBuilder

    # ── Same paths as train_pastis.py ──
    config_model_path    = "./training/configs/config_test-Atomiser_Atos_One.yaml"
    bands_yaml_path      = "./data/bands_info/bands.yaml"
    configs_dataset_path = "./data/Tiny_BigEarthNet/configs_dataset_u_regular.yaml"

    config_model = read_yaml(config_model_path)

    # Match the multi_temporal the run used (CLI override in train script).
    # Force it here to be explicit; change if your 0.38 run used another value.
    if "dataset" not in config_model:
        config_model["dataset"] = {}
    config_model["dataset"].setdefault("multi_temporal", 10)

    # Build the lookup EXACTLY as train_pastis.py does.
    lookup_table = Lookup_encoding(
        read_yaml(configs_dataset_path),
        read_yaml(bands_yaml_path),
        config_model,
    )
    _register_all_resolutions(lookup_table, TokenBuilder)

    # S1 third band (VV_VH) is registered only when use_s1 — mirror that.
    if not args.no_s1:
        lookup_table.register_abstract_channel("VV_VH")

    return lookup_table, config_model

"""
Shared resolution registration — use in EVERY script (training, viz, probes)
=============================================================================

Two lookup-table mechanisms depend on how resolutions are registered:

1. Position grids (Lookup_encoding.register_modality / TokenBuilder.REFERENCE_SIZES).
   Grids registered after initialisation get the next free offset, so their
   ORDER changes the offsets of later grids. One shared, append-only list
   keeps every script's offsets identical.

2. Resolution indices (Lookup_encoding.table_resolution). The table is built
   once from the reference list in init_config (10 m -> 1, 15 m -> 2, 0.1 m -> 3,
   0.5 m -> 4, 30 m -> 5, 0.2 m -> 6, 1.6 m -> 7). get_resolution_idx() only LOOKS
   UP, returning LEARNED_RESOLUTION_IDX = 0 for anything else; it never
   registers. Without register_resolution(), a new GSD such as 5 m silently
   shares index 0 with non-optical channels (and, in ResolutionEncoder, a GSD
   of 0 m). register_resolution() APPENDS new indices, so existing indices
   (and checkpoints that rely on them) are unchanged.

RULE: only ever APPEND to KNOWN_RESOLUTIONS. Never reorder or insert.
Call register_all_resolutions(lookup) BEFORE building the model (the
resolution encoder reads the table at construction).

Usage:
    from training.utils.resolution_registry import register_all_resolutions
    lookup_table = Lookup_encoding(...)
    register_all_resolutions(lookup_table)
"""

from training.utils.datasets.token_builder import TokenBuilder

# (GSD in m/px, reference grid size). APPEND ONLY.
KNOWN_RESOLUTIONS = [
    (1.0, 2048),
    (2.5, 2048),
    (10.0, 2048),
    (20.0, 2048),
    (30.0, 2048),
    (5.0, 2048),        # AgriPotential (super-resolved Sentinel-2), appended
]


def register_all_resolutions(lookup_table, verbose=True):
    """Register every known position grid and resolution index, in a fixed order."""
    before = dict(lookup_table.table_resolution)
    for res, ref_size in KNOWN_RESOLUTIONS:
        TokenBuilder.REFERENCE_SIZES[res] = ref_size
        lookup_table.get_or_register_modality(res, ref_size)
        lookup_table.register_resolution(res)          # get_resolution_idx alone never registers
    changed = {k: (before[k], lookup_table.table_resolution[k]) for k in before
               if before[k] != lookup_table.table_resolution[k]}
    if changed:                                          # cannot happen with append-only registration
        raise RuntimeError(f"[resolution_registry] existing resolution indices changed: {changed}")
    if verbose:
        idx = {res: lookup_table.get_resolution_idx(res) for res, _ in KNOWN_RESOLUTIONS}
        print("[resolution_registry] resolution indices: "
              + ", ".join(f"{r:g} m -> {i}" for r, i in idx.items()))
    return lookup_table

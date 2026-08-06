"""
Centralized token building utilities for remote sensing datasets.

Uses a reference grid system where all crops at a given resolution
extract coordinates from a shared reference grid (e.g., 512×512).
This ensures consistent coordinate spaces across different crop sizes.
"""

import torch
import einops


class TokenBuilder:
    """
    Centralized token builder for remote sensing datasets.

    Uses reference grid indexing: instead of creating position encodings
    for each crop size, maintains reference grids (e.g., 512×512) and
    extracts windows from them.

    Benefits:
    - All crops share the same coordinate space
    - No dynamic modality registration for different crop sizes
    - Consistent positioning between training and validation
    - Handles edge crops naturally
    - Auto-registers unseen resolutions (e.g., from resolution augmentation)

    Args:
        lookup_table: Lookup_encoding instance for position/spectral/query offsets

    Example:
        >>> lookup = Lookup_encoding(config)
        >>> builder = TokenBuilder(lookup)
        >>> # 240×240 crop extracts indices [136:376] from 512 reference
        >>> x_idx, y_idx = builder.get_position_coordinates((12, 240, 240), 10.0)
        >>> tokens = builder.build_tokens(image, label, resolution=10.0, ...)
    """

    # Reference grid sizes per resolution (large enough for any expected crop).
    # Unseen resolutions are auto-registered with default ref_size=512.
    REFERENCE_SIZES = {
        10.0: 512,  # Sentinel-2/Sentinel-1/EnMAP at 10m
        15.0: 512,
        0.1:512,
        0.5:512,
        30.0:512,
        0.2:512,
        1.6: 512
    }

    # Default reference grid size for auto-registered resolutions
    DEFAULT_REF_SIZE = 512

    def __init__(self, lookup_table):
        """
        Initialize TokenBuilder with reference grid system.

        Args:
            lookup_table: Lookup_encoding instance
        """
        self.lookup = lookup_table

        # Pre-register reference grids
        self._register_reference_grids()

    def _register_reference_grids(self):
        """Pre-register reference grid sizes for all known resolutions."""
        print("[TokenBuilder] Registering reference grids:")
        for resolution, ref_size in sorted(self.REFERENCE_SIZES.items()):
            offset = self.lookup.get_or_register_modality(resolution, ref_size)
            print(f"  ({resolution}m, {ref_size}px) → offset {offset}")

    def _ensure_resolution_registered(self, resolution):
        """
        Ensure a resolution is registered in the reference grid system.

        Auto-registers unseen resolutions with DEFAULT_REF_SIZE.
        Called by get_position_coordinates and get_query_coordinates
        before accessing REFERENCE_SIZES.

        Args:
            resolution: float - GSD in meters/pixel

        Returns:
            ref_size: int - reference grid size for this resolution
        """
        if resolution not in self.REFERENCE_SIZES:
            ref_size = self.DEFAULT_REF_SIZE
            self.REFERENCE_SIZES[resolution] = ref_size
            offset = self.lookup.get_or_register_modality(resolution, ref_size)
            print(f"[TokenBuilder] Auto-registered resolution {resolution}m "
                  f"→ ref_size={ref_size}, offset={offset}")

        return self.REFERENCE_SIZES[resolution]

    # =========================================================================
    # COORDINATE GENERATION (Reference Grid Indexing)
    # =========================================================================

    def get_position_coordinates(self, image_shape, resolution):
        """
        Generate position coordinates by extracting from reference grid.

        All crops at a given resolution extract their coordinates from
        a shared reference grid, ensuring consistent coordinate space.

        Args:
            image_shape: (C, H, W) or (H, W) - actual crop dimensions
            resolution: float - GSD in meters/pixel

        Returns:
            x_indices: [C, H, W, 1] - x position indices from reference grid
            y_indices: [C, H, W, 1] - y position indices from reference grid

        Example:
            Reference grid: 512×512 (indices 0-511, center at 256)

            Crop 240×240:
            - Center: 256, Half: 120
            - Extract: [136:376, 136:376]
            - Indices: [136, 137, ..., 255, 256, 257, ..., 375]

            Crop 160×240 (edge crop):
            - Y: [176:336] (80 pixels each side of 256)
            - X: [136:376] (120 pixels each side of 256)
            - Both use indices from SAME reference grid
        """
        # Handle both (C, H, W) and (H, W)
        if len(image_shape) == 3:
            C, H, W = image_shape
        elif len(image_shape) == 2:
            H, W = image_shape
            C = 1
        else:
            raise ValueError(f"image_shape must be (C,H,W) or (H,W), got {image_shape}")

        # Auto-register if needed
        ref_size = self._ensure_resolution_registered(resolution)

        # Validate crop fits in reference
        if H > ref_size or W > ref_size:
            raise ValueError(
                f"Crop size ({H}×{W}) exceeds reference size ({ref_size}×{ref_size}) "
                f"at resolution {resolution}m. Increase REFERENCE_SIZES[{resolution}]."
            )

        # Get global offset for reference grid
        global_offset = self.lookup.get_or_register_modality(resolution, ref_size)

        # Calculate extraction window (centered in reference grid)
        ref_center = ref_size // 2
        half_h = H // 2
        half_w = W // 2

        # Extract indices from reference grid
        # For 240×240 from 512: [256-120:256+120] = [136:376]
        y_start = ref_center - half_h
        y_end = y_start + H  # Ensures exactly H elements
        x_start = ref_center - half_w
        x_end = x_start + W  # Ensures exactly W elements

        y_coords = torch.arange(y_start, y_end, dtype=torch.float32)
        x_coords = torch.arange(x_start, x_end, dtype=torch.float32)

        # Validate extracted window size
        assert len(y_coords) == H, f"y_coords length {len(y_coords)} != H {H}"
        assert len(x_coords) == W, f"x_coords length {len(x_coords)} != W {W}"

        # Create grids and add global offset
        x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing="xy")
        x_grid = x_grid + global_offset
        y_grid = y_grid + global_offset

        # Expand to [C, H, W, 1] format
        x_indices = einops.repeat(x_grid, "h w -> c h w 1", c=C)
        y_indices = einops.repeat(y_grid, "h w -> c h w 1", c=C)

        return x_indices, y_indices

    def get_query_coordinates(self, image_shape, resolution):
        """
        Generate query coordinate indices using reference grid.

        All crops at the same resolution share the same query offset
        (from the reference grid registration).

        Args:
            image_shape: (C, H, W) or (H, W) - image/crop dimensions
            resolution: float - GSD in meters/pixel

        Returns:
            query_indices: [C, H, W, 1] - query offset for all pixels

        Note:
            All pixels in a given resolution share the same query offset,
            regardless of crop size. This is a modality-level identifier.
        """
        # Handle both (C, H, W) and (H, W)
        if len(image_shape) == 3:
            C, H, W = image_shape
        elif len(image_shape) == 2:
            H, W = image_shape
            C = 1
        else:
            raise ValueError(f"image_shape must be (C,H,W) or (H,W), got {image_shape}")

        # Auto-register if needed
        ref_size = self._ensure_resolution_registered(resolution)

        global_offset = self.lookup.get_query_offset(resolution, ref_size)

        # All pixels get the same query offset
        return torch.full((C, H, W, 1), global_offset, dtype=torch.float32)

    # =========================================================================
    # HIGH-LEVEL TOKEN CONSTRUCTION
    # =========================================================================

    def build_tokens(
        self,
        image,
        label,
        resolution,
        spectral_indices,
        resolution_idx,
        time_idx=-1,
    ):
        """
        Build complete token array for image data.

        Token format: [value, x, y, spectral_idx, label, query_idx, resolution_idx, time_idx]
                       col 0    1  2       3          4        5            6            7

        Args:
            image: [C, H, W] - image data (reflectance/radiance values)
            label: [H, W] - per-pixel labels
            resolution: float - GSD in meters/pixel
            spectral_indices: [C] - spectral lookup indices for each band
            resolution_idx: int - resolution group index
            time_idx: int - temporal index (-1 for N/A)

        Returns:
            tokens: [C*H*W, 8] - flattened token array

        Example:
            >>> image = torch.randn(12, 240, 240)  # 12 bands
            >>> label = torch.randint(0, 15, (240, 240))
            >>> spectral_indices = torch.tensor([...])  # 12 wavelength indices
            >>> tokens = builder.build_tokens(
            ...     image, label, resolution=10.0,
            ...     spectral_indices=spectral_indices,
            ...     resolution_idx=0, time_idx=-1
            ... )
            >>> tokens.shape  # [12*240*240, 8] = [691200, 8]
        """
        C, H, W = image.shape

        # Get position and query coordinates (from reference grid)
        x_indices, y_indices = self.get_position_coordinates((C, H, W), resolution)
        query_indices = self.get_query_coordinates((C, H, W), resolution)

        # Expand spectral indices to [C*H*W]
        spectral_coords = spectral_indices.repeat_interleave(H * W)

        # Expand label to [C, H, W]
        label_expanded = label.unsqueeze(0).expand(C, -1, -1)

        # Create resolution and time columns
        resolution_col = torch.full((C, H, W, 1), resolution_idx, dtype=torch.float32)
        time_col = torch.full((C, H, W, 1), time_idx, dtype=torch.float32)

        # Concatenate all features
        # [value, x, y, spectral, label, query, resolution, time]
        tokens = torch.cat([
            image.unsqueeze(-1),                           # [C, H, W, 1] - col 0
            x_indices.float(),                              # [C, H, W, 1] - col 1
            y_indices.float(),                              # [C, H, W, 1] - col 2
            spectral_coords.view(C, H, W, 1).float(),      # [C, H, W, 1] - col 3
            label_expanded.unsqueeze(-1).float(),          # [C, H, W, 1] - col 4
            query_indices.float(),                          # [C, H, W, 1] - col 5
            resolution_col,                                 # [C, H, W, 1] - col 6
            time_col,                                       # [C, H, W, 1] - col 7
        ], dim=-1)

        # Flatten to [C*H*W, 8]
        return einops.rearrange(tokens, "c h w f -> (c h w) f")

    def build_queries(
        self,
        label,
        resolution,
        first_spectral_idx,
        resolution_idx,
        time_idx=-1,
    ):
        """
        Build query token array.

        Query tokens are used for reconstruction/prediction tasks. They have
        the same format as image tokens but with value=0 (to be predicted).

        Token format: [value, x, y, spectral_idx, label, query_idx, resolution_idx, time_idx]
                       col 0    1  2       3          4        5            6            7

        Args:
            label: [H, W] - per-pixel labels
            resolution: float - GSD in meters/pixel
            first_spectral_idx: int - spectral index to use (typically first band)
            resolution_idx: int - resolution group index
            time_idx: int - temporal index (-1 for N/A)

        Returns:
            queries: [H*W, 8] - query token array

        Example:
            >>> label = torch.randint(0, 15, (240, 240))
            >>> queries = builder.build_queries(
            ...     label, resolution=10.0,
            ...     first_spectral_idx=0,
            ...     resolution_idx=0, time_idx=-1
            ... )
            >>> queries.shape  # [240*240, 8] = [57600, 8]
        """
        H, W = label.shape

        # Get position and query coordinates (from reference grid)
        x_indices, y_indices = self.get_position_coordinates((1, H, W), resolution)
        query_indices = self.get_query_coordinates((1, H, W), resolution)

        # Build query tokens (value=0, single channel)
        queries = torch.cat([
            torch.zeros(1, H, W, 1),                                      # col 0 - value (to predict)
            x_indices.float(),                                            # col 1 - x position
            y_indices.float(),                                            # col 2 - y position
            torch.full((1, H, W, 1), first_spectral_idx, dtype=torch.float),  # col 3 - spectral
            label.unsqueeze(0).unsqueeze(-1).float(),                     # col 4 - label
            query_indices.float(),                                        # col 5 - query offset
            torch.full((1, H, W, 1), resolution_idx, dtype=torch.float),  # col 6 - resolution
            torch.full((1, H, W, 1), time_idx, dtype=torch.float),        # col 7 - time
        ], dim=-1)

        # Flatten to [H*W, 8]
        return einops.rearrange(queries, "c h w f -> (c h w) f")


    """
    Extension to TokenBuilder for sparse (irregularly positioned) tokens.

    ADD THE METHODS BELOW INTO THE EXISTING TokenBuilder CLASS in
    `token_builder.py`. They follow the same conventions as `build_tokens` and
    `build_queries`:

        Token format: [value, x, y, spectral_idx, label, query_idx,
                       resolution_idx, time_idx]

    But for sparse inputs:
        - One token per (point, channel) pair — for single-channel modalities
          (LIDAR elevation), that's just one token per point.
        - Positions are passed in pixel-equivalent coordinates (continuous,
          not integer), already in the reference grid's coordinate frame.
        - The reference-grid centering and global offset are applied here to
          match the convention used by build_tokens for dense rasters.

    Why pixel-equivalent positions rather than world coordinates? Because
    build_tokens generates positions via meshgrid over [0, H) × [0, W), then
    centers them in the reference grid. For sparse points to share the same
    coordinate frame, they should be passed in the same [0, H) × [0, W)
    pixel-equivalent space, NOT in absolute world coords (which would have
    arbitrary magnitudes that the Fourier features can't handle).
    """




    # ============================================================================
    # ADD INTO TokenBuilder CLASS
    # ============================================================================

    def build_sparse_tokens(
        self,
        values,                # [N] or [N, C] — point values (e.g., elevation)
        positions,             # [N, 2] — (x_pix, y_pix) in patch-local pixel coords
        labels,                # [N] — per-point labels
        resolution,            # float — GSD context (0.2m for LIDAR aligned with VHR)
        spectral_indices,      # [C] or scalar — spectral lookup index per channel
        resolution_idx,        # int — resolution group index
        patch_size_px,         # int — reference patch size in pixels (250 for FRACTAL)
        time_idx=-1,
        return_number=None,        # NEW: [N] int LIDAR return numbers (1..number_of_returns)
        number_of_returns=None,    # NEW: [N] int LIDAR total returns per pulse
        intensity_override=None,  # NEW: [N] per-point raw/normalized value. When
                                   # provided, col 6 carries THIS value (broadcast
                                   # across channels) instead of the constant
                                   # resolution_idx. Intended for LIDAR intensity —
                                   # resolution is constant/uninformative for LIDAR,
                                   # so this repurposes that column rather than
                                   # adding a second token channel.
    ):
        """
        Build tokens for sparse/irregular inputs (e.g., LIDAR points).

        Echo encoding:
            When both `return_number` and `number_of_returns` are provided,
            each token's column 7 (normally `time_idx`) is replaced with the
            echo index from `self.lookup.get_echo_idx(r, t)`. This index
            references a pre-computed (a, b) continuous encoding stored in
            the Lookup_encoding's echo table (built once via
            `Lookup_encoding.build_echo_continuous_lut()`).

            The downstream encoder is expected to detect LIDAR tokens via
            their spectral_idx (col 3) and route col 7 to the echo
            continuous-encoding LUT (followed by Fourier features) instead
            of the time encoder for those tokens.

            If either return_number or number_of_returns is None, behavior
            is identical to the legacy version: col 7 is set to time_idx
            for all tokens.

        Intensity override (NEW):
            When `intensity_override` is provided, column 6 (normally
            `resolution_idx`) carries this per-point continuous value
            instead of the constant resolution index. Resolution is
            constant/uninformative for LIDAR (fixed GSD), so this
            repurposes that column to carry the second per-point LIDAR
            value (intensity) without doubling token count via a second
            channel. If None, column 6 is filled with resolution_idx as
            before (legacy/default behavior — used by every existing
            caller, e.g. FRACTAL).

            The downstream TokenProcessor must be a variant that knows to
            route column 6 for these tokens through an intensity encoder
            instead of the categorical resolution embedding (e.g.
            DalesTokenProcessor), otherwise column 6 will be silently
            misinterpreted as an out-of-range resolution index.

        Args:
            values: [N] for single-channel, or [N, C] for multi-channel sparse data.
                    For LIDAR elevation: [N] tensor of z-values (normalized).
            positions: [N, 2] — (x_pix, y_pix) coordinates in patch-local pixel space.
                      Should be in [0, patch_size_px) range. The reference-grid
                      centering offset is added here, NOT by the caller.
            labels: [N] — per-point class labels (post-remap, in [0, num_classes)
                    or IGNORE_INDEX).
            resolution: float — GSD this modality is registered at. For LIDAR
                        aligned with VHR, use 0.2.
            spectral_indices: [C] or scalar — lookup indices for each channel.
                              For LIDAR with one elevation channel: scalar (int)
                              or [1].
            resolution_idx: int — resolution group id from the lookup table.
            patch_size_px: int — the nominal pixel size of the patch at this
                           resolution (e.g., 250 for a 50m patch at 0.2m). Used to
                           center positions in the reference grid the same way
                           build_tokens does for dense rasters of shape (H, W).
            time_idx: int — temporal index. -1 for atemporal data (LIDAR).
                     IGNORED if echo info is provided (col 7 is then filled
                     per-point from the echo lookup).
            return_number: [N] int array or tensor — LIDAR return number for
                           each point (1-indexed; 1 <= r <= number_of_returns).
                           Pass None to keep legacy time_idx behavior.
            number_of_returns: [N] int array or tensor — total returns from
                               the same pulse for each point. Pass None to
                               keep legacy time_idx behavior.
            intensity_override: [N] tensor or None — per-point continuous
                                value (e.g. normalized LIDAR intensity) to
                                place in column 6 INSTEAD of resolution_idx.
                                Pass None to keep legacy resolution_idx
                                behavior in column 6.

        Returns:
            tokens: [N*C, 8] (or [N, 8] for single-channel input) — same token
                    format as build_tokens.
        """
        # ── Coerce shapes ────────────────────────────────────────────
        if values.dim() == 1:
            values = values.unsqueeze(-1)             # [N, 1]
        N, C = values.shape

        if isinstance(spectral_indices, int):
            spectral_indices = torch.tensor([spectral_indices], dtype=torch.long)
        elif spectral_indices.dim() == 0:
            spectral_indices = spectral_indices.view(1)
        assert spectral_indices.shape[0] == C, (
            f"spectral_indices has {spectral_indices.shape[0]} channels, "
            f"but values has C={C}"
        )

        # ── Auto-register the resolution (matches build_tokens behavior) ─
        ref_size = self._ensure_resolution_registered(resolution)
        if patch_size_px > ref_size:
            raise ValueError(
                f"patch_size_px ({patch_size_px}) exceeds reference grid "
                f"({ref_size}) at resolution {resolution}m. "
                f"Increase REFERENCE_SIZES[{resolution}]."
            )

        # ── Apply reference-grid centering (mirror of get_position_coordinates) ─
        # In build_tokens, a dense raster of size (H, W) is centered in the
        # reference grid: pixel (i, j) → reference position (i + (ref_size//2 -
        # H//2), j + (ref_size//2 - W//2)) + global_offset. We do the same here.
        global_offset = self.lookup.get_or_register_modality(resolution, ref_size)
        ref_center = ref_size // 2
        half = patch_size_px // 2
        origin = ref_center - half                    # same offset build_tokens uses

        x_pix = positions[:, 0].float() + origin + global_offset   # [N]
        y_pix = positions[:, 1].float() + origin + global_offset   # [N]

        # ── Query offset (same for all tokens at this resolution) ────
        query_offset = self.lookup.get_query_offset(resolution, ref_size)

        # ── Build the per-token "time / echo" column ─────────────────
        # If echo info is provided, look up the echo_idx for each (r, t) pair
        # via Lookup_encoding.get_echo_idx and use that as col 7. Otherwise,
        # fall back to broadcasting the scalar time_idx across all tokens
        # (legacy behavior for non-LIDAR sparse modalities).
        echo_provided = (return_number is not None and number_of_returns is not None)
        if echo_provided:
            if isinstance(return_number, torch.Tensor):
                r_arr = return_number.detach().cpu().long().tolist()
            else:
                r_arr = [int(v) for v in return_number]
            if isinstance(number_of_returns, torch.Tensor):
                t_arr = number_of_returns.detach().cpu().long().tolist()
            else:
                t_arr = [int(v) for v in number_of_returns]
            assert len(r_arr) == N and len(t_arr) == N, (
                f"return_number / number_of_returns length mismatch: "
                f"got {len(r_arr)} and {len(t_arr)}, expected {N}"
            )
            # Per-point lookup is O(1) dict access; the Python loop runs once
            # per __getitem__ (not per training step), so the cost is small
            # relative to LAZ I/O. ~16k points → ~5ms.
            echo_indices = [self.lookup.get_echo_idx(r, t)
                            for r, t in zip(r_arr, t_arr)]
            time_col_per_point = torch.tensor(echo_indices, dtype=torch.float32)  # [N]
        else:
            time_col_per_point = torch.full((N,), float(time_idx),
                                             dtype=torch.float32)

        # ── Repeat per-point fields to [N*C] ─────────────────────────
        # values: [N, C] → flatten to [N*C, 1]; channels vary fastest
        val_flat       = einops.rearrange(values, "n c -> (n c) 1").float()
        x_flat         = einops.repeat(x_pix,   "n -> (n c)", c=C).unsqueeze(-1)
        y_flat         = einops.repeat(y_pix,   "n -> (n c)", c=C).unsqueeze(-1)
        label_flat     = einops.repeat(labels.float(), "n -> (n c)", c=C).unsqueeze(-1)
        spectral_flat  = einops.repeat(spectral_indices.float(), "c -> (n c)", n=N).unsqueeze(-1)
        query_flat     = torch.full((N * C, 1), float(query_offset))

        # ── Build the per-token "resolution / intensity" column ──────
        # If intensity_override is provided, col 6 carries the per-point
        # continuous value (broadcast across channels) instead of the
        # constant resolution_idx. Otherwise, legacy behavior (constant
        # resolution_idx for every token) is preserved exactly.
        if intensity_override is not None:
            assert intensity_override.shape[0] == N, (
                f"intensity_override has {intensity_override.shape[0]} "
                f"entries, expected N={N}"
            )
            resolution_flat = einops.repeat(
                intensity_override.float(), "n -> (n c)", c=C
            ).unsqueeze(-1)
        else:
            resolution_flat = torch.full((N * C, 1), float(resolution_idx))

        time_flat      = einops.repeat(time_col_per_point,
                                        "n -> (n c)", c=C).unsqueeze(-1)

        tokens = torch.cat([
            val_flat,        # col 0
            x_flat,          # col 1
            y_flat,          # col 2
            spectral_flat,   # col 3
            label_flat,      # col 4
            query_flat,      # col 5
            resolution_flat, # col 6  (intensity_override for LIDAR if given,
                             #          resolution_idx otherwise)
            time_flat,       # col 7  (echo_idx for LIDAR, time_idx otherwise)
        ], dim=-1)
        return tokens


    def build_sparse_queries(
        self,
        positions,             # [N, 2] in patch-local pixel coords
        labels,                # [N] per-point labels
        resolution,
        first_spectral_idx,
        resolution_idx,
        patch_size_px,
        time_idx=-1,
    ):
        """
        Build query tokens for sparse outputs (per-point segmentation).

        Identical structure to build_queries() but for irregular positions.
        Used when the target is per-point classification (FRACTAL LIDAR).

        Note: queries don't carry echo info — that's an INPUT modality
        feature, not a query feature. Col 7 of queries holds time_idx as
        before (typically -1 for FRACTAL). Queries also always hold the
        constant resolution_idx in col 6 (no intensity_override support
        here — intensity is an INPUT token feature, not a query feature).
        """
        N = positions.shape[0]

        ref_size = self._ensure_resolution_registered(resolution)
        if patch_size_px > ref_size:
            raise ValueError(
                f"patch_size_px ({patch_size_px}) exceeds reference grid "
                f"({ref_size}) at resolution {resolution}m."
            )

        global_offset = self.lookup.get_or_register_modality(resolution, ref_size)
        query_offset  = self.lookup.get_query_offset(resolution, ref_size)
        ref_center    = ref_size // 2
        half          = patch_size_px // 2
        origin        = ref_center - half

        x_pix = positions[:, 0].float() + origin + global_offset   # [N]
        y_pix = positions[:, 1].float() + origin + global_offset   # [N]

        queries = torch.cat([
            torch.zeros(N, 1),                                     # col 0: value (to predict)
            x_pix.unsqueeze(-1),                                   # col 1
            y_pix.unsqueeze(-1),                                   # col 2
            torch.full((N, 1), float(first_spectral_idx)),         # col 3
            labels.float().unsqueeze(-1),                          # col 4
            torch.full((N, 1), float(query_offset)),               # col 5
            torch.full((N, 1), float(resolution_idx)),             # col 6
            torch.full((N, 1), float(time_idx)),                   # col 7
        ], dim=-1)
        return queries

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    @staticmethod
    def subsample_queries(queries, max_queries, ignore_index=255,
                              prioritize_valid=True, return_indices=False):
            """
            Subsample query tokens with optional prioritization of valid labels.

            Args:
                queries: [N, 8] - query token array
                max_queries: int - maximum number of queries to return
                ignore_index: int - label value to consider invalid (default: 255)
                prioritize_valid: bool - if True, prioritize valid labels
                return_indices: bool - if True, also return the kept row indices
                                into the original `queries` tensor. Default False
                                preserves the original (queries-only) return.

            Returns:
                subsampled_queries: [max_queries, 8]
                (and, if return_indices: kept_indices [max_queries] long)
            """
            if queries.shape[0] <= max_queries:
                if return_indices:
                    return queries, torch.arange(queries.shape[0])
                return queries

            if not prioritize_valid:
                perm = torch.randperm(queries.shape[0])[:max_queries]
                if return_indices:
                    return queries[perm], perm
                return queries[perm]

            query_labels = queries[:, 4]
            valid_mask = (query_labels != ignore_index)
            valid_indices = torch.where(valid_mask)[0]
            invalid_indices = torch.where(~valid_mask)[0]

            if len(valid_indices) >= max_queries:
                perm = torch.randperm(len(valid_indices))[:max_queries]
                selected = valid_indices[perm]
            else:
                n_invalid_needed = max_queries - len(valid_indices)
                n_invalid_needed = min(n_invalid_needed, len(invalid_indices))
                if n_invalid_needed > 0:
                    invalid_perm = torch.randperm(len(invalid_indices))[:n_invalid_needed]
                    selected = torch.cat([valid_indices, invalid_indices[invalid_perm]])
                    selected = selected[torch.randperm(len(selected))]
                else:
                    selected = valid_indices

            if return_indices:
                return queries[selected], selected
            return queries[selected]

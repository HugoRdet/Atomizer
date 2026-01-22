import torch
import pytorch_lightning as pl

class Lookup_encoding(pl.LightningModule):
    def __init__(self, modalities_config, bands_info, config_model):
        super().__init__()
        self.config = modalities_config
        self.bands_info = bands_info
        self.modalities = None
        self.table = None
        self.pixel_coords_table = None  # NEW: Table for (x, y) pixel coordinates
        self.table_wave = None
        self.table_queries = None
        self.nb_tokens_queries = config_model["Atomiser"]["spatial_latents"]

        self.init_config()
        self.init_lookup_table()
        # Initialize the coordinate table alongside the existing index tables
        self.init_pixel_coords_table() 
        self.init_lookup_table_wave()
        self.init_queries_lookup_table()

    def init_config(self):
        modalities = []
        # Manual entry for 0.2m GSD and 512x512 size
        modalities.append((0.2, 512))
        modalities.append((0.2, 28))
        self.modalities = modalities
    
    def init_lookup_table(self):
        table = dict()
        idx_torch_array = 0
        for couple in self.modalities:
            resolution, size = couple
            res_key = int(resolution * 1000)
            table[(res_key, size)] = idx_torch_array
            idx_torch_array += size
        self.table = table

    def init_pixel_coords_table(self):
        """
        NEW: Initializes a lookup table that maps each modality to a 
        static global (x, y) coordinate grid.
        """
        coords_table = dict()
        
        for resolution, size in self.modalities:
            res_key = int(resolution * 1000)
            
            # Generate a 2D grid of global indices [size, size]
            # y corresponds to row index, x corresponds to column index
            grid_y, grid_x = torch.meshgrid(
                torch.arange(size, dtype=torch.float32),
                torch.arange(size, dtype=torch.float32),
                indexing='ij'
            )
            
            # Flatten to [size*size, 2] to match token order
            # Each entry is [x_idx, y_idx]
            coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
            
            coords_table[(res_key, size)] = coords
            
        self.pixel_coords_table = coords_table

    def get_pixel_coords(self, resolution, size):
        """
        NEW: Directly returns the (x, y) global coordinates for a given
        resolution and image size.
        """
        res_key = int(resolution * 1000)
        key = (res_key, size)
        
        if key not in self.pixel_coords_table:
            raise ValueError(f"Coords for resolution {resolution} and size {size} not found.")
            
        return self.pixel_coords_table[key]

    def get_grid_pos(self, resolution, size):
        """Original 1D index getter (kept for backward compatibility)"""
        res_key = int(resolution * 1000)
        key = (res_key, size)
        if key not in self.table:
            raise ValueError(f"Resolution {res_key} and size {size} not found.")
        
        idx = self.table[key]
        return torch.arange(idx, idx + size, dtype=torch.float32)

    def init_queries_lookup_table(self):
        table = dict()
        idx_torch_array = 0
        for resolution, size in self.modalities:
            res_key = int(resolution * 1000)
            table[(res_key, size)] = idx_torch_array
            idx_torch_array += self.nb_tokens_queries
        self.table_queries = table

    def init_lookup_table_wave(self):
        table = dict()
        idx_torch_array = 0
        for sat in self.bands_info:
            sat_content = self.bands_info[sat]
            for band in sat_content:
                band_content = sat_content[band]
                if "bandwidth" not in band_content or "central_wavelength" not in band_content:
                    continue
                bandwidth = band_content["bandwidth"]
                central_wavelength = band_content["central_wavelength"]
                key = (int(bandwidth), int(central_wavelength))
                if key not in table:
                    table[key] = idx_torch_array
                    idx_torch_array += 1
        self.table_wave = table
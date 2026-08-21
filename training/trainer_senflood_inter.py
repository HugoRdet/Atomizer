# training/trainer_SENFLOOD_skip_inter.py
from training.trainer_SENFLOOD_skip import Model_SenFlood_Skip
from training.atomiser.Atomizer_skip_inter import Atomizer_skip_inter

class Model_SenFlood_Skip_Inter(Model_SenFlood_Skip):
    """
    Identical to Model_SenFlood_Skip except self.encoder is the
    attention-exposing Atomizer_skip_inter subclass. No new/renamed
    parameters, so checkpoints trained with Model_SenFlood_Skip load here
    unchanged via load_from_checkpoint(strict=False, ...).
    """
    def __init__(self, config, wand, name, transform, lookup_table,
                 class_names=None, band_group_lut=None):
        super().__init__(config, wand, name, transform, lookup_table, class_names)
        # Replace the encoder built by the parent __init__
        self.encoder = Atomizer_skip_inter(config=self.config, lookup_table=self.lookup_table)
        if band_group_lut is not None:
            self.encoder.band_group_lut = band_group_lut

    def forward_with_attention(self, batch):
        return self.encoder.forward_with_attention(batch)

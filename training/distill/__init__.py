"""
training.distill -- DUNE-style co-distillation of frozen teachers into the Atomiser U-Net.

Teachers: OlmoEarth (Sentinel tiles) and DINOv3 (FLAIR-HUB aerial RGB).

Modules (import them explicitly, e.g. `from training.distill.distill_heads import DistillStudent`):
    distill_heads      LatentQueryHead (teacher head, DUNE projector analogue), DistillStudent, token_centers_from_tokens
    distill_losses     PatchStandardizer (teacher norm), distill_loss_per_image, aggregate_teacher_losses, resid_corr
    distill_optim      DUNE optimizer helpers: parameter groups, per-parameter clipping, set_lr_wd
    distill_schedule   DUNE iteration count, lr scaling, warmup + cosine, teacher-norm momentum schedule
    teachers           OlmoEarthTeacher, DinoV3Teacher
    trainer_distill    Model_Distill (LightningModule), DistillDataModule, StopAtIters, SaveEncoder
    check_distill_data sanity checks / z-score statistics (script)
    smoke_distill      synthetic end-to-end smoke test (script)

This file deliberately imports nothing: importing the package must not pull in torch, Lightning or the teachers.
"""

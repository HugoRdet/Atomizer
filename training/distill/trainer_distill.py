"""
trainer_distill.py -- Lightning module + data module + callbacks for the DUNE-style distillation of Atomiser_Unet.

Where it goes: training/distill/trainer_distill.py   (conventions follow trainer_BURNSCARS.py / train_burnscars_unet.py)

Training step (DUNE 'no data sharing', 'own_data' teacher assignment):
  * the round-robin sampler gives one homogeneous batch per Lightning step, alternating Sentinel / FLAIR;
    accumulate_grad_batches=2 therefore makes every OPTIMIZER step contain one batch of each dataset (DUNE's balanced batch);
    Lightning divides each micro-batch loss by 2, i.e. the average over (image, teacher) pairs of DUNE's aggregation;
  * teacher -> standardized targets (PatchStandardizer, EMA momentum schedule 1.0 -> 0.001);
  * student.encode -> heads at the teacher's token centres -> 0.5 cos + 0.5 smooth-L1 per image, averaged;
  * DUNE's AdamW (betas 0.9/0.99, eps 1e-15), lr = lr * batch / 256 with warmup + cosine, weight decay on group 0 only,
    PER-PARAMETER gradient clipping.
Termination is by optimizer steps (DUNE's pseudo-epoch: epoch_size images), through StopAtIters.

FLAIR views and teachers (distill.flair_views, distill.flair_teachers):
  * every FLAIR batch takes one view {crop, up} drawn with its weight (same on all ranks): a crop x crop patch, whose
    DINOv3 input is upsampled `up` times, so one teacher token covers 16/up crop pixels (grid crop*up/16); each view
    may set its own batch size (`batch`, default batch_per_dataset);
  * every FLAIR batch is supervised by EVERY teacher of flair_teachers (e.g. dino = SAT-493M, pano = Panopticon on
    R, G, B, NIR upsampled by AnyUp to one token per pano_patch x pano_patch crop pixels; dino_web = LVD-1689M),
    each with its own standardizer and projector head; the batch loss is their mean. A DINOv3 teacher `<key>` reads
    distill.<key>_ckpt and distill.<key>_norm (`dino` keeps dino_ckpt / dino_norm);
  * validation: one FLAIR loader per view; logs val_loss_<tag> (tag flair_c<crop>_x<up>) and, with several teachers,
    val_loss_<tag>_<teacher>.
Without flair_views, the single view is (flair_crop, 16 / flair_patch): the previous behaviour.

Student: Atomiser_Unet (DistillStudent) by default; with PTv3.enabled: true in the config, the AtomizerPTv3 student of
distill_heads_ptv3.py (full U-Net, decoder level-0 features averaged over each teacher patch).
"""
import contextlib
import os
import time
from collections import deque

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from training.distill.distill_heads import DistillStudent, token_centers_from_tokens
from training.distill.distill_losses import PatchStandardizer, distill_loss_per_image, resid_corr
from training.distill.distill_optim import clip_per_parameter, make_optimizer, set_lr_wd
from training.distill.distill_schedule import cosine_scheduler, dune_lr_schedule, tnorm_momentum_schedule
from training.distill.teachers import DinoV3Teacher, OlmoEarthTeacher, PanopticonTeacher, SmartiesTeacher
from training.utils.datasets.token_grouping import RoundRobinDistributedBatchSampler
from training.utils.datasets.utils_dataset_distill import (DistillConcat, FixedView, FlairAtoms, MixedViewBatchSampler,
                                                           OlmoSentinelAtoms, collate_distill)

DINO_PATCH = 16


def parse_flair_views(dc):
    """distill.flair_views -> [dict(crop, up, patch, weight, batch, tag)]; default: one view (flair_crop, 16/flair_patch)."""
    bpd = int(dc["batch_per_dataset"])
    raw = dc.get("flair_views")
    if not raw:
        patch = int(dc.get("flair_patch", DINO_PATCH))
        if DINO_PATCH % patch:
            raise ValueError(f"flair_patch {patch} must divide {DINO_PATCH}")
        raw = [dict(crop=int(dc["flair_crop"]), up=DINO_PATCH // patch)]
    views = []
    for v in raw:
        crop, up = int(v["crop"]), int(v.get("up", 1))
        if up < 1 or DINO_PATCH % up:
            raise ValueError(f"flair view {v}: up must divide {DINO_PATCH}")
        patch = DINO_PATCH // up
        if crop % patch:
            raise ValueError(f"flair view {v}: crop {crop} not divisible by the teacher patch {patch}")
        views.append(dict(crop=crop, up=up, patch=patch, weight=float(v.get("weight", 1.0)),
                          batch=int(v.get("batch", bpd)), tag=f"flair_c{crop}_x{up}"))
    tags = [v["tag"] for v in views]
    if len(set(tags)) != len(tags):
        raise ValueError(f"duplicate flair views {tags}")
    return views


class Model_Distill(pl.LightningModule):
    TEACHER_DIMS = {"olmo": 768, "dino": 1024}
    PANO_DIM = 768
    SMARTIES_DIM = 768

    def __init__(self, config, lookup_table, name="distill"):
        super().__init__()
        self.strict_loading = False
        self.config, self.lookup_table, self.name = config, lookup_table, name
        dc = self.dc = config["distill"]
        self.views = parse_flair_views(dc)
        self.view_by_key = {(v["crop"], v["up"]): v for v in self.views}
        self.flair_teachers = [str(t) for t in dc.get("flair_teachers", ["dino"])]
        if not self.flair_teachers or any(not (t.startswith("dino") or t == "pano") for t in self.flair_teachers):
            raise ValueError(f"flair_teachers: DINOv3 keys 'dino*' and/or 'pano', got {self.flair_teachers}")
        self.olmo_teachers = [str(t) for t in dc.get("olmo_teachers", ["olmo"])]
        if not self.olmo_teachers or any(t not in ("olmo", "smarties") for t in self.olmo_teachers):
            raise ValueError(f"olmo_teachers: subset of ['olmo', 'smarties'], got {self.olmo_teachers}")
        self.teacher_dims = {**{t: (self.TEACHER_DIMS["olmo"] if t == "olmo" else self.SMARTIES_DIM) for t in self.olmo_teachers},
                             **{t: (self.PANO_DIM if t == "pano" else self.TEACHER_DIMS["dino"]) for t in self.flair_teachers}}
        self.pano_patch = int(dc.get("pano_patch", DINO_PATCH))           # crop pixels per (upsampled) Panopticon token
        # teachers whose grid is set by an AnyUp output side: footprint in tile pixels per token
        self.teacher_patch = {"pano": self.pano_patch, "smarties": int(dc.get("smarties_patch", 2))}
        if "smarties" in self.olmo_teachers:
            sp = self.teacher_patch["smarties"]
            if 128 % sp or 128 // sp < 14:
                raise ValueError(f"smarties_patch {sp}: must divide the 128-px Sentinel tile with >= 14 tokens per side")
        if "pano" in self.flair_teachers:
            for v in self.views:
                if v["crop"] % self.pano_patch or (v["crop"] * v["up"]) % DINO_PATCH:
                    raise ValueError(f"view {v['tag']}: crop not divisible by pano_patch {self.pano_patch}")
                if v["crop"] // self.pano_patch < v["crop"] * v["up"] // DINO_PATCH:
                    raise ValueError(f"view {v['tag']}: pano_patch {self.pano_patch} coarser than Panopticon's own grid")
        self.layout = {
            "olmo": dict(teacher="olmo", k=int(dc["k_olmo"]), patch=int(dc["olmo_patch"]), hw=128),
            "flair": dict(teacher="dino", k=int(dc["k_flair"]), patch=int(dc.get("flair_patch", 16)), hw=int(dc["flair_crop"])),
        }
        if (config.get("PTv3") or {}).get("enabled", False):           # AtomizerPTv3 student (distill_heads_ptv3.py)
            from training.distill.distill_heads_ptv3 import PTv3DistillStudent
            self.student = PTv3DistillStudent(
                config, lookup_table, self.teacher_dims,
                teacher_patches={L["teacher"]: L["patch"] for L in self.layout.values()},
                dataset_teacher={name: L["teacher"] for name, L in self.layout.items()})
        else:
            self.student = DistillStudent(config, lookup_table, self.teacher_dims, n_pre=int(dc.get("n_pre", 1)))
        self.tnorm = nn.ModuleDict({k: PatchStandardizer(d) for k, d in self.teacher_dims.items()})
        print(f"[Distill] FLAIR teachers {self.flair_teachers} | views "
              + ", ".join(f"{v['tag']} (grid {v['crop'] * v['up'] // DINO_PATCH}, batch {v['batch']}, w {v['weight']:g})"
                          for v in self.views))
        self._teachers = {}                      # plain dict: teachers are NOT registered (not in state_dict, not in DDP)
        self.lr_sched = self.tnorm_sched = None
        self.iters = None
        self._hist = {"olmo": deque(maxlen=400), "flair": deque(maxlen=400)}
        self._val_acc = []
        self._viz, self._last = {}, None        # validation visualization buffers
        self._t_last, self.bs_global = None, None

    # ------------------------------------------------------------------ teachers (lazy, on the right device)
    def _teacher(self, key):
        if key not in self._teachers:
            dc = self.dc
            if key == "olmo":
                t = OlmoEarthTeacher(dc["olmo_ckpt"], int(dc["olmo_patch"]))
            elif key == "smarties":              # SMARTIES on the 12 S2 bands, AnyUp to the smarties_patch grid
                t = SmartiesTeacher(dc.get("smarties_dir", "./checkpoints/smarties/SMARTIES-v1-ViT-B"), dc["olmo_stats"],
                                    dc.get("smarties_anyup_dir", dc.get("pano_anyup_dir", "./checkpoints/anyup")),
                                    int(dc.get("smarties_q_chunk", 256)), int(dc.get("smarties_anyup_batch", 2)))
            elif key == "pano":                  # Panopticon on R, G, B, NIR, AnyUp to the pano_patch grid
                t = PanopticonTeacher(dc.get("pano_dir", "./checkpoints/panopticon"), dc["flair_stats"],
                                      dc.get("pano_wavelengths", [664, 559, 493, 842]),
                                      dc.get("pano_anyup_dir", "./checkpoints/anyup"), int(dc.get("pano_q_chunk", 256)),
                                      int(dc.get("pano_anyup_batch", 2)))
            else:                                # DINOv3 teacher: <key>_ckpt / <key>_norm; default view's upsampling
                t = DinoV3Teacher(dc[f"{key}_ckpt"], dc.get(f"{key}_norm", "auto"),
                                  student_patch=int(dc.get("flair_patch", DINO_PATCH)))
                print(f"[Distill] teacher {key}: {dc[f'{key}_ckpt']}")
            t.eval()
            for p in t.parameters():
                p.requires_grad_(False)
            self._teachers[key] = t
        t = self._teachers[key]
        if next(t.parameters()).device != self.device:
            t.to(self.device)
        return t

    # ------------------------------------------------------------------ one micro-batch
    def _step(self, batch, train):
        """Returns (loss, metrics averaged over teachers, dataset name, view tag, {teacher: (loss, metrics)})."""
        name = batch["dataset_name"]
        L = self.layout[name]
        tokens = batch["groups"][next(iter(batch["groups"]))]["tokens"]
        tin = batch["teacher"]
        if name == "flair":
            key = tuple(int(x) for x in batch.get("view", (L["hw"], DINO_PATCH // L["patch"])))
            v = self.view_by_key.get(key)
            if v is None:
                raise RuntimeError(f"FLAIR batch view {key} is not in flair_views {list(self.view_by_key)}")
            teachers, hw, patch, tag = self.flair_teachers, v["crop"], v["patch"], v["tag"]
            tin = dict(tin, up=float(v["up"]))
        else:
            teachers, hw, patch, tag = list(self.olmo_teachers), L["hw"], L["patch"], name
        if tokens.shape[1] % (hw * hw):
            raise RuntimeError(f"{tag}: {tokens.shape[1]} tokens do not match a {hw}x{hw} tile")
        batch["patch"] = patch                                                    # default pooling footprint
        m = int(float(self.dc.get("m_factor", 1.3)) * L["k"])
        d0, geom = self.student.encode(batch, L["k"], m)                          # student runs once per batch
        qs = {}
        mom = float(self.tnorm_sched[min(int(self.global_step), len(self.tnorm_sched) - 1)]) if train else 0.0
        no_amp = torch.autocast(self.device.type, enabled=False) if self.dc.get("teacher_fp32", False) else contextlib.nullcontext()
        per_t, losses, last = {}, [], {}
        for t in teachers:
            pt = self.teacher_patch.get(t, patch)                                 # this teacher's footprint (tile px)
            g = hw // pt
            if pt not in qs:
                qs[pt] = token_centers_from_tokens(tokens, hw, hw, pt)            # [B, g*g, 2]
            q = qs[pt]
            with torch.no_grad(), no_amp:
                tgt = self._teacher(t)(dict(tin, out_side=g) if t in self.teacher_patch else tin)   # [B, g, g, D]
            assert tgt.shape[1] == g and tgt.shape[2] == g, f"{t}/{tag}: teacher grid {tuple(tgt.shape[1:3])} != expected {g}"
            tgt = self.tnorm[t](tgt.flatten(1, 2), mom)                           # standardized [B, g*g, D]
            pred = self.student.predict(t, d0, geom, q)
            loss_i, mets = distill_loss_per_image(pred, tgt)
            losses.append(loss_i.mean())
            per_t[t] = (losses[-1].detach(), mets)
            if not train and self.trainer.is_global_zero:
                last[t] = (tgt.detach(), pred.detach(), g)
        if not train:
            self._last = last
        mets = {k: torch.stack([m[k] for _, m in per_t.values()]).mean() for k in next(iter(per_t.values()))[1]}
        return torch.stack(losses).mean(), mets, name, tag, per_t

    @staticmethod
    def _bs(batch):
        return int(next(iter(batch["groups"].values()))["tokens"].shape[0])

    def training_step(self, batch, batch_idx):
        loss, mets, name, tag, per_t = self._step(batch, True)
        bs = self._bs(batch)
        self._hist[name].append(float(loss.detach()))
        kw = dict(on_step=True, on_epoch=False, logger=True, batch_size=bs)
        self.log(f"train_loss_{name}", loss, prog_bar=True, **kw)
        self.log(f"train_cos_{name}", mets["cos"], **kw)
        self.log(f"train_fid_{name}", mets["fidelity"], **kw)
        if tag != name:                                                           # FLAIR view
            self.log(f"train_loss_{tag}", loss, **kw)
        if len(per_t) > 1:                                                        # several teachers on this batch
            for t, (lt, mt) in per_t.items():
                self.log(f"train_loss_{name}_{t}", lt, **kw)
                self.log(f"train_cos_{name}_{t}", mt["cos"], **kw)
        if name == "flair" and self.global_step % int(self.dc.get("log_every", 10)) == 0:
            c = resid_corr(self._hist["olmo"], self._hist["flair"])
            if c == c:                                                            # not NaN
                self.log("teacher_loss_corr", c, **kw)
        return loss

    def on_validation_epoch_start(self):
        self._val_acc, self._viz = [], {}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, mets, name, tag, per_t = self._step(batch, False)
        bs = self._bs(batch)
        # per-view keys are logged by ONE loader each; the combined val_loss is logged once, in on_validation_epoch_end
        # (Lightning forbids logging the same key from two dataloaders with different metadata)
        kw = dict(on_step=False, on_epoch=True, logger=True, sync_dist=True, add_dataloader_idx=False, batch_size=bs)
        self.log(f"val_loss_{tag}", loss, **kw)
        self.log(f"val_fid_{tag}", mets["fidelity"], **kw)
        if len(per_t) > 1:
            for t, (lt, mt) in per_t.items():
                self.log(f"val_loss_{tag}_{t}", lt, **kw)
                self.log(f"val_fid_{tag}_{t}", mt["fidelity"], **kw)
        self._val_acc.append(loss.detach())
        if batch_idx == 0 and self.trainer.is_global_zero:
            for t in per_t:
                self._stash_viz(tag if len(per_t) == 1 else f"{tag}_{t}", batch, t)
        return loss

    def on_validation_epoch_end(self):
        if self._val_acc:
            self.log("val_loss", torch.stack(self._val_acc).mean(), prog_bar=True, logger=True, sync_dist=True)
        self._val_acc = []
        if self.trainer.is_global_zero and self._viz and not self.trainer.sanity_checking:
            self._render_viz()
        self._viz = {}

    # ------------------------------------------------------------------ optimizer (DUNE)
    def configure_optimizers(self):
        dc = self.dc
        world = int(getattr(self.trainer, "world_size", 1))
        accum = int(dc.get("grad_accum", 1))
        w = sum(v["weight"] for v in self.views)
        flair_b = sum(v["weight"] * v["batch"] for v in self.views) / w             # expected FLAIR batch (views)
        bs_global = int(round(accum * (int(dc["batch_per_dataset"]) + flair_b) * world))   # images per optimizer step
        self.bs_global = bs_global
        min_lr, wfrac = float(dc.get("min_lr", 1e-6)), float(dc.get("warmup_frac", 0.2))
        peak = float(dc["lr"]) * bs_global / 256.0                                   # DUNE: lr * batch / 256
        if dc.get("steps") is not None:                                              # schedule defined in optimizer STEPS
            self.iters = int(dc["steps"])
            warm = int(dc["warmup_steps"]) if dc.get("warmup_steps") is not None else int(wfrac * self.iters)
            self.lr_sched = cosine_scheduler(peak, min_lr, self.iters, warm)
        else:                                                                        # legacy: epochs x epoch_size images
            self.lr_sched, self.iters = dune_lr_schedule(float(dc["lr"]), bs_global, float(dc["epochs"]), int(dc["epoch_size"]), min_lr, wfrac)
            warm = int(wfrac * self.iters)
        self.tnorm_sched = tnorm_momentum_schedule(self.iters, float(dc.get("tnorm_start", 1.0)), float(dc.get("tnorm_end", 0.001)))
        print(f"[Distill] global batch {bs_global} images/step = 2 datasets x {accum} accum x {dc['batch_per_dataset']} per GPU x {world} GPUs | "
              f"{self.iters} optimizer steps ({self.iters * bs_global / 1e6:.2f}M images) | peak lr {peak:.2e}, warmup {warm} steps")
        return make_optimizer(self, betas=tuple(float(b) for b in dc.get("betas", [0.9, 0.99])),
                              eps=float(dc.get("eps", 1e-15)), fused=torch.cuda.is_available())

    def on_before_optimizer_step(self, optimizer):
        step = int(self.trainer.global_step)
        lr = float(self.lr_sched[min(step, len(self.lr_sched) - 1)])
        set_lr_wd(optimizer, lr, float(self.dc.get("wd", 0.03)))
        kw = dict(on_step=True, on_epoch=False, logger=True, batch_size=1)
        self.log("lr", lr, **kw)
        grads = [p.grad.detach().norm() for p in self.parameters() if p.grad is not None]
        if grads:
            g = torch.stack(grads)
            self.log("gnorm_mean", g.mean(), **kw)
            self.log("gnorm_max", g.max(), **kw)
        # progress / schedule / throughput (rank 0 only where per-rank values are not meaningful)
        seen = step * (self.bs_global or 0)
        self.log("images_seen", float(seen), rank_zero_only=True, **kw)
        self.log("tnorm_momentum", float(self.tnorm_sched[min(step, len(self.tnorm_sched) - 1)]), rank_zero_only=True, **kw)
        now = time.time()
        if self._t_last is not None and self.bs_global:
            self.log("throughput_img_s", self.bs_global / max(now - self._t_last, 1e-9), rank_zero_only=True, **kw)
        self._t_last = now
        if step % int(self.dc.get("log_every", 10)) == 0:
            for k, nz in self.tnorm.items():
                self.log(f"tnorm_std_{k}", float(nz.std.mean()), rank_zero_only=True, **kw)
            if torch.cuda.is_available():
                self.log("gpu_mem_gib", torch.cuda.max_memory_allocated() / 2 ** 30, rank_zero_only=True, **kw)

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        # DUNE clips each parameter's gradient separately (not the global norm)
        clip_per_parameter(self.parameters(), float(self.dc.get("clip", 1.0)))

    def on_fit_start(self):
        """Run-level facts for the loggers (wandb config / hparams.yaml)."""
        if self.trainer.is_global_zero:
            info = {"world_size": self.trainer.world_size, "global_batch_images": self.bs_global, "total_optimizer_steps": self.iters,
                    "peak_lr": float(self.lr_sched.max()),
                    "student_params_M": sum(p.numel() for p in self.student.net.parameters()) / 1e6,
                    "heads_params_M": sum(p.numel() for p in self.student.heads.parameters()) / 1e6}
            for lg in self.trainer.loggers:
                lg.log_hyperparams(info)

    # ------------------------------------------------------------------ validation figures
    def _stash_viz(self, name, batch, teacher):
        """Keep up to 4 samples of the first validation batch of each view / teacher (CPU) for the PCA figure."""
        if name in self._viz or not self._last or teacher not in self._last:
            return
        tgt, pred, g = self._last[teacher]
        n = min(4, tgt.shape[0])
        tin = batch["teacher"]
        if batch["dataset_name"] == "flair":
            img = tin["rgb"][:n].float().div(255).permute(0, 2, 3, 1)                 # aerial RGB
        else:
            img = tin["s2n"][:n, :, :, 0, :][..., [2, 1, 0]].float()                  # Sentinel-2 B04, B03, B02 (OlmoEarth order)
        self._viz[name] = dict(img=img.cpu(), tgt=tgt[:n].float().cpu(), pred=pred[:n].float().cpu(), g=g)
        dense = getattr(self.student, "last_dense", None)          # PTv3 student: per-pixel features [n, H, W, C]
        if dense is not None:
            self._viz[name]["dense"] = dense[:n].float().cpu()

    def _render_viz(self):
        """Rows: input | teacher PCA | student PCA | 1 - cosine [| student pixels PCA].
        One PCA basis, fitted on the (standardized) teacher tokens, is applied to both teacher and student (same axes, so
        the hues are comparable). The student is stretched with its OWN 1-99 % range: early in training its predictions
        are much less spread than the teacher's and would look uniformly grey on the teacher's range; the title gives
        the spread ratio std(student) / std(teacher) along those axes. PTv3 student: a 5th column shows its per-pixel
        level-0 features (before the average over each teacher patch), with their own PCA basis and rank-equalized
        colours (each colour channel uses its full range, whatever the outliers)."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        try:
            from pytorch_lightning.loggers import WandbLogger
        except ImportError:
            WandbLogger = ()
        out = os.path.join(self.trainer.default_root_dir, "viz")
        os.makedirs(out, exist_ok=True)
        for name, v in self._viz.items():
            tgt, pred, g, img = v["tgt"], v["pred"], v["g"], v["img"]
            n, M, D = tgt.shape
            X = tgt.reshape(-1, D)
            mu = X.mean(0)
            _, V = torch.linalg.eigh((X - mu).T @ (X - mu))
            V = V[:, -3:].flip(-1)                                                    # top-3 principal axes
            zt, zp = (tgt - mu) @ V, (pred - mu) @ V
            lo = torch.quantile(zt.reshape(-1, 3), 0.01, dim=0)
            hi = torch.quantile(zt.reshape(-1, 3), 0.99, dim=0)
            norm = lambda z: ((z - lo) / (hi - lo + 1e-6)).clamp(0, 1).reshape(n, g, g, 3)  # noqa: E731
            ct = norm(zt)
            lo_p = torch.quantile(zp.reshape(-1, 3), 0.01, dim=0)                     # student's own range
            hi_p = torch.quantile(zp.reshape(-1, 3), 0.99, dim=0)
            cp = ((zp - lo_p) / (hi_p - lo_p + 1e-6)).clamp(0, 1).reshape(n, g, g, 3)
            spread = float(zp.reshape(-1, 3).std(0).mean() / (zt.reshape(-1, 3).std(0).mean() + 1e-12))
            err = (1 - F.cosine_similarity(pred, tgt, dim=-1)).reshape(n, g, g)
            dense = v.get("dense")
            cd = None
            if dense is not None:                                                     # per-pixel student features
                Xd = dense.reshape(-1, dense.shape[-1])
                sub = Xd[torch.randperm(Xd.shape[0])[:100000]]                        # PCA fit on a subsample
                mud = sub.mean(0)
                _, Vd = torch.linalg.eigh((sub - mud).T @ (sub - mud))
                zd = (Xd - mud) @ Vd[:, -3:].flip(-1)                                 # [P, 3]
                rank = torch.empty_like(zd)                                           # histogram equalization
                order = zd.argsort(dim=0)
                rank.scatter_(0, order, torch.arange(zd.shape[0], dtype=zd.dtype).unsqueeze(1).expand(-1, 3))
                cd = (rank / max(zd.shape[0] - 1, 1)).reshape(*dense.shape[:3], 3)
            ncol = 4 if cd is None else 5
            fig, ax = plt.subplots(n, ncol, figsize=(2.2 * ncol, 2.2 * n), squeeze=False)
            for i in range(n):
                im = img[i]
                lo_i, hi_i = torch.quantile(im.reshape(-1), 0.02), torch.quantile(im.reshape(-1), 0.98)
                ax[i, 0].imshow(((im - lo_i) / (hi_i - lo_i + 1e-6)).clamp(0, 1).numpy())
                ax[i, 1].imshow(ct[i].numpy(), interpolation="nearest")
                ax[i, 2].imshow(cp[i].numpy(), interpolation="nearest")
                ax[i, 3].imshow(err[i].numpy(), cmap="magma", vmin=0, vmax=1, interpolation="nearest")
                if cd is not None:
                    ax[i, 4].imshow(cd[i].numpy(), interpolation="nearest")
                for j in range(ncol):
                    ax[i, j].axis("off")
            titles = (["input", "teacher PCA", f"student PCA (spread x{spread:.2f})", "1 - cos"]
                      + (["student pixels PCA"] if cd is not None else []))
            for j, t in enumerate(titles):
                ax[0, j].set_title(t, fontsize=9)
            fig.suptitle(f"{name}  step {int(self.global_step)}", fontsize=9)
            plt.tight_layout()
            path = os.path.join(out, f"step{int(self.global_step):07d}_{name}.png")
            fig.savefig(path, dpi=100)
            plt.close(fig)
            for lg in self.trainer.loggers:
                if WandbLogger and isinstance(lg, WandbLogger):
                    lg.log_image(key=f"viz/{name}", images=[path], step=int(self.global_step))

    def encoder_state_dict(self):
        return self.student.encoder_state_dict()


# ---------------------------------------------------------------------------------------------------------
class DistillDataModule(pl.LightningDataModule):
    def __init__(self, config, lookup_table, dataset_config, stats_required=True):
        super().__init__()
        self.dc, self.look_up, self.dataset_config, self.stats_required = config["distill"], lookup_table, dataset_config, stats_required
        self._ready = False

    def setup(self, stage=None):
        if self._ready:
            return
        dc, common = self.dc, dict(look_up=self.look_up, dataset_config=self.dataset_config)
        sr = self.stats_required
        self.views = parse_flair_views(dc)
        crops = sorted({v["crop"] for v in self.views})
        bd = dc.get("band_drop") or {}
        for split, aug in (("train", True), ("val", False)):
            setattr(self, f"olmo_{split}", OlmoSentinelAtoms(dc["olmo_index"], dc["olmo_tar_dir"], split, aug=aug,
                                                             stats_path=dc["olmo_stats"], stats_required=sr,
                                                             band_drop=bd.get("olmo"), **common))
            setattr(self, f"flair_{split}", FlairAtoms(dc["flair_index"], dc["flair_root"], split, int(dc["flair_crop"]), aug=aug,
                                                       stats_path=dc["flair_stats"], stats_required=sr,
                                                       aerial_section=dc.get("aerial_section", "bands_flairhub_aerial"),
                                                       share_s2=bool(dc.get("share_s2", False)), crops=crops,
                                                       band_drop=bd.get("flair"), elevation=dc.get("flair_elevation"),
                                                       **common))
        self._ready = True

    def _loader(self, ds, **kw):
        nw = int(self.dc.get("workers", 8))
        return DataLoader(ds, collate_fn=collate_distill, num_workers=nw, pin_memory=torch.cuda.is_available(),
                          persistent_workers=nw > 0 and kw.get("persistent", False), prefetch_factor=2 if nw > 0 else None,
                          **{k: v for k, v in kw.items() if k != "persistent"})

    def train_dataloader(self):
        sampler = MixedViewBatchSampler(
            len(self.olmo_train), len(self.flair_train), int(self.dc["batch_per_dataset"]), self.views,
            num_replicas=self.trainer.world_size, rank=self.trainer.global_rank, seed=int(self.dc.get("seed", 42)))
        return self._loader(DistillConcat([self.olmo_train, self.flair_train]), batch_sampler=sampler, persistent=True)

    def val_dataloader(self):
        """Loader 0: Sentinel; then one FLAIR loader per view (fixed view, the view's batch size)."""
        W, r = self.trainer.world_size, self.trainer.global_rank
        loaders = [self._loader(self.olmo_val, batch_size=int(self.dc["batch_per_dataset"]),
                                sampler=DistributedSampler(self.olmo_val, num_replicas=W, rank=r, shuffle=False))]
        for v in self.views:
            ds = FixedView(self.flair_val, v["crop"], v["up"])
            loaders.append(self._loader(ds, batch_size=v["batch"],
                                        sampler=DistributedSampler(ds, num_replicas=W, rank=r, shuffle=False)))
        return loaders


# ---------------------------------------------------------------------------------------------------------
class StopAtIters(pl.Callback):
    """Stop after the DUNE iteration count (the module computes it in configure_optimizers)."""

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if pl_module.iters is not None and trainer.global_step >= pl_module.iters:
            trainer.should_stop = True


class SaveEncoder(pl.Callback):
    """Every n optimizer steps, save the ENCODER only (heads, teacher norms and optimizer excluded)."""

    def __init__(self, dirpath, every_n_steps):
        self.dirpath, self.every, self._last = dirpath, int(every_n_steps), -1

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        s = trainer.global_step
        if trainer.is_global_zero and s > 0 and s % self.every == 0 and s != self._last:
            self._last = s
            os.makedirs(self.dirpath, exist_ok=True)
            path = os.path.join(self.dirpath, f"encoder_step{s}.pth")
            torch.save(pl_module.encoder_state_dict(), path)
            print(f"[Distill] saved encoder to {path}")


class TimeLimitCheckpoint(pl.Callback):
    """Save a full checkpoint and stop before the job's wall-clock limit.

    start_time: the reference time (the launcher passes the time the script started, so model building and data setup
    count too); limit_s: seconds after start_time at which to save and stop. The check runs right after an optimizer
    step (never in the middle of a gradient accumulation), every `check_every` optimizer steps; rank 0's clock decides
    and its decision is broadcast, so every DDP rank saves and stops at the same step.
    Writes <dirpath>/timelimit.ckpt (Lightning checkpoint, for --resume) and encoder_timelimit_step<N>.pth."""

    def __init__(self, dirpath, limit_s, start_time=None, check_every=10):
        self.dirpath, self.limit = dirpath, float(limit_s)
        self.start = start_time
        self.check_every = int(check_every)
        self._seen, self._n, self.done = -1, 0, False

    def on_train_start(self, trainer, pl_module):
        if self.start is None:
            self.start = time.time()
        if trainer.is_global_zero:
            fmt = lambda t: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))  # noqa: E731
            print(f"[TimeLimit] job started {fmt(self.start)}, training starts {fmt(time.time())} "
                  f"(step {trainer.global_step}); will save and stop at {fmt(self.start + self.limit)}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.done:
            return
        step = trainer.global_step
        if step == self._seen:                 # no optimizer step since the last call (inside a gradient accumulation)
            return
        self._seen = step
        self._n += 1
        if self._n % self.check_every:
            return
        over = trainer.strategy.broadcast(time.time() - self.start >= self.limit, src=0)
        if not over:
            return
        os.makedirs(self.dirpath, exist_ok=True)
        path = os.path.join(self.dirpath, "timelimit.ckpt")
        trainer.save_checkpoint(path)                                        # every rank calls it (DDP)
        if trainer.is_global_zero:
            torch.save(pl_module.encoder_state_dict(), os.path.join(self.dirpath, f"encoder_timelimit_step{step}.pth"))
            print(f"[TimeLimit] {(time.time() - self.start) / 3600:.2f} h elapsed: saved {path} at step {step}, "
                  f"stopping. Resume with --resume {path} (or --auto_resume)")
        trainer.should_stop = True
        self.done = True

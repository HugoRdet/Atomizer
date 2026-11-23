"""
Multi-task trainer for Atomiser_UnetMulti
=========================================

One dataset per batch (batch["dataset"]); the model's head for that dataset
produces the logits.

Loss per task (config multitask.tasks.<name>.loss):
    "ce"       cross-entropy, ignore 255 (default)
    "ordinal"  cumulative binary cross-entropy (AgriPotential): logit k models
               P(y > k), k = 0 .. C-2; prediction = number of sigmoids > 0.5

Task weighting (config multitask.loss_weighting):
    "uncertainty"  Kendall et al. (2018): one learned log-variance s_t per task;
                   a batch of task t contributes exp(-s_t) * L_t + s_t. s_t starts at
                   0 (weight 1), has no weight decay, and is logged as weight_<task>.
    "fixed"        L_t * multitask.tasks.<name>.weight (default 1)

Metrics, per task, on labelled pixels (computed from confusion matrices, so
they are exact under DDP):
    mIoU       mean IoU over the task's classes, a class absent from both labels and
               predictions counting as 0 (PLANTEUR convention); classes listed in
               tasks.<name>.miou_exclude are left out (PASTIS: [0], background)
    for ordinal targets also acc_pm1 (+-1 class) and mae: logged when the loss is
    ordinal, the primary metric is acc_pm1 / mae, or tasks.<name>.ordinal_metrics: true
Primary metric per task (multitask.tasks.<name>.primary: mIoU | acc_pm1); the
checkpoint monitor val_score = mean of the tasks' primary validation metrics.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from transformers import get_cosine_schedule_with_warmup


class ConfusionMatrix(torchmetrics.Metric):
    """Confusion matrix over labelled pixels (rows = truth). A torchmetrics state, NOT
    a module buffer: DDP broadcasts buffers from rank 0 at every forward, which would
    overwrite the other ranks' counts. Cross-rank reduction is torchmetrics' own sum
    at compute(), the standard path Lightning expects."""
    full_state_update = False

    def __init__(self, n):
        super().__init__()
        self.n = n
        self.add_state("m", default=torch.zeros(n, n, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, pred, target, ignore=255):
        v = target != ignore
        t, p = target[v].long(), pred[v].long().clamp(0, self.n - 1)
        self.m += torch.bincount(t * self.n + p, minlength=self.n * self.n).view(self.n, self.n)

    def compute(self):
        return self.m.clone()


def metrics_from_confusion(m, miou_exclude=()):
    tp = m.diag().double()
    union = m.sum(0).double() + m.sum(1).double() - tp
    iou = torch.where(union > 0, tp / union.clamp_min(1), torch.zeros_like(tp))
    keep = torch.ones_like(iou, dtype=torch.bool)
    for c in miou_exclude:
        keep[int(c)] = False
    n = m.sum().clamp_min(1).double()
    C = m.shape[0]
    cls = torch.arange(C, device=m.device)
    dist_ = (cls.view(-1, 1) - cls.view(1, -1)).abs().double()      # |true - pred|
    return {"mIoU": iou[keep].mean(), "iou": iou, "acc_exact": tp.sum() / n,
            "acc_pm1": (m.double() * (dist_ <= 1)).sum() / n,
            "mae": (m.double() * dist_).sum() / n}


class MultiTaskTrainer(pl.LightningModule):
    def __init__(self, config, lookup_table, name="multitask"):
        super().__init__()
        from training.atomiser.Atomiser_unet_multi import Atomiser_UnetMulti
        self.config, self.name = config, name
        self.mt = config["multitask"]
        self.tasks = list(self.mt["tasks"])
        self.encoder = Atomiser_UnetMulti(config=config, lookup_table=lookup_table)
        missing = set(self.tasks) - set(self.encoder.datasets)
        if missing:
            raise ValueError(f"multitask.tasks {missing} not in Atomiser.unet_datasets")
        self.n_classes = {t: self.encoder.datasets[t]["num_classes"] for t in self.tasks}
        self.loss_type = {t: str(self.mt["tasks"][t].get("loss", "ce")).lower() for t in self.tasks}
        self.primary = {t: self.mt["tasks"][t].get("primary",
                        "acc_pm1" if self.loss_type[t] == "ordinal" else "mIoU") for t in self.tasks}
        # ordinal metrics (acc_pm1, mae): for ordinal targets, whatever the loss trains them
        self.ordinal_metrics = {t: bool(self.mt["tasks"][t].get(
            "ordinal_metrics", self.loss_type[t] == "ordinal" or self.primary[t] in ("acc_pm1", "mae")))
            for t in self.tasks}
        # classes left out of the mIoU average (PASTIS convention: background 0)
        self.miou_exclude = {t: [int(c) for c in self.mt["tasks"][t].get("miou_exclude", [])]
                             for t in self.tasks}
        self.weighting = str(self.mt.get("loss_weighting", "uncertainty")).lower()
        if self.weighting not in ("uncertainty", "fixed"):
            raise ValueError("multitask.loss_weighting must be 'uncertainty' or 'fixed'")
        self.log_vars = nn.ParameterDict({t: nn.Parameter(torch.zeros(())) for t in self.tasks})
        if self.weighting == "fixed":
            for p in self.log_vars.values():
                p.requires_grad_(False)
        self.fixed_w = {t: float(self.mt["tasks"][t].get("weight", 1.0)) for t in self.tasks}
        self.conf = nn.ModuleDict({f"{s}_{t}": ConfusionMatrix(self.n_classes[t])
                                   for s in ("train", "val", "test") for t in self.tasks})
        self.val_names = list(self.tasks)                     # dataloader_idx -> task
        self.lr = float(config["trainer"]["lr"])
        self.weight_decay = float(config["trainer"]["weight_decay"])
        print(f"[MultiTask] tasks={self.tasks} weighting={self.weighting} "
              f"losses={self.loss_type} primary={self.primary}")

    # ------------------------------------------------------------------ loss
    def _task_loss(self, task, logits, labels):
        valid = labels != 255
        if not valid.any():
            return logits.float().sum() * 0.0, logits.argmax(-1)
        if self.loss_type[task] == "ordinal":
            C = self.n_classes[task]
            z = logits[..., :C - 1].float()
            k = torch.arange(C - 1, device=z.device)
            target = (labels[valid].unsqueeze(-1) > k).float()
            loss = F.binary_cross_entropy_with_logits(z[valid], target)
            pred = (torch.sigmoid(z) > 0.5).sum(-1)
        else:
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1),
                                   ignore_index=255)
            pred = logits.argmax(-1)
        return loss, pred

    def _step(self, batch, split, task):
        out = self.encoder(batch, training=(split == "train"))
        labels = batch["queries"][:, :, 4].long()
        labels = torch.where(batch["queries_mask"], torch.full_like(labels, 255), labels)
        loss, pred = self._task_loss(task, out["predictions"], labels)
        self.conf[f"{split}_{task}"].update(pred, labels)
        return loss

    # ------------------------------------------------------------------ steps
    # NOTE (DDP): per-step losses are logged WITHOUT sync_dist. With sync_dist=True,
    # Lightning computes the cross-rank mean lazily, when something reads the metric:
    # the progress bar does so on rank 0 only (before the module's epoch-end hook), so
    # rank 0 entered a reduce while the other ranks entered the torchmetrics gather of
    # on_train_epoch_end -> deadlock. The only cross-rank reductions left are the
    # confusion matrices, computed by every rank in the same hooks, in the same order.
    def training_step(self, batch, batch_idx):
        task = batch["dataset"]
        loss = self._step(batch, "train", task)
        if self.weighting == "uncertainty":
            s = self.log_vars[task]
            total = torch.exp(-s) * loss + s
        else:
            total = self.fixed_w[task] * loss
        B = batch["queries"].shape[0]                       # explicit: the batch dict is ambiguous
        self.log(f"train_loss_{task}", loss, on_step=False, on_epoch=True, logger=True,
                 batch_size=B)
        self.log("train_loss", total, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=B)
        return total

    def _print_progress(self):
        """Rank 0 only, outside the sanity check; False when no Trainer is attached."""
        t = getattr(self, "_trainer", None)
        return t is not None and t.is_global_zero and not t.sanity_checking

    def on_validation_epoch_start(self):
        import time
        self._val_t0 = time.time()
        if self._print_progress():
            print(f"[MultiTask] epoch {self.current_epoch}: validation started", flush=True)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        task = batch["dataset"]
        if self._print_progress() and batch_idx % 50 == 0:
            import time
            n = self.trainer.num_val_batches[dataloader_idx]
            print(f"[MultiTask]   validation {task}: batch {batch_idx}/{n} "
                  f"({time.time() - self._val_t0:.0f}s)", flush=True)
        loss = self._step(batch, "val", task)
        self.log(f"val_loss_{task}", loss, on_step=False, on_epoch=True, logger=True,
                 add_dataloader_idx=False, batch_size=batch["queries"].shape[0])
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        task = batch["dataset"]
        loss = self._step(batch, "test", task)
        self.log(f"test_loss_{task}", loss, on_step=False, on_epoch=True, logger=True,
                 add_dataloader_idx=False, batch_size=batch["queries"].shape[0])
        return loss

    # ------------------------------------------------------------------ epoch ends
    def _log_split(self, split):
        scores = []
        for t in self.tasks:
            cm = self.conf[f"{split}_{t}"]
            if cm.update_count == 0:          # no batch of this task in this split (same on every rank)
                continue
            m = cm.compute()
            cm.reset()
            if m.sum() == 0:
                continue
            r = metrics_from_confusion(m, self.miou_exclude[t])
            keys = ["mIoU", "acc_exact"] + (["acc_pm1", "mae"] if self.ordinal_metrics[t] else [])
            for k in keys:
                self.log(f"{split}_{k}_{t}", r[k].float(), on_epoch=True, logger=True,
                         prog_bar=(k == self.primary[t] and split == "val"))
            scores.append(r[self.primary[t]].float())
            if split == "test":
                print(f"[MultiTask] TEST {t}: " + ", ".join(f"{k}={float(r[k]):.4f}" for k in keys)
                      + " | per-class IoU " + " ".join(f"{float(v):.3f}" for v in r["iou"]))
        if scores and split != "train":
            self.log(f"{split}_score", torch.stack(scores).mean(), on_epoch=True, prog_bar=True,
                     logger=True)

    def on_train_epoch_start(self):
        dm = getattr(self.trainer, "datamodule", None)
        if dm is not None and getattr(dm, "train_batch_sampler", None) is not None:
            dm.train_batch_sampler.set_epoch(self.current_epoch)

    def on_train_epoch_end(self):
        self._log_split("train")
        for t in self.tasks:
            self.log(f"weight_{t}", torch.exp(-self.log_vars[t].detach()), on_epoch=True, logger=True)

    def on_validation_epoch_end(self):
        self._log_split("val")
        if self._print_progress():
            import time
            print(f"[MultiTask] epoch {self.current_epoch}: validation done in "
                  f"{time.time() - self._val_t0:.0f}s", flush=True)

    def on_test_epoch_end(self):
        self._log_split("test")

    # ------------------------------------------------------------------ optimiser
    def configure_optimizers(self):
        lv = [p for p in self.log_vars.values() if p.requires_grad]
        lv_ids = {id(p) for p in lv}
        main = [p for p in self.parameters() if p.requires_grad and id(p) not in lv_ids]
        groups = [{"params": main, "weight_decay": self.weight_decay}]
        if lv:
            groups.append({"params": lv, "weight_decay": 0.0})
        opt = torch.optim.AdamW(groups, lr=self.lr)
        total = int(self.config["trainer"].get("total_steps") or self.trainer.estimated_stepping_batches)
        warm = int(self.config.get("optimizer", {}).get("warmup_steps", max(1, int(0.05 * total))))
        print(f"[MultiTask] LR schedule: total_steps={total}, warmup={warm}, peak_lr={self.lr}")
        sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warm, num_training_steps=total)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

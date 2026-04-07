from __future__ import annotations

import math
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_cosine_schedule_with_warmup


def is_dist():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist() else 0


def get_world_size():
    return dist.get_world_size() if is_dist() else 1


def is_main_process():
    return get_rank() == 0


class Stage1Trainer:
    def __init__(
        self,
        model,
        criterion,
        train_loader,
        val_loader,
        lr: float,
        weight_decay: float,
        epochs: int,
        grad_accum_steps: int,
        max_grad_norm: float,
        warmup_ratio: float,
        save_dir: str,
        bf16: bool = False,
        fp16: bool = False,
        log_every: int = 20,
        eval_every: int = 500,
        save_every: int = 1000,
        distributed: bool = False,
        local_rank: int = 0,
        train_sampler=None,
    ):
        self.distributed = distributed
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        if self.distributed:
            self.model = DDP(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
                broadcast_buffers=False,
            )

        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.grad_accum_steps = grad_accum_steps
        self.max_grad_norm = max_grad_norm
        self.warmup_ratio = warmup_ratio
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.bf16 = bf16
        self.fp16 = fp16
        self.log_every = log_every
        self.eval_every = eval_every
        self.save_every = save_every
        self.train_sampler = train_sampler

        self.use_amp = bf16 or fp16
        self.autocast_dtype = torch.bfloat16 if bf16 else torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=fp16)

        model_for_optim = self.unwrap_model()
        self.optimizer = torch.optim.AdamW(
            model_for_optim.trainable_parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        total_steps = math.ceil(len(self.train_loader) / self.grad_accum_steps) * self.epochs
        warmup_steps = int(total_steps * self.warmup_ratio)

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        self.global_step = 0
        self.total_optim_steps_per_epoch = math.ceil(len(self.train_loader) / self.grad_accum_steps)

    def unwrap_model(self):
        return self.model.module if isinstance(self.model, DDP) else self.model

    def reduce_mean(self, value: float) -> float:
        if not self.distributed:
            return value
        t = torch.tensor([value], dtype=torch.float64, device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= get_world_size()
        return t.item()

    def reduce_sum_tensor(self, value: float) -> float:
        t = torch.tensor([value], dtype=torch.float64, device=self.device)
        if self.distributed:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return t.item()

    def train(self):
        for epoch in range(self.epochs):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            self.train_one_epoch(epoch)
            self.evaluate(epoch)
            self.save_checkpoint(f"epoch_{epoch+1}.pt")

    def train_one_epoch(self, epoch: int):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        running = {"loss": 0.0, "lm_loss": 0.0, "align_loss": 0.0}
        running_count = 0
        log_start_time = time.time()
        epoch_start_time = time.time()

        for step, batch in enumerate(self.train_loader, start=1):
            with torch.autocast(
                device_type="cuda",
                dtype=self.autocast_dtype,
                enabled=self.use_amp,
            ):
                outputs = self.model(
                    ms_feat=batch["ms_feat"],
                    sar_feat=batch["sar_feat"],
                    prompt_text=batch["prompt_text"],
                    target_text=batch["target_text"],
                )
                loss_dict = self.criterion(
                    lm_loss=outputs["lm_loss"],
                    ms_global=outputs["ms_global"],
                    sar_global=outputs["sar_global"],
                )
                loss = loss_dict["loss"] / self.grad_accum_steps

            if self.fp16:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            running["loss"] += loss_dict["loss"].item()
            running["lm_loss"] += loss_dict["lm_loss"].item()
            running["align_loss"] += loss_dict["align_loss"].item()
            running_count += 1

            if step % self.grad_accum_steps == 0:
                if self.fp16:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

                if self.global_step % self.log_every == 0:
                    elapsed = time.time() - log_start_time
                    mean_loss = self.reduce_mean(running["loss"] / max(running_count, 1))
                    mean_lm = self.reduce_mean(running["lm_loss"] / max(running_count, 1))
                    mean_align = self.reduce_mean(running["align_loss"] / max(running_count, 1))
                    avg_step_time = elapsed / max(running_count, 1)
                    lr = self.optimizer.param_groups[0]["lr"]
                    epoch_step = math.ceil(step / self.grad_accum_steps)

                    if is_main_process():
                        print(
                            f"[train] epoch={epoch+1} "
                            f"step={self.global_step} "
                            f"epoch_step={epoch_step}/{self.total_optim_steps_per_epoch} "
                            f"loss={mean_loss:.4f} "
                            f"lm_loss={mean_lm:.4f} "
                            f"align_loss={mean_align:.4f} "
                            f"lr={lr:.6e} "
                            f"log_time={elapsed:.1f}s "
                            f"avg_step_time={avg_step_time:.3f}s"
                        )

                    running = {"loss": 0.0, "lm_loss": 0.0, "align_loss": 0.0}
                    running_count = 0
                    log_start_time = time.time()

                if self.global_step % self.eval_every == 0:
                    self.evaluate(epoch)
                    self.model.train()

                if self.global_step % self.save_every == 0:
                    self.save_checkpoint(f"step_{self.global_step}.pt")

        if is_main_process():
            epoch_elapsed = time.time() - epoch_start_time
            print(f"[epoch_done] epoch={epoch+1} epoch_time={epoch_elapsed:.1f}s")

    @torch.no_grad()
    def evaluate(self, epoch: int):
        self.model.eval()

        total = {"loss": 0.0, "lm_loss": 0.0, "align_loss": 0.0}
        count = 0
        val_start_time = time.time()

        for batch in self.val_loader:
            with torch.autocast(
                device_type="cuda",
                dtype=self.autocast_dtype,
                enabled=self.use_amp,
            ):
                outputs = self.model(
                    ms_feat=batch["ms_feat"],
                    sar_feat=batch["sar_feat"],
                    prompt_text=batch["prompt_text"],
                    target_text=batch["target_text"],
                )
                loss_dict = self.criterion(
                    lm_loss=outputs["lm_loss"],
                    ms_global=outputs["ms_global"],
                    sar_global=outputs["sar_global"],
                )

            total["loss"] += loss_dict["loss"].item()
            total["lm_loss"] += loss_dict["lm_loss"].item()
            total["align_loss"] += loss_dict["align_loss"].item()
            count += 1

        total_loss_sum = self.reduce_sum_tensor(total["loss"])
        total_lm_sum = self.reduce_sum_tensor(total["lm_loss"])
        total_align_sum = self.reduce_sum_tensor(total["align_loss"])
        total_count = self.reduce_sum_tensor(count)

        if is_main_process() and total_count > 0:
            val_elapsed = time.time() - val_start_time
            print(
                f"[val] epoch={epoch+1} "
                f"loss={total_loss_sum/total_count:.4f} "
                f"lm_loss={total_lm_sum/total_count:.4f} "
                f"align_loss={total_align_sum/total_count:.4f} "
                f"val_time={val_elapsed:.1f}s"
            )

    def save_checkpoint(self, name: str):
        if not is_main_process():
            return
        ckpt_path = self.save_dir / name
        model_to_save = self.unwrap_model()
        torch.save(
            {
                "model": model_to_save.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "global_step": self.global_step,
            },
            ckpt_path,
        )
        print(f"[ckpt] saved to {ckpt_path}")

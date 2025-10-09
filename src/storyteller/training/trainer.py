"""
Trainer class for training the Storyteller model.

This module implements a flexible trainer with support for:
- Mixed precision training
- Gradient accumulation
- Gradient checkpointing
- Learning rate scheduling
- Checkpoint management
- Logging (console + MLflow)
"""

import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import AMP components based on PyTorch version
try:
    from torch.amp import autocast, GradScaler

    HAS_UNIFIED_AMP = True
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

    HAS_UNIFIED_AMP = False

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class Trainer:
    """
    Trainer for language models with modern optimization techniques.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        use_amp: bool = True,
        amp_dtype: str = "bfloat16",
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        save_dir: str = "checkpoints",
        save_every_n_steps: int = 5000,
        eval_every_n_steps: int = 1000,
        log_every_n_steps: int = 100,
        keep_last_n_checkpoints: int = 3,
        use_mlflow: bool = False,
        mlflow_experiment_name: Optional[str] = None,
        mlflow_run_name: Optional[str] = None,
        mlflow_tracking_uri: Optional[str] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: The model to train
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            use_amp: Whether to use automatic mixed precision
            amp_dtype: AMP dtype ('float16' or 'bfloat16')
            gradient_accumulation_steps: Number of steps to accumulate gradients
            max_grad_norm: Maximum gradient norm for clipping
            save_dir: Directory to save checkpoints
            save_every_n_steps: Save checkpoint every N steps
            eval_every_n_steps: Evaluate every N steps
            log_every_n_steps: Log metrics every N steps
            keep_last_n_checkpoints: Number of recent checkpoints to keep
            use_mlflow: Whether to use MLflow experiment tracking
            mlflow_experiment_name: MLflow experiment name
            mlflow_run_name: MLflow run name
            mlflow_tracking_uri: MLflow tracking server URI (optional)
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device) if isinstance(device, str) else device
        self.device_type = self.device.type

        # Mixed precision configuration - handle MPS, CUDA, and CPU
        self.use_amp = self._configure_amp(use_amp, amp_dtype)
        self.amp_dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16

        # GradScaler only for CUDA with float16
        # MPS and bfloat16 don't need/support GradScaler
        self.scaler = None
        if self.use_amp and self.device_type == "cuda" and amp_dtype == "float16":
            if HAS_UNIFIED_AMP:
                self.scaler = GradScaler(self.device_type)
            else:
                self.scaler = GradScaler()

        # Training config
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm

        # Checkpointing
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_every_n_steps = save_every_n_steps
        self.keep_last_n_checkpoints = keep_last_n_checkpoints
        self.saved_checkpoints = []

        # Logging
        self.eval_every_n_steps = eval_every_n_steps
        self.log_every_n_steps = log_every_n_steps

        # MLflow
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE
        if self.use_mlflow:
            if mlflow_tracking_uri:
                mlflow.set_tracking_uri(mlflow_tracking_uri)

            # Set experiment
            experiment_name = mlflow_experiment_name or "storyteller"
            mlflow.set_experiment(experiment_name)

            # Start run
            mlflow.start_run(run_name=mlflow_run_name)

            # Log model config and training params
            if hasattr(model, "config"):
                mlflow.log_params(
                    {
                        f"model/{k}": v
                        for k, v in model.config.__dict__.items()
                        if isinstance(v, (int, float, str, bool))
                    }
                )

            mlflow.log_params(
                {
                    "batch_size": self.train_dataloader.batch_size,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "max_grad_norm": max_grad_norm,
                    "use_amp": use_amp,
                    "amp_dtype": amp_dtype,
                }
            )

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")

    def _configure_amp(self, use_amp: bool, amp_dtype: str) -> bool:
        """
        Configure automatic mixed precision based on device and dtype.

        Returns:
            Whether AMP should be enabled
        """
        if not use_amp:
            return False

        # CPU doesn't support AMP
        if self.device_type == "cpu":
            print("  Warning: AMP not supported on CPU, disabling...")
            return False

        # MPS supports AMP but with limitations
        if self.device_type == "mps":
            # MPS only supports float16, not bfloat16
            if amp_dtype == "bfloat16":
                print(
                    "  Warning: MPS doesn't support bfloat16, using float16 instead..."
                )
                self.amp_dtype = torch.float16
            return True

        # CUDA supports both float16 and bfloat16
        if self.device_type == "cuda":
            # Check if bfloat16 is supported
            if amp_dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
                print(
                    "  Warning: GPU doesn't support bfloat16, falling back to float16..."
                )
                self.amp_dtype = torch.float16
            return True

        return False

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0
        total_moe_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_dataloader, desc=f"Epoch {self.epoch}")

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass with mixed precision
            # Use device_type for PyTorch 2.0+, otherwise old API
            autocast_kwargs = {"enabled": self.use_amp}
            if HAS_UNIFIED_AMP:
                autocast_kwargs["device_type"] = self.device_type
                autocast_kwargs["dtype"] = self.amp_dtype
            else:
                autocast_kwargs["dtype"] = self.amp_dtype

            with autocast(**autocast_kwargs):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_dict=True,
                )
                loss = outputs["loss"]

                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update weights
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()

                self.optimizer.zero_grad()
                self.global_step += 1

                # Logging
                if self.global_step % self.log_every_n_steps == 0:
                    self._log_metrics(
                        {
                            "train/loss": loss.item()
                            * self.gradient_accumulation_steps,
                            "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                            "train/global_step": self.global_step,
                        },
                        outputs,
                    )

                # Evaluation
                if self.global_step % self.eval_every_n_steps == 0:
                    val_metrics = self.evaluate()
                    self._log_metrics(val_metrics, prefix="val")
                    self.model.train()

                # Checkpointing
                if self.global_step % self.save_every_n_steps == 0:
                    self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")

            # Accumulate metrics
            total_loss += loss.item() * self.gradient_accumulation_steps
            if outputs.get("moe_loss") is not None:
                total_moe_loss += outputs["moe_loss"].item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                }
            )

        return {
            "train/epoch_loss": total_loss / num_batches,
            "train/epoch_moe_loss": total_moe_loss / num_batches
            if total_moe_loss > 0
            else 0,
        }

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate on validation set.

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0
        total_moe_loss = 0
        num_batches = 0

        for batch in tqdm(self.val_dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Use same autocast configuration as training
            autocast_kwargs = {"enabled": self.use_amp}
            if HAS_UNIFIED_AMP:
                autocast_kwargs["device_type"] = self.device_type
                autocast_kwargs["dtype"] = self.amp_dtype
            else:
                autocast_kwargs["dtype"] = self.amp_dtype

            with autocast(**autocast_kwargs):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_dict=True,
                )

            total_loss += outputs["loss"].item()
            if outputs.get("moe_loss") is not None:
                total_moe_loss += outputs["moe_loss"].item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_moe_loss = total_moe_loss / num_batches if total_moe_loss > 0 else 0

        # Calculate perplexity
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        metrics = {
            "val/loss": avg_loss,
            "val/perplexity": perplexity,
        }

        if avg_moe_loss > 0:
            metrics["val/moe_loss"] = avg_moe_loss

        return metrics

    def train(self, num_epochs: int):
        """
        Train for multiple epochs.

        Args:
            num_epochs: Number of epochs to train
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device_type} ({self.device})")
        print(f"Mixed precision: {self.use_amp}")
        if self.use_amp:
            print(f"  AMP dtype: {self.amp_dtype}")
            print(f"  GradScaler: {'enabled' if self.scaler else 'disabled'}")
        print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        print(
            f"Effective batch size: {self.train_dataloader.batch_size * self.gradient_accumulation_steps}"
        )

        start_time = time.time()

        for epoch in range(num_epochs):
            self.epoch = epoch
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train epoch
            epoch_metrics = self.train_epoch()

            # Evaluate
            val_metrics = self.evaluate()

            # Log epoch metrics
            all_metrics = {**epoch_metrics, **val_metrics, "epoch": epoch}
            self._log_metrics(all_metrics)

            # Save best model
            if val_metrics["val/loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val/loss"]
                self.save_checkpoint("best_model.pt")
                print(f"✓ New best validation loss: {self.best_val_loss:.4f}")

        elapsed_time = time.time() - start_time
        print(f"\nTraining completed in {elapsed_time / 3600:.2f} hours")

        # Save final model
        self.save_checkpoint("final_model.pt")

        # End MLflow run
        if self.use_mlflow:
            # Log final model
            mlflow.pytorch.log_model(self.model, "model")
            mlflow.end_run()

    def _log_metrics(
        self,
        metrics: Dict[str, float],
        outputs: Optional[Dict] = None,
        prefix: str = "",
    ):
        """Log metrics to console and MLflow."""
        # Console logging
        if self.global_step % self.log_every_n_steps == 0:
            log_str = f"Step {self.global_step}: "
            log_str += ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            if outputs and outputs.get("moe_stats"):
                # Log expert utilization for first MoE layer
                stats = outputs["moe_stats"][0]
                if "expert_balance_metric" in stats:
                    log_str += f", expert_balance: {stats['expert_balance_metric']:.2f}"

        # MLflow logging
        if self.use_mlflow:
            mlflow.log_metrics(metrics, step=self.global_step)

            # Log MoE statistics if available
            if outputs and outputs.get("moe_stats"):
                for i, stats in enumerate(outputs["moe_stats"]):
                    if "expert_balance_metric" in stats:
                        mlflow.log_metric(
                            f"moe/layer_{i}_balance",
                            stats["expert_balance_metric"],
                            step=self.global_step,
                        )
                    if "routing_entropy" in stats:
                        mlflow.log_metric(
                            f"moe/layer_{i}_entropy",
                            stats["routing_entropy"],
                            step=self.global_step,
                        )

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint_path = self.save_dir / filename

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
            if self.scheduler
            else None,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
            "config": self.model.config.__dict__,
        }

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Checkpoint saved: {checkpoint_path}")

        # Track saved checkpoints
        if "step" in filename:
            self.saved_checkpoints.append(checkpoint_path)

            # Remove old checkpoints
            if len(self.saved_checkpoints) > self.keep_last_n_checkpoints:
                old_checkpoint = self.saved_checkpoints.pop(0)
                if old_checkpoint.exists():
                    old_checkpoint.unlink()

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        # Use weights_only=False for compatibility with older PyTorch versions
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
        except TypeError:
            # Older PyTorch versions don't have weights_only parameter
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.scaler and checkpoint.get("scaler_state_dict"):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]

        print(f"✓ Checkpoint loaded: {checkpoint_path}")
        print(f"  Resuming from step {self.global_step}, epoch {self.epoch}")

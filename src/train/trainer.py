import os
import uuid
from typing import Literal, cast
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from src.datasets.base_dataset import Batch
from src.experiments.base_experiment import BaseExperiment
from src.train.evaluator import Evaluator
from src.train.history import EpochLosses, SingleEpochHistory, TrainHistory
import signal
import sys


class GracefulKiller:
    def __init__(self, max_interrupts=3):
        self.received_signal = False
        self.interrupt_count = 0
        self.max_interrupts = max_interrupts
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum, frame):
        self.interrupt_count += 1
        if self.interrupt_count >= self.max_interrupts:
            print(f"Received Ctrl+C {self.max_interrupts} times. Stopping immediately.")
            sys.exit(1)
        else:
            remaining = self.max_interrupts - self.interrupt_count
            print(
                f"Ctrl+C received. Finishing current epoch. Press {remaining} more times to stop immediately."
            )
            self.received_signal = True

    def should_stop(self):
        return self.received_signal


class Trainer:
    def __init__(self, experiment: BaseExperiment):
        self.experiment = experiment
        self.config = experiment.base_config
        self.yaml_config = experiment.yaml_config

        self.dataloader_train = experiment._create_dataloader("train")
        self.dataloader_val = experiment._create_dataloader("val")
        self.dataloader_test = experiment._create_dataloader("test")

        self.model = experiment.model
        self.device = experiment.get_device()
        self.loss_name = experiment.get_loss_name()
        self.optimizer = experiment.create_optimizer()
        self.scheduler = experiment.create_scheduler(self.optimizer)
        self.killer = GracefulKiller()

    def _log_intermediate(self, batch: int, n_batches: int, evaluator: Evaluator):
        loss = evaluator.get_latest_loss()
        running = evaluator.get_running_loss()
        print(
            f"Batch {batch + 1}/{n_batches} {self.loss_name}_loss: {loss:.2f} running: {running:.2f}\r",
            end="",
        )

    def _preprocess_labels(self, batch: Batch):
        return batch.target

    def _augment_batch(self, batch: Batch) -> Batch:
        return self._gaussian_noise_augmentation(batch)

    def _gaussian_noise_augmentation(self, batch):
        if self.config.whiteNoiseSD > 0:
            input = batch.input
            noised_input = input + (
                torch.randn(input.shape, device=input.device) * self.config.whiteNoiseSD
            )
            batch.input = noised_input

        if self.config.constantOffsetSD > 0:
            input = batch.input
            offset_input = input + (
                torch.randn([input.shape[0], 1, input.shape[2]], device=input.device)
                * self.config.constantOffsetSD
            )
            batch.input = offset_input
        return batch

    def _train_epoch(self, epoch: int, data_loader: DataLoader):
        self.model.train()
        scaler = torch.GradScaler(device=self.device)
        evaluator = self.experiment.create_evaluator("train")

        for i, batch in enumerate(data_loader):
            batch = cast(Batch, batch).to(self.device)
            self.optimizer.zero_grad()
            with torch.autocast(
                device_type=self.device, dtype=torch.float16, enabled=self.config.amp
            ):
                batch.target = self._preprocess_labels(batch)
                batch = self._augment_batch(batch)

                # Make predictions for this batch
                with torch.enable_grad():
                    # calculate gradient for whole model (but only optimize parts)
                    outputs = self.model.forward(batch)

            loss = self.model.compute_loss(outputs, batch, confidence_thresholding=True)

            scaler.scale(loss.loss).backward()

            if self.config.gradient_clipping is not None:
                torch.nn.utils.clip_grad_norm_(  # type: ignore
                    self.model.parameters(), self.config.gradient_clipping
                )

            # Adjust learning weights
            scaler.step(self.optimizer)
            scaler.update()

            self._after_batch_complete(epoch * len(data_loader) + i)

            evaluator.track_batch(outputs, loss, batch)
            if (
                i % self.config.log_every_n_batches
                == self.config.log_every_n_batches - 1
            ):
                self._log_intermediate(i, len(self.dataloader_train), evaluator)
        results = evaluator.evaluate()
        evaluator.clean_up()
        return results

    def _after_batch_complete(self, iter: int):
        return

    def _after_epoch_complete(self, epoch: int):
        return

    def evaluate_epoch(self, mode: Literal["val", "test"]) -> list[SingleEpochHistory]:
        dataloader = self.dataloader_val if mode == "val" else self.dataloader_test
        self.model.eval()
        evaluator = self.experiment.create_evaluator(mode)

        for i, batch in enumerate(dataloader):
            batch = cast(Batch, batch).to(self.device)

            with torch.no_grad():
                outputs = self.model.forward(batch)
            loss = self.model.compute_loss(outputs, batch)
            evaluator.track_batch(outputs, loss, batch)
            if (
                i % self.config.log_every_n_batches
                == self.config.log_every_n_batches - 1
            ):
                self._log_intermediate(i, len(dataloader), evaluator)

        results = evaluator.evaluate()
        evaluator.clean_up()
        return [results]

    def _get_wandb_metrics(
        self, epoch: SingleEpochHistory | list[SingleEpochHistory], prefix: str
    ):
        def add_prefix_to_dict_keys(d: dict, prefix: str):
            return {f"{prefix}_{k}": v for k, v in d.items()}

        epoch = epoch if isinstance(epoch, SingleEpochHistory) else epoch[0]
        loss_avg = epoch.get_average()
        epoch_val_metrics = loss_avg.metrics

        wandb_metrics = {
            f"{prefix}_{self.loss_name}_loss": loss_avg.loss,
        }
        wandb_metrics.update(add_prefix_to_dict_keys(epoch_val_metrics, prefix))
        return wandb_metrics

    def _log_epoch_wandb(self, losses: EpochLosses):
        metrics = self._get_wandb_metrics(losses.val_losses, "val")
        metrics.update(self._get_wandb_metrics(losses.train_losses, "train"))
        wandb.log(metrics)

    def get_relevant_metric(
        self,
        epoch_hists: SingleEpochHistory | list[SingleEpochHistory],
    ):
        epoch_hist = (
            epoch_hists
            if isinstance(epoch_hists, SingleEpochHistory)
            else epoch_hists[0]
        )
        return (
            epoch_hist.get_average().loss
            if self.config.best_model_metric == "loss"
            else epoch_hist.get_average().metrics[self.config.best_model_metric]
        )

    def train(self):
        history: list[EpochLosses] = (
            self.experiment.checkpoint_history.epochs
            if self.experiment.checkpoint_history is not None
            else []
        )
        best_model_val_metric = float(
            "inf" if self.config.minimize_best_model_metric else "-inf"
        )
        best_model_path = os.path.join(
            self.yaml_config.cache_dir,
            "model_checkpoints",
            str(uuid.uuid4()),
            "best_model.pt",
        )
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

        best_model_epoch = -1

        for epoch in range(self.config.epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.epochs}")
            train_losses = self._train_epoch(epoch, self.dataloader_train)
            self._after_epoch_complete(epoch)

            val_losses = self.evaluate_epoch("val")
            self.scheduler.step()

            print(
                f"\n\n{'='*20}\nFinished Epoch {epoch + 1}/{self.config.epochs} "
                f"train {self.loss_name}-loss: {train_losses.get_average().loss} ",
                end="",
            )
            for i, val_loss in enumerate(val_losses):
                print(
                    f"val {self.loss_name}-loss {i}: {val_loss.get_average().loss}",
                    end="",
                )
            print()
            epoch_losses = EpochLosses(train_losses, val_losses)
            history.append(epoch_losses)
            self._log_epoch_wandb(epoch_losses)
            if self.config.return_best_model:
                curr_epoch_val_metric = self.get_relevant_metric(val_losses)

                is_better = (
                    curr_epoch_val_metric < best_model_val_metric
                    if self.config.minimize_best_model_metric
                    else curr_epoch_val_metric > best_model_val_metric
                )
                if is_better:
                    best_model_val_metric = curr_epoch_val_metric
                    self._save_best_model(best_model_path)
                    print(f"\n\nSaving model checkpoint at {best_model_path}\n")
                    best_model_epoch = epoch

            if (
                self.config.early_stopping_patience is not None
                and len(epoch_losses) >= self.config.early_stopping_patience
            ):
                relevant_metric_history = [
                    self.get_relevant_metric(epoch_loss.val_losses)
                    for epoch_loss in history
                ][-self.config.early_stopping_patience :]

                # Adapt basline metric via early_stopping_epsilon
                if self.config.minimize_best_model_metric:
                    relevant_metric_history[0] -= self.config.early_stopping_delta
                else:
                    relevant_metric_history[0] += self.config.early_stopping_delta
                best_index = (
                    np.argmin(relevant_metric_history)
                    if self.config.minimize_best_model_metric
                    else np.argmax(relevant_metric_history)
                )
                if best_index == 0:
                    print(
                        f"\nEarly stopping after {epoch} epochs (len history {len(history)}) ({self.config.early_stopping_patience} epochs without improvement in validation {self.config.best_model_metric} metrics)"
                    )
                    break
            if self.killer.should_stop():
                print("Early stopping due to user interrupt")
                break

        return self._load_best_model_and_test(
            history, best_model_path, best_model_epoch
        )

    def _save_best_model(self, best_model_path):
        torch.save(self.model.state_dict(), best_model_path)

    def _load_best_model_and_test(self, history, best_model_path, best_model_epoch):
        if self.config.return_best_model:
            self.model.load_state_dict(torch.load(best_model_path))
            os.remove(best_model_path)
            print("Loaded model with best validation loss of this experiment from disk")

        val_losses = self.evaluate_epoch("val")
        test_losses = self.evaluate_epoch("test")
        wandb.log(self._get_wandb_metrics(val_losses, "val"))
        wandb.log(self._get_wandb_metrics(test_losses, "test"))
        for i, test_loss in enumerate(test_losses):
            print(f"\nTest loss ({self.loss_name}) {i}: {test_loss.get_average().loss}")
        return self.model, TrainHistory(history, test_losses, best_model_epoch)

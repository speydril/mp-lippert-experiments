import os
import uuid
from typing import Literal, cast

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from torch.nn import functional as F


from src.models.auto_sam_model import SAMBatch, norm_batch
from src.models.base_model import Loss, ModelOutput
from src.experiments.self_learning_experiment import SelfLearningExperiment
from src.datasets.base_dataset import Batch
from src.experiments.base_experiment import BaseExperiment
from src.train.evaluator import Evaluator
from src.train.history import EpochLosses, SingleEpochHistory, TrainHistory
from src.train.trainer import Trainer
from src.util.datatset_helper import generate_unsup_data


class SelfTrainer(Trainer):
    def __init__(self, experiment: SelfLearningExperiment):
        super().__init__(experiment)
        self.experiment = cast(SelfLearningExperiment, experiment)

        # Disable grad for teacher model
        for p in self.model.parameters():
            p.requires_grad = False

    def _update_ema(self, i_iter: int):
        ema_decay = min(
            1 - 1 / (i_iter + 1),
            self.experiment.config.ema_decay_origin,
        )
        for t_params, s_params in zip(
            self.model.parameters(), self.experiment.student_model.parameters()
        ):
            t_params.data = ema_decay * t_params.data + (1 - ema_decay) * s_params.data

    def _backward_batch_student(self, batch: Batch):
        self.optimizer.zero_grad()

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

        # Make predictions for this batch
        with torch.enable_grad():
            # calculate gradient for whole model (but only optimize parts)
            outputs = self.experiment.student_model.forward(batch)

        loss = self.experiment.student_model.compute_loss(outputs, batch)
        loss.loss.backward()

        if self.config.gradient_clipping is not None:
            torch.nn.utils.clip_grad_norm_(  # type: ignore
                self.experiment.student_model.parameters(),
                self.config.gradient_clipping,
            )

        # Adjust learning weights
        self.optimizer.step()
        return outputs, loss
    
    # Taken from https://github.com/usr922/FST/blob/main/semi_seg/train_semi.py
    # Line: 326
    def _augment_batch(self, batch: SAMBatch):
        assert batch.target is not None

        # apply strong data augmentation: cutout or cutmix
        if np.random.uniform(0, 1) < 0.5:
            image_u_aug, label_u_aug = \
                generate_unsup_data(batch.input, batch.target)
        else:
            image_u_aug = batch.input
            label_u_aug = batch.target
        
        return SAMBatch(
            input=image_u_aug,
            target=label_u_aug,
            original_size=batch.original_size,
            image_size=batch.image_size,)
        

    def _self_learning_epoch(self, data_loader: DataLoader, epoch: int):
        self.model.eval()
        self.experiment.student_model.train()
        evaluator = self.experiment.create_evaluator("train")
        iter_unlabeled = iter(self.experiment.unlabeled_loader)

        for i, batch in enumerate(data_loader):
            iteration = epoch * len(data_loader) + i
            batch = cast(Batch, batch).to(self.device)
            unlabeled_batch = next(iter_unlabeled).to(self.device)

            # Create pseudo labels for unlabeled data
            # The input data is not augmented yet, this is done after pseudo labels are created
            with torch.no_grad():
                unlabeled_batch.target = norm_batch(self.model.forward(unlabeled_batch).logits)
                
            # Batches are combined to one batch
            batch = self._merge_batches(cast(SAMBatch, batch), unlabeled_batch)
            batch = self._augment_batch(batch)

            outputs, loss = self._backward_batch_student(batch)

            # Adjust weights of teacher model
            with torch.no_grad():  # Disable gradient tracking
                self._update_ema(iteration)

            evaluator.track_batch(outputs, loss, batch)

            if (
                i % self.config.log_every_n_batches
                == self.config.log_every_n_batches - 1
            ):
                self._log_intermediate(
                    i, len(self.dataloader_train), evaluator
                )
        results = evaluator.evaluate()
        evaluator.clean_up()
        return results
    
    def _merge_batches(self, gt_batch: SAMBatch, pseudo_batch: SAMBatch) -> SAMBatch:
        assert gt_batch.target is not None and pseudo_batch.target is not None
        
        # Align tensors
        if gt_batch.target.dim() == 3:
            gt_batch.target = gt_batch.target.unsqueeze(1)
        if pseudo_batch.target.dim() == 3:
            pseudo_batch.target = pseudo_batch.target.unsqueeze(1)
        if gt_batch.input.dim() == 3:
            gt_batch.input = gt_batch.input.unsqueeze(1)
        if pseudo_batch.input.dim() == 3:
            pseudo_batch.input = pseudo_batch.input.unsqueeze(1)
            
        size = pseudo_batch.target.shape[2:]
        resized_gts = F.interpolate(
            (
                gt_batch.target.unsqueeze(dim=1)
                if gt_batch.target.dim() != gt_batch.target.dim()
                else gt_batch.target
            ),
            size,
            mode="nearest",
        )
        
        input = torch.cat([gt_batch.input, pseudo_batch.input], dim=0)    
        target = torch.cat([resized_gts, pseudo_batch.target], dim=0)
        return SAMBatch(input, target, original_size=gt_batch.original_size, image_size=gt_batch.image_size)
        
        

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

        def get_relevant_metric(epoch_hist: SingleEpochHistory):
            return (
                epoch_hist.get_average().loss
                if self.config.best_model_metric == "loss"
                else epoch_hist.get_average().metrics[self.config.best_model_metric]
            )

        best_model_epoch = -1

        for epoch in range(self.config.epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.epochs}")

            train_losses = self._self_learning_epoch(self.dataloader_train, epoch)
            val_losses = self.evaluate_epoch("val")
            self.scheduler.step()

            print(
                f"\n\n{'='*20}\nFinished Epoch {epoch + 1}/{self.config.epochs} "
                f"train {self.loss_name}-loss: {train_losses.get_average().loss} "
                f"val {self.loss_name}-loss: {val_losses.get_average().loss}"
            )
            epoch_losses = EpochLosses(train_losses, val_losses)
            history.append(epoch_losses)
            self._log_epoch_wandb(epoch_losses)
            if self.config.return_best_model:
                curr_epoch_val_metric = get_relevant_metric(val_losses)

                is_better = (
                    curr_epoch_val_metric < best_model_val_metric
                    if self.config.minimize_best_model_metric
                    else curr_epoch_val_metric > best_model_val_metric
                )
                if is_better:
                    best_model_val_metric = curr_epoch_val_metric
                    torch.save(self.model.state_dict(), best_model_path)
                    print(f"\n\nSaving model checkpoint at {best_model_path}\n")
                    best_model_epoch = epoch

            if (
                self.config.early_stopping_patience is not None
                and len(epoch_losses) >= self.config.early_stopping_patience
            ):
                relevant_metric_history = [
                    get_relevant_metric(epoch_loss.val_losses) for epoch_loss in history
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
                        f"\nEarly stopping after {epoch} epochs ({self.config.early_stopping_patience} epochs without improvement in validation {self.config.best_model_metric} metrics)"
                    )
                    break

        if self.config.return_best_model:
            self.model.load_state_dict(torch.load(best_model_path))
            os.remove(best_model_path)
            os.rmdir(os.path.dirname(best_model_path))
            print("Loaded model with best validation loss of this experiment from disk")

        test_losses = self.evaluate_epoch("test")
        wandb.log(self._get_wandb_metrics(test_losses, "test"))
        print(f"\nTest loss ({self.loss_name}): {test_losses.get_average().loss}")
        return self.model, TrainHistory(history, test_losses, best_model_epoch)

from tokenize import Single
from typing import Literal, cast

import numpy as np
import torch


from src.datasets.ukbiobank_dataset import BiobankBatch
from src.datasets.online_st_dataset import OnlineSTMixedBatch
from src.models.auto_sam_model import norm_batch
from src.experiments.self_learning_experiment import SelfLearningExperiment
from src.datasets.base_dataset import Batch
from src.train.trainer import Trainer
from src.util.datatset_helper import generate_unsup_data
from src.train.history import SingleEpochHistory


class SelfTrainer(Trainer):
    def __init__(self, experiment: SelfLearningExperiment):
        super().__init__(experiment)
        self.experiment = cast(SelfLearningExperiment, experiment)

        # Disable grad for teacher model
        for p in self.experiment.teacher_model.parameters():
            p.requires_grad = False

    def evaluate_epoch(
        self, mode: Literal["val"] | Literal["test"]
    ) -> list[SingleEpochHistory]:
        results = super().evaluate_epoch(mode)

        dataloader = (
            self.experiment.val_biobank_loader
            if mode == "val"
            else self.experiment.test_biobank_loader
        )

        self.model.eval()
        evaluator = self.experiment.create_evaluator(mode)

        for i, batch in enumerate(dataloader):
            batch = cast(Batch, batch).to(self.device)
            batch.target = self._preprocess_labels(batch)

            with torch.no_grad():
                outputs = self.model.forward(batch)
            loss = self.model.compute_loss(outputs, batch)
            evaluator.track_batch(outputs, loss, batch)
            if (
                i % self.config.log_every_n_batches
                == self.config.log_every_n_batches - 1
            ):
                self._log_intermediate(i, len(dataloader), evaluator)

        pseudo_label_results = evaluator.evaluate()
        evaluator.clean_up()
        results.append(pseudo_label_results)
        return results

    def _get_wandb_metrics(
        self, epoch: SingleEpochHistory | list[SingleEpochHistory], prefix: str
    ):
        if isinstance(epoch, SingleEpochHistory):
            return super()._get_wandb_metrics(epoch, prefix)

        def add_prefix_to_dict_keys(d: dict, prefix: str, suffix: str):
            return {f"{prefix}_{k}_{suffix}": v for k, v in d.items()}

        wandb_metrics = {}
        for i, epoch_history in enumerate(epoch):
            loss_avg = epoch_history.get_average()
            epoch_val_metrics = loss_avg.metrics

            wandb_metrics[f"{prefix}_{self.loss_name}_loss_{i}"] = loss_avg.loss
            wandb_metrics.update(
                add_prefix_to_dict_keys(epoch_val_metrics, prefix, str(i))
            )

        loss_sum_of_avgs = epoch[0].get_average()
        loss_sum_of_avgs += epoch[1].get_average()
        loss_sum_of_avgs /= 2
        epoch_val_metrics = loss_sum_of_avgs.metrics
        wandb_metrics[f"{prefix}_{self.loss_name}_loss_avg"] = loss_sum_of_avgs.loss
        wandb_metrics.update(add_prefix_to_dict_keys(epoch_val_metrics, prefix, "avg"))

        return wandb_metrics

    def get_relevant_metric(
        self, epoch_hists: SingleEpochHistory | list[SingleEpochHistory]
    ):
        assert isinstance(epoch_hists, list) and len(epoch_hists) == 2

        loss_sum = epoch_hists[0].get_average()
        loss_sum += epoch_hists[1].get_average()

        metrics = self._get_wandb_metrics(epoch_hists, "val")

        return (
            loss_sum.loss
            if self.config.best_model_metric == "loss"
            else metrics[self.config.best_model_metric]
        )

    # Taken from https://github.com/usr922/FST/blob/main/semi_seg/train_semi.py
    # Line: 326
    def _augment_batch(self, batch: Batch) -> Batch:
        assert batch.target is not None

        # apply strong data augmentation: cutout or cutmix
        if np.random.uniform(0, 1) < 0.5:
            batch.input, batch.target = generate_unsup_data(batch.input, batch.target)

        return super()._augment_batch(batch)

    def _preprocess_labels(self, batch: Batch):
        if isinstance(batch, BiobankBatch) and not isinstance(
            batch, OnlineSTMixedBatch
        ):
            # Fill in the gt flag
            batch = OnlineSTMixedBatch(
                **batch.__dict__, is_gt=torch.zeros(batch.input.shape[0])
            )
        assert isinstance(batch, OnlineSTMixedBatch) and batch.target is not None
        self.experiment.teacher_model.eval()
        indices = torch.where(batch.is_gt == 0)
        with torch.no_grad():
            batch_to_label = Batch(batch.input[indices], None)
            targets = batch.target.unsqueeze(1)
            teacher_logits = self.experiment.teacher_model.forward(
                batch_to_label
            ).logits
            teacher_logits = torch.nn.functional.interpolate(
                teacher_logits,
                size=targets.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            targets[indices] = norm_batch(teacher_logits)
            return targets

    def _after_batch_complete(self, iter: int):
        if self.experiment.config.constant_ema_decay:
            teacher_ratio = self.experiment.config.ema_decay
        else:
            teacher_ratio = max(self.experiment.config.ema_decay, 1 - (1 / (iter + 1)))

        # Adjust weights of teacher model
        with torch.no_grad():  # Disable gradient tracking
            for t_params, s_params in zip(
                self.experiment.teacher_model.parameters(), self.model.parameters()
            ):
                t_params.data = (
                    teacher_ratio * t_params.data + (1 - teacher_ratio) * s_params.data
                )

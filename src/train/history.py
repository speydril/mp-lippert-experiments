from math import nan
from typing import NamedTuple, Optional, Union
from dataclasses import dataclass

from matplotlib.pylab import f


class DecodedPredictionBatch(NamedTuple):
    predictions: list[str]
    targets: Optional[list[str]]


@dataclass
class MetricEntry:
    metrics: dict[str, float]
    loss: float = 0

    def __iadd__(self, other: "MetricEntry"):
        for key, value in other.metrics.items():
            if key in self.metrics and self.metrics[key] is not None:
                self.metrics[key] += value
            else:
                self.metrics[key] = value
        self.loss += other.loss
        return self

    def __truediv__(self, other: float):
        metrics_copy = dict(self.metrics)
        for key, value in metrics_copy.items():
            if other != 0:
                metrics_copy[key] /= other
            else:
                metrics_copy[key] = nan
        return MetricEntry(metrics_copy, self.loss / other if other != 0 else nan)


class SingleEpochHistory:
    def __init__(self):
        self.metrics: list[MetricEntry] = []
        self._total_loss = MetricEntry({})
        self._total_loss_count = 0
        self.decoded: list[Union[DecodedPredictionBatch, None]] = []

    def add_batch_metric(
        self, loss: MetricEntry, decoded: Optional[DecodedPredictionBatch] = None
    ):
        self.metrics.append(loss)
        self._total_loss += loss
        self._total_loss_count += 1
        self.decoded.append(decoded)

    def get_average(self):
        return self._total_loss / self._total_loss_count

    def get_last(self):
        return self.metrics[-1]

    def to_dict(self):
        def get_batch(i: int):
            entry = self.decoded[i]
            if entry is not None:
                if hasattr(entry, "__dict__"):
                    return {
                        **vars(entry),
                        **entry._asdict(),
                    }
                return entry._asdict()
            return {}

        return {
            "history": [
                {**metric.__dict__, "batch": get_batch(i)}
                for i, metric in enumerate(self.metrics)
            ],
            "average": self.get_average().__dict__,
        }

    def plot_metric_as_hist(self, metric_key: str, title: str, plt_ax):
        metric = [
            item.metrics[metric_key]
            for item in self.metrics
            if metric_key in item.metrics
        ]
        # Histogram for data1
        plt_ax.hist(metric, bins=10, color="blue", alpha=0.7)
        num_ignored = len(self.metrics) - len(metric)
        plt_ax.set_title(
            title
            + (
                f" (ignored {num_ignored} batches w/o {metric_key})"
                if num_ignored > 0
                else ""
            )
        )
        plt_ax.set_xlabel(metric_key)
        plt_ax.set_ylabel("Frequency")

    def save_plot_metric_as_hist(self, metric_key: str, title: str, out_path: str):
        import matplotlib.pyplot as plt

        # Create a figure and a set of subplots
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        self.plot_metric_as_hist(metric_key, title, ax)

        # Display the histograms
        plt.tight_layout()
        plt.savefig(out_path)


class EpochLosses(NamedTuple):
    train_losses: SingleEpochHistory
    val_losses: SingleEpochHistory | list[SingleEpochHistory]

    def to_dict(self):
        return {
            "train": self.train_losses.to_dict(),
            "val": (
                self.val_losses.to_dict()
                if self.val_losses is SingleEpochHistory
                else [val.to_dict() for val in self.val_losses]  # type: ignore
            ),
        }


class TrainHistory(NamedTuple):
    epochs: list[EpochLosses]
    test_losses: SingleEpochHistory | list[SingleEpochHistory]
    epoch_index_of_test_model: int

    def to_dict(self):
        return {
            "epochs": [epoch.to_dict() for epoch in self.epochs],
            "test": (
                [self.test_losses.to_dict()]
                if self.test_losses is SingleEpochHistory
                else [test.to_dict() for test in self.test_losses]  # type: ignore
            ),
            "epoch_index_of_test_model": self.epoch_index_of_test_model,
        }

    @classmethod
    def from_json(cls, json_path: str):
        import json

        with open(json_path, "r") as f:
            data = json.load(f)

        epochs = data["epochs"]
        test_losses = data["test"]

        def get_decoded(batch):
            try:
                if "batch" in batch and batch["batch"] == True:
                    return DecodedPredictionBatch(**batch["batch"])
            except:
                pass
            return None

        test_history = SingleEpochHistory()
        for batch in test_losses["history"]:
            test_history.add_batch_metric(
                MetricEntry(batch["metrics"], batch["loss"]),
                get_decoded(batch),
            )

        epoch_histories: list[EpochLosses] = []

        for epoch in epochs:
            train_epoch_history = SingleEpochHistory()
            for batch in epoch["train"]["history"]:
                train_epoch_history.add_batch_metric(
                    MetricEntry(batch["metrics"], batch["loss"]),
                    get_decoded(batch),
                )

            val_epoch_history = SingleEpochHistory()
            for batch in epoch["val"]["history"]:
                val_epoch_history.add_batch_metric(
                    MetricEntry(batch["metrics"], batch["loss"]),
                    get_decoded(batch),
                )

            epoch_history = EpochLosses(
                train_losses=train_epoch_history, val_losses=val_epoch_history
            )
            epoch_histories.append(epoch_history)

        return cls(
            epochs=epoch_histories,
            test_losses=test_history,
            epoch_index_of_test_model=data["epoch_index_of_test_model"],
        )

    def plot(self, out_path: str):
        import matplotlib.pyplot as plt

        if len(self.epochs) == 0:
            return

        metric_keys = set()
        for epoch in self.epochs:
            metric_keys = metric_keys.union(
                epoch.train_losses.get_average().metrics.keys()
            )
            metric_keys = metric_keys.union(
                epoch.val_losses.get_average().metrics.keys()
                if isinstance(epoch.val_losses, SingleEpochHistory)
                else epoch.val_losses[0].get_average().metrics.keys()
            )

        # Creating a figure and subplots
        fig, axs = plt.subplots(
            nrows=len(metric_keys), ncols=1, figsize=(10, 5 * len(metric_keys))
        )
        num_epochs = len(self.epochs)

        for i, metric_key in enumerate(metric_keys):
            ax = axs[i] if len(metric_keys) > 1 else axs  # type: ignore

            def plot_loss(
                label: str,
                linestyle: str,
                marker: str,
                histories: list[SingleEpochHistory],
                metric_key: str,
            ):
                averages = [history.get_average() for history in histories]
                losses = [
                    epoch.metrics[metric_key]
                    for epoch in averages
                    if metric_key in epoch.metrics
                ]
                ax.plot(  # type: ignore
                    losses,
                    label=label,
                    linestyle=linestyle,
                    marker=marker,
                )

            plot_loss(
                f"{metric_key} (train)",
                "-",
                "o",
                [epoch.train_losses for epoch in self.epochs],
                metric_key,
            )

            if isinstance(self.epochs[0].val_losses, list):
                for i in range(len(self.epochs[0].val_losses)):
                    plot_loss(
                        f"{metric_key} (validation {i})",
                        "-",
                        ".",
                        [epoch.val_losses[i] for epoch in self.epochs],  # type: ignore
                        metric_key,
                    )
            else:
                plot_loss(
                    f"{metric_key} (validation)",
                    "-",
                    ".",
                    [epoch.val_losses for epoch in self.epochs],  # type: ignore
                    metric_key,
                )

            def plot_test_loss(
                label: str,
                linestyle: str,
                marker: str,
                losses: SingleEpochHistory,
                metric_key: str,
            ):
                loss_avgs: list = [None] * len(self.epochs)
                loss_avgs[self.epoch_index_of_test_model] = (
                    losses.get_average().metrics[metric_key]
                )
                ax.plot(  # type: ignore
                    loss_avgs,
                    label=label,
                    linestyle=linestyle,
                    marker=marker,
                    markersize=16,
                )

            if isinstance(self.test_losses, SingleEpochHistory):
                plot_test_loss(
                    f"{metric_key} (test)",
                    "-",
                    "x",
                    self.test_losses,
                    metric_key,
                )
            else:
                for i, test_loss in enumerate(self.test_losses):
                    plot_test_loss(
                        f"{metric_key} (test {i})",
                        "-",
                        "x",
                        test_loss,
                        metric_key,
                    )

            ax.grid()  # type: ignore
            ax.set_xlabel("Epochs")  # type: ignore
            ax.set_ylabel(metric_key)  # type: ignore
            ax.set_title(f"{metric_key} history")  # type: ignore
            ax.legend()  # type: ignore
            x_ticks = list(range(0, num_epochs, max(1, num_epochs // 10)))
            ax.set_xticks(x_ticks, [str(i) for i in x_ticks])  # type: ignore

        plt.tight_layout()
        plt.savefig(out_path)

    def plot_metric_histograms(self, out_dir: str, metric_key: str):
        import os
        import matplotlib.pyplot as plt

        out_dir = os.path.join(out_dir, metric_key)
        os.makedirs(out_dir, exist_ok=True)

        if isinstance(self.test_losses, SingleEpochHistory):
            self.test_losses.save_plot_metric_as_hist(
                metric_key,
                "Test set",
                os.path.join(out_dir, "test_histogram.png"),
            )
        else:
            for i, test_loss in enumerate(self.test_losses):
                test_loss.save_plot_metric_as_hist(
                    metric_key,
                    f"Test set {i}",
                    os.path.join(out_dir, f"test_histogram_{i}.png"),
                )
        fig, ax = plt.subplots(len(self.epochs), 2, figsize=(10, len(self.epochs) * 5))
        for i, epoch in enumerate(self.epochs):
            epoch.train_losses.plot_metric_as_hist(
                metric_key,
                f"Train (ep. {i})",
                ax[i, 0],  # type: ignore
            )
            if isinstance(epoch.val_losses, SingleEpochHistory):
                epoch.val_losses.plot_metric_as_hist(
                    metric_key,
                    f"Val (ep. {i})",
                    ax[i, 1],  # type: ignore
                )
            else:
                for j, val_loss in enumerate(epoch.val_losses):
                    val_loss.plot_metric_as_hist(
                        metric_key,
                        f"Val {j} (ep. {i})",
                        ax[i, 1],  # type: ignore
                    )

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "val_train_histograms.png"))

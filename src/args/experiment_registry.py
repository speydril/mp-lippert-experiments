from typing import Type
from src.experiments.pixel_lines_experiment import PixelLinesExperiment
from src.experiments.self_learning_experiment_debug import SelfLearningExperimentDebug
from src.experiments.self_learning_experiment import SelfLearningExperiment
from src.experiments.ukbiobank_experiment import UkBioBankExperiment
from src.experiments.aria_experiment import ARIAExperiment
from src.experiments.resnet_filter_experiment import ResnetFilterExperiment
from src.experiments.multi_ds_vessel_experiment import MultiDsVesselExperiment
from src.experiments.hrf_experiment import HrfExperiment
from src.experiments.drive_experiment import DriveExperiment
from src.experiments.refuge_experiment import RefugeExperiment
from src.experiments.mnist_experiment import MnistExperiment
from src.experiments.base_experiment import BaseExperiment

experiments: dict[str, Type[BaseExperiment]] = {
    "mnist": MnistExperiment,
    "refuge": RefugeExperiment,
    "drive": DriveExperiment,
    "multi_ds_vessel": MultiDsVesselExperiment,
    "hrf": HrfExperiment,
    "resnet_filter": ResnetFilterExperiment,
    "uk_biobank_experiment": UkBioBankExperiment,
    "aria": ARIAExperiment,
    "self_learning_experiment": SelfLearningExperiment,
    "self_learning_experiment_debug": SelfLearningExperimentDebug,
    "pixel_lines_experiment": PixelLinesExperiment
}

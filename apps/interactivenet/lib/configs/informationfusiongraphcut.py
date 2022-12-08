import logging
from typing import Any, Dict, Union

import lib.infers
import lib.trainers

from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer import InferTask

logger = logging.getLogger(__name__)


class InformationFusionGraphCut(TaskConfig):
    def init(
        self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs
    ):
        super().init(name, model_dir, conf, planner, **kwargs)

        self.labels = {
            "tumor": 1,
        }

        # Parameters
        self.median_shape = (512, 512, 32)
        self.target_spacing = (0.6875, 0.6875, 3.600001096725464)
        self.relax_bbox = 0.1
        self.divisble_using = (16, 16, 4)
        self.clipping = []
        self.intensity_mean = 0
        self.intensity_std = 0
        self.tmp_folder = "/tmp/"

    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        task: InferTask = lib.infers.InformationFusionGraphCut(
            median_shape=self.median_shape,
            target_spacing=self.target_spacing,
            relax_bbox=self.relax_bbox,
            divisble_using=self.divisble_using,
            clipping=self.clipping,
            intensity_mean=self.intensity_mean,
            intensity_std=self.intensity_std,
            labels=self.labels,
        )
        return task

    def trainer(self) -> None:
        return None

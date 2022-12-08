import logging
import os
import glob
from typing import Any, Dict, Union

import lib.infers
import lib.trainers
from monai.networks.nets import DynUNet

from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer import InferTask
logger = logging.getLogger(__name__)


class InteractiveNet(TaskConfig):
    def init(
        self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, studies: str, **kwargs
    ):
        super().init(name, model_dir, conf, planner, **kwargs)

        self.labels = {
            "tumor": 1,
        }

        modality = studies.split("/")[-1].split("_")[1]
        logger.info(f"Using modality: {modality}")
        # Parameters
        if modality == "MRI":
            self.median_shape = (512, 512, 32)
            self.target_spacing = (0.6875, 0.6875, 3.600001096725464)
            self.relax_bbox = 0.1
            self.divisble_using = (16, 16, 4)
            self.clipping = []
            self.intensity_mean = 0
            self.intensity_std = 0
            self.kernels = [[3, 3, 1], [3, 3, 1], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
            self.strides = [[1, 1, 1], [2, 2, 1], [2, 2, 1], [2, 2, 2], [2, 2, 2]]
        elif modality == "CT":
            self.median_shape = (512, 512, 114.5)
            self.target_spacing = (0.7734375, 0.7734375, 1)
            self.relax_bbox = 0.1
            self.divisble_using = (16, 16, 8)
            self.clipping = [-44.0, 142.0]
            self.intensity_mean = 41.436787
            self.intensity_std = 51.599117
            self.kernels = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
            self.strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 1]]

        self.tmp_folder = "/tmp/"
        self.studies = studies + "/meta/"
        
        if not os.path.exists(self.studies):
            os.makedirs(self.studies)

        # Model Files
        if "ensemble" in name:
            self.ensemble = True
            self.path = [
                model for model in glob.glob(f"{self.model_dir}/{modality}/interactivenet*")
            ]
        else:
            fold = 0
            self.ensemble = False
            self.path = [
                os.path.join(
                    self.model_dir, f"{self.model_dir}/{modality}/interactivenet_{fold}.pt"
                ),
            ]

        if "tta" in name:
            self.tta = True
        else:
            self.tta = False

        # Network
        self.network = DynUNet(
            spatial_dims=3,
            in_channels=2,
            out_channels=2,
            kernel_size=self.kernels,
            strides=self.strides,
            upsample_kernel_size=self.strides[1:],
            filters=[4, 8, 16, 32, 64, 128],
            norm_name="instance",
            act_name="leakyrelu",
        )

    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        task: InferTask = lib.infers.InteractiveNet(
            path=self.path,
            network=self.network,
            ensemble=self.ensemble,
            tta=self.tta,
            median_shape=self.median_shape,
            target_spacing=self.target_spacing,
            relax_bbox=self.relax_bbox,
            divisble_using=self.divisble_using,
            clipping=self.clipping,
            intensity_mean=self.intensity_mean,
            intensity_std=self.intensity_std,
            labels=self.labels,
            tmp_folder=self.tmp_folder,
            studies=self.studies
        )
        return task

    def trainer(self) -> None:
        return None

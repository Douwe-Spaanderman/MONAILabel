from http import client
import logging
import numpy as np
import torch
import os
import time
import json

from typing import Callable, Sequence, Union, Tuple, List

from lib.transforms.transforms import (
    AnnotationToChanneld,
    Resamplingd,
    BoudingBoxd,
    NormalizeValuesd,
    EGDMapd,
    OriginalSized,
    TestTimeFlippingd,
    SaveIntermediated,
)
from monai.inferers import Inferer, SimpleInferer
from monai.data import MetaTensor
from monai.transforms import (
    AsDiscreted,
    AddChanneld,
    EnsureTyped,
    LoadImaged,
    CastToTyped,
    ConcatItemsd,
    ToTensord,
    ToNumpyd,
    MeanEnsembled,
    KeepLargestConnectedComponentd,
    FillHolesd,
    SqueezeDimd,
)

from monai.apps.deepgrow.transforms import (
    AddGuidanceFromPointsd,
)

from monailabel.interfaces.tasks.infer import InferTask, InferType

logger = logging.getLogger(__name__)

from typing import Any, Dict, Sequence, Tuple

from monailabel.transform.writer import Writer

class InteractiveNet(InferTask):
    """ """

    def __init__(
        self,
        path,
        network=None,
        type=InferType.DEEPGROW,
        labels=None,
        dimension=3,
        ensemble: bool = False,
        tta: bool = False,
        median_shape: Tuple[float] = (128, 128, 64),
        target_spacing: Tuple[float] = (1.0, 1.0, 1.0),
        relax_bbox: Union[float, Tuple[float]] = 0.1,
        divisble_using: Union[int, Tuple[int]] = (16, 16, 8),
        clipping: List[float] = [],
        intensity_mean: float = 0,
        intensity_std: float = 0,
        tmp_folder: str = "/tmp/",
        studies: str = "/tmp/",
        description="Volumetric Interactivenet",
        **kwargs,
    ):
        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
            input_key="image",
            output_label_key="mask",
            output_json_key="result",
            **kwargs,
        )

        self.ensemble = ensemble
        self.tta = tta
        self.median_shape = median_shape
        self.target_spacing = target_spacing
        self.relax_bbox = relax_bbox
        self.divisble_using = divisble_using
        self.clipping = clipping
        self.intensity_mean = intensity_mean
        self.intensity_std = intensity_std
        if self.clipping:
            self.ct = True
        else:
            self.ct = False

        self.tmp_folder = tmp_folder
        self.studies = studies

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        t = [
            LoadImaged(keys="image"),
        ]

        self.add_cache_transform(t, data)

        t.extend(
            [
                AddGuidanceFromPointsd(
                    ref_image="image",
                    guidance="annotation",
                    depth_first=False,
                    dimensions=3,
                ),
                AnnotationToChanneld(
                    ref_image="image", guidance="annotation", method="interactivenet"
                ),
                AddChanneld(keys=["image"]),
                ToNumpyd(keys=["image", "annotation"]),
                Resamplingd(
                    keys=["image", "annotation"],
                    pixdim=self.target_spacing,
                ),
                BoudingBoxd(
                    keys=["image", "annotation"],
                    on="annotation",
                    relaxation=self.relax_bbox,
                    divisiblepadd=self.divisble_using,
                ),
                NormalizeValuesd(
                    keys=["image"],
                    clipping=self.clipping,
                    mean=self.intensity_mean,
                    std=self.intensity_std,
                ),
                EGDMapd(
                    keys=["annotation"],
                    image="image",
                    lamb=1,
                    iter=4,
                    logscale=True,
                    ct=self.ct,
                ),
                CastToTyped(
                    keys=["image", "annotation"], dtype=(np.float32, np.float32)
                ),
            ]
        )

        if self.tta:
            t.extend(
                [
                    TestTimeFlippingd(keys=["image", "annotation"]),
                ]
            )
            dim = 1
        else:
            dim = 0

        t.extend(
            [
                ConcatItemsd(keys=["image", "annotation"], name="image", dim=dim),
                ToTensord(keys=["image"]),
            ]
        )
        return t

    def inferer(self, data=None) -> Inferer:
        return SimpleInferer()

    def post_transforms(self, data=None) -> Sequence[Callable]:
        t = [
            EnsureTyped(keys="mask", device=data.get("device") if data else None),
        ]

        if self.tta:
            t.extend(
                [
                    TestTimeFlippingd(keys=["mask"], back=True),
                ]
            )

        if self.ensemble or self.tta:
            t.extend(
                [
                    MeanEnsembled(keys="mask"),
                ]
            )

        t.extend(
            [
                OriginalSized(
                    img_key="mask",
                    ref_meta="image",
                    discreet=True,
                    device=data.get("device") if data else None,
                ),
                #SaveIntermediated(
                #    keys=["weights"],
                #    name="weights",
                #    tmp_folder=self.tmp_folder,
                #    softmax=True,
                #),
                KeepLargestConnectedComponentd(keys="mask"),
                FillHolesd(keys="mask"),
                SqueezeDimd(keys="mask"),
            ]
        )

        return t

    def run_inferer(self, data, convert_to_batch=True, device="cuda"):
        if self.tta:
            convert_to_batch = False

        if self.ensemble:
            pred = []
            models = self.path
            for model in models:
                self.model = model
                output = super().run_inferer(
                    data, convert_to_batch=convert_to_batch, device=device
                )

                pred.append(output["mask"])

            output["mask"] = MetaTensor(torch.stack(pred), affine=output["mask"].affine)
        else:
            super().run_inferer(data, convert_to_batch=convert_to_batch, device=device)

        return data

    def get_path(self):
        if not self.path:
            return None

        paths = self.path

        if len(paths) == 1:
            for path in reversed(paths):
                if path and os.path.exists(path):
                    return path
        else:
            if self.model in paths and os.path.exists(self.model):
                return self.model
        return None

    def writer(self, data: Dict[str, Any], extension=None, dtype=None) -> Tuple[Any, Any]:
        """
        You can provide your own writer.  However, this writer saves the prediction/label mask to file
        and fetches result json
        :param data: typically it is post processed data
        :param extension: output label extension
        :param dtype: output label dtype
        :return: tuple of output_file and result_json
        """
        logger.info("Writing Result...")
        if extension is not None:
            data["result_extension"] = extension
        if dtype is not None:
            data["result_dtype"] = dtype
        if self.labels is not None:
            data["labels"] = self.labels

        writer = Writer(label=self.output_label_key)

        json_file = {
            "ts" : int(time.time()),
            "latencies" : data["latencies"],
            "interactions" : data["foreground"]
        }

        name = data["image_path"].split("/")[-1].split(".")[0]
        client = data["client_id"]
        studies = self.studies + f"/{client}/" 

        if not os.path.exists(studies):
            os.makedirs(studies)

        json_loc = studies + f"{name}.json"
        with open(json_loc, 'w', encoding='utf-8') as f:
            json.dump(json_file, f, ensure_ascii=False, indent=4)

            
        return writer(data)
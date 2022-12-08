
import logging

from typing import Union, Tuple, List

from lib.transforms.transforms import (
    AnnotationToChanneld,
    Resamplingd,
    BoudingBoxd,
    NormalizeValuesd,
    EGDMapd,
    OriginalSized,
    LoadWeightsd,
    InformationFusionGraphCutd,
)
from monai.transforms import (
    Compose,
    AddChanneld,
    LoadImaged,
    ToNumpyd,
    KeepLargestConnectedComponentd,
    FillHolesd,
    SqueezeDimd,
)

from monai.apps.deepgrow.transforms import (
    AddGuidanceFromPointsd,
)

from monailabel.interfaces.tasks.infer import InferTask, InferType

logger = logging.getLogger(__name__)


class InformationFusionGraphCut(InferTask):
    """
    Defines a generic Scribbles Likelihood based segmentor infertask
    """

    def __init__(
        self,
        type=InferType.DEEPGROW,
        labels=None,
        dimension=3,
        median_shape: Tuple[float] = (128, 128, 64),
        target_spacing: Tuple[float] = (1.0, 1.0, 1.0),
        relax_bbox: Union[float, Tuple[float]] = 0.1,
        divisble_using: Union[int, Tuple[int]] = (16, 16, 8),
        clipping: List[float] = [],
        intensity_mean: float = 0,
        intensity_std: float = 0,
        tmp_folder: str = "/tmp/",
        description="A refinement method using Information Fusion + GraphCut for initial segmentations",
    ):
        super().__init__(
            path=None,
            network="place_holder",
            labels=labels,
            type=type,
            dimension=dimension,
            description=description,
        )

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

    def pre_transforms(self, data):
        t = [
            LoadImaged(keys=["image"]),
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
                    ref_image="image", guidance="annotation", method="refinement"
                ),
                AddChanneld(keys=["image"]),
                ToNumpyd(keys=["image", "annotation"]),
                Resamplingd(
                    keys=["image", "annotation"],
                    pixdim=self.target_spacing,
                ),
                LoadWeightsd(
                    key="weights",
                    ref_image="image",
                    tmp_folder=self.tmp_folder,
                    device=data.get("device") if data else None,
                ),
                BoudingBoxd(
                    keys=["image", "annotation", "weights"],
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
                    powerof=2,
                    backup=True,
                ),
            ]
        )

        return t

    def post_transforms(self, data):
        return [
            KeepLargestConnectedComponentd(keys="pred"),
            FillHolesd(keys="pred"),
            OriginalSized(
                img_key="pred",
                ref_meta="image",
                discreet=False,
                label=True,
                device=data.get("device") if data else None,
            ),
            SqueezeDimd(keys="pred"),
        ]

    def inferer(self, data):
        return Compose(
            [
                InformationFusionGraphCutd(
                    image="image",
                    cue_map="annotation",
                    interations="annotation_backup",
                    prediction="weights",
                ),
            ]
        )

    def is_valid(self):
        return True

    def _get_network(self, device):
        return None

from monai.apps.deepgrow.transforms import (
    AddGuidanceFromPointsd,
    Fetch2DSliced,
    SpatialCropGuidanced,
    ResizeGuidanced,
    AddGuidanceSignald,
    RestoreLabeld
)
from monai.inferers import SimpleInferer
from monai.transforms import (
    LoadImaged,
    AsChannelFirstd,
    AddChanneld,
    Spacingd,
    Activationsd,
    AsDiscreted,
    ToNumpyd,
    Resized,
    NormalizeIntensityd,
    AsChannelLastd
)
from server.interface import InferenceEngine


# In many cases people like to use something existing.. and only define, pre/post transforms + inferer
# Or you can write your InferenceEngine (e.g. run multiple/chained inferences...)
class DeepgrowInferenceEngine(InferenceEngine):
    def __init__(self, model):
        super().__init__(model=model)

    def pre_transforms(self):
        return [
            LoadImaged(keys='image'),
            AsChannelFirstd(keys='image'),
            Spacingd(keys='image', pixdim=[1.0, 1.0], mode='bilinear'),
            AddGuidanceFromPointsd(ref_image='image', guidance='guidance', dimensions=2),
            Fetch2DSliced(keys='image', guidance='guidance'),
            AddChanneld(keys='image'),
            SpatialCropGuidanced(keys='image', guidance='guidance', spatial_size=[256, 256]),
            Resized(keys='image', spatial_size=[256, 256], mode='area'),
            ResizeGuidanced(guidance='guidance', ref_image='image'),
            NormalizeIntensityd(keys='image', subtrahend=208, divisor=388),
            AddGuidanceSignald(image='image', guidance='guidance')
        ]

    def inferer(self):
        return SimpleInferer()

    def post_transforms(self):
        return [
            Activationsd(keys='pred', sigmoid=True),
            AsDiscreted(keys='pred', threshold_values=True, logit_thresh=0.5),
            ToNumpyd(keys='pred'),
            RestoreLabeld(keys='pred', ref_image='image', mode='nearest'),
            AsChannelLastd(keys='pred')
        ]

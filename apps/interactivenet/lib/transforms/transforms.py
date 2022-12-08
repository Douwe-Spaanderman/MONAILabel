# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import warnings
import pickle
import math
import os
from itertools import combinations

from monai.transforms.transform import MapTransform, Transform
from monai.transforms import (
    NormalizeIntensity,
    GaussianSmooth,
    Flip,
    AsDiscrete,
    Activations,
)
from monai.data import MetaTensor
import numpy as np
import GeodisTK

import numpymaxflow
from skimage.transform import resize
from nibabel import affines
import numpy.linalg as npl
import torch

logger = logging.getLogger(__name__)


def resample_label(label, shape, anisotrophy_flag):
    reshaped = np.zeros(shape, dtype=np.uint8)
    n_class = np.max(label)
    if anisotrophy_flag:
        shape_2d = shape[:-1]
        depth = label.shape[-1]
        reshaped_2d = np.zeros((*shape_2d, depth), dtype=np.uint8)

        for class_ in range(1, int(n_class) + 1):
            for depth_ in range(depth):
                mask = label[0, :, :, depth_] == class_
                resized_2d = resize(
                    mask.astype(float),
                    shape_2d,
                    order=1,
                    mode="edge",
                    cval=0,
                    clip=True,
                    anti_aliasing=False,
                )
                reshaped_2d[:, :, depth_][resized_2d >= 0.5] = class_
        for class_ in range(1, int(n_class) + 1):
            mask = reshaped_2d == class_
            resized = resize(
                mask.astype(float),
                shape,
                order=0,
                mode="constant",
                cval=0,
                clip=True,
                anti_aliasing=False,
            )
            reshaped[resized >= 0.5] = class_
    else:
        for class_ in range(1, int(n_class) + 1):
            mask = label[0] == class_
            resized = resize(
                mask.astype(float),
                shape,
                order=1,
                mode="edge",
                cval=0,
                clip=True,
                anti_aliasing=False,
            )
            reshaped[resized >= 0.5] = class_

    reshaped = np.expand_dims(reshaped, 0)
    return reshaped


def resample_image(image, shape, anisotrophy_flag):
    resized_channels = []
    if anisotrophy_flag:
        for image_c in image:
            resized_slices = []
            for i in range(image_c.shape[-1]):
                image_c_2d_slice = image_c[:, :, i]
                image_c_2d_slice = resize(
                    image_c_2d_slice,
                    shape[:-1],
                    order=3,
                    mode="edge",
                    cval=0,
                    clip=True,
                    anti_aliasing=False,
                )
                resized_slices.append(image_c_2d_slice)
            resized = np.stack(resized_slices, axis=-1)
            resized = resize(
                resized,
                shape,
                order=0,
                mode="constant",
                cval=0,
                clip=True,
                anti_aliasing=False,
            )
            resized_channels.append(resized)
    else:
        for image_c in image:
            resized = resize(
                image_c,
                shape,
                order=3,
                mode="edge",
                cval=0,
                clip=True,
                anti_aliasing=False,
            )
            resized_channels.append(resized)
    resized = np.stack(resized_channels, axis=0)
    return resized


def resample_annotation(image, affine, new_spacing, shape):
    resized_channels = []
    for image_d in image:
        resized = np.zeros(shape)
        new_affine = affines.rescale_affine(
            affine, image_d.shape, new_spacing, new_shape=shape
        )

        inds_x, inds_y, inds_z = np.where(image_d > 0.5)
        for i, j, k in zip(inds_x, inds_y, inds_z):
            old_vox2new_vox = npl.inv(new_affine).dot(affine)
            new_point = np.rint(
                affines.apply_affine(old_vox2new_vox, [i, j, k])
            ).astype(int)

            for i in range(len(new_point)):
                if new_point[i] < 0:
                    new_point[i] = 0
                elif new_point[i] >= shape[i]:
                    new_point[i] = shape[i] - 1

            resized[new_point[0], new_point[1], new_point[2]] = 1

        resized_channels.append(resized)

    resized = np.stack(resized_channels, axis=0)
    return resized


class OriginalSized(Transform):
    """
    Return the label to the original image shape
    """

    def __init__(
        self,
        img_key,
        ref_meta,
        keep_key=None,
        label: bool = True,
        discreet: bool = True,
        device=None,
    ) -> None:
        self.img_key = img_key
        self.ref_meta = ref_meta
        self.keep_key = keep_key
        self.label = label
        self.discreet = discreet
        self.device = device
        self.as_discrete = AsDiscrete(argmax=True)

    def __call__(self, data):
        """
        Apply the transform to `img` using `meta`.
        """

        d = dict(data)

        img = d[self.img_key]
        meta = d[f"{self.ref_meta}_meta_dict"]

        if (np.array(img[0, :].shape) != np.array(meta["final_bbox_shape"])).all():
            raise ValueError(
                "image and metadata don't match so can't restore to original size"
            )

        new_size = tuple(meta["new_dim"])
        box_start = meta["final_bbox"]
        padding = [
            box_start[0],
            [
                new_size[0] - box_start[1][0],
                new_size[1] - box_start[1][1],
                new_size[2] - box_start[1][2],
            ],
        ]

        old_size = img.shape[1:]
        zero_padding = np.array(meta["zero_padding"])
        zero_padding = [
            [zero_padding[0][0], zero_padding[0][1], zero_padding[0][2]],
            [
                old_size[0] - zero_padding[1][0],
                old_size[1] - zero_padding[1][1],
                old_size[2] - zero_padding[1][2],
            ],
        ]

        if img.shape[0] > 1:
            method = [np.max(img[0])]
            for channel in img[1:]:
                method.append(np.min(channel))
        else:
            method = [np.min(img)]

        new_img = []
        for i, channel in enumerate(img):
            box = channel[
                zero_padding[0][0] : zero_padding[1][0],
                zero_padding[0][1] : zero_padding[1][1],
                zero_padding[0][2] : zero_padding[1][2],
            ]
            new_img.append(
                np.pad(
                    box,
                    (
                        (padding[0][0], padding[1][0]),
                        (padding[0][1], padding[1][1]),
                        (padding[0][2], padding[1][2]),
                    ),
                    constant_values=method[i],
                )
            )

        img = np.stack(new_img, axis=0)

        if img[0].shape != new_size:
            raise ValueError("New img and new size do know have the same size??")

        if self.keep_key:
            cache = img.copy()

        if self.discreet:
            img = self.as_discrete(img)

        new_img = []
        for i, channel in enumerate(img):
            if self.label or self.discreet:
                new_img.append(
                    torch.tensor(
                        resample_label(
                            channel[None, :],
                            meta["org_dim"],
                            anisotrophy_flag=meta["anisotrophy_flag"],
                        )[0],
                        dtype=torch.float32,
                        device=self.device,
                    )
                )
            else:
                new_img.append(
                    torch.tensor(
                        resample_image(
                            channel[None, :],
                            meta["org_dim"],
                            anisotrophy_flag=meta["anisotrophy_flag"],
                        )[0],
                        dtype=torch.float32,
                        device=self.device,
                    )
                )

        new_img = torch.stack(new_img, dim=0)

        d[self.img_key] = MetaTensor(new_img, meta.get("original_affine"))

        meta_dict = d.get(f"{self.img_key}_meta_dict")
        if meta_dict is None:
            meta_dict = dict()
            d[f"{self.img_key}_meta_dict"] = meta_dict
        meta_dict["affine"] = meta.get("original_affine")

        if self.keep_key:
            cache = np.stack(cache, axis=0)
            d[self.keep_key] = MetaTensor(
                torch.tensor(cache), meta.get("original_affine")
            )

        return d


class InformationFusionGraphCutd(MapTransform):
    """
    This transform class takes list of annotations to array.
    That code is in:
    """

    def __init__(self, image, cue_map, interations, prediction, new_key="pred") -> None:
        super().__init__(image)
        self.image = image
        self.cue_map = cue_map
        self.interations = interations
        self.prediction = prediction
        self.new_key = new_key

    def __call__(self, data):
        d = dict(data)

        cue_maps = d[self.cue_map]
        interactions = d[self.interations]
        predictions = d[self.prediction]
        if cue_maps.shape[0] == 1:
            logger.info(
                f"INF - adding empty background map, because no background interactions where provided"
            )
            background = np.zeros(cue_maps.shape[1:])
            cue_maps = np.stack([cue_maps[0], background])
            interactions = np.stack([interactions[0], background])

        seed = np.zeros(cue_maps.shape[1:], np.uint8)
        seed[interactions[1] > 0] = 2
        seed[interactions[0] > 0] = 3
        seed = np.asarray([seed == 2, seed == 3], np.uint8)
        # seed = np.transpose(seed, [3, 1, 2, 0])

        foreground = np.maximum(cue_maps[0], predictions[1])
        background = np.maximum(cue_maps[1], predictions[0])

        # prob = np.asarray(np.stack([foreground, background]), dtype=np.float32)
        prob = np.asarray(np.stack([background, foreground]), dtype=np.float32)
        softmax = np.exp(prob) / np.sum(np.exp(prob), axis=0)
        softmax = np.exp(softmax) / np.sum(np.exp(softmax), axis=0)

        # prob = np.transpose(softmax, [3, 1, 2, 0])

        # img = np.asarray(d[self.image][0], dtype=np.float32)
        img = np.asarray(d[self.image], dtype=np.float32)
        # img = np.transpose(img, [2, 0, 1])

        lamda = 5.0
        sigma = 0.1
        # connectivity = np.asarray(d[f"{self.image}_meta_dict"]["new_spacing"], dtype=np.float32)
        connectivity = 6

        refined_pred = numpymaxflow.maxflow_interactive(
            img, prob, seed, lamda, sigma, connectivity
        )[None, :]
        # refined_pred = maxflow.interactive_maxflow3d(img, prob, seed, (5.0, 0.1))
        # refined_pred = np.transpose(refined_pred, [1, 2, 0])[None,:]

        d[self.new_key] = MetaTensor(
            torch.tensor(refined_pred),
            affine=d[self.image].affine,
            meta=d[f"{self.image}_meta_dict"],
        )
        d[f"{self.new_key}_meta_dict"] = d[f"{self.image}_meta_dict"]

        return d


class LoadWeightsd(MapTransform):
    """
    This transform class takes list of annotations to array.
    That code is in:
    """

    def __init__(
        self,
        key,
        ref_image,
        tmp_folder,
        extention="pt",
        device=None,
    ) -> None:
        super().__init__(key)
        self.key = key
        self.ref_image = ref_image
        self.tmp_folder = tmp_folder
        self.extention = extention
        self.device = device

    def __call__(self, data):
        d = dict(data)

        weight_path = f"/{self.tmp_folder}/{self.key}.{self.extention}"
        if not os.path.isfile(weight_path):
            raise ValueError("No weight file found, please run interactivenet first")

        weight = torch.load(weight_path, map_location=self.device)
        img = d[self.ref_image]

        if weight.shape[-3:] != img.shape[-3:]:
            raise ValueError(
                "Weights don't match the size of the image shape, might be still on the previous img"
            )

        if type(img).__module__ == torch.__name__:
            d[self.key] = MetaTensor(
                weight, affine=img.affine, meta=d[f"{self.ref_image}_meta_dict"]
            )
        elif type(img).__module__ == np.__name__:
            d[self.key] = weight.cpu().numpy()
        else:
            raise KeyError(
                f"Please provide {self.ref_image} as an array or tensor not as {type(img)}"
            )

        d[f"{self.key}_meta_dict"] = d[f"{self.ref_image}_meta_dict"]

        return d


class SaveIntermediated(MapTransform):
    """
    This transform class takes list of annotations to array.
    That code is in:
    """

    def __init__(self, keys, name, tmp_folder, softmax=False, extention="pt") -> None:
        super().__init__(keys)
        self.keys = keys
        self.name = name
        self.tmp_folder = tmp_folder
        self.softmax = softmax
        self.extention = extention
        self.activation = Activations(softmax=True)

    def __call__(self, data):
        d = dict(data)
        data_type = None
        keys = list(self.key_iterator(d))
        for key in keys:
            if data_type is None:
                data_type = type(d[key])
            elif not isinstance(d[key], data_type):
                raise TypeError("All items in data must have the same type.")

            if self.softmax:
                weights = self.activation(d[key])
                torch.save(weights, f"/{self.tmp_folder}/{self.name}.{self.extention}")
            else:
                torch.save(d[key], f"/{self.tmp_folder}/{self.name}.{self.extention}")

        return d


class TestTimeFlippingd(MapTransform):
    """
    This transform class takes list of annotations to array.
    That code is in:
    """

    def __init__(
        self,
        keys,
        all_dimensions=True,
        back=False,
    ) -> None:
        super().__init__(keys)
        self.keys = keys
        self.all_dimensions = all_dimensions
        self.back = back

    def __call__(self, data):
        d = dict(data)
        data_type = None
        keys = list(self.key_iterator(d))
        for key in keys:
            if data_type is None:
                data_type = type(d[key])
            elif not isinstance(d[key], data_type):
                raise TypeError("All items in data must have the same type.")

        if self.all_dimensions:
            spatial_axis = [0, 1, 2]
        else:
            spatial_axis = [0, 1]

        all_combinations = []
        for n in range(len(spatial_axis) + 1):
            all_combinations += list(combinations(spatial_axis, n))

        for key in keys:
            image = d[key]
            if not self.back:
                new_image = [image]
                for spatial_axis in all_combinations[1:]:
                    flipping = Flip(spatial_axis=spatial_axis)
                    new_image += flipping(image)[None, :]

                d[key] = torch.stack(new_image)
            else:
                if len(image.shape) == 5:
                    image = image[None, :]

                new_image = []
                for img in image:
                    new_image += [img[0]]
                    for idx, spatial_axis in enumerate(all_combinations[1:], 1):
                        flipping = Flip(spatial_axis=spatial_axis)
                        current_img = img[idx]
                        new_image += torch.stack(
                            [flipping(i[None, :]) for i in current_img], dim=1
                        )

                d[key] = torch.stack(new_image)

        return d


class AnnotationToChanneld(MapTransform):
    """
    This transform class takes list of annotations to array.
    That code is in:
    """

    def __init__(self, ref_image, guidance, method="interactivenet") -> None:
        super().__init__(guidance)
        self.ref_image = ref_image
        self.guidance = guidance
        self.method = method

    def __call__(self, data):
        d = dict(data)
        click_map = []

        for clicks in d[self.guidance]:  # pos and neg
            if clicks:
                if click_map and self.method == "interactivenet":
                    logger.info(
                        f"PRE - Transform (AnnotationToChanneld): Discarding negatives clicks because of method {self.method}"
                    )
                    continue

                annotation_map = torch.zeros(d[self.ref_image].shape)
                if len(clicks) < 6 and self.method == "interactivenet":
                    raise KeyError("please provide 6 interactions")

                for click in clicks:
                    annotation_map[click[0], click[1], click[2]] = 1

                click_map.append(annotation_map)

        d[self.guidance] = MetaTensor(
            torch.stack(click_map, dim=0),
            affine=d[self.ref_image].affine,
            meta=d[f"{self.ref_image}_meta_dict"],
        )
        d[f"{self.guidance}_meta_dict"] = d[f"{self.ref_image}_meta_dict"]

        return d


class NormalizeValuesd(MapTransform):
    """
    This transform class takes NNUNet's preprocessing method for reference.
    That code is in:
    https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/preprocessing/preprocessing.py
    """

    def __init__(
        self,
        keys,
        clipping=[],
        mean=0,
        std=0,
        nonzero=True,
        channel_wise=True,
    ) -> None:
        super().__init__(keys)
        self.keys = keys
        self.nonzero = nonzero
        self.channel_wise = channel_wise
        self.clipping = clipping
        self.mean = mean
        self.std = std
        self.normalize_intensity = NormalizeIntensity(nonzero=True, channel_wise=True)

    def __call__(self, data):
        d = dict(data)
        data_type = None
        keys = list(self.key_iterator(d))
        for key in keys:
            if data_type is None:
                data_type = type(d[key])
            elif not isinstance(d[key], data_type):
                raise TypeError("All items in data must have the same type.")

        for key in self.keys:
            image = d[key]
            if self.clipping:
                d[f"{key}_EGD"] = (image - self.mean) / self.std
                image = np.clip(image, self.clipping[0], self.clipping[1])
                image = (image - self.mean) / self.std
                image = MetaTensor(
                    image, affine=d[f"{key}_meta_dict"]["affine"], meta=d[f"{key}_meta_dict"]
                )
            else:
                image = self.normalize_intensity(image.copy())

            d[key] = image

        return d


class Resamplingd(MapTransform):
    """
    This transform class takes NNUNet's preprocessing method for reference.
    That code is in:
    https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/preprocessing/preprocessing.py
    """

    def __init__(
        self,
        keys,
        pixdim,
    ) -> None:
        super().__init__(keys)
        self.keys = keys
        self.target_spacing = pixdim

    def calculate_new_shape(self, spacing_ratio, shape):
        new_shape = (spacing_ratio * np.array(shape)).astype(int).tolist()
        return new_shape

    def check_anisotrophy(self, spacing):
        def check(spacing):
            return np.max(spacing) / np.min(spacing) >= 3

        return check(spacing) or check(self.target_spacing)

    def sanity_in_mask(self, annotation, label):
        sanity = []
        for i, annotation_d in enumerate(annotation):
            label_d = label[i]
            idx_x, idx_y, idx_z = np.where(annotation_d > 0.5)
            sanity_d = []
            for x, y, z in zip(idx_x, idx_y, idx_z):
                sanity_d.append(label_d[x, y, z] == 1)

            sanity.append(not any(sanity_d))

        return not any(sanity)

    def __call__(self, data):
        if len(self.keys) == 3:
            image, annotation, label = self.keys
            nimg, npnt, nseg = image, annotation, label
        elif len(self.keys) == 2:
            image, annotation = self.keys
            nimg, npnt = image, annotation
        else:
            image = self.keys
            name = image

        d = dict(data)
        image = d[image]
        image_spacings = d[f"{nimg}_meta_dict"]["pixdim"][1:4].tolist()
        print(
            f"Original Spacing: {image_spacings} \t Target Spacing: {self.target_spacing}"
        )

        if "annotation" in self.keys:
            annotation = d["annotation"]
            annotation[annotation < 0] = 0
        elif "point" in self.keys:
            annotation = d["point"]
            annotation[annotation < 0] = 0

        if "label" in self.keys:
            label = d["label"]
            label[label < 0] = 0
        elif "seg" in self.keys:
            label = d["seg"]
            label[label < 0] = 0
        elif "mask" in self.keys:
            label = d["mask"]
            label[label < 0] = 0

        # calculate shape
        original_shape = image.shape[1:]
        resample_flag = False
        anisotrophy_flag = False

        if self.target_spacing != image_spacings:
            # resample
            resample_flag = True
            spacing_ratio = np.array(image_spacings) / np.array(self.target_spacing)
            resample_shape = self.calculate_new_shape(spacing_ratio, original_shape)
            print(f"Original Shape: {original_shape} \t Target Shape: {resample_shape}")
            anisotrophy_flag = self.check_anisotrophy(image_spacings)
            image = resample_image(image, resample_shape, anisotrophy_flag)

            if "label" in self.keys or "seg" in self.keys or "mask" in self.keys:
                label = resample_label(label, resample_shape, anisotrophy_flag)

            if "annotation" in self.keys:
                annotation = resample_annotation(
                    d["annotation"],
                    d[f"{nimg}_meta_dict"]["affine"].numpy(),
                    self.target_spacing,
                    resample_shape,
                )

        new_meta = {
            "org_spacing": np.array(image_spacings),
            "org_dim": np.array(original_shape),
            "new_spacing": np.array(self.target_spacing),
            "new_dim": np.array(resample_shape),
            "resample_flag": resample_flag,
            "anisotrophy_flag": anisotrophy_flag,
        }

        d[f"{nimg}"] = image
        d[f"{nimg}_meta_dict"].update(new_meta)

        if "annotation" in self.keys:
            d["annotation"] = annotation

        if "label" in self.keys:
            d["label"] = label
        elif "seg" in self.keys:
            d["seg"] = label
        elif "mask" in self.keys:
            d["mask"] = label

        return d


class EGDMapd(MapTransform):
    """
    This transform class takes NNUNet's preprocessing method for reference.
    That code is in:
    https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/preprocessing/preprocessing.py
    """

    def __init__(
        self,
        keys,
        image,
        lamb=1,
        iter=4,
        logscale=True,
        ct=False,
        backup=False,
        powerof=False,
    ) -> None:
        super().__init__(keys)
        self.keys = keys
        self.image = image
        self.lamb = lamb
        self.iter = iter
        self.logscale = logscale
        self.backup = backup
        self.ct = ct
        self.powerof = powerof
        self.gaussiansmooth = GaussianSmooth(sigma=1)

    def __call__(self, data):
        d = dict(data)
        data_type = None
        keys = list(self.key_iterator(d))
        for key in keys:
            if data_type is None:
                data_type = type(d[key])
            elif not isinstance(d[key], data_type):
                raise TypeError("All items in data must have the same type.")

        for key in self.keys:
            if self.backup:
                d[f"{key}_backup"] = d[key].copy()

            if "new_spacing" in d[f'{self.image}_meta_dict'].keys():
                spacing = d[f'{self.image}_meta_dict']["new_spacing"]
            else:
                spacing = np.asarray(d[f'{self.image}_meta_dict']["pixdim"][1:4])

            if f"{self.image}_EGD" in d.keys():
                image = d[f"{self.image}_EGD"]
                del d[f'{self.image}_EGD']
            else:
                image = d[self.image]
                
            if len(d[key].shape) == 4:
                for idx in range(d[key].shape[0]):
                    img = image[idx]

                    GD = GeodisTK.geodesic3d_raster_scan(
                        img.astype(np.float32),
                        d[key][idx].astype(np.uint8),
                        spacing.astype(np.float32),
                        self.lamb,
                        self.iter,
                    )
                    if self.powerof:
                        GD = GD**self.powerof

                    if self.logscale == True:
                        GD = np.exp(-GD)

                    d[key][idx, :, :, :] = GD
            else:
                GD = GeodisTK.geodesic3d_raster_scan(
                    image.astype(np.float32),
                    d[key].astype(np.uint8),
                    spacing.astype(np.float32),
                    self.lamb,
                    self.iter,
                )
                if self.powerof:
                    GD = GD**self.powerof

                if self.logscale == True:
                    GD = np.exp(-GD)

                d[key] = GD

        d[key] = MetaTensor(
            d[key], affine=d[self.image].affine, meta=d[f"{self.image}_meta_dict"]
        )

        print(
            f"Geodesic Distance Map with lamd: {self.lamb}, iter: {self.iter} and logscale: {self.logscale}"
        )
        return d


class BoudingBoxd(MapTransform):
    """
    This transform class takes NNUNet's preprocessing method for reference.
    That code is in:
    https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/preprocessing/preprocessing.py
    """

    def __init__(
        self,
        keys,
        on,
        relaxation=0,
        divisiblepadd=None,
    ) -> None:
        super().__init__(keys)
        self.keys = keys
        self.on = on

        if isinstance(relaxation, float):
            relaxation = [relaxation] * 3

        if divisiblepadd:
            if isinstance(divisiblepadd, int):
                divisiblepadd = [divisiblepadd] * 3

        self.relaxation = relaxation
        self.divisiblepadd = divisiblepadd

    def calculate_bbox(self, data):
        inds_x, inds_y, inds_z = np.where(data > 0.5)

        bbox = np.array(
            [
                [np.min(inds_x), np.min(inds_y), np.min(inds_z)],
                [np.max(inds_x), np.max(inds_y), np.max(inds_z)],
            ]
        )

        return bbox

    def calculate_relaxtion(self, bbox_shape, anisotropic=False):
        relaxation = [0] * len(bbox_shape)
        for i, axis in enumerate(range(len(bbox_shape))):
            relaxation[axis] = math.ceil(bbox_shape[axis] * self.relaxation[axis])

            if anisotropic and i == 2: # This is only possible with Z on final axis 
                check = 3
            else:
                check = 8

            if relaxation[axis] < check:
                print(f"relaxation was to small: {relaxation[axis]}, so adjusting it to {check}")
                relaxation[axis] = check

        return relaxation

    def relax_bbox(self, data, bbox, relaxation):
        bbox = np.array(
            [
                [
                    bbox[0][0] - relaxation[0],
                    bbox[0][1] - relaxation[1],
                    bbox[0][2] - relaxation[2],
                ],
                [
                    bbox[1][0] + relaxation[0],
                    bbox[1][1] + relaxation[1],
                    bbox[1][2] + relaxation[2],
                ],
            ]
        )
        for axis in range(len(bbox[0])):
            if bbox[0, axis] == bbox[1, axis]:
                bbox[0, axis] = bbox[0, axis] - 1
                bbox[1, axis] = bbox[1, axis] + 1
                warnings.warn(
                    f"Bounding box has the same size in {axis} axis so extending axis by 1 both direction"
                )

        # Remove below zero and higher than shape because of relaxation
        bbox[bbox < 0] = 0
        largest_dimension = [
            int(x) if x <= data.shape[i] else data.shape[i]
            for i, x in enumerate(bbox[1])
        ]
        bbox = np.array([bbox[0].tolist(), largest_dimension])

        zeropadding = np.zeros(3)
        if self.divisiblepadd:
            for axis in range(len(self.divisiblepadd)):
                expand = True
                while expand == True:
                    bbox_shape = np.subtract(bbox[1][axis], bbox[0][axis])
                    residue = bbox_shape % self.divisiblepadd[axis]
                    if residue != 0:
                        residue = self.divisiblepadd[axis] - residue
                        if residue < 2:
                            neg = bbox[0][axis] - 1
                            if neg >= 0:
                                bbox[0][axis] = neg
                            else:
                                pos = bbox[1][axis] + 1
                                if pos <= data.shape[axis]:
                                    bbox[1][axis] = pos
                                else:
                                    zeropadding[axis] = zeropadding[axis] + residue
                                    warnings.warn(
                                        f"bbox doesn't fit in the image for axis {axis}, adding zero padding {residue}"
                                    )
                                    expand = False
                        else:
                            neg = bbox[0][axis] - 1
                            if neg >= 0:
                                bbox[0][axis] = neg

                            pos = bbox[1][axis] + 1
                            if pos <= data.shape[axis]:
                                bbox[1][axis] = pos

                            if neg <= 0 and pos > data.shape[axis]:
                                zeropadding[axis] = zeropadding[axis] + residue
                                warnings.warn(
                                    f"bbox doesn't fit in the image for axis {axis}, adding zero padding {residue}"
                                )
                                expand = False
                    else:
                        expand = False

        padding = np.zeros((2, 3), dtype=int)
        if any(zeropadding > 0):
            for idx, value in enumerate(zeropadding):
                x = int(value / 2)
                y = int(value - x)
                padding[0][idx] = x
                padding[1][idx] = y

        return bbox, padding

    def extract_bbox_region(self, data, bbox, padding):
        new_region = data[
            bbox[0][0] : bbox[1][0], bbox[0][1] : bbox[1][1], bbox[0][2] : bbox[1][2]
        ]

        new_region = np.pad(
            new_region,
            (
                (padding[0][0], padding[1][0]),
                (padding[0][1], padding[1][1]),
                (padding[0][2], padding[1][2]),
            ),
            "constant",
        )

        return new_region

    def __call__(self, data):
        d = dict(data)
        output = []
        data_type = None
        keys = list(self.key_iterator(d))
        for key in keys:
            if data_type is None:
                data_type = type(d[key])
            elif not isinstance(d[key], data_type):
                raise TypeError("All items in data must have the same type.")
            output.append(d[key])

        bbox = self.calculate_bbox(d[self.on][0])
        bbox_shape = np.subtract(bbox[1], bbox[0])
        relaxation = self.calculate_relaxtion(bbox_shape, d[f"{key}_meta_dict"]["anisotrophy_flag"])

        print(
            f"Original bouding box at location: {bbox[0]} and {bbox[1]} \t shape of bbox: {bbox_shape}"
        )
        final_bbox, zeropadding = self.relax_bbox(d[self.on][0], bbox, relaxation)
        final_bbox_shape = np.subtract(final_bbox[1], final_bbox[0])
        print(
            f"Bouding box at location: {final_bbox[0]} and {final_bbox[1]} \t bbox is relaxt with: {relaxation} \t and zero_padding: {zeropadding} \t and made divisible with: {self.divisiblepadd} \t shape after cropping: {final_bbox_shape}"
        )
        for key in self.keys:
            if len(d[key].shape) == 4:
                new_dkey = []
                for idx in range(d[key].shape[0]):
                    new_dkey.append(
                        self.extract_bbox_region(d[key][idx], final_bbox, zeropadding)
                    )
                d[key] = np.stack(new_dkey, axis=0)
                final_size = d[key].shape[1:]
            else:
                d[key] = self.extract_bbox_region(d[key], final_bbox, zeropadding)
                final_size = d[key].shape

            d[f"{key}_meta_dict"]["bbox"] = bbox
            d[f"{key}_meta_dict"]["bbox_shape"] = bbox_shape
            d[f"{key}_meta_dict"]["bbox_relaxation"] = self.relaxation
            d[f"{key}_meta_dict"]["final_bbox"] = final_bbox
            d[f"{key}_meta_dict"]["final_bbox_shape"] = final_bbox_shape
            d[f"{key}_meta_dict"]["zero_padding"] = zeropadding
            d[f"{key}_meta_dict"]["final_size"] = final_size

        return d


class LoadPreprocessed(MapTransform):
    """
    This transform class takes NNUNet's preprocessing method for reference.
    That code is in:
    https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/preprocessing/preprocessing.py
    """

    def __init__(
        self,
        keys,
        new_keys,
    ) -> None:
        super().__init__(keys)
        self.keys = keys
        if len(self.keys) != 2:
            raise ValueError(
                f"LoadPreprocessed data assumes the data has 2 keys with npz and metadata, this is not the case as there are {len(self.keys)} provided"
            )

        self.new_keys = new_keys
        self.meta_keys = [x + "_meta_dict" for x in new_keys] + [
            x + "_transforms" for x in new_keys
        ]

    def read_pickle(self, filename):
        with open(filename, "rb") as handle:
            b = pickle.load(handle)

        return b

    def __call__(self, data):
        d = dict(data)
        new_d = {}
        for key in self.keys:
            current_data = d[key]
            if current_data.suffix == ".npz":
                image_data = np.load(d[key])
                old_keys = list(image_data.keys())
                if not len(old_keys) == len(self.new_keys):
                    raise KeyError(
                        "Old keys and new keys have not te same length in preprocessed data loader"
                    )

                if set(old_keys) == set(self.new_keys):
                    for new_key in self.new_keys:
                        new_d[new_key] = image_data[new_key]

                else:
                    warnings.warn(
                        "old keys do not match new keys, however were the right length so just applying it in order"
                    )
                    for old_key, new_key in zip(old_keys, self.new_keys):
                        new_d[new_key] = image_data[old_key]

            elif current_data.suffix == ".pkl":
                metadata = self.read_pickle(d[key])
                old_keys = list(metadata.keys())
                if not len(old_keys) == len(self.meta_keys):
                    raise KeyError(
                        "Old keys and new keys have not te same length in preprocessed data loader"
                    )

                if set(old_keys) == set(self.meta_keys):
                    for new_key in self.meta_keys:
                        new_d[new_key] = metadata[new_key]

                else:
                    warnings.warn(
                        "old keys do not match new keys, however were the right length so just applying it in order"
                    )
                    for old_key, new_key in zip(old_keys, self.meta_keys):
                        new_d[new_key] = metadata[old_key]
            else:
                raise ValueError("Neither npz or pkl in preprocessed loader")

        return new_d

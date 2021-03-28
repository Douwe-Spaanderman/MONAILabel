import logging
import pathlib
import tempfile

import SimpleITK
import numpy as np

from monai.data import write_nifti


# TODO:: Move to MONAI ??
class Writer:
    def __init__(
            self,
            image='pred',
            json=None,
            ref_image=None,
            key_extension='result_extension',
            key_dtype='result_dtype',
            key_compress='result_compress',
            meta_key_postfix="meta_dict"
    ):
        self.image = image
        self.json = json
        self.ref_image = ref_image if ref_image else image

        # User can specify through params
        self.key_extension = key_extension
        self.key_dtype = key_dtype
        self.key_compress = key_compress
        self.meta_key_postfix = meta_key_postfix

    def __call__(self, data):
        logger = logging.getLogger(self.__class__.__name__)

        file_ext = ''.join(pathlib.Path(data['image_path']).suffixes)
        dtype = data.get(self.key_dtype, None)
        compress = data.get(self.key_compress, False)
        file_ext = data.get(self.key_extension, file_ext)
        logger.debug('Result ext: {}'.format(file_ext))

        image_np = data[self.image]
        meta_dict = data.get(f"{self.ref_image}_{self.meta_key_postfix}")
        affine = meta_dict.get("affine") if meta_dict else None
        logger.debug('Image: {}; Data Image: {}'.format(image_np.shape, data[self.image].shape))

        output_file = tempfile.NamedTemporaryFile(suffix=file_ext).name
        logger.debug('Saving Image to: {}'.format(output_file))

        if file_ext.lower() in ['.nii', '.nii.gz']:
            logger.debug('Using MONAI write_nifti...')
            write_nifti(image_np, output_file, affine=affine, output_dtype=dtype)
        else:
            if len(image_np.shape) > 2:
                image_np = image_np.transpose()
            if dtype:
                image_np = image_np.astype(dtype)

            result_image = SimpleITK.GetImageFromArray(image_np)

            # https://github.com/RSIP-Vision/medio/blob/master/medio/metadata/affine.py#L108-L121
            if affine is not None:
                convert_aff_mat = np.diag([-1, -1, 1, 1])
                if affine.shape[0] == 3:
                    convert_aff_mat = np.diag([-1, -1, 1])
                affine = convert_aff_mat @ affine

                dim = affine.shape[0] - 1
                _origin_key = (slice(-1), -1)
                _m_key = (slice(-1), slice(-1))

                origin = affine[_origin_key]
                spacing = np.linalg.norm(affine[_m_key] @ np.eye(dim), axis=0)
                direction = affine[_m_key] @ np.diag(1 / spacing)

                spacing = spacing.tolist()
                direction = direction.flatten().tolist()

                logger.debug('Affine: {}'.format(affine))
                logger.debug('Origin: {}'.format(origin))
                logger.debug('Spacing: {}'.format(spacing))
                logger.debug('Direction: {}'.format(direction))

                result_image.SetDirection(direction)
                result_image.SetSpacing(spacing)
                result_image.SetOrigin(origin)

            SimpleITK.WriteImage(result_image, output_file, compress)

        return output_file, data.get(self.json, {})


class ClassificationWriter:
    def __init__(
            self,
            image='pred',
            label_names=None
    ):
        self.image = image
        self.label_names = label_names

    def __call__(self, data):
        result = []
        for label in data[self.image]:
            result.append(self.label_names[int(label)])
        return None, {'prediction': result}

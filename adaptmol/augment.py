import albumentations as A
from albumentations.augmentations.geometric.functional import safe_rotate_enlarged_img_size, _maybe_process_in_chunks, \
                                                              keypoint_rotate
import cv2
import math
import random
import numpy as np




class CropWhite(A.DualTransform):
    """
    Albumentations transform that crops white (or specified color) borders from an image
    and applies uniform padding.

    Args:
        value (tuple): Background color to crop, default (255, 255, 255).
        pad (int): Padding to add after cropping. Default 0.
        p (float): Probability of applying the transform. Default 1.0.
    """
    def __init__(self, value=(255,255,255), pad=0, p=1.0):
        super(CropWhite, self).__init__(p=p)
        self.value = value
        self.pad = pad
        assert pad >= 0

    def update_params(self, params, **kwargs):
        super().update_params(params, **kwargs)
        assert "image" in kwargs
        img = kwargs["image"]
        height, width, _ = img.shape
        # thr = 80
        # x = (img < thr).any(axis=2).astype(np.uint8) 

        x = (img != self.value).sum(axis=2)
        if x.sum() == 0:
            return params
        row_sum = x.sum(axis=1)
        top = 0
        while row_sum[top] == 0 and top+1 < height:
            top += 1
        bottom = height
        while row_sum[bottom-1] == 0 and bottom-1 > top:
            bottom -= 1
        col_sum = x.sum(axis=0)
        left = 0
        while col_sum[left] == 0 and left+1 < width:
            left += 1
        right = width
        while col_sum[right-1] == 0 and right-1 > left:
            right -= 1
        # crop_top = max(0, top - self.pad)
        # crop_bottom = max(0, height - bottom - self.pad)
        # crop_left = max(0, left - self.pad)
        # crop_right = max(0, width - right - self.pad)
        # params.update({"crop_top": crop_top, "crop_bottom": crop_bottom,
        #                "crop_left": crop_left, "crop_right": crop_right})
        params.update({"crop_top": top, "crop_bottom": height - bottom,
                       "crop_left": left, "crop_right": width - right})
        return params

    def apply(self, img, crop_top=0, crop_bottom=0, crop_left=0, crop_right=0, **params):
        height, width, _ = img.shape
        img = img[crop_top:height - crop_bottom, crop_left:width - crop_right]
        img = A.augmentations.pad_with_params(
            img, self.pad, self.pad, self.pad, self.pad, border_mode=cv2.BORDER_CONSTANT, value=self.value)
        return img

    def apply_to_keypoint(self, keypoint, crop_top=0, crop_bottom=0, crop_left=0, crop_right=0, **params):
        x, y, angle, scale = keypoint[:4]
        return x - crop_left + self.pad, y - crop_top + self.pad, angle, scale

    def get_transform_init_args_names(self):
        return ('value', 'pad')



class SaltAndPepperNoise(A.DualTransform):
    """
    Randomly scatter fixed-color dots across an image to simulate salt-and-pepper noise.

    Args:
        num_dots (int): Maximum number of noise dots to add.
        value (tuple): Color of the noise dots. Default (0, 0, 0).
        p (float): Probability of applying the transform. Default 0.5.
    """

    def __init__(self, num_dots, value=(0, 0, 0), p=0.5):
        super().__init__(p)
        self.num_dots = num_dots
        self.value = value

    def apply(self, img, **params):
        height, width, _ = img.shape
        num_dots = random.randrange(self.num_dots + 1)
        for i in range(num_dots):
            x = random.randrange(height)
            y = random.randrange(width)
            img[x, y] = self.value
        return img

    def apply_to_keypoint(self, keypoint, **params):
        return keypoint

    def get_transform_init_args_names(self):
        return ('value', 'num_dots')
    



import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import PIL
from glob import glob
from os import path


class RaggedImageDataset(Dataset):
    def __init__(
        self,
        image_path,
        batch_size=16,
        largest_side_res=512,
        smallest_side_res=64,
        max_aspect_ratio=10.0,
        min_aspect_ratio=0.2,
        ext=".jpg",
        bucketing_metric=torch.median,
    ):
        """
        Returns tensor batches of equal shape, from underlying images with different aspect ratios.

        see: https://github.com/pytorch/vision/blob/main/references/detection/group_by_aspect_ratio.py

        Does a first pass over all images in path/*ext, getting their dimensions and checking that they are not corrupt.
        """

        image_files = glob(path.join(image_path, f"**/*{ext}"), recursive=True)
        widths, heights = get_images_dimensions(image_files)

        error_indeces = torch.where(widths == 0)[0]
        if len(error_indeces) > 0:
            print(
                "Warning!, the aspect ratios of the following images were unable to be read:\n"
            )
            print("".join([f"\t'{image_files[i]}'\n" for i in error_indeces]))

        aspect_ratios = widths / heights
        aspect_ratios, aspect_ratios_indeces = aspect_ratios.sort()
        _mask = torch.logical_and(
            torch.logical_and(aspect_ratios != 0, min_aspect_ratio < aspect_ratios),
            aspect_ratios < max_aspect_ratio,
        )
        aspect_ratios_indeces = aspect_ratios_indeces[_mask]
        widths = widths[aspect_ratios_indeces]
        heights = heights[aspect_ratios_indeces]
        widths, heights = clamp_by_max_res(widths, heights, largest_side_res)
        widths, heights = clamp_by_min_res(widths, heights, smallest_side_res)
        self.aspect_ratios = widths / heights
        self.widths = widths
        self.heights = heights

        self.image_files = [image_files[i] for i in aspect_ratios_indeces]

        indeces = torch.arange(self.widths.shape[0])
        split_indeces = torch.split(indeces, batch_size)

        width_buckets = torch.IntTensor(
            [bucketing_metric(self.widths[si]) for si in split_indeces]
        )
        height_buckets = torch.IntTensor(
            [bucketing_metric(self.heights[si]) for si in split_indeces]
        )
        bucketed_indeces = indeces // batch_size

        self.bucketed_widths = width_buckets[bucketed_indeces]
        self.bucketed_heights = height_buckets[bucketed_indeces]
        self.bucketed_aspect_ratios = self.bucketed_widths / self.bucketed_heights

    def __getitem__(self, idx):
        """
        returns an image
        """

        # return self.

    def __len__(self):
        return len(self.image_files)


def clamp_by_max_res(widths, heights, largest_side_res):
    widths = widths.clone()
    heights = heights.clone()
    widths, heights = _clamp_by_max(widths, heights, largest_side_res)
    heights, widths = _clamp_by_max(heights, widths, largest_side_res)
    return widths, heights


def clamp_by_min_res(widths, heights, smallest_side_res):
    widths = widths.clone()
    heights = heights.clone()
    widths, heights = _clamp_by_min(widths, heights, smallest_side_res)
    heights, widths = _clamp_by_min(heights, widths, smallest_side_res)
    return widths, heights


def _clamp_by_min(a, b, _min):
    smaller_a = a <= b
    adjusted_widths = torch.clamp_min(a[smaller_a], _min)
    grow_ratio = adjusted_widths / a[smaller_a]
    a[smaller_a] = adjusted_widths
    b_mod = b[smaller_a] * grow_ratio
    b[smaller_a] = torch.round(b_mod).to(b.dtype)
    return a, b


def _clamp_by_max(a, b, _max):
    bigger_a = a >= b
    adjusted_widths = torch.clamp_max(a[bigger_a], _max)
    shrink_ratio = adjusted_widths / a[bigger_a]
    a[bigger_a] = adjusted_widths
    b_mod = b[bigger_a] * shrink_ratio
    b[bigger_a] = torch.round(b_mod).to(b.dtype)
    return a, b


def get_images_dimensions(image_files):
    """
    If the image fails to load, sets the width to 0 and height to -1
    """
    widths = torch.zeros(len(image_files), dtype=torch.int)
    heights = torch.zeros(len(image_files), dtype=torch.int)
    for i, image_file in enumerate(image_files):
        try:
            with Image.open(image_file) as im:
                width, height = im.size
        except PIL.UnidentifiedImageError:
            widths[i] = 0
            heights[i] = -1
            continue
        widths[i] = width
        heights[i] = height
    return widths, heights

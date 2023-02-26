import unittest
import matplotlib.pyplot as plt
from os import path
import torch
from torch.utils.data import DataLoader

torch.random.seed()

from ragged_image_dataset.dataset import RaggedImageDataset, RandomBatchwiseSampler
from ragged_image_dataset.dataset import clamp_by_max_res, clamp_by_min_res

TESTIMAGE_DIR = path.join(path.dirname(__file__), "testimages/")

class TestRaggedDataset(unittest.TestCase):
    def testClampRes(self):
        _max = 7
        widths = torch.IntTensor([1, 3, 8, 7, 9, 17])
        heights = torch.IntTensor([3, 3, 7, 8, 9, 13])
        original_ars = widths / heights
        mWidths, mHeights = clamp_by_max_res(widths.clone(), heights.clone(), _max)
        new_ars = mWidths / mHeights
        self.assertTrue(torch.allclose(original_ars, new_ars, atol=0.2))
        self.assertEqual(_max, mHeights.max().item())
        self.assertEqual(_max, mWidths.max().item())

    def testClampRandMax(self):
        n = 100
        _max = 70
        widths = torch.randint(1, n, (n,))
        heights = torch.randint(1, n, (n,))
        original_ars = widths / heights
        mWidths, mHeights = clamp_by_max_res(widths.clone(), heights.clone(), _max)
        new_ars = mWidths / mHeights
        self.assertLessEqual(mHeights.max().item(), _max)
        self.assertLessEqual(mWidths.max().item(), _max)

    def testClampRandMin(self):
        n = 1000
        _min = n // 4
        widths = torch.randint(2, n, (n,))
        heights = torch.randint(2, n, (n,))
        original_ars = widths / heights
        mWidths, mHeights = clamp_by_min_res(widths, heights, n // 2)
        new_ars = mWidths / mHeights
        self.assertLessEqual(_min, mHeights.min().item())
        self.assertLessEqual(_min, mWidths.min().item())
        self.assertTrue(torch.allclose(original_ars, new_ars, atol=0.2))

    def testLoadImages(self):
        bs = 4
        max_res = 128
        min_res = 64
        ds = RaggedImageDataset(TESTIMAGE_DIR, batch_size=8, largest_side_res=max_res, smallest_side_res=min_res)
        old_size = None
        for i, (im, label) in enumerate(ds):
            new_size = im.shape
            print(label)
            print(new_size)
            if i % bs != 0:
                self.assertEqual(old_size, new_size)
            old_size = new_size

    def testLoadImagesWithDataloader(self):
        bs = 4
        max_res = 128
        min_res = 64
        ds = RaggedImageDataset(TESTIMAGE_DIR, batch_size=8, largest_side_res=max_res, smallest_side_res=min_res)
        dl = DataLoader(ds, sampler=RandomBatchwiseSampler(len(ds), bs), num_workers=8, batch_size=bs)
        for b_i, (im_b, label_b) in enumerate(dl):
            print("b: ", im_b.shape)
            print("b: ", label_b)


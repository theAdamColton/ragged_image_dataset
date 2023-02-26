import matplotlib.pyplot as plt
import torch

import sys

sys.path.append("..")
from ragged_image_dataset.dataset import RaggedImageDataset

bs = 512
ds = RaggedImageDataset(
    "/home/figes/Desktop/ukr_images/", batch_size=bs, bucketing_metric=torch.median
)

# plots cdf of aspect_ratios
nnth = torch.quantile(ds.aspect_ratios, 0.99)
mask = ds.aspect_ratios < nnth
ar = ds.aspect_ratios[mask]
bar = ds.bucketed_aspect_ratios[mask]
plt.plot(ar, [i / len(ds.aspect_ratios) for i in range(len(ar))])
plt.plot(
    bar,
    [i / len(ds.aspect_ratios) for i in range(len(bar))],
    label="bucketed aspect ratios",
)
plt.xlabel("aspect ratio")
plt.ylabel("percentile")
plt.show()

# plots cdf of heights and bucketed heights
_, sorted_idxs = ds.heights.sort()
plt.plot(ds.heights[sorted_idxs], [i / len(ds) for i in range(len(ds))], label="h")
plt.plot(
    ds.bucketed_heights[sorted_idxs], [i / len(ds) for i in range(len(ds))], label="bh"
)
plt.legend()
plt.xlabel("resolution")
plt.ylabel("percentile")
plt.show()

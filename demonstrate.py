import matplotlib.pyplot as plt
import torch

import sys

sys.path.append("..")
from ragged_image_dataset.dataset import RaggedImageDataset

bs = 3
ds = RaggedImageDataset(
    "./tests/testimages/", batch_size=bs, bucketing_metric=torch.median, largest_side_res=256
)

# plots cdf of aspect_ratios
#nnth = torch.quantile(ds.aspect_ratios, 0.99)
#mask = ds.aspect_ratios < nnth
plt.plot(ds.aspect_ratios, [i / len(ds.aspect_ratios) for i in range(len(ds.aspect_ratios))], label="Original aspect ratios")
plt.plot(
    ds.bucketed_aspect_ratios,
    [i / len(ds.aspect_ratios) for i in range(len(ds.bucketed_aspect_ratios))],
    label="bucketed aspect ratios",
)
plt.xlabel("aspect ratio")
plt.ylabel("percentile")
plt.title("CDF of Original Vs. Bucketed Aspect Ratios")
plt.legend()
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

for im, label in ds:
    print(label)
    print(im.shape)
    plt.imshow(im.movedim(0, -1))
    plt.show()

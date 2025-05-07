#将deer.jpg换成自己的自己照片的路径即可得出分割后的图片，输出分析的时间和sse
import time
import numpy as np
from PIL import Image
from kmeans_segmentation import segment_image

t0 = time.perf_counter()

img = np.array(Image.open("deer.jpg").convert("RGB"))
seg, info = segment_image(img, K=4, colorspace="lab", palette="vivid")

elapsed = time.perf_counter() - t0

Image.fromarray(seg).save("deer_seg.png")
side = np.concatenate([img, seg], axis=1)
Image.fromarray(side).save("deer_compare.png")

print(f"SSE = {info['sse']:.1f}")
print(f"Runtime = {elapsed:.3f} seconds")

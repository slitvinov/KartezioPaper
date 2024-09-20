from matplotlib.patches import Rectangle
from roifile import ImagejRoi
import cv2
import matplotlib.pyplot as plt
import sys

img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
rois = ImagejRoi.fromfile(sys.argv[2])
fig, ax = plt.subplots()
ax.imshow(img, cmap='gray')
for roi in rois:
    roi.plot(ax, color='red')
plt.show()

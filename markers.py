import cv2
import numpy as np
from matplotlib import pyplot as plt

TARGET_COLOR_BGR = [55, 60, 90]
THRESHOLD = 0.8

STEREO_PAIR = ('./data/training/frames/left/frame_40.png',
    './data/training/frames/centre/frame_1.png')

# detector
mser = cv2.MSER_create(
    _delta=5,
    _min_area=60,
    _max_area=1000,
    _max_variation=0.25,
    _min_diversity=.2,
    _max_evolution=200,
    _area_threshold=0.003,
    _edge_blur_size=5
)

# Load image
left_img, right_img = map(cv2.imread, STEREO_PAIR)
if left_img is None or right_img is None:
    print('Could not load, check path')
    exit(0)

# Display input
plt.figure('Input stereo pair')
plt.imshow(cv2.cvtColor(np.hstack((left_img, right_img)), cv2.COLOR_BGR2RGB))

# HSV
plt.figure('hsv')
hsv = cv2.cvtColor(left_img, cv2.COLOR_BGR2HSV)
plt.imshow(np.hstack(cv2.split(hsv)), cmap='gray')

# LUV
plt.figure('luv')
luv = cv2.cvtColor(left_img, cv2.COLOR_BGR2Luv)
plt.imshow(np.hstack(cv2.split(luv)), cmap='gray')

# Pre-process input
# skip for now

target = luv[...,0]
_, boxes = mser.detectRegions(target)
vis = np.copy(target)
for bbox in boxes:
    vis = cv2.rectangle(vis, bbox, (255,0,0), 1)

"""
# Distance based filtering
mask = np.zeros(left_img.shape[:2], dtype=bool)
distance  = abs(left_img[...,0] - 55) / 55
distance += abs(left_img[...,1] - 60) / 60 < THRESHOLD
distance += abs(left_img[...,2] - 90) / 90 < THRESHOLD
mask = distance < THRESHOLD

vis = np.copy(left_img)
vis[mask] = (255, 255, 255)
"""

plt.figure('Left image output')
plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
plt.show()






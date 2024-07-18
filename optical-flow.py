import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import time

# ShiTomasi corner detection
config_st = {"maxCorners": 100, "qualityLevel": 0.3, "minDistance": 7, "blockSize": 7}

# Lucas-Kanade optical flow
config_lk = {
    "winSize": (15, 15),
    "maxLevel": 2,
    "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
}

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Take first frame and find keypoints
index = 2139
source = cv2.imread(f"/home/michael/Stuff/ffmpeg-tutorial/frames/realsense/{index}.jpg")


src_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)

start = time.time()
p_src = cv2.goodFeaturesToTrack(src_gray, mask=None, **config_st)
end = time.time()
print(f"New features in {(end - start) * 1000}ms")

# Create a mask image for drawing purposes
mask = np.zeros_like(source)

while True:
    index += 1
    if index == 2190:
        break

    target = cv2.imread(f"/home/michael/Stuff/ffmpeg-tutorial/frames/realsense/{index}.jpg")

    dst_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    start = time.time()
    p_dst, status, err = cv2.calcOpticalFlowPyrLK(
        src_gray, dst_gray, p_src, None, **config_lk
    )
    end = time.time()
    print(f"Finished in {(end - start) * 1000}ms")

    if p_dst is not None:
        p_dst = p_dst[status == 1]
        p_src = p_src[status == 1]

    for i, (dst, src) in enumerate(zip(p_dst, p_src)):
        x_dst, y_dst = dst
        x_src, y_src = src

        mask = cv2.line(
            mask,
            (int(x_src), int(y_src)),
            (int(x_dst), int(y_dst)),
            color[i].tolist(),
            2,
        )
        target = cv2.circle(target, (int(x_src), int(y_src)), 5, color[i].tolist(), -1)

    result = cv2.add(target, mask)
    cv2.imshow("zal", result)
    if cv2.waitKey(0) & 0xFF == ord("q"):
        break

    # Update the previous frame and previous points
    src_gray = np.copy(dst_gray)
    p_src = p_dst.reshape(-1, 1, 2)

cv2.destroyAllWindows()
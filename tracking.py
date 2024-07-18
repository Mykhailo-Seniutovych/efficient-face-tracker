import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import time


def create_tracker():
    tracker_types = [
        "BOOSTING",
        "MIL",
        "KCF",
        "TLD",
        "MEDIANFLOW",
        "GOTURN",
        "MOSSE",
        "CSRT",
    ]
    tracker_type = "CSRT"

    if tracker_type == "BOOSTING":
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == "MIL":
        tracker = cv2.TrackerMIL_create()
    if tracker_type == "KCF":
        tracker = cv2.TrackerKCF_create()
    if tracker_type == "TLD":
        tracker = cv2.TrackerTLD_create()
    if tracker_type == "MEDIANFLOW":
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == "GOTURN":
        tracker = cv2.TrackerGOTURN_create()
    if tracker_type == "MOSSE":
        tracker = cv2.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()

    return tracker


video = cv2.VideoCapture("/home/michael/Stuff/ffmpeg-tutorial/videos/2/right_camera.mp4")

start_time_seconds = 0 * 60 + 18
if not video.isOpened():
    print("Error: Could not open video.")
    exit()
fps = video.get(cv2.CAP_PROP_FPS)
start_frame = int(start_time_seconds * fps)

video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

tracker = create_tracker()

ok, frame = video.read()
if not ok:
    print("Cannot read video file")
    video.exit()

bbox = cv2.selectROI(frame, False)
tracker.init(frame, bbox)

while True:
    ret, frame = video.read()
    if not ret:
        break

    start_time = time.time()
    object_found, bbox = tracker.update(frame)
    end_time = time.time()

    print("tracked ", bbox, " in ", (end_time - start_time) * 1000, "ms")
    if object_found:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

    cv2.imshow("Frame", frame)
    if cv2.waitKey() & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()

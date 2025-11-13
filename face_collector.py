import time
import os
import sys

import cv2
import numpy as np

from frames_reader import ImageFilesFrameReader
from frames_writer import DummyFramesWriter
from face_tracker import FaceTracker, Config, do_nothing


if len(sys.argv) < 3:
    print("Wrong params")
    sys.exit(0)

input_folder = sys.argv[1]
output_folder = sys.argv[2]
os.makedirs(output_folder, exist_ok=True)

frames_reader = ImageFilesFrameReader(images_dir=input_folder, start_image_index=1)
frames_writer = DummyFramesWriter()

files = [
    int(os.path.splitext(f)[0]) for f in os.listdir(output_folder) if os.path.isfile(os.path.join(output_folder, f))
]
files = sorted(files)

frame_index = files[-1] if len(files) > 0 else 0
initial_frame_index = frame_index


def save_frame(frame: cv2.typing.MatLike, x1: int, y1: int, w: int, h: int):
    global frame_index
    frame_index += 1
    cv2.imwrite(f"{output_folder}/{frame_index}.jpg", frame)


config = Config(action_highlight_enabled=False, detection_frequency=5)
face_tracker = FaceTracker(config, frames_reader, frames_writer, on_frame_detected_action=save_frame)

try:
    start = time.time()
    face_tracker.execute()
    end = time.time()
    print(f"Executed in {(end-start):.2f} seconds")
finally:
    frames_writer.flush_buffer()

# if initial_frame_index == frame_index:
#     os.rmdir(output_folder)

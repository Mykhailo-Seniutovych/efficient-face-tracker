import time
import cv2
import numpy as np

from frames_reader import ImageFilesFrameReader, VideoFramesReader
from frames_writer import ImageFilesFramesWriter, VideoFramesWriter, InteractiveFramesWriter
from face_tracker import FaceTracker, Config

frames_reader = ImageFilesFrameReader(images_dir="./data/test/scenario_5", start_image_index=0)
frames_reader = VideoFramesReader(
    video_path="data/public/videos/footage.mp4",
    start_time_in_sec=0,
)

frames_writer = ImageFilesFramesWriter(output_dir="./data/temp", buffer_size=50, img_size=(480, 640))
frames_writer = VideoFramesWriter(
    video_path="data/public/videos/footage-out.mp4",
    buffer_size=50,
    img_size=(480, 640),
    fps=30,
)
# frames_writer = InteractiveFramesWriter(reader=frames_reader)

config = Config(action_highlight_enabled=False, detection_frequency=7)
face_tracker = FaceTracker(config, frames_reader, frames_writer)

try:
    start = time.time()
    face_tracker.execute()
    end = time.time()
    print(f"Executed in {(end-start):.2f} seconds")
finally:
    frames_writer.flush_buffer()
    cv2.destroyAllWindows()

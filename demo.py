import time
import cv2
import numpy as np

from frames_reader import ImageFilesFrameReader, VideoFramesReader
from frames_writer import ImageFilesFramesWriter, VideoFramesWriter, InteractiveFramesWriter
from face_tracker import FaceTracker

frames_reader = ImageFilesFrameReader(
    images_dir="./data/test/scenario_5",
    # start_image_index=1637,
    start_image_index=0,
)
frames_reader = VideoFramesReader(
    # video_path="/home/michael/Stuff/ffmpeg-tutorial/videos/lex-33-lviv/lviv-2024-06-27/13:18/right_camera.mp4",
    video_path="/home/michael/Stuff/ffmpeg-tutorial/videos/lex-33-lviv/lviv-2024-07-10/1/external.mp4",
    # video_path="/home/michael/Stuff/ffmpeg-tutorial/videos/lex-33-lviv/old-school/1/fwd_camera.mp4",
    # start_time_in_sec=5 * 60 + 3,
    start_time_in_sec=0,
)

frames_writer = ImageFilesFramesWriter(output_dir="./data/temp", buffer_size=50, img_size=(640, 480))
frames_writer = VideoFramesWriter(video_path="./data/temp/out5.mp4", buffer_size=50, img_size=(640, 480), fps=15)
# frames_writer = InteractiveFramesWriter(reader=frames_reader)
face_tracker = FaceTracker(frames_reader, frames_writer)

try:
    start = time.time()
    face_tracker.execute()
    end = time.time()
    print(f"Executed in {(end-start):.2f} seconds")
finally:
    frames_writer.flush_buffer()
    cv2.destroyAllWindows()

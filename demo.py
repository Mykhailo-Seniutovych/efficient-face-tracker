import cv2
import numpy as np

from frames_reader import ImageFilesFrameReader, VideoFramesReader
from frames_writer import ImageFilesFramesWriter, VideoFramesWriter, InteractiveFramesWriter
from face_tracker import FaceTracker

frames_reader = ImageFilesFrameReader(images_dir="./data/test/scenario_5", start_image_index=1637)
frames_reader = VideoFramesReader(
    video_path="/home/michael/Stuff/ffmpeg-tutorial/videos/lex-33-lviv/lviv-2024-07-10/1/external.mp4",
    start_time_in_sec=6 * 60 + 49,
)

frames_writer = ImageFilesFramesWriter(output_dir="./data/temp", buffer_size=50, img_size=(640, 480))
frames_writer = VideoFramesWriter(video_path="./data/temp/out.mp4", buffer_size=50, img_size=(640, 480), fps=15)
frames_writer = InteractiveFramesWriter(reader=frames_reader)
face_tracker = FaceTracker(frames_reader, frames_writer)

try:
    face_tracker.execute()
finally:
    frames_writer.flush_buffer()
    cv2.destroyAllWindows()

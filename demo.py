import time
import cv2
import numpy as np

from frames_reader import ImageFilesFrameReader, VideoFramesReader
from frames_writer import ImageFilesFramesWriter, VideoFramesWriter, InteractiveFramesWriter
from face_tracker import FaceTracker, Config, draw_bbox, blur_and_draw_bbox, blur_bbox, do_nothing

frames_reader = ImageFilesFrameReader(images_dir="./data/test/scenario_5", start_image_index=0)
frames_reader = VideoFramesReader(
    video_path="/home/michael/Stuff/ffmpeg-tutorial/videos/lex-33-lviv/old-school/1/fwd_camera.mp4"
)

frames_writer = ImageFilesFramesWriter(output_dir="./data/temp/fac", buffer_size=50, img_size=(640, 480))
# frames_writer = VideoFramesWriter(
#     video_path="data/temp/fac1-new.realsense.mp4",
#     buffer_size=50,
#     img_size=(640, 480),
#     fps=15,
# )
# frames_writer = InteractiveFramesWriter(reader=frames_reader)

config = Config(action_highlight_enabled=False, detection_frequency=5)
face_tracker = FaceTracker(config, frames_reader, frames_writer, on_frame_detected_action=do_nothing)

try:
    start = time.time()
    face_tracker.execute()
    end = time.time()
    print(f"Executed in {(end-start):.2f} seconds")
finally:
    frames_writer.flush_buffer()
    cv2.destroyAllWindows()

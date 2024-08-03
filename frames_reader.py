from typing import Protocol

import cv2


class FramesReader(Protocol):
    def next_frame(self) -> tuple[bool, cv2.typing.MatLike]:
        pass

    def seek_backward(self, frames_offset: int):
        pass


class ImageFilesFrameReader(FramesReader):
    def __init__(self, images_dir: str, start_image_index: int = 0):
        self.__images_dir = images_dir
        self.__current_frame_index = start_image_index

    def next_frame(self) -> tuple[bool, cv2.typing.MatLike]:
        frame = cv2.imread(f"{self.__images_dir}/{self.__current_frame_index}.jpg")
        self.__current_frame_index += 1
        return True, frame

    def seek_backward(self, frames_offset: int):
        self.__current_frame_index -= frames_offset
        assert self.__current_frame_index >= 0


class VideoFramesReader(FramesReader):
    def __init__(self, video_path: str, start_time_in_sec: int = 0):
        self.__capture = cv2.VideoCapture(video_path)
        fps = self.__capture.get(cv2.CAP_PROP_FPS)
        start_frame_number = int(start_time_in_sec * fps)
        self.__capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
        self.__current_frame_index = start_frame_number

    def __del__(self):
        self.__capture.release()

    def next_frame(self) -> tuple[bool, cv2.typing.MatLike]:
        self.__current_frame_index += 1
        return self.__capture.read()

    def seek_backward(self, frames_offset: int):
        self.__current_frame_index -= frames_offset
        assert self.__current_frame_index >= 0
        self.__capture.set(cv2.CAP_PROP_POS_FRAMES, self.__current_frame_index)

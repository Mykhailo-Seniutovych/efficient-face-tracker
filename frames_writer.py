from typing import Protocol

import cv2
import numpy as np

from frames_reader import FramesReader


class FramesWriter(Protocol):
    def write_frame(self, frame: cv2.typing.MatLike) -> bool:
        pass

    def flush_buffer(self):
        pass

    def seek_backward(self, frames_offset: int):
        pass


class ImageFilesFramesWriter(FramesWriter):
    def __init__(self, output_dir: str, buffer_size: int, img_size: tuple[int, int]):
        rgb_channels = 3
        self.__output_dir = output_dir
        self.__buffer = np.zeros((buffer_size, img_size[1], img_size[0], rgb_channels), dtype=np.uint8)
        self.__buffer_size = buffer_size
        self.__frame_buffer_index = -1
        self.__frame_index = -1

    def __del__(self):
        self.flush_buffer()

    def write_frame(self, frame: cv2.typing.MatLike) -> bool:
        self.__frame_index += 1
        self.__frame_buffer_index += 1

        if self.__frame_buffer_index >= self.__buffer_size:
            self.flush_buffer()

        self.__buffer[self.__frame_buffer_index] = frame
        return True

    def flush_buffer(self):
        for i in range(min(self.__frame_buffer_index + 1, self.__buffer_size)):
            img_index = self.__frame_index - (self.__frame_buffer_index - i) + 1
            cv2.imwrite(f"{self.__output_dir}/{img_index}.jpg", self.__buffer[i])
        self.__frame_buffer_index = 0

    def seek_backward(self, frames_offset: int):
        assert frames_offset >= 0

        self.__frame_buffer_index -= frames_offset
        assert self.__frame_buffer_index >= -1
        assert self.__frame_buffer_index < self.__buffer_size

        self.__frame_index -= frames_offset
        assert self.__frame_index >= -1


class VideoFramesWriter(FramesWriter):
    def __init__(self, video_path: str, buffer_size: int, img_size: tuple[int, int], fps=15):
        rgb_channels = 3
        self.__video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc("m", "p", "4", "v"), fps, img_size)
        self.__buffer = np.zeros((buffer_size, img_size[1], img_size[0], rgb_channels), dtype=np.uint8)
        self.__buffer_size = buffer_size
        self.__frame_buffer_index = -1
        self.__frame_index = -1

    def __del__(self):
        self.flush_buffer()

    def write_frame(self, frame: cv2.typing.MatLike) -> bool:
        self.__frame_index += 1
        self.__frame_buffer_index += 1

        if self.__frame_buffer_index >= self.__buffer_size:
            self.flush_buffer()

        self.__buffer[self.__frame_buffer_index] = frame
        return True

    def flush_buffer(self):
        for i in range(min(self.__frame_buffer_index + 1, self.__buffer_size)):
            self.__video.write(self.__buffer[i])
        self.__frame_buffer_index = 0

    def seek_backward(self, frames_offset: int):
        assert frames_offset >= 0

        self.__frame_buffer_index -= frames_offset
        assert self.__frame_buffer_index >= -1
        assert self.__frame_buffer_index < self.__buffer_size

        self.__frame_index -= frames_offset
        assert self.__frame_index >= -1


class InteractiveFramesWriter(FramesWriter):
    """
    Used for testing, to visualize detection and tracking frame by frame
    """

    def __init__(self, reader: FramesReader):
        self.__frame_index = 0
        self.__reader = reader

    def write_frame(self, frame: cv2.typing.MatLike) -> bool:
        self.__frame_index += 1
        cv2.imshow("Frame", InteractiveFramesWriter.__named_frame(frame, self.__frame_index))
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            return False
        elif key == 81:  # left arrow
            self.__reader.seek_backward(2)
            self.__frame_index -= 2
            assert self.__frame_index >= 0
        elif key == 83:  # right arrow
            pass
        return True

    def flush_buffer(self):
        pass

    def seek_backward(self, frames_offset: int):
        pass

    def __named_frame(frame: cv2.typing.MatLike, number: int) -> cv2.typing.MatLike:
        named_frame = frame.copy()
        cv2.putText(
            named_frame,
            str(number),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
        return named_frame

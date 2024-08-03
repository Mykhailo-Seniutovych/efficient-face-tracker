import time
import os
from typing import Protocol

import numpy as np
import cv2
import face_detection

from frames_reader import FramesReader
from frames_writer import FramesWriter

IS_PIXEL_MOVING_THRESHOLD = 50
AMOUNT_OF_MOVING_PIXELS_THRESHOLD = 20


def named_frame(frame: cv2.typing.MatLike, number: int) -> cv2.typing.MatLike:
    named_frame = frame.copy()
    cv2.putText(
        named_frame,
        str(number),
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return named_frame


class FaceTracker:
    def __init__(self, frames_reader: FramesReader, frames_writer: FramesWriter):
        self.__frames_reader = frames_reader
        self.__frames_writer = frames_writer

    def execute(self):
        is_read, previous_frame = self.__frames_reader.next_frame()
        if not is_read:
            return

        while True:
            is_read, frame = self.__frames_reader.next_frame()
            if not is_read:
                break

            prev_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = np.abs(np.float32(gray) - np.float32(prev_gray))
            diff = np.uint8(diff)
            print("max", np.max(diff))
            ret, diff_clear = cv2.threshold(diff, IS_PIXEL_MOVING_THRESHOLD, 255, cv2.THRESH_BINARY)
            diff_clear = cv2.medianBlur(diff_clear, 3)
            amount_of_moving_pixels = len(diff_clear[diff_clear > 0])
            print("am", amount_of_moving_pixels)

            diff_frame = np.clip(diff * 2, 0, 255)
            diff_frame = named_frame(diff_frame, amount_of_moving_pixels)

            previous_frame = frame
            is_written = self.__frames_writer.write_frame(diff_frame)
            if not is_written:
                break

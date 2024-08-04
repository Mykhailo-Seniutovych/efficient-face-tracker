import time
import os
from typing import Protocol
from enum import Enum
from collections import deque

import numpy as np
import cv2
import cv2.legacy
import face_detection

from frames_reader import FramesReader
from frames_writer import FramesWriter

IS_PIXEL_MOVING_THRESHOLD = 30
AMOUNT_OF_MOVING_PIXELS_THRESHOLD = 20
AMOUNT_OF_MOVING_FRAMES_THRESHOLD = 2

DETECTION_CONFIDENCE_THRESHOLD = 0.7
DETECTOR = "DSFDDetector"
DETECTION_FREQUENCY = 7
MIN_FRAME_TO_FACE_SIZE_RATIO = 3.0

ACTION_HIGHLIGHT_ENABLED = False
REPORT_PROGRESS_INTERVAL = 100
USE_BACKWARD_TRACKING = True


class FrameColors:
    NO_MOVEMENT = (169, 169, 169)
    HAS_MOVEMENT = (0, 255, 255)
    DETECTOR_RUN = (0, 0, 255)
    TRACKER_RUN = (255, 0, 0)


class FaceTracker:
    def __init__(self, frames_reader: FramesReader, frames_writer: FramesWriter):
        self.__frames_reader = frames_reader
        self.__frames_writer = frames_writer
        self.__moving_frames_count = 0
        self.__has_movement = False
        self.__detector = face_detection.build_detector(
            DETECTOR, confidence_threshold=DETECTION_CONFIDENCE_THRESHOLD, nms_iou_threshold=0.3
        )
        self.__frame_index = -1
        self.__prev_faces_tracked = False
        self.__frame_buffer = deque(maxlen=DETECTION_FREQUENCY)

    def execute(self):
        is_read, frame = self.__read_next_frame()
        if not is_read:
            return
        self.__frame_index += 1
        self.__frame_width = frame.shape[1]
        self.__frame_height = frame.shape[0]

        self.__prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        while True:
            is_read, frame = self.__read_next_frame()
            if not is_read:
                break
            self.__frame_index += 1

            current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.__check_movement(current_gray)

            if self.__has_movement:
                output_frame = frame.copy()
                self.__highlight_frame(output_frame, FrameColors.HAS_MOVEMENT)
                faces = self.__detect_faces(frame, output_frame)
                success = self.__track_faces(frame, faces, output_frame)
                self.__prev_gray = current_gray
                if not success:
                    break
            else:
                self.__highlight_frame(frame, FrameColors.NO_MOVEMENT)
                self.__prev_gray = current_gray
                is_written = self.__frames_writer.write_frame(frame)
                if not is_written:
                    break

            if self.__frame_index % REPORT_PROGRESS_INTERVAL == 0:
                print(f"Processed {self.__frame_index}/{self.__frames_reader.frames_count} frames")

    def __read_next_frame(self) -> tuple[bool, cv2.typing.MatLike]:
        is_read, frame = self.__frames_reader.next_frame()
        if is_read:
            self.__frame_buffer.append(frame)
        return is_read, frame

    def __check_movement(self, current_gray: cv2.typing.MatLike):
        diff = np.abs(np.float32(current_gray) - np.float32(self.__prev_gray))
        diff = np.uint8(diff)
        ret, diff_clear = cv2.threshold(diff, IS_PIXEL_MOVING_THRESHOLD, 255, cv2.THRESH_BINARY)
        diff_clear = cv2.medianBlur(diff_clear, 3)
        amount_of_moving_pixels = len(diff_clear[diff_clear > 0])
        frame_has_movement = amount_of_moving_pixels >= IS_PIXEL_MOVING_THRESHOLD
        self.__moving_frames_count = self.__moving_frames_count + 1 if frame_has_movement else 0
        self.__has_movement = self.__moving_frames_count >= AMOUNT_OF_MOVING_FRAMES_THRESHOLD

    def __detect_faces(
        self, frame: cv2.typing.MatLike, output_frame: cv2.typing.MatLike
    ) -> list[tuple[int, int, int, int]]:
        should_run_detector = self.__frame_index % DETECTION_FREQUENCY == 0
        if not should_run_detector:
            return []

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        model_detections = self.__detector.detect(frame_rgb)
        result: tuple[int, int, int, int] = []
        for detection in model_detections:
            x1, y1, x2, y2 = int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3])
            face_area = (x2 - x1) * (y2 - y1)
            frame_area = self.__frame_width * self.__frame_height
            size_ratio = frame_area / face_area
            # the model sometimes falsely detects unrealistically large areas on fisheye camera images, we should filter them out
            if size_ratio > MIN_FRAME_TO_FACE_SIZE_RATIO:
                result.append((x1, y1, x2, y2))
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        self.__highlight_frame(output_frame, FrameColors.DETECTOR_RUN)
        return result

    def __track_faces(
        self,
        frame: cv2.typing.MatLike,
        detected_faces: list[tuple[int, int, int, int]],
        output_frame: cv2.typing.MatLike,
    ) -> bool:
        is_written = self.__frames_writer.write_frame(output_frame)
        if not is_written:
            return False

        if len(detected_faces) == 0:
            self.__prev_faces_tracked = False
            return True

        # backward track
        if USE_BACKWARD_TRACKING and not self.__prev_faces_tracked:
            trackers = []
            for face_bbox in detected_faces:
                x1, y1, x2, y2 = face_bbox
                x, y, w, h = (x1, y1, x2 - x1, y2 - y1)
                tracker = cv2.legacy.TrackerMedianFlow_create()
                tracker.init(frame, (x, y, w, h))
                trackers.append(tracker)

            offset = self.__frames_writer.seek_backward(DETECTION_FREQUENCY)
            prev_frames = []
            for idx in range(DETECTION_FREQUENCY - offset, DETECTION_FREQUENCY - 1):
                prev_frames.append(self.__frame_buffer[idx])

            tracked_faces: list[list[tuple[float, float, float, float]]] = []
            for prev_frame in reversed(prev_frames):
                tracked_faces.append([])
                for tracker in trackers:
                    object_found, bbox = tracker.update(prev_frame)
                    if object_found:
                        tracked_faces[-1].append(bbox)

            assert len(tracked_faces) == len(prev_frames)
            index = 0
            for frame_faces in reversed(tracked_faces):
                prev_frame = prev_frames[index]
                index += 1
                for face in frame_faces:
                    x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                    cv2.rectangle(prev_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                self.__highlight_frame(prev_frame, FrameColors.TRACKER_RUN)
                is_written = self.__frames_writer.write_frame(prev_frame)
                assert is_written == True

            self.__prev_faces_tracked = True

            # currently detected frame
            self.__highlight_frame(output_frame, FrameColors.DETECTOR_RUN)
            is_written = self.__frames_writer.write_frame(output_frame)
            if not is_written:
                return False

        # forward track
        trackers = []
        for face_bbox in detected_faces:
            x1, y1, x2, y2 = face_bbox
            x, y, w, h = (x1, y1, x2 - x1, y2 - y1)
            tracker = cv2.legacy.TrackerMedianFlow_create()
            tracker.init(frame, (x, y, w, h))
            trackers.append(tracker)

        for _ in range(DETECTION_FREQUENCY - 1):
            is_read, frame = self.__read_next_frame()
            if not is_read:
                return False
            self.__frame_index += 1

            output_frame = frame.copy()
            self.__highlight_frame(output_frame, FrameColors.TRACKER_RUN)

            faces = []
            for tracker in trackers:
                object_found, bbox = tracker.update(frame)
                if object_found:
                    faces.append(bbox)

            if len(faces) > 0:
                for face in faces:
                    x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                    cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            self.__prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            is_written = self.__frames_writer.write_frame(output_frame)
            if not is_written:
                return False

        return True

    def __highlight_frame(self, frame_bgr: cv2.typing.MatLike, color: FrameColors):
        if ACTION_HIGHLIGHT_ENABLED:
            cv2.rectangle(frame_bgr, (0, 0), (self.__frame_width, self.__frame_height), color, 10)

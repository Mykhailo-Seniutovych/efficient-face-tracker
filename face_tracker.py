import time
import os
from typing import Protocol
from enum import Enum
from collections import deque
from dataclasses import dataclass

import numpy as np
import cv2
import cv2.legacy
import face_detection

from frames_reader import FramesReader
from frames_writer import FramesWriter


@dataclass
class Config:
    is_pixel_moving_threshold: str = 30
    """This value is used to determine if a difference between two pixels constitutes as movement"""

    amount_of_moving_pixels_threshold: int = 20
    """How many moving pixels should be on a given frame, to consider this frame as having movement"""

    amount_of_moving_frames_threshold: int = 2
    """
    How many consecutive frames should have movement, to detect that the movement is currently happening on the video.
    Used to reduce the noise, when the movement was detected only on one frame.
    """

    detection_confidence_threshold: float = 0.7
    """The confidence threshold for face detector"""

    detector_name = "DSFDDetector"
    """The name of the detector to use, can be "DSFDDetector","RetinaNetResNet50", "RetinaNetMobileNetV1" """

    detection_frequency: int = 7
    """How often the detector should be run in terms of frames, for higher fps, videos this number should be larger"""

    min_frame_to_face_size_ratio: float = 3.0
    """
    The DSFDDetector detector sometimes falsely detects very large bounding boxes. The face cannot be so large .
    This value is used to filter out the bounding boxes when the ratio between frame area and bounding box is too large.
    """

    action_highlight_enabled: bool = False
    """
    Used to for debugging to mark each action on the frame 
    (e.g., no movement detected, movement detected, detector runs, tracker runs)
    """

    report_progress_interval: int = 100
    """The amount of frames to be processed to report progress to the console"""

    use_backward_tracking: bool = True
    """
    When face is detected, we can track the frames not only forward in the video, 
    but also go back in time before previous detection was run to ensure that we did not miss any frames. 
    """

    max_tracking_pixel_distance: int = 100
    """
    When face is tracked sometimes median flow can wrongly catch some random bbox on the image. 
    This value is used for filtering such bound boxes.
    When the distance between left top point of the previous and current bounding box is greater than this value, the bounding box will be excluded
    """


class FrameAction:
    NO_MOVEMENT = (169, 169, 169), "NO MOV"
    HAS_MOVEMENT = (0, 255, 255), "   MOV"
    DETECTOR_RUN = (0, 0, 255), "DETECT"
    TRACKER_RUN = (255, 0, 0), " TRACK"


class FaceTracker:
    def __init__(self, cfg: Config, frames_reader: FramesReader, frames_writer: FramesWriter):
        self.__cfg = cfg
        self.__frames_reader = frames_reader
        self.__frames_writer = frames_writer
        self.__moving_frames_count = 0
        self.__has_movement = False
        self.__detector = face_detection.build_detector(
            self.__cfg.detector_name,
            confidence_threshold=self.__cfg.detection_confidence_threshold,
            nms_iou_threshold=0.3,
        )
        self.__frame_index = -1
        self.__prev_faces_tracked = False
        self.__frame_buffer = deque(maxlen=self.__cfg.detection_frequency)

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
                faces = self.__detect_faces(frame, output_frame)
                success = self.__track_faces(frame, faces, output_frame)

                self.__prev_gray = current_gray
                if not success:
                    break
            else:
                self.__highlight_frame(frame, FrameAction.NO_MOVEMENT)
                self.__prev_gray = current_gray
                is_written = self.__frames_writer.write_frame(frame)
                if not is_written:
                    break

            if self.__frame_index % self.__cfg.report_progress_interval == 0:
                print(f"Processed {self.__frame_index}/{self.__frames_reader.frames_count} frames")

    def __read_next_frame(self) -> tuple[bool, cv2.typing.MatLike]:
        is_read, frame = self.__frames_reader.next_frame()
        if is_read:
            self.__frame_buffer.append(frame)
        return is_read, frame

    def __check_movement(self, current_gray: cv2.typing.MatLike):
        diff = np.abs(np.float32(current_gray) - np.float32(self.__prev_gray))
        diff = np.uint8(diff)
        ret, diff_clear = cv2.threshold(diff, self.__cfg.is_pixel_moving_threshold, 255, cv2.THRESH_BINARY)
        diff_clear = cv2.medianBlur(diff_clear, 3)
        amount_of_moving_pixels = len(diff_clear[diff_clear > 0])
        frame_has_movement = amount_of_moving_pixels >= self.__cfg.is_pixel_moving_threshold
        self.__moving_frames_count = self.__moving_frames_count + 1 if frame_has_movement else 0
        self.__has_movement = self.__moving_frames_count >= self.__cfg.amount_of_moving_frames_threshold

    def __detect_faces(
        self, frame: cv2.typing.MatLike, output_frame: cv2.typing.MatLike
    ) -> list[tuple[int, int, int, int]]:
        should_run_detector = self.__frame_index % self.__cfg.detection_frequency == 0
        if not should_run_detector:
            self.__highlight_frame(output_frame, FrameAction.HAS_MOVEMENT)
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
            if size_ratio > self.__cfg.min_frame_to_face_size_ratio:
                result.append((x1, y1, x2, y2))
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        self.__highlight_frame(output_frame, FrameAction.DETECTOR_RUN)
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
        if self.__cfg.use_backward_tracking and not self.__prev_faces_tracked:
            trackers: list[MedianFlowTracker] = []
            for face_bbox in detected_faces:
                x1, y1, x2, y2 = face_bbox
                x, y, w, h = (x1, y1, x2 - x1, y2 - y1)
                tracker = MedianFlowTracker(frame, (x, y, w, h))
                trackers.append(tracker)

            offset = self.__frames_writer.seek_backward(self.__cfg.detection_frequency)
            prev_frames = []
            for idx in range(self.__cfg.detection_frequency - offset, self.__cfg.detection_frequency - 1):
                prev_frames.append(self.__frame_buffer[idx])

            tracked_faces: list[list[tuple[float, float, float, float]]] = []
            for prev_frame in reversed(prev_frames):
                tracked_faces.append([])
                for tracker in trackers:
                    object_found, bbox = tracker.update(prev_frame)
                    if object_found and tracker.distance_moved() <= self.__cfg.max_tracking_pixel_distance:
                        tracked_faces[-1].append(bbox)

            assert len(tracked_faces) == len(prev_frames)
            index = 0
            for frame_faces in reversed(tracked_faces):
                prev_frame = prev_frames[index]
                index += 1
                for face in frame_faces:
                    x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                    cv2.rectangle(prev_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                self.__highlight_frame(prev_frame, FrameAction.TRACKER_RUN)
                is_written = self.__frames_writer.write_frame(prev_frame)
                assert is_written == True

            self.__prev_faces_tracked = True

            # currently detected frame
            self.__highlight_frame(output_frame, FrameAction.DETECTOR_RUN)
            is_written = self.__frames_writer.write_frame(output_frame)
            if not is_written:
                return False

        # forward track
        trackers: list[MedianFlowTracker] = []
        for face_bbox in detected_faces:
            x1, y1, x2, y2 = face_bbox
            x, y, w, h = (x1, y1, x2 - x1, y2 - y1)
            tracker = MedianFlowTracker(frame, (x, y, w, h))
            trackers.append(tracker)

        for _ in range(self.__cfg.detection_frequency - 1):
            is_read, frame = self.__read_next_frame()
            if not is_read:
                return False
            self.__frame_index += 1

            output_frame = frame.copy()
            self.__highlight_frame(output_frame, FrameAction.TRACKER_RUN)

            faces = []
            for tracker in trackers:
                object_found, bbox = tracker.update(frame)
                if object_found and tracker.distance_moved() <= self.__cfg.max_tracking_pixel_distance:
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

    def __highlight_frame(self, frame_bgr: cv2.typing.MatLike, action: FrameAction):
        if self.__cfg.action_highlight_enabled:
            cv2.rectangle(frame_bgr, (0, 0), (self.__frame_width, self.__frame_height), action[0], 10)
            cv2.putText(
                frame_bgr,
                action[1],
                (self.__frame_width - 90, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                action[0],
                1,
                cv2.LINE_AA,
            )


class MedianFlowTracker:
    def __init__(self, frame: cv2.typing.MatLike, bbox: tuple[int, int, int, int]):
        self.__tracker = cv2.legacy.TrackerMedianFlow_create()
        self.__tracker.init(frame, bbox)
        self.__prev_point = np.array([bbox[0], bbox[1]])
        self.__distance_moved = 0

    def update(self, frame: cv2.typing.MatLike) -> tuple[bool, tuple[int, int, int, int]]:
        obj_found, bbox = self.__tracker.update(frame)
        if obj_found:
            curr_point = np.array([bbox[0], bbox[1]])
            self.__distance_moved = np.linalg.norm(curr_point - self.__prev_point)
        return (obj_found, bbox)

    def distance_moved(self) -> float:
        return self.__distance_moved

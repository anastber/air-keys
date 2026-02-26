"""
Hand landmark extraction using MediaPipe.

This module provides functionality to extract 21 3D hand landmarks from webcam frames
and normalize them for consistent coordinate system.
"""

from typing import Any

import cv2
import mediapipe as mp
import numpy as np


class LandmarkExtractor:
    """Extracts and normalizes hand landmarks using MediaPipe."""

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """
        Initialize MediaPipe hands detector using the new tasks API.

        Args:
            static_image_mode: If True, treats input as static images.
            max_num_hands: Maximum number of hands to detect.
            min_detection_confidence: Minimum confidence for hand detection.
            min_tracking_confidence: Minimum confidence for hand tracking.
        """
        # Create hand detector using the new tasks API
        import os
        model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'hand_landmarker.task')
        model_path = os.path.abspath(model_path)

        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO if not static_image_mode else mp.tasks.vision.RunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self.frame_timestamp = 0

    def extract_landmarks(self, image: np.ndarray) -> list[dict[str, Any]] | None:
        """
        Extract hand landmarks from an image.

        Args:
            image: Input image as numpy array (BGR format from OpenCV).

        Returns:
            List of dictionaries containing normalized landmarks for each detected hand,
            or None if no hands are detected. Each dictionary contains:
            - 'landmarks': List of 21 landmarks, each with 'x', 'y', 'z' coordinates
            - 'handedness': 'Left' or 'Right'
        """
        # Convert BGR to RGB (MediaPipe expects RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Increment frame timestamp for video mode
        self.frame_timestamp += 1

        # Detect hand landmarks
        detection_result = self.detector.detect_for_video(mp_image, self.frame_timestamp)

        if not detection_result.hand_landmarks:
            return None

        hands_data = []

        # Extract landmarks for each detected hand
        for i, hand_landmarks in enumerate(detection_result.hand_landmarks):
            # Get handedness if available
            handedness = 'Unknown'
            if detection_result.handedness and i < len(detection_result.handedness):
                handedness_info = detection_result.handedness[i]
                if handedness_info:
                    handedness = handedness_info[0].category_name

            # Normalize landmarks relative to image dimensions
            normalized_landmarks = self._normalize_landmarks_new_api(
                hand_landmarks, image.shape[:2]
            )

            hand_data = {
                'landmarks': normalized_landmarks,
                'handedness': handedness
            }
            hands_data.append(hand_data)

        return hands_data

    def _normalize_landmarks_new_api(
        self, hand_landmarks, image_shape: tuple
    ) -> list[dict[str, float]]:
        """
        Normalize landmarks to absolute positions in image coordinates.
        Updated for new MediaPipe tasks API.

        Args:
            hand_landmarks: List of MediaPipe landmark objects from new API.
            image_shape: (height, width) of the input image.

        Returns:
            List of 21 normalized landmarks with x, y, z coordinates.
            - x, y are absolute positions normalized to [0, 1] range
            - z is absolute depth value
        """
        landmarks = []
        height, width = image_shape

        for landmark in hand_landmarks:
            # Keep landmarks in absolute normalized coordinates [0, 1]
            # MediaPipe already provides these in the correct format
            normalized_landmark = {
                'x': landmark.x,  # Already normalized to [0, 1]
                'y': landmark.y,  # Already normalized to [0, 1]
                'z': landmark.z,  # Depth value
            }
            landmarks.append(normalized_landmark)

        return landmarks

    def _normalize_landmarks(
        self, hand_landmarks, image_shape: tuple
    ) -> list[dict[str, float]]:
        """
        Legacy normalize method for backwards compatibility.
        """
        return self._normalize_landmarks_new_api(hand_landmarks, image_shape)

    def get_landmark_connections(self) -> list[tuple]:
        """
        Get MediaPipe hand landmark connections for visualization.

        Returns:
            List of tuples representing connections between landmarks.
        """
        # Hand connections for the new API (same as before)
        return [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)  # Palm
        ]

    def close(self):
        """Clean up MediaPipe resources."""
        if hasattr(self.detector, 'close'):
            self.detector.close()


class WebcamCapture:
    """Handles webcam capture for real-time landmark extraction."""

    def __init__(self, camera_index: int = 0):
        """
        Initialize webcam capture.

        Args:
            camera_index: Index of the camera to use (0 for default webcam).
        """
        self.camera_index = camera_index
        self.cap = None

    def start_capture(self) -> bool:
        """
        Start webcam capture.

        Returns:
            True if capture started successfully, False otherwise.
        """
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            return False

        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        return True

    def get_frame(self) -> np.ndarray | None:
        """
        Capture a single frame from the webcam.

        Returns:
            Frame as numpy array (BGR format) or None if capture failed.
        """
        if not self.cap or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        return frame

    def stop_capture(self):
        """Stop webcam capture and release resources."""
        if self.cap:
            self.cap.release()
            self.cap = None


def format_landmarks_for_json(landmarks_data: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Format landmarks data for JSON serialization.

    Args:
        landmarks_data: List of hand data dictionaries from extract_landmarks.

    Returns:
        Dictionary formatted for JSON transmission containing timestamp and hands data.
    """
    import time

    formatted_data = {
        'timestamp': time.time(),
        'hands': []
    }

    if landmarks_data:
        for hand_data in landmarks_data:
            formatted_hand = {
                'handedness': hand_data['handedness'],
                'landmarks': hand_data['landmarks']
            }
            formatted_data['hands'].append(formatted_hand)

    return formatted_data

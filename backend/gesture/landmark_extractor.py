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

        model_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "models", "hand_landmarker.task"
        )
        model_path = os.path.abspath(model_path)

        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO
            if not static_image_mode
            else mp.tasks.vision.RunningMode.IMAGE,
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
        detection_result = self.detector.detect_for_video(
            mp_image, self.frame_timestamp
        )

        if not detection_result.hand_landmarks:
            return None

        hands_data = []

        # Extract landmarks for each detected hand
        for i, hand_landmarks in enumerate(detection_result.hand_landmarks):
            # Get handedness if available
            handedness = "Unknown"
            if detection_result.handedness and i < len(detection_result.handedness):
                handedness_info = detection_result.handedness[i]
                if handedness_info:
                    handedness = handedness_info[0].category_name

            normalized_landmarks = self._normalize_landmarks(hand_landmarks)

            hand_data = {"landmarks": normalized_landmarks, "handedness": handedness}
            hands_data.append(hand_data)

        return hands_data

    def _normalize_landmarks(self, hand_landmarks) -> list[dict[str, float]]:
        """
        Convert MediaPipe landmark objects to plain dicts.

        Args:
            hand_landmarks: List of MediaPipe landmark objects.

        Returns:
            List of 21 landmarks with x, y, z coordinates.
            - x, y are already normalized to [0, 1] by MediaPipe
            - z is the depth value
        """
        return [
            {"x": landmark.x, "y": landmark.y, "z": landmark.z}
            for landmark in hand_landmarks
        ]

    def close(self):
        """Clean up MediaPipe resources."""
        if hasattr(self.detector, "close"):
            self.detector.close()


def format_landmarks_for_json(landmarks_data: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Format landmarks data for JSON serialization.

    Args:
        landmarks_data: List of hand data dictionaries from extract_landmarks.

    Returns:
        Dictionary formatted for JSON transmission containing timestamp and hands data.
    """
    import time

    formatted_data = {"timestamp": time.time(), "hands": []}

    if landmarks_data:
        for hand_data in landmarks_data:
            formatted_hand = {
                "handedness": hand_data["handedness"],
                "landmarks": hand_data["landmarks"],
            }
            # Pass through gesture classification if the caller attached one
            # (see backend.gesture.classifier) — absent until a model is trained.
            if "gesture" in hand_data:
                formatted_hand["gesture"] = hand_data["gesture"]
                formatted_hand["confidence"] = hand_data["confidence"]
            formatted_data["hands"].append(formatted_hand)

    return formatted_data

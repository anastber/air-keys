"""
Tests for landmark extraction and normalization functionality.
"""

from unittest.mock import Mock, patch

import numpy as np

from backend.gesture.landmark_extractor import (
    LandmarkExtractor,
    WebcamCapture,
    format_landmarks_for_json,
)


class TestLandmarkExtractor:
    """Test suite for LandmarkExtractor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = LandmarkExtractor()

    def test_initialization(self):
        """Test LandmarkExtractor initialization with default parameters."""
        assert self.extractor.mp_hands is not None
        assert self.extractor.hands is not None

    def test_initialization_with_custom_params(self):
        """Test LandmarkExtractor initialization with custom parameters."""
        extractor = LandmarkExtractor(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.8,
        )
        assert extractor is not None

    def test_normalize_landmarks(self):
        """Test landmark normalization relative to wrist position."""
        # Create mock hand landmarks
        mock_landmarks = self._create_mock_landmarks()
        image_shape = (480, 640)  # height, width

        # Test normalization
        normalized = self.extractor._normalize_landmarks(mock_landmarks, image_shape)

        # Should have 21 landmarks (standard MediaPipe hand landmarks)
        assert len(normalized) == 21

        # Each landmark should have x, y, z coordinates
        for landmark in normalized:
            assert 'x' in landmark
            assert 'y' in landmark
            assert 'z' in landmark
            assert isinstance(landmark['x'], float)
            assert isinstance(landmark['y'], float)
            assert isinstance(landmark['z'], float)

        # Wrist (index 0) should be at origin after normalization
        wrist = normalized[0]
        assert abs(wrist['x']) < 1e-6  # Should be approximately 0
        assert abs(wrist['y']) < 1e-6  # Should be approximately 0
        assert abs(wrist['z']) < 1e-6  # Should be approximately 0

    def test_normalize_landmarks_coordinate_ranges(self):
        """Test that normalized coordinates are in reasonable ranges."""
        mock_landmarks = self._create_mock_landmarks()
        image_shape = (480, 640)

        normalized = self.extractor._normalize_landmarks(mock_landmarks, image_shape)

        # Normalized x, y coordinates should typically be in range [-1, 1]
        # (since they're relative to wrist and normalized by image dimensions)
        for landmark in normalized:
            assert -2.0 <= landmark['x'] <= 2.0  # Allow some margin
            assert -2.0 <= landmark['y'] <= 2.0  # Allow some margin
            # Z coordinates can vary more widely
            assert -1.0 <= landmark['z'] <= 1.0

    def test_normalize_landmarks_consistency(self):
        """Test that normalization is consistent across different image sizes."""
        mock_landmarks = self._create_mock_landmarks()

        # Test with different image sizes
        shapes = [(240, 320), (480, 640), (720, 1280)]
        normalized_results = []

        for shape in shapes:
            normalized = self.extractor._normalize_landmarks(mock_landmarks, shape)
            normalized_results.append(normalized)

        # The relative positions should be similar regardless of image size
        # (We'll check that the patterns are consistent, not exact values due to
        # different normalization factors)
        for i in range(len(normalized_results[0])):
            landmark_240p = normalized_results[0][i]
            landmark_480p = normalized_results[1][i]
            landmark_720p = normalized_results[2][i]

            # Check that the signs are consistent (relative direction from wrist)
            if abs(landmark_480p['x']) > 0.01:  # Only check non-zero values
                assert np.sign(landmark_240p['x']) == np.sign(landmark_480p['x'])
                assert np.sign(landmark_480p['x']) == np.sign(landmark_720p['x'])

            if abs(landmark_480p['y']) > 0.01:
                assert np.sign(landmark_240p['y']) == np.sign(landmark_480p['y'])
                assert np.sign(landmark_480p['y']) == np.sign(landmark_720p['y'])

    @patch('backend.gesture.landmark_extractor.cv2.cvtColor')
    def test_extract_landmarks_no_hands(self, mock_cvtColor):
        """Test extract_landmarks when no hands are detected."""
        # Mock image
        mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cvtColor.return_value = mock_image

        # Mock MediaPipe to return no hands
        with patch.object(self.extractor.hands, 'process') as mock_process:
            mock_result = Mock()
            mock_result.multi_hand_landmarks = None
            mock_process.return_value = mock_result

            result = self.extractor.extract_landmarks(mock_image)
            assert result is None

    @patch('backend.gesture.landmark_extractor.cv2.cvtColor')
    def test_extract_landmarks_with_hands(self, mock_cvtColor):
        """Test extract_landmarks when hands are detected."""
        # Mock image
        mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cvtColor.return_value = mock_image

        # Mock MediaPipe to return hands
        with patch.object(self.extractor.hands, 'process') as mock_process:
            mock_result = Mock()
            mock_result.multi_hand_landmarks = [self._create_mock_landmarks()]

            # Mock handedness
            mock_handedness = Mock()
            mock_handedness.classification = [Mock()]
            mock_handedness.classification[0].label = 'Right'
            mock_result.multi_handedness = [mock_handedness]

            mock_process.return_value = mock_result

            result = self.extractor.extract_landmarks(mock_image)

            assert result is not None
            assert len(result) == 1
            assert result[0]['handedness'] == 'Right'
            assert 'landmarks' in result[0]
            assert len(result[0]['landmarks']) == 21

    def test_get_landmark_connections(self):
        """Test that landmark connections are returned."""
        connections = self.extractor.get_landmark_connections()
        assert isinstance(connections, list)
        assert len(connections) > 0
        # Each connection should be a tuple of two integers
        for connection in connections:
            assert isinstance(connection, tuple)
            assert len(connection) == 2
            assert isinstance(connection[0], int)
            assert isinstance(connection[1], int)

    def _create_mock_landmarks(self):
        """Create mock MediaPipe hand landmarks for testing."""
        mock_landmarks = Mock()
        mock_landmarks.landmark = []

        # Create 21 mock landmarks (standard MediaPipe hand landmarks)
        landmark_positions = [
            # Wrist
            (0.5, 0.5, 0.0),
            # Thumb
            (0.4, 0.6, -0.1), (0.3, 0.7, -0.2), (0.2, 0.8, -0.3), (0.1, 0.9, -0.4),
            # Index finger
            (0.6, 0.4, -0.1), (0.7, 0.3, -0.2), (0.8, 0.2, -0.3), (0.9, 0.1, -0.4),
            # Middle finger
            (0.6, 0.3, -0.1), (0.7, 0.2, -0.2), (0.8, 0.1, -0.3), (0.9, 0.0, -0.4),
            # Ring finger
            (0.6, 0.5, -0.1), (0.7, 0.6, -0.2), (0.8, 0.7, -0.3), (0.9, 0.8, -0.4),
            # Pinky
            (0.6, 0.6, -0.1), (0.7, 0.7, -0.2), (0.8, 0.8, -0.3), (0.9, 0.9, -0.4),
        ]

        for x, y, z in landmark_positions:
            mock_landmark = Mock()
            mock_landmark.x = x
            mock_landmark.y = y
            mock_landmark.z = z
            mock_landmarks.landmark.append(mock_landmark)

        return mock_landmarks


class TestWebcamCapture:
    """Test suite for WebcamCapture class."""

    def test_initialization(self):
        """Test WebcamCapture initialization."""
        webcam = WebcamCapture(camera_index=0)
        assert webcam.camera_index == 0
        assert webcam.cap is None

    def test_initialization_custom_index(self):
        """Test WebcamCapture initialization with custom camera index."""
        webcam = WebcamCapture(camera_index=1)
        assert webcam.camera_index == 1

    @patch('backend.gesture.landmark_extractor.cv2.VideoCapture')
    def test_start_capture_success(self, mock_VideoCapture):
        """Test successful webcam capture start."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_VideoCapture.return_value = mock_cap

        webcam = WebcamCapture()
        result = webcam.start_capture()

        assert result is True
        assert webcam.cap == mock_cap
        # Verify camera properties are set
        mock_cap.set.assert_any_call(mock_VideoCapture.return_value.CAP_PROP_FPS, 30)

    @patch('backend.gesture.landmark_extractor.cv2.VideoCapture')
    def test_start_capture_failure(self, mock_VideoCapture):
        """Test failed webcam capture start."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_VideoCapture.return_value = mock_cap

        webcam = WebcamCapture()
        result = webcam.start_capture()

        assert result is False

    def test_get_frame_no_capture(self):
        """Test get_frame when capture is not started."""
        webcam = WebcamCapture()
        frame = webcam.get_frame()
        assert frame is None

    def test_stop_capture(self):
        """Test stopping webcam capture."""
        webcam = WebcamCapture()
        mock_cap = Mock()
        webcam.cap = mock_cap

        webcam.stop_capture()

        mock_cap.release.assert_called_once()
        assert webcam.cap is None


class TestFormatLandmarksForJson:
    """Test suite for format_landmarks_for_json function."""

    def test_format_empty_landmarks(self):
        """Test formatting empty landmarks data."""
        result = format_landmarks_for_json([])

        assert 'timestamp' in result
        assert 'hands' in result
        assert isinstance(result['timestamp'], float)
        assert isinstance(result['hands'], list)
        assert len(result['hands']) == 0

    def test_format_landmarks_with_data(self):
        """Test formatting landmarks data with actual hand data."""
        mock_landmarks_data = [
            {
                'handedness': 'Right',
                'landmarks': [
                    {'x': 0.0, 'y': 0.0, 'z': 0.0},
                    {'x': 0.1, 'y': 0.1, 'z': 0.1},
                ]
            }
        ]

        result = format_landmarks_for_json(mock_landmarks_data)

        assert 'timestamp' in result
        assert 'hands' in result
        assert len(result['hands']) == 1
        assert result['hands'][0]['handedness'] == 'Right'
        assert len(result['hands'][0]['landmarks']) == 2
        assert result['hands'][0]['landmarks'][0] == {'x': 0.0, 'y': 0.0, 'z': 0.0}

    def test_format_multiple_hands(self):
        """Test formatting data with multiple hands."""
        mock_landmarks_data = [
            {
                'handedness': 'Right',
                'landmarks': [{'x': 0.0, 'y': 0.0, 'z': 0.0}]
            },
            {
                'handedness': 'Left',
                'landmarks': [{'x': 0.1, 'y': 0.1, 'z': 0.1}]
            }
        ]

        result = format_landmarks_for_json(mock_landmarks_data)

        assert len(result['hands']) == 2
        assert result['hands'][0]['handedness'] == 'Right'
        assert result['hands'][1]['handedness'] == 'Left'

    def test_format_preserves_landmark_structure(self):
        """Test that landmark structure is preserved during formatting."""
        mock_landmarks_data = [
            {
                'handedness': 'Right',
                'landmarks': [
                    {'x': 0.123, 'y': -0.456, 'z': 0.789},
                    {'x': 1.0, 'y': -1.0, 'z': 0.5},
                ]
            }
        ]

        result = format_landmarks_for_json(mock_landmarks_data)
        landmarks = result['hands'][0]['landmarks']

        assert landmarks[0] == {'x': 0.123, 'y': -0.456, 'z': 0.789}
        assert landmarks[1] == {'x': 1.0, 'y': -1.0, 'z': 0.5}

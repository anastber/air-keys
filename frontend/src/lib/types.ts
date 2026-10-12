// Shared shapes for the hand-tracking data flowing from the backend's
// /ws/landmarks stream. Used by the camera component, the instrument that
// plays audio from it, and the gesture-training UI that records samples.

export interface Landmark {
  x: number;
  y: number;
  z: number;
}

export interface HandData {
  handedness: string;
  landmarks: Landmark[];
  // Present only once a gesture classifier has been trained and is
  // confident about this hand's pose (see backend.gesture.classifier).
  gesture?: string;
  confidence?: number;
}

export interface LandmarkData {
  timestamp: number;
  hands: HandData[];
  error?: string;
}

// The active webcam, as reported by the browser once the stream starts.
// Recorded with collected samples so evaluation can be broken down by
// camera (see components/DatasetCollector.tsx).
export interface CameraInfo {
  label: string;
  width: number;
  height: number;
}

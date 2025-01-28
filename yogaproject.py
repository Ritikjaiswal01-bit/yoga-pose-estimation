import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Tree Pose ideal angles
tree_pose_template = {
    "left_knee_angle": 45,  # The knee should be bent to ~45 degrees
    "left_hip_angle": 90   # The raised leg should be close to a right angle at the hip
}

# OpenCV video capture
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Detect pose
        results = pose.process(image)

        # Convert back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Process pose landmarks
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            height, width, _ = image.shape

            # Get key points
            left_hip = [
                int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * width),
                int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * height)
            ]
            left_knee = [
                int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * width),
                int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * height)
            ]
            left_ankle = [
                int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * width),
                int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * height)
            ]

            # Calculate angles for the Tree Pose
            left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            left_hip_angle = calculate_angle(
                [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * width,
                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * height],
                left_hip,
                left_knee
            )

            # Check alignment with the ideal angles
            tolerance = 10
            feedback = []
            if abs(left_knee_angle - tree_pose_template["left_knee_angle"]) <= tolerance:
                knee_color = (0, 255, 0)  # Green for correct knee pose
                feedback.append("Knee angle is correct!")
            else:
                knee_color = (0, 0, 255)  # Red for incorrect knee pose
                feedback.append("Adjust your left knee to ~45 degrees.")

            if abs(left_hip_angle - tree_pose_template["left_hip_angle"]) <= tolerance:
                hip_color = (0, 255, 0)  # Green for correct hip alignment
                feedback.append("Hip angle is correct!")
            else:
                hip_color = (0, 0, 255)  # Red for incorrect hip alignment
                feedback.append("Adjust your left hip to ~90 degrees.")

            # Draw lines with appropriate color
            cv2.line(image, tuple(left_hip), tuple(left_knee), knee_color, 5)
            cv2.line(image, tuple(left_knee), tuple(left_ankle), knee_color, 5)
            cv2.line(image, tuple(left_hip), tuple(left_knee), hip_color, 5)

            # Display feedback
            for i, comment in enumerate(feedback):
                cv2.putText(image, comment, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            # Display the calculated angles
            cv2.putText(image, f"Knee Angle: {int(left_knee_angle)}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, knee_color, 2, cv2.LINE_AA)
            cv2.putText(image, f"Hip Angle: {int(left_hip_angle)}", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hip_color, 2, cv2.LINE_AA)

        # Show the image
        cv2.imshow('Tree Pose Estimation', image)

        # Quit with 'q' key
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()

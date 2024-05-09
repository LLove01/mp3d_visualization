import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import sys

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS

def calculate_2d_angle(A, B, C):
    """Calculate the angle between three 2D points based on their X and Y coordinates."""
    BA = np.array([A[0] - B[0], A[1] - B[1]])
    BC = np.array([C[0] - B[0], C[1] - B[1]])
    cosine_angle = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Image at {image_path} not found.")
        sys.exit(1)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0)
    results = pose.process(image)

    if not results.pose_world_landmarks:
        print("No pose landmarks detected.")
        sys.exit(1)

    landmarks = results.pose_world_landmarks.landmark

    # Prepare the plot for 2D visualization
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_aspect('equal')

    # Plot all connections
    for connection in POSE_CONNECTIONS:
        start_point = landmarks[connection[0]]
        end_point = landmarks[connection[1]]
        ax.plot([start_point.x, end_point.x],
                [start_point.y, end_point.y],
                color='grey', alpha=0.5)

    # Plot all keypoints
    for landmark in landmarks:
        ax.scatter(landmark.x, landmark.y, c='grey', alpha=0.5, s=10)

    # Define keypoints of interest
    keypoints_of_interest = [mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE]
    connections_of_interest = [(mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
                               (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE)]

    # Highlight selected connections
    for connection in connections_of_interest:
        start_point = landmarks[connection[0].value]
        end_point = landmarks[connection[1].value]
        ax.plot([start_point.x, end_point.x],
                [start_point.y, end_point.y],
                color='blue', linewidth=2)

    # Highlight selected keypoints
    for keypoint in keypoints_of_interest:
        landmark = landmarks[keypoint.value]
        ax.scatter(landmark.x, landmark.y, c='red', s=40)

    # Calculate and display the right knee angle
    hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
           landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
             landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
    right_knee_angle = calculate_2d_angle(hip, knee, ankle)

    # Add annotation at the second keypoint (RIGHT_KNEE)
    angle_text = f"Knee Angle: {right_knee_angle:.2f}°"
    ax.text(knee[0], knee[1] + 0.1, angle_text, color='black',
            ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('2D Pose Estimation Plot')
    ax.set_xlim([-1, 1])
    ax.set_ylim([1, -1])  # Invert the Y-axis

    plt.show()
    print("Right Knee Angle:", right_knee_angle)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python plot_2d.py <path_to_image>")
        sys.exit(1)
    image_path = sys.argv[1]
    process_image(image_path)

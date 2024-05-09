import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import sys

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

def calculate_midpoint_2d(pointA, pointB):
    """Calculate the midpoint between two 2D points."""
    return [(pointA[0] + pointB[0]) / 2, (pointA[1] + pointB[1]) / 2]

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

    # Calculate midpoints and draw vectors
    hips_midpoint = calculate_midpoint_2d(
        [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y],
        [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y])
    shoulders_midpoint = calculate_midpoint_2d(
        [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
        [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])

    # Draw the trunk vector
    ax.plot([hips_midpoint[0], shoulders_midpoint[0]],
            [hips_midpoint[1], shoulders_midpoint[1]],
            color='blue', linewidth=2, label='Trunk Vector')

    # Calculate and draw vertical reference line
    vertical_point = [hips_midpoint[0], shoulders_midpoint[1]]  
    ax.plot([hips_midpoint[0], hips_midpoint[0]],
            [hips_midpoint[1], vertical_point[1]],
            color='green', linestyle='--', linewidth=2, label='Vertical Reference')

    # Calculate and display the trunk vertical angle
    trunk_vertical_angle = calculate_2d_angle(shoulders_midpoint, hips_midpoint, vertical_point)
    angle_text = f"Trunk Vertical Angle: {trunk_vertical_angle:.2f}°"

    # Adjusted text position directly on the plot without an arrow
    ax.text(hips_midpoint[0], hips_midpoint[1] + 0.1, angle_text, color='black', 
            ha='center', va='top', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('2D Pose Estimation with Skeleton Highlighted')
    ax.legend()
    ax.set_xlim([-1, 1])
    ax.set_ylim([1, -1])

    plt.show()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python plot_2d.py <path_to_image>")
        sys.exit(1)
    image_path = sys.argv[1]
    process_image(image_path)

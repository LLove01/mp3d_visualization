import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS

def calculate_3d_angle(A, B, C):
    """Calculate the angle between three 3D points."""
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    AB = B - A
    BC = C - B
    dot_product = np.dot(AB, BC)
    magnitude_AB = np.linalg.norm(AB)
    magnitude_BC = np.linalg.norm(BC)
    angle_rad = np.arccos(dot_product / (magnitude_AB * magnitude_BC))
    angle_deg = np.degrees(angle_rad)
    return int(angle_deg)

def process_image(image_path):
    """Process the image and visualize selected keypoints and connections."""
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

    # Prepare the plot for 3D visualization
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')

    # Plot all connections with a lighter color
    for connection in POSE_CONNECTIONS:
        start_point = landmarks[connection[0]]
        end_point = landmarks[connection[1]]
        ax.plot([start_point.x, end_point.x],
                [start_point.z, end_point.z],
                [-start_point.y, -end_point.y],
                color='grey', alpha=0.5)

    # Plot all keypoints with a lighter color
    for landmark in landmarks:
        ax.scatter(landmark.x, landmark.z, -landmark.y, c='grey', alpha=0.5, s=10)

    # Define keypoints of interest
    keypoints_of_interest = [mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE]
    connections_of_interest = [(mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
                               (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE)]

    # Highlight selected connections
    for connection in connections_of_interest:
        start_point = landmarks[connection[0].value]
        end_point = landmarks[connection[1].value]
        ax.plot([start_point.x, end_point.x],
                [start_point.z, end_point.z],
                [-start_point.y, -end_point.y],
                color='blue')

    # Highlight selected keypoints
    for keypoint in keypoints_of_interest:
        landmark = landmarks[keypoint.value]
        ax.scatter(landmark.x, landmark.z, -landmark.y, c='red', s=40)

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.set_title('3D Pose Estimation Plot')
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)

    plt.show()

    # Calculate and print the right knee angle
    hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
           landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
           landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]
    knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z]
    ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
             landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
             landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z]

    right_knee_angle = calculate_3d_angle(hip, knee, ankle)
    print("Right Knee Angle:", right_knee_angle)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python display_angle.py <path_to_image>")
        sys.exit(1)
    image_path = sys.argv[1]  # Use the command-line argument for the image path
    process_image(image_path)
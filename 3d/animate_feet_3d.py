import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys
from matplotlib.animation import FuncAnimation, PillowWriter

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS

def calculate_vector_angle(vec1, vec2):
    """ Calculate the angle between two vectors. """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot = np.dot(vec1, vec2)
    mag_vec1 = np.linalg.norm(vec1)
    mag_vec2 = np.linalg.norm(vec2)
    cos_angle = dot / (mag_vec1 * mag_vec2)
    angle = np.arccos(np.clip(cos_angle, -1, 1))
    return np.degrees(angle)

def process_image(image_path):
    """Process the image and visualize the angle between the heel and foot index (toe) in a rotating 3D plot."""
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

    left_foot_vector = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x - landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y - landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y,
                        landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].z - landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].z]
    right_foot_vector = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x - landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y - landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].z - landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].z]

    feet_angle = calculate_vector_angle(left_foot_vector, right_foot_vector)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')

    def update(frame):
        plt.cla()
        ax.view_init(elev=10, azim=frame)

        # Plot skeleton
        for connection in POSE_CONNECTIONS:
            start_point = landmarks[connection[0]]
            end_point = landmarks[connection[1]]
            ax.plot([start_point.x, end_point.x],
                    [start_point.z, end_point.z],
                    [-start_point.y, -end_point.y],
                    color='grey', alpha=0.5)

        # Draw keypoints
        for landmark in landmarks:
            ax.scatter(landmark.x, landmark.z, -landmark.y, c='grey', alpha=0.5, s=10)

        # Draw foot vectors
        ax.plot([landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x],
                [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].z, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].z],
                [-landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y, -landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y],
                color='blue', linewidth=2)

        ax.plot([landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x],
                [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].z, landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].z],
                [-landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y, -landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y],
                color='red', linewidth=2)

        ax.text2D(0.05, 0.95, f"Feet Angle: {feet_angle:.2f}°", transform=ax.transAxes, color='black', fontsize=12)

        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')
        ax.set_title('3D Pose Estimation with Feet Angle')
        ax.set_xlim3d(-1, 1)
        ax.set_ylim3d(-1, 1)
        ax.set_zlim3d(-1, 1)

    anim = FuncAnimation(fig, update, frames=np.arange(-90, 0, 1), interval=100)
    anim.save('feet_angle_rotation.gif', writer=PillowWriter(fps=20))

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python display_angle.py <path_to_image>")
        sys.exit(1)
    image_path = sys.argv[1]
    process_image(image_path)

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

def calculate_midpoint(landmarkA, landmarkB):
    """ Calculate the midpoint between two landmarks in 3D space. """
    return {
        'x': (landmarkA.x + landmarkB.x) / 2,
        'y': (landmarkA.y + landmarkB.y) / 2,
        'z': (landmarkA.z + landmarkB.z) / 2
    }

def process_image(image_path):
    """Process the image and visualize the vertical angle in a rotating 3D plot."""
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

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')

    hips_midpoint = calculate_midpoint(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])
    shoulders_midpoint = calculate_midpoint(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])

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

        # Draw trunk line
        ax.plot([hips_midpoint['x'], shoulders_midpoint['x']],
                [hips_midpoint['z'], shoulders_midpoint['z']],
                [-hips_midpoint['y'], -shoulders_midpoint['y']],
                color='blue', linewidth=2)

        # Draw vertical line
        ax.plot([hips_midpoint['x'], hips_midpoint['x']],
                [hips_midpoint['z'], hips_midpoint['z']],
                [-hips_midpoint['y'], -hips_midpoint['y'] + 0.5],
                color='green', linewidth=2)

        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')
        ax.set_title('3D Pose Estimation with Vertical Trunk Angle')
        ax.set_xlim3d(-1, 1)
        ax.set_ylim3d(-1, 1)
        ax.set_zlim3d(-1, 1)

    anim = FuncAnimation(fig, update, frames=np.arange(-90, 0, 1), interval=100)
    anim.save('pose_vertical_angle_rotation.gif', writer=PillowWriter(fps=20))

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python display_angle.py <path_to_image>")
        sys.exit(1)
    image_path = sys.argv[1]
    process_image(image_path)

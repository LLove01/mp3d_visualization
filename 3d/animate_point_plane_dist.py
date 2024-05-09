import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import sys

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS

def calculate_plane_normal(vec1, vec2):
    normal = np.cross(vec1, vec2)
    normal = normal / np.linalg.norm(normal)
    return normal

def point_to_plane_distance(point, plane_point, normal):
    vec = point - plane_point
    dot_product = np.dot(vec, normal)
    return dot_product

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Image at {image_path} not found.")
        sys.exit(1)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    results = pose.process(image)

    if not results.pose_world_landmarks:
        print("No pose landmarks detected.")
        sys.exit(1)

    landmarks = results.pose_world_landmarks.landmark
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')

    # Define vectors for the torso plane
    shoulder_vector = np.array([
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x - landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z - landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z,
        -(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y - landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)
    ])
    hip_vector = np.array([
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x - landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z - landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z,
        -(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y - landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)
    ])

    # Calculate the normal of the torso plane
    torso_normal = calculate_plane_normal(shoulder_vector, hip_vector)
    right_shoulder_point = np.array([
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z,
        -landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
    ])

    # Define elbow midpoint
    elbow_midpoint = (np.array([
        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z,
        -landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y
    ]) + np.array([
        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z,
        -landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y
    ])) / 2

    # Calculate the perpendicular distance from the elbow midpoint to the torso plane
    distance = point_to_plane_distance(elbow_midpoint, right_shoulder_point, torso_normal)
    print(distance)
    closest_point_on_plane = elbow_midpoint - distance * torso_normal

    # Plot vectors defining the plane
    quiver_length = 0.2  # Increase quiver length
    ax.quiver(*right_shoulder_point, *shoulder_vector, color='blue', length=quiver_length, normalize=True)
    ax.quiver(*right_shoulder_point, *hip_vector, color='green', length=quiver_length, normalize=True)

    # Highlight the elbow midpoint and closest point on the plane
    ax.scatter(*elbow_midpoint, color='red', s=20, label='Elbow Midpoint')
    ax.scatter(*closest_point_on_plane, color='purple', s=20, label='Closest Point on Plane')

    # Dotted line from elbow midpoint to closest point on plane
    ax.plot(*zip(elbow_midpoint, closest_point_on_plane), color='red', linestyle='--')

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
        ax.scatter(landmark.x, landmark.z, -landmark.y, color='grey', alpha=0.5, s=10)

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.set_title('3D Pose Estimation with Vectors and Plane')
    ax.legend()
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    plt.show()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python display_torso_plane.py <path_to_image>")
        sys.exit(1)
    image_path = sys.argv[1]
    process_image(image_path)

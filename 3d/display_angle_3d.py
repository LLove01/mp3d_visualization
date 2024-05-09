import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import sys
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_3d_angle(A, B, C):
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    AB = B - A
    BC = C - B
    cos_angle = np.dot(AB, BC) / (np.linalg.norm(AB) * np.linalg.norm(BC))
    angle = np.arccos(cos_angle)
    return np.degrees(angle)

def process_image(image_path, keypoints_of_interest, save_path):
    if len(keypoints_of_interest) != 3:
        print("Error: keypoints_of_interest must contain exactly 3 keypoints.")
        sys.exit(1)

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Image at {image_path} not found.")
        sys.exit(1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    results = pose.process(image)

    if not results.pose_landmarks:
        print("No pose landmarks detected.")
        return  # Exiting if no landmarks detected

    landmarks = results.pose_landmarks.landmark

    # Draw all keypoints with a lighter color
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(200, 200, 200),thickness=3),
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(100, 100, 100), thickness=3, circle_radius=3)
    )

    # Define connections of interest based on keypoints_of_interest
    connections_of_interest = []
    for i in range(len(keypoints_of_interest) - 1):
        connections_of_interest.append(
            (keypoints_of_interest[i].value, keypoints_of_interest[i + 1].value)
        )

    # Draw selected keypoints and connections
    for connection in connections_of_interest:
        start_point = landmarks[connection[0]]
        end_point = landmarks[connection[1]]
        cv2.line(image, 
                 (int(start_point.x * image.shape[1]), int(start_point.y * image.shape[0])),
                 (int(end_point.x * image.shape[1]), int(end_point.y * image.shape[0])),
                 (0, 0, 255), 3)
        for point in [start_point, end_point]:
            cv2.circle(image, 
                       (int(point.x * image.shape[1]), int(point.y * image.shape[0])),
                       4, (255, 0, 0), -1)

    # Calculate the angles between keypoints_of_interest
    angles = []
    for i in range(len(keypoints_of_interest) - 2):
        angle = calculate_3d_angle(
            [landmarks[keypoints_of_interest[i].value].x, landmarks[keypoints_of_interest[i].value].y, landmarks[keypoints_of_interest[i].value].z],
            [landmarks[keypoints_of_interest[i + 1].value].x, landmarks[keypoints_of_interest[i + 1].value].y, landmarks[keypoints_of_interest[i + 1].value].z],
            [landmarks[keypoints_of_interest[i + 2].value].x, landmarks[keypoints_of_interest[i + 2].value].y, landmarks[keypoints_of_interest[i + 2].value].z]
        )
        angles.append(angle)

    # Display the angles next to the keypoints
    for i, keypoint in enumerate(keypoints_of_interest[1:-1]):
        text = f"{int(angles[i])}°"
        x = int(landmarks[keypoint.value].x * image.shape[1]) + 10
        y = int(landmarks[keypoint.value].y * image.shape[0])
        # cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

    # Convert back to BGR for displaying with OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Save the image to disk
    cv2.imwrite(save_path, image)

    # Optionally display the image
    cv2.imshow('Pose with Angles', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Close the pose detector
    pose.close()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python display_angle.py <path_to_image> <save_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    save_path = sys.argv[2]

    # Define the keypoints of interest
    keypoints_of_interest = [
        mp_pose.PoseLandmark.RIGHT_HIP, 
        mp_pose.PoseLandmark.RIGHT_KNEE, 
        mp_pose.PoseLandmark.RIGHT_ANKLE
    ]
    
    process_image(image_path, keypoints_of_interest, save_path)

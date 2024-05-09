import cv2
import mediapipe as mp
import sys
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_3d_vertical_angle(landmarkA, landmarkB):
    """ Calculate the 3D vertical angle between two points. """
    vectorAB = np.array([landmarkB['x'] - landmarkA['x'], landmarkB['y'] - landmarkA['y'], landmarkB['z'] - landmarkA['z']])
    verticalVector = np.array([0, -1, 0])
    dotProduct = np.dot(vectorAB, verticalVector)
    magnitudeAB = np.linalg.norm(vectorAB)
    cosineAngle = dotProduct / magnitudeAB
    angleRadians = np.arccos(np.clip(cosineAngle, -1.0, 1.0))
    angleDegrees = np.round(np.degrees(angleRadians))
    return angleDegrees

def calculate_midpoint(landmarkA, landmarkB):
    """ Calculate the midpoint between two landmarks. """
    midpoint = {
        'x': (landmarkA.x + landmarkB.x) / 2,
        'y': (landmarkA.y + landmarkB.y) / 2,
        'z': (landmarkA.z + landmarkB.z) / 2
    }
    return midpoint

def process_image(image_path):
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
    hips_midpoint = calculate_midpoint(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])
    shoulders_midpoint = calculate_midpoint(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])

    # Convert midpoints for drawing
    image_height, image_width, _ = image.shape
    hips_point = (int(hips_midpoint['x'] * image_width), int(hips_midpoint['y'] * image_height))
    shoulders_point = (int(shoulders_midpoint['x'] * image_width), int(shoulders_midpoint['y'] * image_height))
    vertical_point = (hips_point[0], hips_point[1] - 50)  # 50 pixels up for visualization

    # Draw the trunk vector
    cv2.line(image, hips_point, shoulders_point, (255, 0, 0), 2)
    # cv2.putText(image, "Trunk", shoulders_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Draw the vertical vector
    cv2.line(image, hips_point, vertical_point, (0, 255, 0), 2)
    # cv2.putText(image, "Vertical", vertical_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Calculate and display the trunk vertical angle
    trunk_vertical_angle = calculate_3d_vertical_angle(hips_midpoint, shoulders_midpoint)
    angle_text = f"{trunk_vertical_angle}"
    cv2.putText(image, angle_text, (hips_point[0] + 10, hips_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Show the image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('Pose with Vectors', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python display_angle.py <path_to_image>")
        sys.exit(1)
    image_path = sys.argv[1]
    process_image(image_path)

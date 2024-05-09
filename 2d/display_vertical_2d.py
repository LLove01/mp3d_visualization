import cv2
import mediapipe as mp
import sys
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_2d_vertical_angle(landmarkA, landmarkB):
    """Calculate the 2D vertical angle using the x and y coordinates."""
    vectorAB = np.array([landmarkB['x'] - landmarkA['x'], landmarkB['y'] - landmarkA['y']])
    verticalVector = np.array([0, -1])  # Upward direction in a typical 2D coordinate system
    dotProduct = np.dot(vectorAB, verticalVector)
    magnitudeAB = np.linalg.norm(vectorAB)
    cosineAngle = dotProduct / magnitudeAB
    angleRadians = np.arccos(np.clip(cosineAngle, -1.0, 1.0))
    angleDegrees = np.round(np.degrees(angleRadians))
    return angleDegrees

def calculate_midpoint_2d(landmarkA, landmarkB):
    """Calculate the midpoint between two landmarks using x and y coordinates."""
    midpoint = {
        'x': (landmarkA.x + landmarkB.x) / 2,
        'y': (landmarkA.y + landmarkB.y) / 2
    }
    return midpoint

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

    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    results = pose.process(image)
    if not results.pose_landmarks:
        print("No pose landmarks detected.")
        return  # Exiting if no landmarks detected

    landmarks = results.pose_landmarks.landmark
    hips_midpoint = calculate_midpoint_2d(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])
    shoulders_midpoint = calculate_midpoint_2d(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])

    # Convert midpoints for drawing
    image_height, image_width, _ = image.shape
    hips_point = (int(hips_midpoint['x'] * image_width), int(hips_midpoint['y'] * image_height))
    shoulders_point = (int(shoulders_midpoint['x'] * image_width), int(shoulders_midpoint['y'] * image_height))
    vertical_point = (hips_point[0], shoulders_point[1])  # Top of the image for visualization

    # Draw the trunk vector
    cv2.line(image, hips_point, shoulders_point, (255, 0, 0), 2)

    # Draw the vertical reference
    cv2.line(image, hips_point, vertical_point, (0, 255, 0), 2)

    # Calculate and display the trunk vertical angle
    trunk_vertical_angle = calculate_2d_angle(shoulders_point, hips_point, vertical_point)
    angle_text = f"{trunk_vertical_angle:.1f}"
    cv2.putText(image, angle_text, (hips_point[0] + 10, hips_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Show the image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('Pose with Vectors', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the image
    save_path = image_path.rsplit('.', 1)[0] + "_annotated.jpg"
    cv2.imwrite(save_path, image)
    print(f"Annotated image saved to {save_path}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python display_vertical_2d.py <path_to_image>")
        sys.exit(1)
    image_path = sys.argv[1]
    process_image(image_path)

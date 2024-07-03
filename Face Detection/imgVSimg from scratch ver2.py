from PIL import Image
import numpy as np
from skimage.feature import hog
from skimage.transform import resize
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import cv2


def recognize(img1_path, img2_path):
    def load_image(filepath):
        image = Image.open(filepath).convert('L')  # Convert to grayscale
        return np.array(image)

    def detect_face(image):
        # Apply Canny edge detector
        edges = cv2.Canny(image, 100, 200)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Assume the largest contour is the face
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)

        return x, y, w, h  # Return the coordinates and size of the detected face

    def extract_features(image, x, y, w, h):
        face_region = image[y:y + h, x:x + w]
        resized_face = resize(face_region, (128, 128))  # Resize to a standard size
        hog_features, hog_image = hog(resized_face, pixels_per_cell=(16, 16),
                                      cells_per_block=(2, 2), visualize=True)
        return hog_features

    def compare_features(features1, features2, threshold=0.3):
        similarity = cosine(features1, features2)
        return similarity < threshold

    # Load images
    img1 = load_image(img1_path)
    img2 = load_image(img2_path)

    # Detect faces
    x1, y1, w1, h1 = detect_face(img1)
    x2, y2, w2, h2 = detect_face(img2)

    # Extract features
    features1 = extract_features(img1, x1, y1, w1, h1)
    features2 = extract_features(img2, x2, y2, w2, h2)

    # Compare features
    return compare_features(features1, features2)


# Example usage
img1_path = "C:\\Users\\Abdul\\OneDrive\\Pictures\\th (1).jpeg"
img2_path = "C:\\Users\\Abdul\\OneDrive\\Pictures\\th.jpeg"

is_match = recognize(img1_path, img2_path)

if is_match:
    print("The faces match.")
else:
    print("The faces don't match.")

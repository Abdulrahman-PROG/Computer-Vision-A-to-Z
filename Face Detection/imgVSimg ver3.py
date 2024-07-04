import numpy as np
import cv2
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def read_image(filepath):
    try:
        img = cv2.imread(filepath)
        if img is None:
            raise FileNotFoundError(f"Failed to read image at {filepath}")
        return img
    except FileNotFoundError:
        raise FileNotFoundError(f"Failed to read image at {filepath}")


def bgr_to_rgb(img):
    return img[:, :, ::-1]


def detect_faces(img_rgb):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        raise ValueError("No faces detected in the image")

    x, y, w, h = faces[0]
    return x, y, x + w, y + h


def encode_image(face_img):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(face_img, None)
    if descriptors is None:
        raise ValueError("No keypoints detected in the face image")
    return keypoints, descriptors


def compare_encodings(descriptors1, descriptors2, threshold):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    if len(matches) == 0:
        return False
    distances = [m.distance for m in matches]
    average_distance = np.mean(distances)
    return average_distance < threshold


def recognize(img1_path, img2_path, threshold):
    try:
        img1 = read_image(img1_path)
        img2 = read_image(img2_path)

        img1_rgb = bgr_to_rgb(img1)
        img2_rgb = bgr_to_rgb(img2)

        x1, y1, x2, y2 = detect_faces(img1_rgb)
        face_img1 = img1_rgb[y1:y2, x1:x2]

        x1, y1, x2, y2 = detect_faces(img2_rgb)
        face_img2 = img2_rgb[y1:y2, x1:x2]

        face_img1_gray = cv2.cvtColor(face_img1, cv2.COLOR_BGR2GRAY)
        face_img2_gray = cv2.cvtColor(face_img2, cv2.COLOR_BGR2GRAY)

        keypoints1, descriptors1 = encode_image(face_img1_gray)
        keypoints2, descriptors2 = encode_image(face_img2_gray)

        # Debugging: Visualize keypoints
        img1_with_keypoints = cv2.drawKeypoints(face_img1_gray, keypoints1, None, color=(0, 255, 0), flags=0)
        img2_with_keypoints = cv2.drawKeypoints(face_img2_gray, keypoints2, None, color=(0, 255, 0), flags=0)

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(img1_with_keypoints, cmap='gray')
        plt.title('Face 1 Keypoints')

        plt.subplot(1, 2, 2)
        plt.imshow(img2_with_keypoints, cmap='gray')
        plt.title('Face 2 Keypoints')

        plt.show()

        match = compare_encodings(descriptors1, descriptors2, threshold)

        # Print whether the faces match or not
        if match:
            print(f"Faces in '{img1_path}' and '{img2_path}' match.")
        else:
            print(f"Faces in '{img1_path}' and '{img2_path}' do not match.")

        return match

    except Exception as e:
        print(f"Error: {e}")
        return False


# Example validation set
validation_pairs = [
    ("C:\\Users\\Abdul\\OneDrive\\Pictures\\th (1).jpeg", "C:\\Users\\Abdul\\OneDrive\\Pictures\\th.jpeg", True),
    ("C:\\Users\\Abdul\\OneDrive\\Pictures\\th (1).jpeg", "C:\\Users\\Abdul\\OneDrive\\Pictures\\1.jpg",
     False),
    # Add more pairs as needed
]

thresholds = np.arange(55, 63, 1)  # Adjust the range based on initial results
best_threshold = None
best_accuracy = 0

for threshold in thresholds:
    predictions = []
    actuals = []

    for img1_path, img2_path, is_match in validation_pairs:
        result = recognize(img1_path, img2_path, threshold)
        predictions.append(result)
        actuals.append(is_match)

    accuracy = accuracy_score(actuals, predictions)
    print(f"Threshold: {threshold}, Accuracy: {accuracy}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = threshold

print(f"Best threshold: {best_threshold} with accuracy: {best_accuracy}")

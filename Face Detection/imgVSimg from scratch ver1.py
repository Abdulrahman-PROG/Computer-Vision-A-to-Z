from PIL import Image
import numpy as np

def recognize(img1_path, img2_path):

    def load_image(filepath):
        image = Image.open(filepath).convert('L')  # Convert to grayscale
        return np.array(image)

    def detect_face(image, template):
        h, w = template.shape
        max_corr = -1
        best_x = best_y = 0
        for y in range(image.shape[0] - h + 1):
            for x in range(image.shape[1] - w + 1):
                region = image[y:y+h, x:x+w]
                corr = np.sum(region * template)
                if corr > max_corr:
                    max_corr = corr
                    best_x, best_y = x, y
        return best_x, best_y  # Return the coordinates of the top-left corner of the detected face

    def extract_features(image, x, y, size=50):
        face_region = image[y:y+size, x:x+size]
        mean = np.mean(face_region)
        std = np.std(face_region)
        return mean, std

    def compare_features(features1, features2, threshold=10):
        mean_diff = abs(features1[0] - features2[0])
        std_diff = abs(features1[1] - features2[1])
        return mean_diff < threshold and std_diff < threshold

    # Load images
    img1 = load_image(img1_path)
    img2 = load_image(img2_path)

    # Dummy template for face detection (in real scenarios, use a proper face template)
    face_template = np.ones((50, 50)) * 255

    # Detect faces
    x1, y1 = detect_face(img1, face_template)
    x2, y2 = detect_face(img2, face_template)

    # Extract features
    features1 = extract_features(img1, x1, y1)
    features2 = extract_features(img2, x2, y2)

    # Compare features
    return compare_features(features1, features2)

# Example usage
img1_path = "C:\\Users\\Abdul\\OneDrive\\Pictures\\th (1).jpeg"
img2_path = "C:\\Users\\Abdul\\OneDrive\\Pictures\\th.jpeg"
img3_path = "C:\\Users\\Abdul\\OneDrive\\Pictures\\th.jpeg"

is_match = recognize(img1_path, img2_path)

if is_match:
    print("The faces match.")
else:
    print("The faces don't match.")

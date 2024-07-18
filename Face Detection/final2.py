import face_recognition
import os

def load_and_encode(image_path):
    if not os.path.isfile(image_path):
        print(f"File not found: {image_path}")
        return None
    try:
        img = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(img)
        if len(encodings) == 0:
            print(f"No faces found in image: {image_path}")
            return None
        return encodings[0]
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def recognize(img1_path, img2_path, tolerance=0.6):
    encoding_img1 = load_and_encode(img1_path)
    encoding_img2 = load_and_encode(img2_path)

    if encoding_img1 is None or encoding_img2 is None:
        print("One or both images do not contain valid face encodings.")
        return False

    # print(f"Comparing encodings with tolerance {tolerance}:")
    # print(f"Encoding 1: {encoding_img1}")
    # print(f"Encoding 2: {encoding_img2}")

    match = face_recognition.compare_faces([encoding_img1], encoding_img2, tolerance=tolerance)[0]
    return match

img1_path = r"C:\Users\Abdul\OneDrive\Pictures\th (2).jpeg"
img2_path = r"C:\Users\Abdul\OneDrive\Pictures\1.jpg"


is_match = recognize(img1_path, img2_path, tolerance=0.6)

if is_match:
    print("The faces match.")
else:
    print("The faces don't match.")

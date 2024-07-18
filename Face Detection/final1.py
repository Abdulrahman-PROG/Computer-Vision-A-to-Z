import face_recognition
import os
import concurrent.futures

def load_and_encode(image_path):
    if not os.path.isfile(image_path):
        print(f"File not found: {image_path}")
        return None
    img = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(img)
    if len(encodings) == 0:
        print(f"No faces found in image: {image_path}")
        return None
    return encodings[0]

def recognize(img1_path, img2_path):
    # Use concurrent futures for parallel loading and encoding
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_img1 = executor.submit(load_and_encode, img1_path)
        future_img2 = executor.submit(load_and_encode, img2_path)

        encoding_img1 = future_img1.result()
        encoding_img2 = future_img2.result()

    if encoding_img1 is None or encoding_img2 is None:
        return False

    # Compare
    match = face_recognition.compare_faces([encoding_img1], encoding_img2)[0]
    return match

img1_path = r"C:\Users\Abdul\OneDrive\Pictures\th (2).jpeg"
img2_path = r"C:\Users\Abdul\OneDrive\Pictures\th (1).jpeg"

is_match = recognize(img1_path, img2_path)

if is_match:
    print("The faces match.")
else:
    print("The faces don't match.")

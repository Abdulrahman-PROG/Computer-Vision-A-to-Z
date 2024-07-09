import face_recognition
import os
import cv2


def recognize_image_against_folder(img_path, folder_path):
    if not os.path.exists(folder_path):
        print(f"Folder path does not exist: {folder_path}")
        return None

    # Load the image using OpenCV
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load image: {img_path}")
        return None

    # Convert the image from BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_encodings = face_recognition.face_encodings(img_rgb)
    if len(img_encodings) == 0:
        print(f"No faces found in the image: {img_path}")
        return None

    img_encoding = img_encodings[0]
    folder_images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                     f.lower().endswith(('jpg', 'jpeg', 'png'))]

    if not folder_images:
        print(f"No images found in the folder: {folder_path}")
        return None

    for folder_img in folder_images:
        folder_img_cv = cv2.imread(folder_img)
        if folder_img_cv is None:
            print(f"Failed to load image: {folder_img}")
            continue

        folder_img_rgb = cv2.cvtColor(folder_img_cv, cv2.COLOR_BGR2RGB)
        folder_img_encodings = face_recognition.face_encodings(folder_img_rgb)
        if len(folder_img_encodings) == 0:
            print(f"No faces found in the image: {folder_img}")
            continue

        match = face_recognition.compare_faces([img_encoding], folder_img_encodings[0])[0]
        if match:
            print(f"Match found: {folder_img}")
            return folder_img

    print("No match found.")
    return None


# Test the function
img_path = r"C:\Users\Abdul\OneDrive\Pictures\th (2).jpeg"
folder_path = r"C:\Users\Abdul\OneDrive\Pictures\db"
matching_image = recognize_image_against_folder(img_path, folder_path)
print(matching_image)

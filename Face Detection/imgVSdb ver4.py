import face_recognition
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

def recognize_image_against_folder(img_path, folder_path):
    if not os.path.exists(folder_path):
        logging.error(f"Folder path does not exist: {folder_path}")
        return False

    try:
        img_encodings = face_recognition.face_encodings(face_recognition.load_image_file(img_path))
    except Exception as e:
        logging.error(f"Error loading image {img_path}: {e}")
        return False

    if len(img_encodings) == 0:
        logging.info(f"No faces found in the image: {img_path}")
        return False

    img_encoding = img_encodings[0]
    folder_images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('jpg', 'jpeg', 'png'))]

    if not folder_images:
        logging.info(f"No images found in the folder: {folder_path}")
        return False

    for folder_img in folder_images:
        try:
            folder_img_encodings = face_recognition.face_encodings(face_recognition.load_image_file(folder_img))
        except Exception as e:
            logging.error(f"Error loading image {folder_img}: {e}")
            continue

        if len(folder_img_encodings) == 0:
            logging.info(f"No faces found in the image: {folder_img}")
            continue

        match = face_recognition.compare_faces([img_encoding], folder_img_encodings[0])[0]
        if match:
            logging.info(f"Match found: {folder_img}")
            return True

    logging.info("No matches found.")
    return False

# Test the function
img_path = r"C:\Users\Abdul\OneDrive\Pictures\th (2).jpeg"
folder_path = r"C:\Users\Abdul\OneDrive\Pictures\db"
is_matching_image = recognize_image_against_folder(img_path, folder_path)
logging.info(f"Is there a matching image? {is_matching_image}")

import face_recognition
import os



def recognize_image_against_folder(img_path, folder_path):
    if not os.path.exists(folder_path):
        print(f"Folder path does not exist: {folder_path}")
        return

    img_encodings = face_recognition.face_encodings(face_recognition.load_image_file(img_path))
    if len(img_encodings) == 0:
        print(f"No faces found in the image: {img_path}")
        return

    img_encoding = img_encodings[0]
    folder_images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                     f.lower().endswith(('jpg', 'jpeg', 'png'))]

    if not folder_images:
        print(f"No images found in the folder: {folder_path}")
        return

    for folder_img in folder_images:
        folder_img_encodings = face_recognition.face_encodings(face_recognition.load_image_file(folder_img))
        if len(folder_img_encodings) == 0:
            print(f"No faces found in the image: {folder_img}")
            continue
        match = face_recognition.compare_faces([img_encoding], folder_img_encodings[0])[0]



    return match




img_path = r"C:\Users\Abdul\OneDrive\Pictures\th (2).jpeg"
folder_path = r"C:\Users\Abdul\OneDrive\Pictures\db"
matching_image = recognize_image_against_folder(img_path, folder_path)
print(matching_image)

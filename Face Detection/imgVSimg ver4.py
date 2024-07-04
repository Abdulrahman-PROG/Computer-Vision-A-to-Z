import os
import PIL.Image
import dlib
import numpy as np

# Assuming the shape predictor file is in the same directory as your script
shape_predictor_68_path = "D:\\face-attendance-system-master\\.venv\\Lib\\site-packages\\face_recognition_models\\models\\shape_predictor_68_face_landmarks.dat"
shape_predictor_5_path = "D:\\face-attendance-system-master\\.venv\\Lib\\site-packages\\face_recognition_models\models\\shape_predictor_5_face_landmarks.dat"

# Check if the files exist
if not os.path.isfile(shape_predictor_68_path):
    raise FileNotFoundError(f"Cannot find shape predictor file: {shape_predictor_68_path}")

if not os.path.isfile(shape_predictor_5_path):
    raise FileNotFoundError(f"Cannot find shape predictor file: {shape_predictor_5_path}")

# Initialize the dlib face detector and facial landmarks predictors
detector = dlib.get_frontal_face_detector()
shape_predictor_68_point = dlib.shape_predictor(shape_predictor_68_path)
shape_predictor_5_point = dlib.shape_predictor(shape_predictor_5_path)

face_encoder = dlib.face_recognition_model_v1("D:\\face-attendance-system-master\\.venv\\Lib\\site-packages\\face_recognition_models\\models\\dlib_face_recognition_resnet_model_v1.dat")


def load_image_file(file, mode='RGB'):
    im = PIL.Image.open(file)
    if mode:
        im = im.convert(mode)
    return np.array(im)


def _raw_face_locations(img, number_of_times_to_upsample=1, model="hog"):
    if model == "cnn":
        return detector(img, number_of_times_to_upsample)
    else:
        return detector(img, number_of_times_to_upsample)


def _raw_face_landmarks(face_image, face_locations=None, model="large"):
    if face_locations is None:
        face_locations = _raw_face_locations(face_image)
    else:
        face_locations = [dlib.rectangle(*face_loc) for face_loc in face_locations]

    if model == "small":
        pose_predictor = shape_predictor_5_point
    else:
        pose_predictor = shape_predictor_68_point

    return [pose_predictor(face_image, face_loc) for face_loc in face_locations]


def face_encodings(face_image, known_face_locations=None, num_jitters=1, model="small"):
    raw_landmarks = _raw_face_landmarks(face_image, known_face_locations, model)
    return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_landmarks]


def compare_faces(known_encodings, candidate_encoding, tolerance=0.6):
    return [np.linalg.norm(known_encoding - candidate_encoding) <= tolerance for known_encoding in known_encodings]


def recognize(img1, img2):
    """
    Recognize whether the faces in img1 and img2 are the same.

    Parameters:
    - img1: First image containing a face
    - img2: Second image containing a face

    Returns:
    - True if the faces match, False otherwise
    """
    # Get face encodings for img1
    encodings_img1 = face_encodings(img1)
    if len(encodings_img1) == 0:
        return False

    # Get face encodings for img2
    encodings_img2 = face_encodings(img2)
    if len(encodings_img2) == 0:
        return False

    # Compare
    match = compare_faces([encodings_img1[0]], encodings_img2[0])[0]

    return match


# Example usage:
img1 = load_image_file("C:\\Users\\Abdul\\OneDrive\\Pictures\\th (2).jpeg")
img2 = load_image_file("C:\\Users\\Abdul\\OneDrive\\Pictures\\th.jpeg")
img3 = load_image_file("C:\\Users\\Abdul\\OneDrive\\Pictures\\1.jpg")

result = recognize(img1, img3)
print("Face recognition result:", result)

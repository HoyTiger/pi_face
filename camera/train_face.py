import math
import os
import os.path
import pickle

import face_recognition
import numpy as np
from PIL import Image, ImageDraw
from PIL import ImageFont
from face_recognition.face_recognition_cli import image_files_in_folder
from sklearn import neighbors
import cv2
from cv2 import face


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
font = ImageFont.truetype('./static/simsun.ttc')

label2name = {}

recognizer = cv2.face.LBPHFaceRecognizer_create()


def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    X = []
    y = []

    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue
        path = os.path.join(train_dir, class_dir)
        for img_path in os.listdir(path):
            PIL_img = Image.open(os.path.join(path, img_path)).convert('L')  # convert it to grayscale
            img_numpy = np.array(PIL_img, 'uint8')

            X.append(img_numpy)
            y.append(int(class_dir))

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    clf = recognizer.train(X, np.array(y))
    recognizer.write('trainer.yml')  # recognizer.save() worked on Mac, but not on Pi

    # # Save the trained KNN classifier
    # if model_save_path is not None:
    #     with open(model_save_path, 'wb') as f:
    #         pickle.dump(clf, f)

    return clf


def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.6):
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
            zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


def show_prediction_labels_on_image(img_path, predictions):
    """
    Shows the face recognition results visually.
    :param img_path: path to image to be recognized
    :param predictions: results of the predict function
    :return:
    """
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        name = name

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255), font=font)

    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    pil_image.show()


if __name__ == "__main__":
    # STEP 1: Train the KNN classifier and save it to disk
    # Once the model is trained and saved, you can skip this step next time.
    # print("Training KNN classifier...")
    # classifier = train("static/datasets", model_save_path="trained_knn_model.clf", n_neighbors=2)
    # print("Training complete!")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    img = cv2.imread('1.jpeg')
    recognizer.read('trainer.yml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_recognition.face_locations(img)

    for (top, right, bottom, left) in faces:
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

        id, confidence = recognizer.predict(gray[top:bottom, left:right])
        print(id, confidence)


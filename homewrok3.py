import cv2
import face_recognition
import mediapipe as mp
import numpy as np
from keras.src.saving import load_model

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

EMOTION_LABELS = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
                  4: 'sad', 5: 'surprise', 6: 'neutral'}
emotion_model = load_model('emotion_model.hdf5', compile=False)


my_img = face_recognition.load_image_file("1.jpg")
img_encoder = face_recognition.face_encodings(my_img)[0]

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
video_capture = cv2.VideoCapture(0)
name, surname, color = "UNKNOWN", "", (255, 255, 0)


def find_face(video):
    global name, surname, color
    gray_img = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=8, minSize=(64, 64))
    face_roi = None

    for (x, y, w, h) in faces:
        face_img = video[y:y + h, x:x + w]
        rgb_face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_face_img)

        if face_encodings:
            match = face_recognition.compare_faces([img_encoder], face_encodings[0])
            if match[0]:
                name = "Damir"
                surname = "Rakhmatullin"
                color = (0, 0, 255)
            else:
                name = "UNKNOWN"
                surname = ""
        else:
            name = "UNKNOWN"
            surname = ""

        cv2.rectangle(video, (x, y), (x + w, y + h), color, 4)
        face_roi = gray_img[y:y + h, x:x + w]

    return faces, face_roi


def count_fingers(hand_landmarks, handedness):
    finger_count = 0
    tip_ids = [4, 8, 12, 16, 20]

    if (handedness == "Right" and hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x) or \
            (handedness == "Left" and hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x):
        finger_count += 1

    for tip in tip_ids[1:]:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            finger_count += 1

    return finger_count


def predict_emotion(face_roi):
    if face_roi is None:
        return "UNKNOWN"
    emotion_img = cv2.resize(face_roi, (64, 64))
    emotion_img = emotion_img / 255.0
    emotion_pred = emotion_model.predict(np.expand_dims(emotion_img, axis=0))
    return EMOTION_LABELS.get(np.argmax(emotion_pred))


while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    faces, face_roi = find_face(frame)
    result = hands.process(frame)

    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            mpDraw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            num_fingers = count_fingers(hand_landmarks, handedness)

            if num_fingers == 1:
                cv2.putText(frame, name, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif num_fingers == 2:
                cv2.putText(frame, name + " " + surname, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif num_fingers == 3:
                emotion = predict_emotion(face_roi)
                if name == "UNKNOWN":
                    cv2.putText(frame, name, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, f"Emotion: {emotion}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Face & Hand Recognition", frame)
    if cv2.waitKey(1) & 0xFF in [ord("q"), ord("й")]:
        break

video_capture.release()
cv2.destroyAllWindows()

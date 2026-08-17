import cv2
import pickle
import os
import csv
import numpy as np
from datetime import datetime
from collections import deque

YUNET_MODEL = "face_detection_yunet.onnx"
SFACE_MODEL = "face_recognition_sface_2021dec.onnx"


def normalize_lighting(bgr_face):
    lab = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


if not os.path.isfile("encodings.pickle"):
    print("'encodings.pickle' pawa jায়নি! Age '2_train_encodings.py' run koro.")
    exit()

with open("encodings.pickle", "rb") as f:
    data = pickle.load(f)
names = data["names"]
prototypes = data["prototypes"]

if not os.path.isfile(YUNET_MODEL) or not os.path.isfile(SFACE_MODEL):
    print("Model file khuje pawa jায়নি! Age '0_check_setup.py' run koro.")
    exit()

detector = cv2.FaceDetectorYN.create(YUNET_MODEL, "", (320, 320), 0.6, 0.3, 5000)
recognizer = cv2.FaceRecognizerSF.create(SFACE_MODEL, "")

MATCH_THRESHOLD = 0.40
MARGIN = 0.05

# Voting window: shobcheye boro (primary/kache thaka) face'r shesh koyekbar
# identification track kora hocche. Ekta random bhul frame'r jonno attendance
# vul hobe na -- consistently same naam ashle tobei mark hobe (koyek second e).
VOTE_WINDOW = 10
CONFIRM_COUNT = 6
recent_votes = deque(maxlen=VOTE_WINDOW)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera open kora jায়নি!")
    exit()

marked_names = set()
print("System is Ready. Looking for faces...")

if not os.path.isfile('Attendance.csv'):
    with open('Attendance.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Time", "Date"])

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(frame)

        primary_name_this_frame = None

        if faces is not None and len(faces) > 0:
            # Attendance shudhu shobcheye boro (samner) face'r jonno count hobe,
            # kintu shob face e box/label dekhano hobe (visual feedback'er jonno)
            primary_face = max(faces, key=lambda f: f[2] * f[3])

            for face in faces:
                aligned = recognizer.alignCrop(frame, face)
                aligned = normalize_lighting(aligned)
                feature = recognizer.feature(aligned)

                best_name = "Unknown"
                best_score = -1.0
                second_score = -1.0

                for i, proto in enumerate(prototypes):
                    score = recognizer.match(feature, proto, cv2.FaceRecognizerSF_FR_COSINE)
                    if score > best_score:
                        second_score = best_score
                        best_score = score
                        best_name = names[i]
                    elif score > second_score:
                        second_score = score

                if len(prototypes) >= 2:
                    confident = (best_score >= MATCH_THRESHOLD) and (best_score - second_score >= MARGIN)
                else:
                    confident = best_score >= MATCH_THRESHOLD

                box = face[0:4].astype(int)
                x, y, bw, bh = box
                is_primary = np.array_equal(face, primary_face)

                if confident:
                    label = best_name
                    color = (0, 255, 0)
                    if is_primary:
                        primary_name_this_frame = best_name
                    if best_name in marked_names:
                        label += " [Present]"
                else:
                    label = "Unknown"
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        recent_votes.append(primary_name_this_frame)

        if len(recent_votes) == VOTE_WINDOW:
            vote_counts = {}
            for v in recent_votes:
                if v is not None:
                    vote_counts[v] = vote_counts.get(v, 0) + 1

            if vote_counts:
                top_name = max(vote_counts, key=vote_counts.get)
                top_count = vote_counts[top_name]

                if top_count >= CONFIRM_COUNT and top_name not in marked_names:
                    now = datetime.now()
                    time_str = now.strftime("%H:%M:%S")
                    date_str = now.strftime("%d-%m-%Y")

                    with open('Attendance.csv', 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([top_name, time_str, date_str])

                    marked_names.add(top_name)
                    print(f"Attendance recorded for {top_name} at {time_str}")
                    recent_votes.clear()

        cv2.imshow('Smart Attendance System', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

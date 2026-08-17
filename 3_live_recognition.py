import cv2
import pickle
import os
import numpy as np

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

# SFace er official cosine threshold ~0.363; ektu strict rakha hoyeche extra safety'r jonno
MATCH_THRESHOLD = 0.40
MARGIN = 0.05  # best match, dwitiyo-shera match theke koto agiye thakte hobe

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera open kora jায়নি!")
    exit()

print("System is Ready. Looking for faces...")
print("(Ei script e score dekhay -- threshold tune korte help hobe)")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(frame)

        if faces is not None:
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

                if confident:
                    label = f"{best_name} ({best_score:.2f})"
                    color = (0, 255, 0)
                else:
                    label = f"Unknown ({best_score:.2f})"
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow('Live Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()

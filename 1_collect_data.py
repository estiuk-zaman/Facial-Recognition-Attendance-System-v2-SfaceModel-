import cv2
import numpy as np
import os
import time

YUNET_MODEL = "face_detection_yunet.onnx"
SFACE_MODEL = "face_recognition_sface_2021dec.onnx"


def normalize_lighting(bgr_face):
    """Low-light / high-light er effect kombate CLAHE apply kora hocche.
    Ei function collection ebong recognition, DUJAYGAY e same vabe use hote hobe."""
    lab = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


if not os.path.isfile(YUNET_MODEL) or not os.path.isfile(SFACE_MODEL):
    print("Model file khuje pawa jায়নি! Age '0_check_setup.py' run kore confirm koro.")
    exit()

# ---------------- User Info ----------------
user_name = input("Enter your name (e.g., Estiuk): ").strip()
save_path = f"dataset/{user_name}"
os.makedirs(save_path, exist_ok=True)

# ---------------- Load Models ----------------
detector = cv2.FaceDetectorYN.create(YUNET_MODEL, "", (320, 320), 0.6, 0.3, 5000)
recognizer = cv2.FaceRecognizerSF.create(SFACE_MODEL, "")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera open kora jায়নি!")
    exit()

# Existing images koyta ache check kore continue kora hocche (overwrite na kore)
existing_files = [f for f in os.listdir(save_path) if f.lower().endswith('.jpg')]
existing_numbers = []
for f in existing_files:
    try:
        existing_numbers.append(int(os.path.splitext(f)[0]))
    except ValueError:
        continue
count = max(existing_numbers) if existing_numbers else 0

# 5-stage guided capture: pose + distance variety ekbar e cover kora hocche
stages = [
    ("Shamne shohoj vabe takao", 25),
    ("Matha AKTU BAME ghurao", 15),
    ("Matha AKTU DANE ghurao", 15),
    ("EKTU KACHE eso camera'r", 15),
    ("EKTU DURE jao camera theke", 15),
]

print(f"\nAge theke {count} ta image ache '{user_name}' er jonno.\n")
print("IMPORTANT: Best accuracy'r jonno, EI PURA SCRIPT TA 2-3 BAR run koro,")
print("protibar ALADA lighting e daঁড়িয়ে:")
print("   1st bar : shadharon (normal) room light")
print("   2nd bar : alo kome (low light)")
print("   3rd bar : beshi alo / janalar dhare (bright light)")
print("Protibar notun images purono gula overwrite na kore JOG hobe.\n")
input("Ready hole ENTER chapo...")

quit_requested = False

try:
    for stage_text, stage_target in stages:
        if quit_requested:
            break

        stage_count = 0
        print(f"\n>>> {stage_text}  ({stage_target} images)")

        # 2 sec preparation somoy — pose thik korar jonno
        prep_start = time.time()
        while time.time() - prep_start < 2:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.putText(frame, "READY: " + stage_text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow('Data Collection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                quit_requested = True
                break

        while stage_count < stage_target and not quit_requested:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            detector.setInputSize((w, h))
            _, faces = detector.detect(frame)

            if faces is not None and len(faces) > 0:
                # Shobcheye boro (kache thaka) face take primary dhora hocche
                face = max(faces, key=lambda f: f[2] * f[3])

                # alignCrop: landmark diye face ke rotate/scale kore standard
                # position e ante hoy -- eta distance/angle variance onek kome dey
                aligned = recognizer.alignCrop(frame, face)
                aligned = normalize_lighting(aligned)

                count += 1
                stage_count += 1
                cv2.imwrite(f"{save_path}/{count}.jpg", aligned)

                box = face[0:4].astype(int)
                x, y, bw, bh = box
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

            cv2.putText(frame, f"{stage_text}: {stage_count}/{stage_target}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Total saved: {count}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow('Data Collection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                quit_requested = True
                break

finally:
    cap.release()
    cv2.destroyAllWindows()

print(f"\nData Collection Successful! Total {count} images saved for {user_name}.")
print("Aro lighting condition e collect korte, script ta abar SAME naam diye run koro.")

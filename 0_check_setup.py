import os

print("=" * 55)
print("Environment Check")
print("=" * 55)

all_ok = True
cv2 = None

# 1. OpenCV import check
try:
    import cv2
    print(f"[OK] OpenCV imported. Version: {cv2.__version__}")
except ImportError:
    print("[FAIL] OpenCV import kora jায়নি.")
    print("       Fix: pip install opencv-contrib-python")
    all_ok = False

if cv2 is not None:
    # 2. FaceDetectorYN check
    if hasattr(cv2, 'FaceDetectorYN'):
        print("[OK] cv2.FaceDetectorYN available")
    else:
        print("[FAIL] cv2.FaceDetectorYN NEI.")
        print("       Fix: pip uninstall opencv-python -y")
        print("            pip install opencv-contrib-python")
        all_ok = False

    # 3. FaceRecognizerSF check
    if hasattr(cv2, 'FaceRecognizerSF'):
        print("[OK] cv2.FaceRecognizerSF available")
    else:
        print("[FAIL] cv2.FaceRecognizerSF NEI.")
        print("       Fix: pip uninstall opencv-python -y")
        print("            pip install opencv-contrib-python")
        all_ok = False

# 4. Model files existence + size sanity check
YUNET_MODEL = "face_detection_yunet.onnx"
SFACE_MODEL = "face_recognition_sface_2021dec.onnx"

def check_model_file(path, expected_min_mb, label):
    global all_ok
    if not os.path.isfile(path):
        print(f"[FAIL] '{path}' ei folder e pawa jায়নি.")
        all_ok = False
        return
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb < expected_min_mb:
        print(f"[FAIL] '{path}' ache kintu shudhu {size_mb:.4f} MB -- eta broken/incomplete")
        print(f"       download (probably a Git LFS pointer file, real model na). Abar download koro.")
        all_ok = False
    else:
        print(f"[OK] '{label}' found ({size_mb:.1f} MB)")

check_model_file(YUNET_MODEL, 0.05, "YuNet detection model")
check_model_file(SFACE_MODEL, 1.0, "SFace recognition model")

# 5. Try actually loading the models (catches corrupt/incompatible files)
if cv2 is not None and hasattr(cv2, 'FaceDetectorYN') and os.path.isfile(YUNET_MODEL):
    try:
        _ = cv2.FaceDetectorYN.create(YUNET_MODEL, "", (320, 320))
        print("[OK] YuNet model successfully load hoyeche")
    except Exception as e:
        print(f"[FAIL] YuNet model load korte giye error: {e}")
        all_ok = False

if cv2 is not None and hasattr(cv2, 'FaceRecognizerSF') and os.path.isfile(SFACE_MODEL):
    try:
        _ = cv2.FaceRecognizerSF.create(SFACE_MODEL, "")
        print("[OK] SFace model successfully load hoyeche")
    except Exception as e:
        print(f"[FAIL] SFace model load korte giye error: {e}")
        all_ok = False

# 6. Camera check
if cv2 is not None:
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("[OK] Camera (index 0) open hocche")
            cap.release()
        else:
            print("[FAIL] Camera open kora jায়নি (index 0).")
            all_ok = False
    except Exception as e:
        print(f"[FAIL] Camera check e error: {e}")
        all_ok = False

print("=" * 55)
if all_ok:
    print("Sob thik ache! Ekhon '1_collect_data.py' run korte paro.")
else:
    print("Upore [FAIL] ja ja dekhacche, seigula age fix kore abar ei script run koro.")
print("=" * 55)

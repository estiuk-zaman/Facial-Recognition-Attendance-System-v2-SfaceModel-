# Facial Recognition Attendance System

A real-time, camera-based attendance system built with OpenCV — detects and recognizes faces live, then automatically logs attendance to a CSV file. Built and tuned to run smoothly on a low-spec, CPU-only laptop (no GPU required).

## Demo

*(add a screenshot or GIF of the live recognition window here)*

## Features

- Real-time face detection and recognition via webcam
- Lighting-robust: CLAHE normalization + face alignment before matching
- Multi-person support with a margin-based matching rule that prevents misidentifying one person as another
- Multi-frame voting before marking attendance, so a single bad frame can't cause a wrong entry
- Attendance automatically logged to `Attendance.csv` with name, time, and date
- Runs fully on CPU — no GPU required

## Why This Tech Stack

This project went through a few iterations before landing on the current approach:

| Attempt | Result |
|---|---|
| LBPH (OpenCV classical) | Worked, but accuracy dropped under different lighting and with multiple people |
| `face_recognition` (dlib) | Blocked by a broken dependency (`face_recognition_models`, unmaintained since 2020, incompatible with newer Python) |
| `insightface` (buffalo_l / buffalo_sc) | Accurate, but the model was too heavy — laggy on CPU-only hardware |
| **OpenCV's own YuNet + SFace** | Lightweight, built into OpenCV, fast on CPU, and accurate enough for real use |

The final pipeline uses `cv2.FaceDetectorYN` (YuNet) for detection and `cv2.FaceRecognizerSF` (SFace) for recognition — both part of OpenCV's own DNN-based face module, so no extra heavy dependencies are needed.

## How It Works

1. **Detect** — YuNet finds the face and 5 facial landmarks in each frame
2. **Align** — the face is warped into a standard pose using those landmarks (`alignCrop`)
3. **Normalize** — CLAHE evens out lighting differences before feature extraction
4. **Encode** — SFace converts the aligned face into a numeric embedding (a vector)
5. **Match** — the new embedding is compared (cosine similarity) against saved per-person prototypes; a match only counts if it clears both a similarity threshold *and* a margin over the next-best match, to avoid mixing up two people
6. **Confirm** — attendance is only marked once the same person is recognized consistently across several frames, not from a single frame

## Project Structure

```
Facial-Recognition-Attendance-System-v2/
├── 0_check_setup.py         # Verifies OpenCV, model files, and camera before anything else
├── 1_collect_data.py        # Captures aligned face images for a new person (multi-pose, multi-session)
├── 2_train_encodings.py     # Builds a face embedding "prototype" for each person
├── 3_live_recognition.py    # Live recognition preview, with match scores shown (for tuning)
├── 4_attendance_system.py   # Full attendance system with CSV logging
├── face_detection_yunet.onnx
├── face_recognition_sface_2021dec.onnx
├── dataset/                 # Collected face images, one folder per person
├── encodings.pickle         # Saved face prototypes (generated)
└── Attendance.csv           # Attendance log (generated)
```

## Requirements

- Python 3.x
- `opencv-contrib-python`
- `numpy`

```bash
pip install opencv-contrib-python numpy
```

## Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/estiuk-zaman/Facial-Recognition-Attendance-System-v2.git
   cd Facial-Recognition-Attendance-System-v2
   ```

2. Download the two model files from [OpenCV Zoo](https://github.com/opencv/opencv_zoo) — **use the browser "Download" button on the GitHub file page, not `curl`/`wget`**, since these files are Git-LFS tracked and a direct link only downloads a small pointer file:
   - `face_detection_yunet_*.onnx` from `models/face_detection_yunet`
   - `face_recognition_sface_2021dec.onnx` from `models/face_recognition_sface`

   Place both files in the project root.

3. Verify everything is ready:
   ```bash
   python 0_check_setup.py
   ```

## Usage

Run the scripts in order:

```bash
python 1_collect_data.py       # Enroll a new person (run 2-3 times, in different lighting)
python 2_train_encodings.py    # Build face prototypes from the collected data
python 3_live_recognition.py   # Test recognition live (shows match scores on screen)
python 4_attendance_system.py  # Run the full attendance system
```

For best accuracy, run `1_collect_data.py` for each person 2-3 times under different lighting conditions (normal, low light, bright light).

## Notes

- Tested on a low-spec, CPU-only laptop (Intel i3, 8GB RAM, no dedicated GPU) — runs smoothly in real time.
- `MATCH_THRESHOLD` and `MARGIN` (in `3_live_recognition.py` / `4_attendance_system.py`) can be tuned if you see too many "Unknown" results, or — less desirably — any mismatches.

## Acknowledgments

- [OpenCV Zoo](https://github.com/opencv/opencv_zoo) for the YuNet and SFace models

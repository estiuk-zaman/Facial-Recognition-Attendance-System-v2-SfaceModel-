import cv2
import os
import pickle
import numpy as np

SFACE_MODEL = "face_recognition_sface_2021dec.onnx"
dataset_path = "dataset"

if not os.path.isfile(SFACE_MODEL):
    print(f"'{SFACE_MODEL}' pawa jায়নি! Age '0_check_setup.py' run koro.")
    exit()

if not os.path.isdir(dataset_path) or len(os.listdir(dataset_path)) == 0:
    print("Kono training data pawa jায়নি! Age '1_collect_data.py' run koro.")
    exit()

recognizer = cv2.FaceRecognizerSF.create(SFACE_MODEL, "")

names = []
prototypes = []

print("Training started...\n")

for user_name in sorted(os.listdir(dataset_path)):
    user_folder = os.path.join(dataset_path, user_name)
    if not os.path.isdir(user_folder):
        continue

    person_features = []

    for image_name in os.listdir(user_folder):
        image_path = os.path.join(user_folder, image_name)
        image = cv2.imread(image_path)
        if image is None:
            continue

        # Ei image gula '1_collect_data.py' theke already ALIGNED crop, tai
        # sorasori feature extract kora hocche -- notun kore detect/align lagbe na
        feature = recognizer.feature(image)
        norm = np.linalg.norm(feature)
        if norm > 0:
            person_features.append(feature / norm)

    if len(person_features) == 0:
        print(f"  [SKIP] {user_name}: kono valid feature paoya jায়নি")
        continue

    person_features = np.array(person_features)

    
    prototype = person_features.mean(axis=0)
    prototype = prototype / np.linalg.norm(prototype)

    names.append(user_name)
    prototypes.append(prototype)
    print(f"  {user_name}: {len(person_features)} images -> 1 prototype")

if len(names) == 0:
    print("\nKono face encode kora jায়নি! Dataset check koro.")
    exit()

data = {"names": names, "prototypes": np.array(prototypes, dtype=np.float32)}
with open("encodings.pickle", "wb") as f:
    pickle.dump(data, f)

print(f"\n✅ Training Successful! {len(names)} jon manusher data save hoyeche.")
print("Encodings saved as 'encodings.pickle'")

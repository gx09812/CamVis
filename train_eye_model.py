import cv2
import os
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
import mediapipe as mp


# Paths

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
MODEL_PATH = os.path.join(BASE_DIR, "eye_model.h5")

for folder in [TRAIN_DIR, VAL_DIR]:
    for cls in ["Open", "Closed"]:
        os.makedirs(os.path.join(folder, cls), exist_ok=True)


# MediaPipe setup

mp_face_mesh = mp.solutions.face_mesh
cap = cv2.VideoCapture(0)

LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]


def crop_eye(frame, landmarks, indices):
    xs = [landmarks[i][0] for i in indices]
    ys = [landmarks[i][1] for i in indices]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    margin = 5
    x_min, y_min = max(0, x_min - margin), max(0, y_min - margin)
    x_max, y_max = min(frame.shape[1], x_max + margin), min(frame.shape[0], y_max + margin)
    eye = frame[y_min:y_max, x_min:x_max]
    if eye.size == 0:
        return None
    eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    eye = cv2.resize(eye, (24, 24))
    return eye



#  Capture Dataset

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    print("Capturing dataset. Press 'o' for Open eye, 'c' for Closed eye, 'q' to finish capture.")
    counts = {"Open": 0, "Closed": 0}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        display_frame = frame.copy()

        if results.multi_face_landmarks:
            landmarks = [(int(p.x * frame.shape[1]), int(p.y * frame.shape[0]))
                         for p in results.multi_face_landmarks[0].landmark]
            left_eye = crop_eye(frame, landmarks, LEFT_EYE_INDICES)
            right_eye = crop_eye(frame, landmarks, RIGHT_EYE_INDICES)

            if left_eye is not None and right_eye is not None:
                combined = np.hstack([left_eye, right_eye])
                cv2.imshow("Eye Capture (Left|Right)", combined)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('o') and results.multi_face_landmarks:
            counts["Open"] += 1
            cv2.imwrite(os.path.join(TRAIN_DIR, "Open", f"open_{counts['Open']}.png"), combined)
            print(f"Saved Open eye image #{counts['Open']}")

        elif key == ord('c') and results.multi_face_landmarks:
            counts["Closed"] += 1
            cv2.imwrite(os.path.join(TRAIN_DIR, "Closed", f"closed_{counts['Closed']}.png"), combined)
            print(f"Saved Closed eye image #{counts['Closed']}")

        elif key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

print(f" Captured {counts['Open']} Open images and {counts['Closed']} Closed images")


#  Train Model

if counts['Open'] > 0 and counts['Closed'] > 0:
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_data = datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(24, 48),   # fixed (height=24, width=48)
        color_mode="grayscale",
        batch_size=32,
        class_mode="binary",
        subset="training"
    )

    val_data = datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(24, 48),   # fixed (height=24, width=48)
        color_mode="grayscale",
        batch_size=32,
        class_mode="binary",
        subset="validation"
    )

    if train_data.samples == 0 or val_data.samples == 0:
        print("❌ Not enough images captured for training and validation. Please capture more images.")
    else:
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(24, 48, 1)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print(" Training model...")
        model.fit(train_data, validation_data=val_data, epochs=10)
        model.save(MODEL_PATH)
        print(f" Model saved: {MODEL_PATH}")
else:
    print("Skipping training as no new images were captured.")


# 3️ Real-Time Detection

cap = cv2.VideoCapture(0)
model = load_model(MODEL_PATH)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    print("🎥 Starting real-time detection. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        left_status, right_status = "No Face", "No Face"

        if results.multi_face_landmarks:
            landmarks = [(int(p.x * frame.shape[1]), int(p.y * frame.shape[0]))
                         for p in results.multi_face_landmarks[0].landmark]
            left_eye = crop_eye(frame, landmarks, LEFT_EYE_INDICES)
            right_eye = crop_eye(frame, landmarks, RIGHT_EYE_INDICES)

            if left_eye is not None and right_eye is not None:
                combined = np.hstack([left_eye, right_eye])
                combined = cv2.resize(combined, (48, 24)) / 255.0  # fixed: width=48, height=24
                combined = np.expand_dims(combined, axis=(0, -1))
                pred = model.predict(combined, verbose=0)[0][0]
                status = "Open" if pred > 0.5 else "Closed"
                left_status = right_status = status

        cv2.putText(frame, f"L:{left_status}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if left_status == "Open" else (0, 0, 255), 2)
        cv2.putText(frame, f"R:{right_status}", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if right_status == "Open" else (0, 0, 255), 2)
        cv2.imshow("Real-Time Eye Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

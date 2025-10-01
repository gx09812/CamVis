import cv2 as vision
from ultralytics import YOLO
import mediapipe as mp
import numpy as np


class SmartVision:
    def __init__(self, input_source=0, object=None, count=None):
        # Load YOLOv8 model
        self.model = YOLO("yolov8n.pt")

        # MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        # MediaPipe FaceMesh (for eyes + iris/retina)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

        # Object filter & counting
        self.object = object
        self.count_limit = count

        # Video input
        self.video = vision.VideoCapture(input_source)
        self.video.set(vision.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(vision.CAP_PROP_FRAME_HEIGHT, 480)

    def read(self):
        ret, frame = self.video.read()
        return ret, frame

    # -------------------------
    # YOLO Object Detection
    # -------------------------
    def process_yolo(self, frame):
        detected_count = 0
        results = self.model(frame, stream=True, conf=0.5, iou=0.4)
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0].item())
                class_name = self.model.names[cls]

                if self.object and class_name not in self.object:
                    continue

                detected_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                label = f"{class_name} {conf:.2f}"

                vision.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                vision.putText(frame, label, (x1, y1 - 10),
                               vision.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if self.count_limit and detected_count >= self.count_limit:
            vision.putText(frame, f"Limit Reached ({self.count_limit})", (10, 70),
                           vision.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame

    # -------------------------
    # Finger Detection
    # -------------------------
    def detect_fingers(self, frame, finger_names=None):
        img_rgb = vision.cvtColor(frame, vision.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            finger_tips = [4, 8, 12, 16, 20]
            finger_pips = [3, 6, 10, 14, 18]
            finger_status = {}

            for tip, pip, name in zip(finger_tips, finger_pips, finger_names):
                if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
                    finger_status[name] = "Up"
                else:
                    finger_status[name] = "Down"

            print("Finger States:", finger_status)

            y = 30
            for name, state in finger_status.items():
                vision.putText(frame, f"{name}: {state}", (10, y),
                               vision.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                y += 25
        return frame

    # -------------------------
    # Face Detection (Haar)
    # -------------------------
    def face_detection(self, frame):
        face_cascade = vision.CascadeClassifier(vision.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = vision.cvtColor(frame, vision.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            vision.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return frame

    # -------------------------
    # Eye + Retina Detection
    # -------------------------
    def eye_retina_detection(self, frame):
        h, w, _ = frame.shape
        rgb_frame = vision.cvtColor(frame, vision.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Define indexes
                LEFT_EYE = [33, 133, 159, 145]
                RIGHT_EYE = [362, 263, 386, 374]
                LEFT_IRIS = [474, 475, 476, 477]
                RIGHT_IRIS = [469, 470, 471, 472]

                # Draw eye outline
                for i in LEFT_EYE + RIGHT_EYE:
                    x, y = int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)
                    vision.circle(frame, (x, y), 2, (0, 255, 0), -1)

                # Calculate iris centers
                def get_center(indices):
                    cx = int(np.mean([face_landmarks.landmark[i].x * w for i in indices]))
                    cy = int(np.mean([face_landmarks.landmark[i].y * h for i in indices]))
                    return (cx, cy)

                left_center = get_center(LEFT_IRIS)
                right_center = get_center(RIGHT_IRIS)

                # Draw retina centers
                vision.circle(frame, left_center, 3, (0, 0, 255), -1)
                vision.circle(frame, right_center, 3, (0, 0, 255), -1)

                # Display coordinates
                vision.putText(frame, f"L iris: {left_center}", (30, 50),
                               vision.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                vision.putText(frame, f"R iris: {right_center}", (30, 80),
                               vision.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return frame

    # -------------------------
    # Helper Info
    # -------------------------
    def help(self):
        print("\n" + "-"*100)
        print(f"\nSmartVision Class - Methods and Usage")
        print(f"\nAvailable methods: process_yolo, detect_fingers, face_detection, eye_retina_detection")
        print(f"\nTotal functions available: {len(self.__class__.__dict__)}")
        print("\n" + "-"*100)
        return

    # -------------------------
    # OpenCV Window Wrappers
    # -------------------------
    def imgshow(self, frame, title="Smart Vision"):
        vision.imshow(title, frame)

    def wait(self, delay=1):
        return vision.waitKey(delay) & 0xFF

    def close(self):
        self.video.release()
        vision.destroyAllWindows()

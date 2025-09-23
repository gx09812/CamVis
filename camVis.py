import cv2 as vision
from ultralytics import YOLO
import mediapipe as mp


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

    def face_detection(self, frame):
        face_cascade = vision.CascadeClassifier(vision.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = vision.cvtColor(frame, vision.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            vision.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return frame

    def help(self):
        # how many function used
        print("Available methods: process_yolo 'SmartVision( input_source=0, object=['name'], count=10)', detect_fingers, face_detection")
        print(f"Total functions available: {len(self.__class__.__dict__)}")

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

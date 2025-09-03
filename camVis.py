import cv2
import os
from ultralytics import YOLO
import mediapipe as mp
import face_recognition


class SmartVision:
    def __init__(self, known_faces_dir="known_faces", cam_index=0):
        # YOLOv8
        self.model = YOLO("yolov8n.pt")

        # MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(max_num_hands=1)

        # Face Recognition
        self.known_encodings = []
        self.known_names = []
        self.load_known_faces(known_faces_dir)

        # Video
        self.video = cv2.VideoCapture(cam_index)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def load_known_faces(self, folder):
        if not os.path.exists(folder):
            print(f"[WARN] '{folder}' folder not found. Skipping face loading.")
            return

        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                self.known_encodings.append(encodings[0])
                self.known_names.append(os.path.splitext(filename)[0])
                print(f"[INFO] Loaded face: {filename}")
            else:
                print(f"[WARN] No face found in {filename}")

    def process_yolo(self, frame):
        results = self.model(frame, stream=True)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                label = f"{self.model.names[cls]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame

    def detect_fingers(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                # Index finger check
                tip_id = 8
                pip_id = 6
                if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[pip_id].y:
                    cv2.putText(frame, "Index Finger Up", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        return frame

    def recognize_faces(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(self.known_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_names[first_match_index]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return frame

    def run(self):
        while True:
            ret, frame = self.video.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame = self.process_yolo(frame)
            frame = self.detect_fingers(frame)
            frame = self.recognize_faces(frame)

            cv2.imshow("Smart Vision", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sv = SmartVision(known_faces_dir="known_faces", cam_index=0)
    sv.run()

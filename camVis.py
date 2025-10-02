import cv2 as vision
from ultralytics import YOLO
import mediapipe as mp
import numpy as np
from typing import Optional, List, Dict, Tuple, Any


class SmartVision:
    def __init__(
        self, 
        input_source: Any = 0, 
        objects: Optional[List[str]] = None,
        count: Optional[int] = None,
        use_yolo: bool = True,
        model_path: str = "yolov8n.pt",
        use_hands: bool = True,
        use_face: bool = True,
        debug: bool = True,
        mirror: bool = True,  # mirror webcam view
    ):
        
        self.debug = debug
        self.objects = objects
        self.count_limit = count
        self.mirror = mirror

        # Load YOLO model
        self.model = None
        if use_yolo:
            try:
                self.model = YOLO(model_path)
            except Exception as e:
                if self.debug:
                    print(f"[SmartVision] YOLO model load failed: {e}")
                self.model = None

        #  Hands
        self.mp_hands = mp.solutions.hands if use_hands else None
        self.hands = None
        self.mp_draw = None
        if use_hands and self.mp_hands:
            self.hands = self.mp_hands.Hands(
                max_num_hands=2,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6,
            )
            self.mp_draw = mp.solutions.drawing_utils

        #  FaceMesh
        self.mp_face_mesh = mp.solutions.face_mesh if use_face else None
        self.face_mesh = None
        if use_face and self.mp_face_mesh:
            try:
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                )
            except Exception as e:
                if self.debug:
                    print(f"[SmartVision] FaceMesh init failed: {e}")
                self.face_mesh = None

        #  face detection 
        try:
            self.face_cascade = vision.CascadeClassifier(
                vision.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
        except Exception:
            self.face_cascade = None
            if self.debug:
                print("[SmartVision] Haar cascade could not be loaded")

        # input Video
        self.video = vision.VideoCapture(input_source)
        if not self.video.isOpened():
            raise RuntimeError(f"Could not open video source: {input_source}")
        self.video.set(vision.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(vision.CAP_PROP_FRAME_HEIGHT, 480)

    # -----------------------------
    # Context manager support
    # -----------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
    
    # -----------------------------
    # Video utilities
    # -----------------------------

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        ret, frame = self.video.read()
        if ret and self.mirror:
            frame = vision.flip(frame, 1)
        return ret, frame

    def imgshow(self, frame: np.ndarray, title: str = "Smart Vision") -> None:
        vision.imshow(title, frame)

    def wait(self, delay: int = 1) -> int:
        return int(vision.waitKey(delay) & 0xFF)

    def close(self) -> None:
        try:
            if self.video and self.video.isOpened():
                self.video.release()
        except Exception:
            pass
        try:
            vision.destroyAllWindows()
        except Exception:
            pass
        try:
            if self.hands:
                self.hands.close()
        except Exception:
            pass
        try:
            if self.face_mesh:
                self.face_mesh.close()
        except Exception:
            pass

    # -----------------------------
    # Object detection
    # -----------------------------

    def detectObjects(self, frame: np.ndarray, draw: bool = True) -> Tuple[np.ndarray, List[Dict]]:
        if not self.model:
            return frame, []

        detected_data: List[Dict] = []
        try:
            results = self.model(frame, stream=True, conf=0.5, iou=0.4)
            for result in results:
                for box in getattr(result, "boxes", []):
                    try:
                        cls = int(box.cls[0].item()) if hasattr(box.cls, "item") else int(box.cls[0])
                    except Exception:
                        cls = int(box.cls) if not hasattr(box.cls, "__len__") else 0
                    class_name = self.model.names.get(cls, str(cls)) if getattr(self.model, "names", None) else str(cls)
                    if self.objects and class_name not in self.objects:
                        continue
                    xyxy = box.xyxy[0] if hasattr(box.xyxy, "__len__") else box.xyxy
                    x1, y1, x2, y2 = map(int, (xyxy[0], xyxy[1], xyxy[2], xyxy[3]))
                    try:
                        conf = float(box.conf[0].item()) if hasattr(box.conf, "item") else float(box.conf[0])
                    except Exception:
                        conf = float(getattr(box, "conf", 0.0))

                    detected_data.append({"class": class_name, "confidence": conf, "bbox": (x1, y1, x2, y2)})
                    if draw:
                        vision.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        vision.putText(frame, f"{class_name} {conf:.2f}", (x1, max(y1 - 10, 0)),
                                       vision.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        except Exception as e:
            if self.debug:
                print(f"[SmartVision] detectObjects error: {e}")

        if self.count_limit and len(detected_data) >= self.count_limit and draw:
            vision.putText(frame, f"Limit Reached ({self.count_limit})", (10, 70),
                           vision.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return frame, detected_data

    # -----------------------------
    # Finger detection
    # -----------------------------

    def detectFingers(self, frame: np.ndarray, finger_names: Optional[List[str]] = None, draw: bool = True) -> Tuple[np.ndarray, Dict[str, Dict[str, str]]]:
        if not self.hands:
            return frame, {}

        img_rgb = vision.cvtColor(frame, vision.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        finger_status: Dict[str, Dict[str, str]] = {}

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_label = "Unknown"
                try:
                    if results.multi_handedness and len(results.multi_handedness) > idx:
                        hand_label = results.multi_handedness[idx].classification[0].label
                except Exception:
                    pass

                if draw and self.mp_draw:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                finger_tips = [4, 8, 12, 16, 20]
                finger_pips = [3, 6, 10, 14, 18]
                names = finger_names or ["Thumb", "Index", "Middle", "Ring", "Pinky"]

                this_hand_status: Dict[str, str] = {}
                for tip, pip, name in zip(finger_tips, finger_pips, names):
                    try:
                        tip_y = hand_landmarks.landmark[tip].y
                        pip_y = hand_landmarks.landmark[pip].y
                        this_hand_status[name] = "Up" if tip_y < pip_y else "Down"
                    except Exception:
                        this_hand_status[name] = "Unknown"

                finger_status[hand_label] = this_hand_status

                if draw:
                    base_y = 30 + idx * 120
                    vision.putText(frame, f"Hand: {hand_label}", (10, base_y - 20),
                                   vision.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    y = base_y
                    for name, state in this_hand_status.items():
                        vision.putText(frame, f"{name}: {state}", (10, y),
                                       vision.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        y += 25

        return frame, finger_status

    # -----------------------------
    # Face detection
    # -----------------------------

    def detectFaces(self, frame: np.ndarray, draw: bool = True) -> Tuple[np.ndarray, List[Dict]]:
        detected_faces: List[Dict] = []
        if self.face_cascade is None:
            return frame, detected_faces

        try:
            gray = vision.cvtColor(frame, vision.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            for (x, y, w, h) in faces:
                detected_faces.append({"bbox": (x, y, x + w, y + h)})
                if draw:
                    vision.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        except Exception as e:
            if self.debug:
                print(f"[SmartVision] detectFaces error: {e}")

        return frame, detected_faces
    
    # -----------------------------
    # Eyes + Iris detection
    # -----------------------------
    def detectEYR(self, frame: np.ndarray, mode: str = "both", draw: bool = True) -> Tuple[np.ndarray, Dict]:
        
        if self.face_mesh is None:
            return frame, {"eyes": {"left": [], "right": []}, "iris": {"left": None, "right": None}}

        h, w = frame.shape[:2]
        rgb_frame = vision.cvtColor(frame, vision.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        data = {"eyes": {"left": [], "right": []}, "iris": {"left": None, "right": None}}

        alpha = 0.6  
        max_jump = 80.0  # px; if jump bigger than this, consider it an outlier and smooth more

        prev_iris = getattr(self, "_prev_iris", {"left": None, "right": None})
        prev_eyes_center = getattr(self, "_prev_eyes_center", {"left": None, "right": None})

        try:
            if results and getattr(results, "multi_face_landmarks", None):
                for face_landmarks in results.multi_face_landmarks:
                    LEFT_EYE = [33, 133, 159, 145]
                    RIGHT_EYE = [362, 263, 386, 374]
                    LEFT_IRIS = [474, 475, 476, 477]
                    RIGHT_IRIS = [469, 470, 471, 472]

                    def center_from_indices(indices):
                        xs = np.array([face_landmarks.landmark[i].x * w for i in indices])
                        ys = np.array([face_landmarks.landmark[i].y * h for i in indices])
                        return float(np.median(xs)), float(np.median(ys))

                   # EyesLeft
                    if mode in ("eyes", "eyesLeft", "both"):
                        data["eyes"]["left"] = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE]
                        if draw and mode in ("eyes", "eyesLeft", "both"):
                            for pt in data["eyes"]["left"]:
                                vision.circle(frame, pt, 2, (0, 255, 0), -1)
                                
                    # EyesRight
                    if mode in ("eyes", "eyesRight", "both"):
                        data["eyes"]["right"] = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE]
                        if draw and mode in ("eyes", "eyesRight", "both"):
                            for pt in data["eyes"]["right"]:
                                vision.circle(frame, pt, 2, (0, 255, 0), -1)
                    
                    # Retina Left
                    if mode in ("retina", "retinaLeft", "both"):
                        iris_left = center_from_indices(LEFT_IRIS)
                        p = prev_iris.get("left")
                        if p is None:
                            smooth_iris_left = iris_left
                        else:
                            dist = np.hypot(iris_left[0] - p[0], iris_left[1] - p[1])
                            a = alpha if dist <= max_jump else max(0.1, alpha * 0.25)
                            smooth_iris_left = (a * iris_left[0] + (1 - a) * p[0],
                                                a * iris_left[1] + (1 - a) * p[1])
                        prev_iris["left"] = smooth_iris_left
                        data["iris"]["left"] = (int(round(smooth_iris_left[0])), int(round(smooth_iris_left[1])))

                        if draw:
                            vision.circle(frame, data["iris"]["left"], 3, (0, 0, 255), -1)
                            vision.putText(frame, f"L: {data['iris']['left']}", (30, 50),
                                           vision.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            
                    # Retina Right  
                    if mode in ("retina", "retinaRight", "both"):
                        iris_right = center_from_indices(RIGHT_IRIS)
                        p = prev_iris.get("right")
                        if p is None:
                            smooth_iris_right = iris_right
                        else:
                            dist = np.hypot(iris_right[0] - p[0], iris_right[1] - p[1])
                            a = alpha if dist <= max_jump else max(0.1, alpha * 0.25)
                            smooth_iris_right = (a * iris_right[0] + (1 - a) * p[0],
                                                 a * iris_right[1] + (1 - a) * p[1])
                        prev_iris["right"] = smooth_iris_right
                        data["iris"]["right"] = (int(round(smooth_iris_right[0])), int(round(smooth_iris_right[1])))

                        if draw:
                            vision.circle(frame, data["iris"]["right"], 3, (0, 0, 255), -1)
                            vision.putText(frame, f"R: {data['iris']['right']}", (30, 80),
                                           vision.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        except Exception as e:
            if self.debug:
                print(f"{self.__class__.__name__} detectEYR error: {e}")

        self._prev_iris = prev_iris
        self._prev_eyes_center = prev_eyes_center

        return frame, data

    def help(self) -> None:
        # Prints usage instructions for the SmartVision class.
       
        print("\n=== SmartVision Help ===\n")
        print("1. Reading frames from camera or video:")
        print("   ret, frame = sv.read()")
        print("\n2. Object detection:")
        print("   frame, objects = sv.detectObjects(frame, draw=True)")
        print("\n3. Finger detection:")
        print("   frame, fingers = sv.detectFingers(frame, draw=True)")
        print("\n4. Face detection:")
        print("   frame, faces = sv.detectFaces(frame, draw=True)")
        print("\n5. Eye and iris detection:")
        print("   frame, data = sv.detectEYR(frame, mode='both', draw=True)")
        print("   Modes: 'eyes', 'eyesLeft', 'eyesRight', 'retina', 'retinaLeft', 'retinaRight', 'both'")
        print("\n6. Show frame:")
        print("   sv.imgshow(frame)")
        print("\n7. Wait for key press:")
        print("   key = sv.wait(delay=1)")
        print("\n8. Close camera and windows:")
        print("   sv.close()")
        print("\nPress 'q' or 'Q' to quit the video stream.")
       
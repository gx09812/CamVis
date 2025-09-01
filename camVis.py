from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt



# Load YOLO model (Nano version for speed)
model = YOLO("yolov8n.pt")  
video = cv2.VideoCapture(0)

video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# add the name in frames
def add_name():
    names ={
        "monitor","keyboard","mouse","buildings"
    }
    return
def process_frame(frame):
    results = model(frame, stream=True)  # stream=True = better for realtime
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to grab frame")
        continue

    frame = process_frame(frame)
    cv2.imshow("YOLOv8 Real-Time Detection", frame)

    # quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

video.release()
cv2.destroyAllWindows()

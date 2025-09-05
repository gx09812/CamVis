from camVis import SmartVision


sv = SmartVision(input_source=0, allowed_classes=["cell phone"], count=10)

while True:
    ret, frame = sv.read()
    if not ret:
        print("Failed to grab frame")
        break
        
    frame = sv.process_yolo(frame)
    # frame = sv.detect_fingers(frame, finger_names=[""])
    sv.imgshow(frame)
    if sv.wait(1) in [ord('q'), ord('Q')]:
        break

if __name__ == "__main__":
    sv.close()

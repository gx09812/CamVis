from camVis import SmartVision


sv = SmartVision()

sv.help()

while True:
    ret, frame = sv.read()
    if not ret:
        print("Failed to grab frame")
        break

    # frame = sv.face_detection(frame)
    # frame = sv.process_yolo(frame)
    frame = sv.eye_retina_detection(frame)

    sv.imgshow(frame)
    if sv.wait(1) in [ord('q'), ord('Q')]:
        break

if __name__ == "__main__":
    sv.close()

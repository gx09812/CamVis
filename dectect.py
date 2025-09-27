from camVis import SmartVision


sv = SmartVision(input_source=0,object=["person"], count=10)

sv.help()

# while True:
#     ret, frame = sv.read()
#     if not ret:
#         print("Failed to grab frame")
#         break
        
#     frame = sv.face_detection(frame)
#     frame = sv.process_yolo(frame)

#     sv.imgshow(frame)
#     if sv.wait(1) in [ord('q'), ord('Q')]:
#         break

# if __name__ == "__main__":
#     sv.close()


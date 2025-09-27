import argparse
from camVis import SmartVision
import pyfiglet


# Terminal Colors & Styles


CYAN = "\033[96m"
MAGENTA = "\033[95m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


# CamVis Banner

ascii_banner = pyfiglet.figlet_format("CamVis", font="slant")
print(f"{CYAN}{BOLD}{ascii_banner}{RESET}")

print(f"{MAGENTA}{BOLD}Usage Example:{RESET} python detect.py --object person car --count 5 --input_source 0 --fingers --face\n")
print(f"{MAGENTA}{BOLD}Options:{RESET}")
print(f"{CYAN}- --object       {BOLD}Specify object(s) to detect (e.g. person car dog){RESET}")
print(f"{CYAN}- --count        {BOLD}Number of objects to detect{RESET}")
print(f"{CYAN}- --input_source {BOLD}Camera input source (default: 0){RESET}")
print(f"{CYAN}- --fingers      {BOLD}Enable finger detection{RESET}")
print(f"{CYAN}- --face         {BOLD}Enable face detection{RESET}")
print(f"{CYAN}- Press 'q' to quit the video stream{RESET}")
print("\n" + "="*100 + "\n")


# Main Function

def main():
    parser = argparse.ArgumentParser(description="CamVis Object Detection")
    parser.add_argument('--object', nargs='+', default=["person"], help="Object(s) to detect (e.g. person car dog)")
    parser.add_argument('--count', type=int, default=10, help="Number of objects to detect")
    parser.add_argument('--input_source', type=int, default=0, help="Camera input source (default: 0)")
    parser.add_argument('--fingers', action='store_true', help="Enable finger detection")
    parser.add_argument('--face', action='store_true', help="Enable face detection")
    args = parser.parse_args()

    # Initialize SmartVision
    sv = SmartVision(input_source=args.input_source, object=args.object, count=args.count)

    # Show available methods
    sv.help()

    
    # Main Video Loop
    try:
        while True:
            ret, frame = sv.read()
            if not ret:
                print(f"{RED}[ERROR]{RESET} Failed to grab frame from camera {args.input_source}")
                break

            # Face Detection
            if args.face:
                frame = sv.face_detection(frame)

            # Finger Detection
            if args.fingers:
                frame = sv.detect_fingers(frame, ["Thumb", "Index", "Middle", "Ring", "Pinky"])

            # Object Detection
            frame = sv.process_yolo(frame)

            # Display frame
            sv.imgshow(frame)

            # Quit on 'q' or 'Q'
            if sv.wait(1) in [ord('q'), ord('Q')]:
                print(f"{YELLOW}[INFO]{RESET} Quitting video stream...")
                break

    except KeyboardInterrupt:
        sv.close()
        print(f"{YELLOW}[INFO]{RESET} Interrupted by user. Exiting...")

    # Cleanup
    sv.close()
    print(f"{GREEN}[INFO]{RESET} CamVis session ended successfully.")


# Entry Point

if __name__ == "__main__":
    main()

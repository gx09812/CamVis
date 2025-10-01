import argparse
import sys

try:
    from camVis import SmartVision
except ImportError:
    print("[ERROR] camVis module not found. Please ensure it is installed and accessible.")
    sys.exit(1)

try:
    import pyfiglet
except ImportError:
    print("[ERROR] pyfiglet module not found. Please install it with 'pip install pyfiglet'.")
    sys.exit(1)

# Terminal Colors & Styles
CYAN = "\033[96m"
MAGENTA = "\033[95m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

def print_banner_and_usage():
    ascii_banner = pyfiglet.figlet_format("CamVis", font="slant")
    print(f"{CYAN}{BOLD}{ascii_banner}{RESET}")
    print(f"{MAGENTA}{BOLD}Usage Example:{RESET} python CamTool.py --object person car --count 5 --input_source 0 --fingers --face --eye_redina\n")
    print(f"{MAGENTA}{BOLD}Options:{RESET}")
    print(f"{CYAN}- --object       {BOLD}Specify object(s) to detect (e.g. person car dog){RESET}")
    print(f"{CYAN}- --count        {BOLD}Number of objects to detect{RESET}")
    print(f"{CYAN}- --input_source {BOLD}Camera input source (default: 0){RESET}")
    print(f"{CYAN}- --fingers      {BOLD}Enable finger detection{RESET}")
    print(f"{CYAN}- --face         {BOLD}Enable face detection{RESET}")
    print(f"{CYAN}- --eye_redina   {BOLD}Enable eye redina detection{RESET}")
    print(f"{CYAN}- Press 'q' to quit the video stream{RESET}")
    print("\n" + "="*100 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="CamVis Object Detection Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--object', nargs='+', default=[""], help="Object(s) to detect (e.g. person car dog)")
    parser.add_argument('--count', type=int, default=0, help="Number of objects to detect")
    parser.add_argument('--input_source', type=int, default=0, help="Camera input source")
    parser.add_argument('--fingers', action='store_true', help="Enable finger detection")
    parser.add_argument('--face', action='store_true', help="Enable face detection")
    parser.add_argument('--eye_retina_detection', action='store_true', help="Enable eye retina detection")
    parser.add_argument('--version', action='version', version='CamVis 1.0', help="Show program version and exit")
    args = parser.parse_args()

    # Show banner and usage if running interactively
    if sys.stdout.isatty():
        print_banner_and_usage()

    sv = None
    try:
        sv = SmartVision(input_source=args.input_source, object=args.object, count=args.count)
        sv.help()

        while True:
            ret, frame = sv.read()
            if not ret or frame is None:
                print(f"{RED}[ERROR]{RESET} Failed to grab frame from camera {args.input_source}")
                break

            if args.face:
                frame = sv.face_detection(frame)

            if args.fingers:
                frame = sv.detect_fingers(frame, ["Thumb", "Index", "Middle", "Ring", "Pinky"])

            if args.eye_retina_detection:
                if hasattr(sv, "eye_retina_detection"):
                    frame = sv.eye_retina_detection(frame)
                else:
                    print(f"{YELLOW}[WARN]{RESET} SmartVision has no method 'eye_retina_detection'.")

            frame = sv.process_yolo(frame)
            sv.imgshow(frame)

            key = sv.wait(1)
            if key in [ord('q'), ord('Q')]:
                print(f"{YELLOW}[INFO]{RESET} Quitting video stream...")
                break

    except KeyboardInterrupt:
        print(f"{YELLOW}[INFO]{RESET} Interrupted by user. Exiting...")
    except Exception as e:
        print(f"{RED}[ERROR]{RESET} Unexpected error: {e}")
    finally:
        if sv is not None:
            sv.close()
        print(f"{GREEN}[INFO]{RESET} CamVis session ended successfully.")

if __name__ == "__main__":
    main()

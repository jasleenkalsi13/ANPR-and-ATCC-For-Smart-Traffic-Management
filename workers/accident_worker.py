# workers/accident_worker.py
import sys, os
proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

def main():
    from accident import AccidentDetectionSystem
    paths = sys.argv[1:]
    if not paths:
        print("No path provided to accident_worker", file=sys.stderr)
        sys.exit(2)
    detector = AccidentDetectionSystem("../models/best.pt")  # adjust path if needed
    try:
        # if detector expects a list
        detector.process_video_with_gui(paths)
    except Exception as e:
        print("[accident_worker] Error:", e, file=sys.stderr)

if __name__ == "__main__":
    main()

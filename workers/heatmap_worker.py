# workers/heatmap_worker.py
import sys, os
proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

def main():
    from heatmap_visualization import process_videos
    from ultralytics import YOLO
    paths = sys.argv[1:]
    if not paths:
        print("No path provided to heatmap_worker", file=sys.stderr)
        sys.exit(2)
    print("[heatmap_worker] Loading model")
    model = YOLO("yolov8n.pt")
    try:
        process_videos(paths, model)
    except Exception as e:
        print("[heatmap_worker] Error:", e, file=sys.stderr)

if __name__ == "__main__":
    main()

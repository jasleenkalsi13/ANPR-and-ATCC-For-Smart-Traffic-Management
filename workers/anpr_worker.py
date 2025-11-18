# workers/anpr_worker.py
import sys, os
proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

def main():
    from anpr_video import start_anpr
    paths = sys.argv[1:]
    if not paths:
        print("No path provided to anpr_worker", file=sys.stderr)
        sys.exit(2)
    for p in paths:
        print("[anpr_worker] Processing:", p)
        try:
            start_anpr([p])   # your start_anpr expects list
        except Exception as e:
            print("[anpr_worker] Error:", e, file=sys.stderr)

if __name__ == "__main__":
    main()

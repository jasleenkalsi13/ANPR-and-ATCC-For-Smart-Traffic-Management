# workers/traffic_worker.py
import sys, os
proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

def main():
    from traffic_violation import main as traffic_main
    paths = sys.argv[1:]
    if not paths:
        print("No path provided to traffic_worker", file=sys.stderr)
        sys.exit(2)
    for p in paths:
        print("[traffic_worker] Processing:", p)
        try:
            traffic_main(p)
        except Exception as e:
            print("[traffic_worker] Error:", e, file=sys.stderr)

if __name__ == "__main__":
    main()

# workers/triple_worker.py
import sys, os
proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

def main():
    from triple_riding import detect_triple_riding
    paths = sys.argv[1:]
    if not paths:
        print("No path provided to triple_worker", file=sys.stderr)
        sys.exit(2)
    for p in paths:
        print("[triple_worker] Processing:", p)
        try:
            detect_triple_riding(p)
        except Exception as e:
            print("[triple_worker] Error:", e, file=sys.stderr)

if __name__ == "__main__":
    main()

# workers/helmet_worker.py
import sys, os

# ensure project root is on path if needed
proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

def main():
    from helmet_detection import main_fun  # your function
    # Accept multiple paths
    paths = sys.argv[1:]
    if not paths:
        print("No path provided to helmet_worker", file=sys.stderr)
        sys.exit(2)
    for p in paths:
        print("[helmet_worker] Processing:", p)
        try:
            main_fun(p)
        except Exception as e:
            print("[helmet_worker] Error:", e, file=sys.stderr)

if __name__ == "__main__":
    main()

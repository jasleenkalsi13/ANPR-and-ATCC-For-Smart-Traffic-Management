import cv2
import numpy as np
from ultralytics import YOLO
import os

# -----------------------------------------------------------
# TRAFFIC LIGHT SIMULATION
# -----------------------------------------------------------

def simulate_traffic_light(frame, light_state, position):
    """Draw traffic light on frame."""
    traffic_light_colors = [(0, 0, 255), (0, 255, 255), (0, 255, 0)]  # R, Y, G
    x, y = position

    cv2.rectangle(frame, (x, y), (x + 50, y + 150), (50, 50, 50), -1)
    for i, color in enumerate(traffic_light_colors):
        cv2.circle(frame, (x + 25, y + 25 + i * 50),
                   15, color if i == light_state else (80, 80, 80), -1)


def determine_signal(total_vehicles):
    """Return signal color based on vehicle count."""
    if total_vehicles < 10:
        return "Green", (0, 255, 0), 2
    elif total_vehicles < 20:
        return "Yellow", (0, 255, 255), 1
    else:
        return "Red", (0, 0, 255), 0


# -----------------------------------------------------------
# FIXED FUNCTION — WORKS FOR LIST OR SINGLE VIDEO
# -----------------------------------------------------------

def load_videos_from_folder(video_inputs):
    """
    Accepts:
      ✔ a single file path
      ✔ a single folder
      ✔ a list of paths
    Returns a clean list of video files.
    """
    valid_ext = ('.mp4', '.avi', '.mov', '.mkv')
    videos = []

    # Convert single string → list
    if isinstance(video_inputs, str):
        video_inputs = [video_inputs]

    # Ensure it's now a list
    if not isinstance(video_inputs, list):
        print("Invalid video input format.")
        return []

    for item in video_inputs:

        # CASE 1 → item is a video file
        if isinstance(item, str) and os.path.isfile(item) and item.lower().endswith(valid_ext):
            videos.append({"path": item, "road_name": os.path.basename(item)})
            continue

        # CASE 2 → item is a folder
        if isinstance(item, str) and os.path.isdir(item):
            for f in os.listdir(item):
                if f.lower().endswith(valid_ext):
                    full = os.path.join(item, f)
                    videos.append({"path": full, "road_name": f})
            continue

        print(f"Skipping invalid input: {item}")

    return videos


# -----------------------------------------------------------
# FRAME PROCESSING
# -----------------------------------------------------------

def process_frame(frame, model, road_name, directions):
    results = model(frame)
    detections = results[0].boxes.data.cpu().numpy()

    vehicle_count = {"car": 0, "truck": 0, "motorcycle": 0, "bus": 0}

    for det in detections:
        x1, y1, x2, y2, conf, class_id = det
        class_id = int(class_id)
        label = model.names[class_id]

        if label in vehicle_count:
            vehicle_count[label] += 1

            center_x = (x1 + x2) / 2
            if center_x < frame.shape[1] / 2:
                directions["left"] += 1
            else:
                directions["right"] += 1

            color = (0, 255, 0) if conf > 0.50 else (0, 0, 255)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            cv2.putText(frame, f"{label} {conf:.2f}",
                        (int(x1), int(y1) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Text overlays
    cv2.putText(frame, f"Road: {road_name}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    y_offset = 50
    for direction, cnt in directions.items():
        cv2.putText(frame, f"{direction.capitalize()}: {cnt}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 25

    total = sum(vehicle_count.values())
    cv2.putText(frame, f"Total Vehicles: {total}", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_offset += 25

    signal_color, signal_rgb, light_state = determine_signal(total)
    cv2.putText(frame, f"Signal: {signal_color}", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, signal_rgb, 2)

    simulate_traffic_light(
        frame,
        light_state,
        position=(frame.shape[1] - 100, 20)
    )

    return frame, vehicle_count


# -----------------------------------------------------------
# VIDEO PROCESSING (MAC SAFE – NO CRASH)
# -----------------------------------------------------------

def process_atcc_videos(video_input_list, model):
    input_files = load_videos_from_folder(video_input_list)
    if not input_files:
        print("No valid videos found.")
        return

    caps = []
    roads = []

    for f in input_files:
        cap = cv2.VideoCapture(f["path"])

        if not cap.isOpened():
            print(f"Could not open: {f['path']}")
            continue

        caps.append(cap)
        roads.append(f["road_name"])

    target_w, target_h = 640, 480

    while True:
        processed_frames = []

        for i, cap in enumerate(caps):
            ret, frame = cap.read()

            if not ret:
                print(f"Finished: {roads[i]}")
                caps[i].release()
                continue

            frame = cv2.resize(frame, (target_w, target_h))
            directions = {"left": 0, "right": 0}

            processed, _ = process_frame(frame, model, roads[i], directions)
            processed_frames.append(processed)

        if len(processed_frames) == 0:
            break

        rows = []
        for i in range(0, len(processed_frames), 2):
            if i + 1 < len(processed_frames):
                row = np.hstack(processed_frames[i:i + 2])
            else:
                row = processed_frames[i]
            rows.append(row)

        max_w = max(r.shape[1] for r in rows)
        max_h = max(r.shape[0] for r in rows)
        rows_resized = [cv2.resize(r, (max_w, max_h)) for r in rows]

        grid = np.vstack(rows_resized)

        cv2.imshow("Traffic Management System", grid)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")

    # PASS MULTIPLE VIDEOS HERE OR A FOLDER
    video_input_list = [
        "/Users/amarjeet/Desktop/archive"   # folder
    ]

    process_atcc_videos(video_input_list, model)

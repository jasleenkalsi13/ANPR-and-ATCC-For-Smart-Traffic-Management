from flask import Flask, render_template, request, redirect, url_for, jsonify, Response
import os
from werkzeug.utils import secure_filename
import cv2
from app_instance import create_app
from dotenv import load_dotenv
import pymysql
import shlex
import subprocess
import sys
import uuid

# Load environment variables from .env file
load_dotenv()

# Initialize the app using the create_app function
app = create_app()

# Fetch camera IPs from environment variables (comma-separated list)
camera_ips_env = os.getenv('LIVE_CCTV_IPS', '')
if camera_ips_env:
    app.config['CAMERA_IPS'] = camera_ips_env.split(',')
else:
    app.config['CAMERA_IPS'] = []

UPLOAD_FOLDER = app.config['UPLOAD_FOLDER']
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



def dbconnection():
     connection = pymysql.connect(host='127.0.0.1',database='traffic_management',user='admin',password='12345678')
     return connection


# Routes
@app.route('/')
def login():
    return render_template('login.html')
from dbconnection import get_connection
@app.route('/validate_login', methods=['POST'])
def validate_login():
    username = request.form.get('username')
    password = request.form.get('password')
    connection = get_connection()
    print(f"number_plate......................{connection}")
    print(username, password)
    with connection.cursor() as cursor:
        # Use parameterized query to prevent SQL injection
        sql_query = "SELECT * FROM login_details WHERE username = %s AND password = %s"
        cursor.execute(sql_query, (username, password))
        result = cursor.fetchone() 
        print("SQL Statement Executed:", sql_query)
        print("Query result:", result)
        if result:
            print(f"✅ User {username} logged in successfully.")
            return redirect(url_for('home'))
            #return render_template('login.html')
        else:
            print("❌ Invalid credentials")
    
    return render_template('login.html', error="Invalid username or password.")


@app.route('/home')
def home():
    return render_template('home.html')
from dbconnection import get_connection

import pymysql

from flask import request, render_template
import pymysql
from dbconnection import get_connection
import glob
@app.route('/search', methods=['GET', 'POST'])
def search_license_plate():
    number_plate = request.form.get('number_plate')
    connection = get_connection()

    vehicle_data = None
    detections = []
    images = []
    error = None

    if request.method == 'POST' and number_plate:
        with connection.cursor(pymysql.cursors.DictCursor) as cursor:

            # Get all matching detection timestamps
            sql_query = "SELECT id, timestamp FROM vehicle_data WHERE number_plate = %s"
            cursor.execute(sql_query, (number_plate,))
            results = cursor.fetchall()

            if results:
                # License number
                vehicle_data = {"license_no": number_plate}

                # All detections with timestamps
                detections = [
                    {"id": row["id"], "detected_at": row["timestamp"]}
                    for row in results
                ]

                # Load related images from static folder
                file_list = glob.glob(f"static/vehicle_images/{number_plate}*.jpg")
                images = [path.replace("static/", "") for path in file_list]

            else:
                error = "No details found for the entered number plate."

    return render_template(
        'search.html',
        vehicle_data=vehicle_data,
        detections=detections,
        images=images,
        error=error
    )


################################################################################
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
@app.route('/upload_video', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('upload_video.html', error="No file selected.")
        file = request.files['file']
        if file and allowed_file(file.filename):
            # Ensure the filename is safe
            filename = secure_filename(file.filename)
            # Save the file in the static/uploads directory
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return render_template('upload_video.html', success=True, filename=filename)
        else:
            return render_template('upload_video.html', error="Invalid file type. Allowed: mp4, avi, mov, mkv.")
    return render_template('upload_video.html')


from flask import render_template
import pymysql
import dbconnection  # your module to get MySQL connection

@app.route('/results', methods=['GET'])
def results():
    connection = get_connection()
    with connection.cursor(pymysql.cursors.DictCursor) as cursor:
        cursor.execute("""
            SELECT number_plate, detected_at, video_file
            FROM vehicle_data
            ORDER BY detected_at DESC
        """)
        rows = cursor.fetchall()

    results_data = []
    for row in rows:
        results_data.append({
            "plate_number": row["number_plate"],
            "timestamp": row["detected_at"],
            "video_file": row["video_file"] if row["video_file"] else "N/A"
        })

    return render_template('results.html', results=results_data)

@app.route('/logout')
def logout():
    return redirect(url_for('login'))

#############################################################################################

def load_videos_from_folder(videos_folder):
    """Load all video files from the provided folder."""
    video_files = []
    for filename in os.listdir(videos_folder):
        if filename.endswith(".mp4"):
            video_files.append({"path": os.path.join(videos_folder, filename), "road_name": filename})
    return video_files

def load_videos_from_folder(videos_folder):
    """Load all video files from the provided folder."""
    video_files = []
    for filename in os.listdir(videos_folder):
        if filename.endswith(".mp4"):
            video_files.append({"path": os.path.join(videos_folder, filename), "road_name": filename})
    return video_files
########################################################################################
from flask import session

@app.route('/live_monitoring', methods=['GET', 'POST'])
def live_monitoring():
    if request.method == 'POST':
        # Save form data in session to repopulate the form after submission
        session['numCameras'] = request.form.get('numCameras', 1)
        session['processType'] = request.form.get('processType', 'anpr')

        # Save camera inputs (IPs or File Uploads)
        num_cameras = int(session['numCameras'])
        session['cameraInputs'] = []
        for i in range(1, num_cameras + 1):
            ip_or_file_key = f"cameraIp{i}" if f"cameraIp{i}" in request.form else f"cameraFile{i}"
            session['cameraInputs'].append({
                'label': f'Camera {i} Input',
                'type': 'text' if f"cameraIp{i}" in request.form else 'file',
                'name': ip_or_file_key,
                'value': request.form.get(ip_or_file_key, '')
            })

        # Perform any processing here
        print("Processing started...")

        # Redirect back to the page with the same form values
        return redirect('/live_monitoring')

    # For GET requests, repopulate the form using session data or defaults
    num_cameras = session.get('numCameras', 1)
    process_type = session.get('processType', 'anpr')
    camera_inputs = session.get('cameraInputs', [
        {'label': 'Camera 1 Input', 'type': 'file', 'name': 'cameraFile1', 'value': ''}
    ])

    return render_template(
        'live_monitoring.html',
        numCameras=num_cameras,
        processType=process_type,
        cameraInputs=camera_inputs
    )

######################################################################################################

#####################################################################################################################
from atcc import *
import os

from anpr_video import PlateFinder  # Ensure PlateFinder is properly imported
from anpr_video import OCR  # Ensure OCR is properly imported
from anpr_video import *
from ultralytics import YOLO


def spawn_worker(module_or_script, args_list, log_subdir="worker_logs"):
    """
    Spawn a detached subprocess that runs: <python> -u -m <module_or_script> <args...>
    module_or_script: module name (e.g. 'anpr_video') OR full script path.
    args_list: list of str args to pass.
    Returns Popen object.
    """
    # Ensure logs folder exists
    logs_dir = os.path.join(app.config['UPLOAD_FOLDER'], log_subdir)
    os.makedirs(logs_dir, exist_ok=True)

    # unique log file per worker
    worker_id = uuid.uuid4().hex[:8]
    out_path = os.path.join(logs_dir, f"worker-{worker_id}.out.log")
    err_path = os.path.join(logs_dir, f"worker-{worker_id}.err.log")

    # Build command: prefer module run (-m) if module_or_script looks like a module name,
    # otherwise run the script path directly.
    python_exe = sys.executable  # ensures same venv interpreter

    if os.path.isfile(module_or_script):
        # run script path: <python> -u <script> <args...>
        cmd = [python_exe, "-u", module_or_script] + args_list
    else:
        # run module: <python> -u -m <module> <args...>
        cmd = [python_exe, "-u", "-m", module_or_script] + args_list

    # spawn detached process (don't block Flask). Keep stdout/stderr logs for debugging.
    fout = open(out_path, "ab")
    ferr = open(err_path, "ab")

    # On macOS/Unix: set close_fds True to fully detach (Windows ok too)
    p = subprocess.Popen(cmd, stdout=fout, stderr=ferr, stdin=subprocess.DEVNULL, close_fds=True)

    print(f"Spawned worker pid: {p.pid} cmd={' '.join(shlex.quote(c) for c in cmd)}")
    print(f"Logs: stdout -> {out_path} stderr -> {err_path}")
    return p

@app.route('/start_processing', methods=['POST'])
def start_processing():
    process_type = (request.form.get('processType') or '').lower()
    try:
        num_cameras = int(request.form.get('numCameras', 1))
    except Exception:
        num_cameras = 1

    saved = []
    for i in range(1, num_cameras+1):
        f = request.files.get(f'cameraFile{i}')
        if f and f.filename:
            filename = secure_filename(f.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            f.save(path)
            saved.append(path)

    # Also accept single 'file' input from upload page
    if 'file' in request.files:
        f = request.files['file']
        if f and f.filename:
            filename = secure_filename(f.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            f.save(path)
            if path not in saved:
                saved.append(path)

    # pick worker and spawn accordingly
    if process_type == 'anpr':
        script = os.path.join(os.path.dirname(__file__), 'workers', 'anpr_worker.py')
    elif process_type == 'atcc':
        script = os.path.join(os.path.dirname(__file__), 'workers', 'atcc_worker.py')
    else:
        # fallback: pick triple worker or return
        script = os.path.join(os.path.dirname(__file__), 'workers', 'triple_worker.py')

    for p in saved:
        spawn_worker(script, [p])

    files_query = ','.join(os.path.basename(s) for s in saved)
    return redirect(url_for('live_monitoring') + f"?files={files_query}&processType={process_type}")

#####################################################################################
   
#####################################################################################

from atcc import *
# ---------- ATCC route ----------
@app.route('/atcc', methods=['POST'])
def atcc_route():
    # accept cameraFile1..N or single 'file'
    uploaded_files = [request.files[key] for key in request.files if key.startswith('cameraFile')]
    if not uploaded_files and 'file' in request.files:
        uploaded_files = [request.files['file']]

    if not uploaded_files:
        return jsonify({"success": False, "message": "No files uploaded for ATCC."}), 400

    saved = []
    script = os.path.join(os.path.dirname(__file__), 'workers', 'atcc_worker.py')
    for f in uploaded_files:
        if f and f.filename:
            filename = secure_filename(f.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            f.save(path)
            saved.append(path)
            # spawn worker (no GUI needed for ATCC usually)
            spawn_worker(script, [path], use_module=False, extra_env={"OPENCV_HEADLESS": "1"})
    return jsonify({"success": True, "files": saved}), 200


# ---------- Traffic Violation route ----------
@app.route('/traffic_violation', methods=['POST'])
def traffic_violation_route():
    uploaded_files = [request.files[key] for key in request.files if key.startswith('cameraFile')]
    if not uploaded_files and 'file' in request.files:
        uploaded_files = [request.files['file']]

    if not uploaded_files:
        return jsonify({"success": False, "message": "No files uploaded for Traffic Violation Detection."}), 400

    saved = []
    script = os.path.join(os.path.dirname(__file__), 'workers', 'traffic_worker.py')
    for f in uploaded_files:
        if f and f.filename:
            filename = secure_filename(f.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            f.save(path)
            saved.append(path)
            # spawn worker (headless recommended)
            spawn_worker(script, [path], use_module=False, extra_env={"OPENCV_HEADLESS": "1"})
    return jsonify({"success": True, "files": saved}), 200

#####################################################################################
from helmet_detection import *  # Import your helmet detection logic
import os

# --- replace helmet_detection route with this ---
@app.route('/helmet_detection', methods=['POST'])
def helmet_detection_route():
    uploaded_files = [request.files[key] for key in request.files if key.startswith('cameraFile')]
    if not uploaded_files and 'file' in request.files:
        uploaded_files = [request.files['file']]

    if not uploaded_files:
        return jsonify({"success": False, "message": "No files uploaded."}), 400

    saved = []
    for f in uploaded_files:
        if f and f.filename:
            filename = secure_filename(f.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            f.save(path)
            saved.append(path)
            # spawn worker script file:
            script = os.path.join(os.path.dirname(__file__), 'workers', 'helmet_worker.py')
            # pass env var to run headless if your module supports it:
            spawn_worker(script, [path], use_module=False, extra_env={"OPENCV_HEADLESS": "0"})
    return jsonify({"success": True, "files": saved}), 200

#####################################################################################

from traffic_violation import *
import os

# =========================
# TRAFFIC VIOLATION DETECTION
# =========================

@app.route('/traffic_violation', methods=['POST'])
def traffic_violation():
    if 'input_video' not in request.files:
        return "No file part", 400

    file = request.files['input_video']
    if file.filename == "":
        return "No selected file", 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    print("Saved:", save_path)

    # Worker logs
    worker_id = str(uuid.uuid4())[:8]
    log_out = os.path.join(WORKER_LOG_FOLDER, f"worker-{worker_id}.out.log") # type: ignore
    log_err = os.path.join(WORKER_LOG_FOLDER, f"worker-{worker_id}.err.log") # type: ignore

    # Spawn worker for traffic violation module
    worker_cmd = [
        sys.executable, "-u", "-m", "traffic_violation_video", save_path
    ]

    print(f"Spawned worker pid: {os.getpid()} cmd={' '.join(worker_cmd)}")
    print(f"Logs: stdout -> {log_out} stderr -> {log_err}")

    with open(log_out, "w") as out, open(log_err, "w") as err:
        subprocess.Popen(worker_cmd, stdout=out, stderr=err)

    # redirect to monitoring
    return redirect(f"/live_monitoring?files={filename}&processType=traffic_violation")

##############################################################################################
from heatmap_visualization import *
import os
@app.route('/heatmap_visualisation', methods=['POST'])
def heatmap_visualisation():
    """
    Handle the request for heatmap visualization and display the processed videos.
    """
    # Retrieve uploaded files
    num_cameras = int(request.form.get('numCameras', 0))
    input_files = []

    for i in range(1, num_cameras + 1):
        file = request.files.get(f'cameraFile{i}')
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            input_files.append(filepath)

    # Initialize YOLO model
    model = YOLO("yolov8n.pt")

    # Process and display videos for heatmap visualization
    process_videos(input_files, model)

    # Provide confirmation once the process is complete
    return "Heatmap visualization displayed in OpenCV window", 200

##################################

from accident import AccidentDetectionSystem
from concurrent.futures import ThreadPoolExecutor
import os

@app.route('/accident_detection', methods=['POST'])
def accident_detection():
    """
    Handle the request for accident detection and process uploaded videos using multithreading.
    """
    try:
        # Retrieve the number of uploaded files
        num_cameras = int(request.form.get('numCameras', 0))
        input_files = []

        # Collect uploaded video files
        for i in range(1, num_cameras + 1):
            file = request.files.get(f'cameraFile{i}')
            if file:
                filepath = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(filepath)
                input_files.append(filepath)

        if not input_files:
            return "No files uploaded. Please upload at least one video.", 400

        # Debugging: Print uploaded file paths
        print("Uploaded files:", input_files)

        # Initialize the AccidentDetectionSystem
        model_path = "../models/best.pt"  # Use your YOLO model path
        detector = AccidentDetectionSystem(model_path, conf_threshold=0.4, enable_gui=False)

        # Define a thread-safe function to process a single video
        def process_single_video(video_path):
            try:
                print(f"Processing video: {video_path}")
                output_path = os.path.join(UPLOAD_FOLDER, f"processed_{os.path.basename(video_path)}")
                detector.process_video_with_gui(video_path, output_path)
                print(f"Completed processing for: {video_path}")
            except Exception as e:
                print(f"Error processing {video_path}: {e}")

        # Use ThreadPoolExecutor to process videos in parallel
        max_threads = min(len(input_files), 6)  # Limit threads to 4 or the number of videos
        with ThreadPoolExecutor(max_threads) as executor:
            executor.map(process_single_video, input_files)

        # GUI Visualization for all videos
        print("Launching GUI visualization...")
        detector.process_video_with_gui(input_files)

        return "Accident Detection completed. Check the GUI window for output.", 200

    except Exception as e:
        print(f"Error during accident detection: {e}")
        return f"An error occurred: {str(e)}", 500

##############################################################################################
from triple_riding import *
import os
@app.route('/triple_riding_detection', methods=['POST'])
def triple_riding_detection_route():
    uploaded_files = [request.files[key] for key in request.files if key.startswith('cameraFile')]
    if not uploaded_files and 'file' in request.files:
        uploaded_files = [request.files['file']]

    if not uploaded_files:
        return jsonify({"success": False, "message": "No files uploaded."}), 400

    saved = []
    script = os.path.join(os.path.dirname(__file__), 'workers', 'triple_worker.py')
    for f in uploaded_files:
        if f and f.filename:
            filename = secure_filename(f.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            f.save(path)
            saved.append(path)
            spawn_worker(script, [path])
    return jsonify({"success": True, "files": saved}), 200


##############################################################################################
import cv2
import numpy as np

# Define button coordinates on the top (adjusted for larger buttons and space between them)
BUTTONS = {
   
    "Heatmap Visualization": (10, 10, 450, 60),
    "Accident Detection": (470, 10, 450, 60),
    "Triple Riding Detection": (930, 10, 450, 60),
    "Helmet Detection": (1390, 10, 450, 60),
    # "Traffic Signal Control": (1850, 10, 450, 60)# Adjusted x-coordinate for spacing
    }

# Store the last clicked button
clicked_button = None

def mouse_callback(event, x, y, flags, param):
    """Mouse callback to detect button clicks."""
    global clicked_button
    if event == cv2.EVENT_LBUTTONDOWN:
        for label, (bx, by, bw, bh) in BUTTONS.items():
            if bx < x < bx + bw and by < y < by + bh:
                clicked_button = label
                print(f"Button clicked: {label}")
                # Trigger specific action based on the button clicked
                if label == "Heatmap Visualization":
                    print("Trigger Heatmap Visualization")
                elif label == "Accident View":
                    print("Trigger Accident View")
                elif label == "Triple Riding Detection":
                    print("Triple Riding Detection")
                elif label == "Helmet Detection":
                    print("Trigger Helmet Detection")
                elif label == "Traffic Signal Control":
                    print("Trigger Traffic Signal Control")
    
def display_videos(video_paths):
    """Display videos in OpenCV with horizontal and vertical concatenation and buttons at the top."""
    cap_list = [cv2.VideoCapture(path) for path in video_paths]

    cv2.namedWindow("ATCC Process - Video Grid", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("ATCC Process - Video Grid", mouse_callback)

    while True:
        frames = []
        for cap in cap_list:
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                frames.append(None)

        # Stop if all videos are done
        if all(frame is None for frame in frames):
            break

        # Filter valid frames and find the minimum height for resizing
        valid_frames = [frame for frame in frames if frame is not None]
        if not valid_frames:
            break

        min_height = min(frame.shape[0] for frame in valid_frames)
        resized_frames = [
            cv2.resize(frame, (int(frame.shape[1] * min_height / frame.shape[0]), min_height))
            if frame is not None else np.zeros((min_height, 1, 3), dtype=np.uint8)  # Black placeholder
            for frame in frames
        ]

        # Group frames into rows of 2
        rows = [resized_frames[i:i+2] for i in range(0, len(resized_frames), 2)]

        # Ensure all frames in a row have the same height and pad if necessary
        padded_rows = []
        for row in rows:
            max_width = max(frame.shape[1] for frame in row)
            padded_row = [
                cv2.copyMakeBorder(frame, 0, 0, 0, max_width - frame.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
                for frame in row
            ]
            # Horizontally concatenate the frames in the row
            padded_rows.append(np.hstack(padded_row))

        # Ensure all rows have the same width by padding
        max_row_width = max(row.shape[1] for row in padded_rows)
        padded_rows = [
            cv2.copyMakeBorder(row, 0, 0, 0, max_row_width - row.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
            for row in padded_rows
        ]

        # Vertically concatenate the rows
        grid_frame = np.vstack(padded_rows)

        # Create a blank canvas for the button area (top of the window)
        button_area = np.zeros((100, grid_frame.shape[1], 3), dtype=np.uint8)  # Button bar height set to 100

        # Draw buttons on the top with larger font size
        font_scale = 1.2  # Increased font scale for bigger text
        for label, (x, y, w, h) in BUTTONS.items():
            cv2.rectangle(button_area, (x, y), (x + w, y + h), (200, 200, 200), -1)  # Gray button
            # Draw the text with a larger font size
            cv2.putText(button_area, label, (x + 10, y + 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3)

        # Combine the button area on top with the video grid below it
        combined_frame = np.vstack((button_area, grid_frame))

        # Display the concatenated grid with buttons at the top
        cv2.imshow("ATCC Process - Video Grid", combined_frame)

        # Wait for user input (mouse clicks or key presses)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Release all video captures and close OpenCV windows
    for cap in cap_list:
        cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    # Create the upload folder if it does not exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
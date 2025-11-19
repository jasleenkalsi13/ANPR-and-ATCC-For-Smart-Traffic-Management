🚦 ANPR & ATCC for Smart Traffic Management

An advanced AI-powered traffic monitoring system integrating Automatic Number Plate Recognition (ANPR) and Automatic Traffic Counting & Classification (ATCC). The system detects traffic violations, recognizes number plates, identifies accidents, counts vehicles, and generates heatmaps — all in real-time using deep learning and computer vision.

⭐ Features
🔹 ANPR (Automatic Number Plate Recognition)

Detects license plates

Extracts plate text via OCR

Supports multiple plate formats

🔹 ATCC (Traffic Counting & Classification)

Counts vehicles in real-time

Classifies cars, bikes, trucks, buses, etc.

Works on live and recorded videos

🔹 Traffic Violation Detection

Helmet detection

Triple riding detection

Wrong lane or rule violation detection

🔹 Accident Detection

Identifies collision events

Generates instant alerts

🔹 Heatmap Visualization

Tracks vehicle movement

Generates traffic density heatmaps

🔹 Flask Web Dashboard

Upload and process videos

View logs, detections, heatmaps

Simple and interactive UI

🧠 Tech Stack

Python 3.9 (Recommended for best compatibility)

OpenCV

YOLOv8

Tesseract OCR / Custom OCR

Flask

NumPy, Pandas, Matplotlib

MySQL (optional for logging)

Development and testing were done on Python 3.9, so using the same version is strongly recommended.

📁 Project Structure
ANPR-and-ATCC-For-Smart-Traffic-Management
│── app.py                 # Flask application
│── anpr_video.py          # ANPR detection script
│── accident.py            # Accident detection module
│── triple_riding.py       # Triple riding module
│── traffic_violation.py   # Violation detection
│── atcc.py                # Traffic counting & classification
│── heatmap_visualization.py
│── utils/                 # Utility functions
│── templates/             # HTML templates for Flask
│── static/                # CSS, JS, Images
│── uploads/               # Uploaded media
│── best/                  # YOLO model files
│── requirements.txt       # Dependencies
└── ...

🛠 Installation & Setup
✔ Recommended Python Version

Use Python 3.9 for maximum compatibility and error-free execution.

1️⃣ Clone the repository
git clone https://github.com/jasleenkalsi13/ANPR-and-ATCC-For-Smart-Traffic-Management.git
cd ANPR-and-ATCC-For-Smart-Traffic-Management

2️⃣ Create a virtual environment (optional but recommended)
macOS / Linux:
python3.9 -m venv venv
source venv/bin/activate

Windows:
python3.9 -m venv venv
venv\Scripts\activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Install Tesseract OCR
macOS:
brew install tesseract

Windows:

Download the EXE installer and install normally.
(Ensure the path is added to system environment variables)

▶️ Usage
Start the Flask Web App
python app.py


Open in browser:

http://127.0.0.1:5000/

Run ANPR only
python anpr_video.py

Run Traffic Counter
python atcc.py

📊 Outputs

Real-time annotated video with detections

Extracted ANPR text

Violation alerts

Accident detection logs

Traffic heatmap visualizations

🤝 Contributing

Pull requests and suggestions are welcome!

📄 License

This project is released under the MIT License.

💡 Author

Jasleen Kaur Kalsi

screenshots:
anpr_video.py - <img width="1162" height="619" alt="image" src="https://github.com/user-attachments/assets/5be4020c-4991-4dff-b636-6f8d80c9c612" />
accident.py - <img width="585" height="456" alt="image" src="https://github.com/user-attachments/assets/ba5112bd-f7bc-4c4e-b46b-922b57f2a9cf" />
atcc.py - <img width="520" height="801" alt="image" src="https://github.com/user-attachments/assets/c7328b40-6111-41d9-97bb-d44fae19e328" />
heatmap_visualization.py - <img width="608" height="497" alt="image" src="https://github.com/user-attachments/assets/6560e64c-de33-48d4-a272-b8881c581b9e" />
helmet_detection.py - <img width="940" height="540" alt="image" src="https://github.com/user-attachments/assets/45129ae0-e83d-4a49-a048-53de5881374b" />
traffic_violation.py - <img width="1251" height="691" alt="image" src="https://github.com/user-attachments/assets/fe76e10e-17c7-4b3a-938a-deff1592ca02" />
triple_riding.py - <img width="726" height="438" alt="image" src="https://github.com/user-attachments/assets/7248d45f-e5f3-408c-81f2-0fac4f3b049f" />
app.py :-
interface - <img width="1428" height="849" alt="image" src="https://github.com/user-attachments/assets/3bde7f20-a6e3-4fe4-a81c-da733996118f" />
search - <img width="1400" height="713" alt="image" src="https://github.com/user-attachments/assets/19e7e267-b649-4e02-8490-6947367c9a09" />
live monitoring - anpr interface - <img width="1419" height="828" alt="image" src="https://github.com/user-attachments/assets/1ed97bed-4fda-4892-8ff4-48eceac62627" />
                - atcc interface - <img width="1427" height="835" alt="image" src="https://github.com/user-attachments/assets/0487af96-fe21-486c-bd16-3cfd08c26b41" />
live monitoring - anpr output - <img width="1184" height="672" alt="image" src="https://github.com/user-attachments/assets/0b43955a-84a2-48f0-8969-41f93c7ccb0f" />
                - atcc output - Heatmap Visualization - <img width="631" height="496" alt="image" src="https://github.com/user-attachments/assets/f0819643-47d1-4b7b-8f7b-624ccd7746d8" />
                - Accident Detection - <img width="612" height="494" alt="image" src="https://github.com/user-attachments/assets/a342adac-3bb3-465e-938b-544669d1ebda" />
                - Triple Riding Detection - <img width="635" height="490" alt="image" src="https://github.com/user-attachments/assets/34bde57a-0214-4035-8d37-1c1f77398cdc" />
                - Helmet Detection - <img width="631" height="497" alt="image" src="https://github.com/user-attachments/assets/b106bf4f-9a07-4b19-85fc-c917b4e54d75" />
                - Traffic Violation Detection - <img width="1251" height="691" alt="image" src="https://github.com/user-attachments/assets/681b0cd6-59b6-43f6-9f81-feafa97b5a49" />
Upload Video - <img width="1426" height="559" alt="image" src="https://github.com/user-attachments/assets/4b649f39-07f9-40b9-8d63-82af6690e8c5" />
Results - <img width="1368" height="634" alt="image" src="https://github.com/user-attachments/assets/784afbd2-1ca1-49ce-9f02-c7e98bffb91c" />
db - <img width="440" height="126" alt="image" src="https://github.com/user-attachments/assets/0607916d-dd52-4d75-ae49-b5b966997b7f" />
















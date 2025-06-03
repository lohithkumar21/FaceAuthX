import os
import cv2
import base64
import sqlite3
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cvzone
from time import time, strftime, localtime

# Configuration
confidence = 0.6
classNames = ["fake", "real"]
model = YOLO("../model/version_1.pt") 

# Flask app initialization
app = Flask(__name__)
CORS(app)  

UPLOAD_FOLDER = "static/uploads"
DATABASE_FOLDER = "static/database_images"
DATABASE_PATH = "database/image_records.db"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATABASE_FOLDER, exist_ok=True)



# Database Initialization
def init_db():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            capture_image TEXT NOT NULL,
            result_image TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


init_db()


def save_to_database(capture_path, result_path):
    """Saves image paths to the database."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    timestamp = strftime("%Y-%m-%d %H:%M:%S", localtime())
    cursor.execute("INSERT INTO images (capture_image, result_image, timestamp) VALUES (?, ?, ?)", 
                   (capture_path, result_path, timestamp))
    conn.commit()
    conn.close()


def process_image(image_path):
    """Processes the uploaded image using YOLO model and saves the output."""
    try:
        img = cv2.imread(image_path)
        results = model(img)
        if not results:
            raise Exception("No results returned from model.")
        result = results[0]
        boxes = result.boxes
        
        if boxes is None or len(boxes) == 0:
            raise Exception("No boxes detected.")

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = classNames[cls_id] if cls_id < len(classNames) else f"class_{cls_id}"

            color = (0, 255, 0) if label.lower() == "real" else (0, 0, 255)
            cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), colorC=color, colorR=color)
            cvzone.putTextRect(img, f'{label.upper()} {int(conf * 100)}%',
                               (max(0, x1), max(35, y1)), scale=2, thickness=4,
                               colorR=color, colorB=color)

        # Save processed image
        output_path = os.path.join(UPLOAD_FOLDER, "output.jpg")
        cv2.imwrite(output_path, img)

        # Save to database
        timestamp = int(time() * 1000)
        capture_path = os.path.join(DATABASE_FOLDER, f"{timestamp}_capture_image.jpg")
        result_path = os.path.join(DATABASE_FOLDER, f"{timestamp}_result_image.jpg")
        cv2.imwrite(capture_path, cv2.imread(image_path))
        cv2.imwrite(result_path, img)
        save_to_database(capture_path, result_path)

        return output_path

    except Exception as e:
        print(f"Error during image processing: {e}")
        raise



@app.route("/")
def index():
    """Renders the main HTML page."""
    return render_template("index.html")


@app.route("/capture", methods=["POST"])
def capture():
    """Handles image capture and processing."""
    try:
        data = request.get_json()
        img_data = data.get("image")

        if not img_data:
            return jsonify({"error": "No image data provided"}), 400

        # Decode base64 image
        img_data = base64.b64decode(img_data.split(",")[1])
        input_path = os.path.join(UPLOAD_FOLDER, "input.jpg")
        with open(input_path, "wb") as f:
            f.write(img_data)

        # Process image and return the output path
        output_path = process_image(input_path)
        return jsonify({"output_image": f"/{output_path}"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/database")
def database():
    """Renders the database page."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM images")
    records = cursor.fetchall()
    conn.close()

    return render_template("database.html", records=records)

@app.route("/delete/<int:id>", methods=["DELETE"])
def delete(id):
    """Deletes the record from the database."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM images WHERE id = ?", (id,))
        conn.commit()
        conn.close()
        return '', 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

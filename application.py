import os
import requests
from flask import Flask, render_template, flash, redirect, send_from_directory, url_for, session, logging, request, session
from flask import Flask, jsonify, send_file
from PIL import Image
import io
import torch
import cv2
from werkzeug.utils import secure_filename
from database_connection import execute_query
import sqlite3
from datetime import datetime
#from yolov5.models import yolov5  # Import the yolov5 model
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# Download punkt resource
nltk.download('punkt')
nltk.download('stopwords')
from difflib import SequenceMatcher
import traceback

app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = r'uploads'
ANNOTATED_FOLDER = r'runs\detect'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ANNOTATED_FOLDER'] = ANNOTATED_FOLDER


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/logout")
def logout():
    session.pop('uname', None)  # Clear the session variable
    return redirect(url_for('index'))


@app.route("/user")
def index_auth():
    return render_template("index_auth.html")


@app.route("/profile")
def profile():
    return render_template("profile.html")


@app.route("/instruct")
def instruct():
    return render_template("instructions.html")

@app.route("/data")
def processed_data():
    return render_template("processed_data.html")

@app.route('/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['ANNOTATED_FOLDER'], filename)

@app.route('/labels/<path:filename>')
def download_file(filename):
    return send_from_directory(app.config['ANNOTATED_FOLDER'], filename)


from flask import render_template
from flask import url_for

@app.route('/get_data', methods=['POST'])
def get_data():
    try:
        name = request.form.get('namee')
        email = request.form.get('email')

        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file'})

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Run YOLOv7 detection script
        detect_command = ( 
            f"python3 yolov7/detect.py "
            f"--weights yolov7_best.pt "
            f"--conf 0.5 --img-size 640 --save-txt --save-conf --source {file_path}"
        )
        os.system(detect_command)
        basepath = r"runs/detect"


        # Find the maximum numbered directory
        max_exp_directory = find_max_exp_number(basepath)

        # Construct the path to the annotated image
        annotated_image_path = os.path.join(basepath, max_exp_directory, filename)
        filename_without_extension = ".".join(filename.split(".")[:-1])
        annotated_label_path = os.path.join(basepath, max_exp_directory, "labels", f"{filename_without_extension}.txt")

        #annotated_label_path = os.path.join(basepath, max_exp_directory,"labels", filename.split(".")[-1].replace(".txt"))
        print(annotated_label_path)

        # Move the annotated image to a different directory accessible by Flask
        annotated_image_destination = os.path.join(app.config['ANNOTATED_FOLDER'], filename)

        # Overwrite the existing destination file, if it exists
        if os.path.exists(annotated_image_destination):
            os.remove(annotated_image_destination)

        os.rename(annotated_image_path, annotated_image_destination)

        # Pass the image URL to the template using url_for

        
        
        # Read the confidence value from the label file
        # Read the class and confidence value from the label file
        with open(annotated_label_path, 'r') as label_file:
            line = label_file.readline().strip()
            if line:  # Check if the line is not empty
                parts = line.split()
                if len(parts) >= 6:  # Ensure there are enough elements in the line
                    class_value = int(parts[0])
                    confidence_value = float(parts[-1])
                    print(class_value,confidence_value)
                else:
                    class_value = 0  # Default class value if the line doesn't have enough elements
                    confidence_value = 0.0  # Default confidence value if the line is not as expected
            else:
                class_value = 0  # Default class value if the line is empty
                confidence_value = 0.0  # Default confidence value if the line is empty
        
        with open(file_path, 'rb') as image_file:
            image_data = image_file.read()
            
        with open(annotated_image_destination, 'rb') as a_image_file:
            a_image_data = a_image_file.read()
            
        with open(annotated_label_path, 'rb') as aimage_file:
            aimage_data = aimage_file.read()
        print("!here i aam ",annotated_image_destination )
        # with open(annotated_image_destination, 'rb') as image_file2:
        #     a_image_data = image_file2.read()
        # print("!here i am ",basepath + filename )

            
        # with open(annotated_image_path, 'rb') as image_file:
        #     annotated_image_data = image_file.read()

        # Insert the image data into the database
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        # Your SQL query to insert data
        insert_query = "INSERT INTO tbl_fetal_data (upload_image, username, annotated_label, annotated_image, date) VALUES (?, ?, ?, ?, ?)"
        print(insert_query)

        # Execute the query with parameters
        execute_query(insert_query, sqlite3.Binary(image_data), session['uname'],aimage_data,sqlite3.Binary(a_image_data), formatted_datetime)
        print("here")
        image_url = url_for('uploaded_file', filename=filename)
        print("Romesh noob hai : ",image_url)
        return render_template('result_template.html', image_url=image_url, labels=annotated_label_path, confidence_value=confidence_value,class_value=class_value)
    
    except Exception as e:

        print("Exception:", e)
        traceback.print_exc()
        
        # Add proper error handling
        error_message = "The model Currently fails to identify any Anomaly or may be Normal brain to be more precise go for further analysis."
        print("Exception:", error_message)
        
        return render_template('result_template.html', error_message=error_message)
     

            
    


@app.route('/pred_page')
def pred_page():
    pred = session.get('pred_label', None)
    f_name = session.get('filename', None)
    return render_template('pred.html', pred=pred, f_name=f_name)



@app.route("/upload", methods=['POST', 'GET'])
def upload():
    try:
        if request.method == 'POST':
            f = request.files['bt_image']
            filename = str(f.filename)

            if filename != '':
                ext = filename.split(".")

                if ext[1] in ALLOWED_EXTENSIONS:
                    filename = secure_filename(f.filename)

                    f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

                    with open(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'rb') as img:
                        predicted = requests.post("http://localhost:5000/predict", files={"file": img}).json()

                    session['pred_label'] = predicted['class_name']
                    session['filename'] = filename

                    return redirect(url_for('templates\pred_page'))

    except Exception as e:
        print("Exception\n")
        print(e, '\n')

    return render_template("upload.html")
    

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
            uname = request.form["uname"]
            passw = request.form["passw"]

            # Check if the user exists in the database (for testing, replace with your actual query)
            login = execute_query("SELECT * FROM tbl_fetal_main WHERE username = ? AND password = ?", uname, passw)

            if login:
                # User exists, perform login logic if needed
                print("Login Success")
                session['uname'] = uname
                return redirect(url_for('index_auth'))
            else:
                # User doesn't exist, display a message
                message = "Username does not exist. Please sign up."
                return render_template("login.html", message=message)

    return render_template("login.html")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/get_results_api_yolov5', methods=['POST'])
def get_results_api_yolov5():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'})

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    img = cv2.imread(file_path)
    img = cv2.resize(img, (512, 512))
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # Inference using the YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'best.pt')
    results = model(img_tensor)
    print(results)

    highest_confidence_index = -1
    highest_confidence = 0.0

    # Find the box with the highest confidence
    if hasattr(results, '__iter__'):
        for i, pred in enumerate(results):
            conf = pred[..., 4]  # confidence scores
            max_conf_in_pred = torch.max(conf).item()
            if max_conf_in_pred > highest_confidence:
                highest_confidence = max_conf_in_pred
                highest_confidence_index = i

    # Draw only the box with the highest confidence
    if highest_confidence_index != -1:
        pred = results[highest_confidence_index]
        conf = pred[..., 4]  # confidence scores
        bbox = pred[..., :4]  # bounding box coordinates

        for c, box in zip(conf, bbox):
            x_min, y_min, x_max, y_max = box  # Unpack bounding box coordinates
            conf_val = float(c)

            # Draw the box with the highest confidence in a different color (e.g., red)
            if conf_val == highest_confidence:
                color = (0, 0, 255)  # Red color (BGR format)
                cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
                label_text = f"Confidence: {conf_val:.2f}"
                cv2.putText(img, label_text, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # (remaining code for saving and returning the annotated image)
    # Save the annotated image
    annotated_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'annotated_' + filename)
    cv2.imwrite(annotated_image_path, img)

    # Send the annotated image as a response
    return send_file(annotated_image_path, mimetype='image/jpeg', as_attachment=True)

@app.route('/get_results_api_yolov7', methods=['POST'])
def get_results_api_yolov7():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'})

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Run YOLOv7 detection script
    detect_command = (
        f"python yolov7\detect.py "
        f"--weights yolov7_best.pt "
        f"--conf 0.5 --img-size 640 --source {file_path}"
    )
    os.system(detect_command)
    basepath = r"runs\detect"

    # Send the annotated image as a response
    annotated_image_path = os.path.join('runs', 'detect', find_max_exp_number(basepath), filename)
    return send_file(annotated_image_path, mimetype='image/jpeg', as_attachment=True)

from pathlib import Path
def find_max_exp_number(base_path, prefix='exp'):
    """Find the maximum numbered directory with a given prefix."""
    max_exp_number = -1
    max_exp_directory = None

    for directory in os.listdir(base_path):
        if directory.startswith(prefix) and directory[len(prefix):].isdigit():
            exp_number = int(directory[len(prefix):])
            if exp_number > max_exp_number:
                max_exp_number = exp_number
                max_exp_directory = directory

    return max_exp_directory



def check_username(username):
    query = "SELECT * FROM tbl_fetal_main WHERE username = ?"
    result = execute_query(query, username)
    return bool(result)

def check_email(email):
    query = "SELECT * FROM tbl_fetal_main WHERE email = ?"
    result = execute_query(query, email)
    return bool(result)

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        uname = request.form['uname']
        mail = request.form['mail']
        passw = request.form['passw']

        # Check if username or email already exist
        if check_username(uname):
            username_exists_message = "Username already exists."
            return render_template("register.html", username_exists_message=username_exists_message)
        
        if check_email(mail):
            email_exists_message = "Email already exists."
            return render_template("register.html", email_exists_message=email_exists_message)
    
        # Insert the new user into the database
        query = "INSERT INTO tbl_fetal_main (username, email, password) VALUES (?, ?, ?)"
        success = execute_query(query, uname, mail, passw)

        if success:
            return redirect(url_for("login"))
        else:
            # Handle the case where the insertion failed (display an error message, redirect, etc.)
            return render_template("register.html", message="Failed to register user.")

    return render_template("register.html")

import base64
import io
def compress_and_encode_blob(image_blob, quality=85):
    try:
        # Convert the BLOB data to a Pillow Image
        image = Image.open(io.BytesIO(image_blob))

        # Convert RGBA to RGB if the image has an alpha channel
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        # Create an in-memory stream to save the compressed image
        compressed_stream = io.BytesIO()

        # Save the image with compression
        image.save(compressed_stream, format='JPEG', quality=quality)

        # Get the compressed image data
        compressed_data = compressed_stream.getvalue()

        # Encode the compressed image data to Base64
        encoded_image = base64.b64encode(compressed_data).decode('utf-8')

        return encoded_image
    except Exception as e:
        print(f"Error compressing and encoding image: {e}")
        return None

@app.route('/get_tabulator_data', methods=["POST"])
def get_tabulator_data():
    try:
        data = request.json
        username = data.get('uname')

        # Now you have the username, you can use it in a database query
        get_data_query = f"""SELECT date, upload_image, annotated_image
                             FROM tbl_fetal_data
                             WHERE username = '{username}'"""
        result = execute_query(get_data_query)

        # Process the result and send a response
        response_data = []

        for row in result:
            date = row[0]  # Access the elements using integer indices
            uploaded_image_blob = row[1]
            annotated_image_blob = row[2]
            
            if date is not None and uploaded_image_blob is not None and annotated_image_blob is not None:

                # Convert blob to base64
                uploaded_image_base64 = compress_and_encode_blob(uploaded_image_blob)
                annotated_image_base64 = compress_and_encode_blob(annotated_image_blob)

                response_data.append({'date': date, 'upload_image': uploaded_image_base64, 'annotated_image': annotated_image_base64})
                app.logger.info('Image data: %s', uploaded_image_base64)  # Add logging statement
        return jsonify({'result': response_data})
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/get_api_details', methods=["GET"])
def get_api_details():
    return render_template("apiDetails.html")

@app.route('/question', methods=["GET"])
def question():
    return render_template("question.html")


@app.route('/process_text_yes', methods=['POST'])
def process_text_yes():
    if request.method == 'POST':
        input_text = request.form.get('input_text')
        given_names = request.form.getlist('names[]') 
        print(given_names)
        if input_text:
            # List of anomaly descriptions
            anomalies = {
            "Arnold Chiari Malformation": "A condition where brain tissue extends into the spinal canal, causing symptoms such as difficulty swallowing or changes in voice.",
            "Arachnoid Cyst": "The observation of unusual sensations or pain that varies with head position, possibly associated with the presence of cysts.",
            "Cerebellar Hypoplasia": "Underdevelopment of the cerebellum, leading to difficulties in the baby's motor coordination or movements.",
            "Cisterna Magna": "An abnormal enlargement of the cisterna magna, a space at the back of the brain, which may be associated with developmental concerns.",
            "Colpocephaly": "Abnormal enlargement of the occipital horns of the brain, potentially leading to noticeable head shape abnormalities.",
            "Encephalocele": "Protrusion of brain tissue through the skull, often visible as a sac-like structure on the baby's head.",
            "Holoprosencephaly": "Observations related to facial abnormalities during ultrasound examinations, indicating incomplete separation of the brain hemispheres.",
            "Hydranencephaly": "The absence of cerebral hemispheres, often leading to severe developmental challenges.",
            "Intracranial Hemorrhage": "Incidents of trauma during pregnancy that might be related to bleeding within the skull, causing symptoms like severe headaches or changes in consciousness.",
            "Intracranial Tumor": "The presence of abnormal growths in the brain, as indicated by ultrasound or other imaging reports.",
            "Mild Ventriculomegaly": "Mild enlargement of the cerebral ventricles, often requiring further evaluation for potential developmental concerns.",
            "Moderate Ventriculomegaly": "Moderate enlargement of the cerebral ventricles, indicating a more pronounced concern for developmental issues.",
            "Polencephaly": "Observations related to the presence of cavities or cysts within the brain, indicating abnormal brain development.",
            "Severe Ventriculomegaly": "Measurements or comments about the size of the cerebral ventricles in ultrasound reports, indicating severe enlargement and potential developmental challenges."
        }



            # Function to find the similarity scores for all anomalies
            similarity_scores = {anomaly: SequenceMatcher(None, input_text.lower(), desc.lower()).ratio() for anomaly, desc in anomalies.items()}

            # Create a list of anomalies with their similarity scores
            matching_anomalies = [{"anomaly": anomaly, "similarity_score": score} for anomaly, score in similarity_scores.items()]
            
            # Update the similarity scores based on the presence of names
            Add_value = 1 / len(given_names)
            print(Add_value)
            for matching_anomaly in matching_anomalies:
                anomaly_name = matching_anomaly["anomaly"]
                if anomaly_name in given_names:
                    matching_anomaly["similarity_score"] += Add_value
                    matching_anomaly["similarity_score"] /= 2


            # Find the anomaly with the maximum similarity score
            max_anomaly = max(matching_anomalies, key=lambda x: x["similarity_score"])

            # Print the result
            result = {
                "input": input_text,
                "matching_anomalies": matching_anomalies,
                "max_anomaly": max_anomaly
            }

            return jsonify(result)
        else:
            result = {
                "input": "No Result found"
            }
            return jsonify(result)



@app.route('/process_text_no', methods=['POST'])
def process_text_no():
    if request.method == 'POST':
        input_text = request.form.get('input_text')
        given_names = request.form.getlist('names[]') 
        print(given_names)
        if input_text:
            # List of anomaly descriptions
            anomalies = {
                "Arnold Chiari Malformation": "A condition where brain tissue extends into the spinal canal, causing symptoms such as difficulty swallowing or changes in voice.",
                "Hydrocephalus": "A buildup of fluid in the brain that leads to increased pressure, potentially caused by trauma during pregnancy.",
                "Intracranial Tumor": "The presence of abnormal growths in the brain, as indicated by ultrasound or other imaging reports.",
                "Arachnoid Cyst": "The observation of unusual sensations or pain that varies with head position, possibly associated with the presence of cysts.",
                "Intracranial Hemorrhage": "Incidents of trauma during pregnancy that might be related to bleeding within the skull, causing symptoms like severe headaches or changes in consciousness.",
                "Cerebellar Hypoplasia": "Underdevelopment of the cerebellum, leading to difficulties in the baby's motor coordination or movements.",
                "Holoprosencephaly": "Observations related to facial abnormalities during ultrasound examinations.",
                "Severe Ventriculomegaly": "Measurements or comments about the size of the cerebral ventricles in ultrasound reports, indicating severe enlargement.",
                "Polencephaly": "Observations related to the presence of cavities or cysts within the brain.",
            }


            # Function to find the similarity scores for all anomalies
            similarity_scores = {anomaly: SequenceMatcher(None, input_text.lower(), desc.lower()).ratio() for anomaly, desc in anomalies.items()}

            # Create a list of anomalies with their similarity scores
            matching_anomalies = [{"anomaly": anomaly, "similarity_score": score} for anomaly, score in similarity_scores.items()]
            
            # Update the similarity scores based on the presence of names
            Add_value = 1 / len(given_names)
            for matching_anomaly in matching_anomalies:
                anomaly_name = matching_anomaly["anomaly"]
                if anomaly_name in given_names:
                    matching_anomaly["similarity_score"] += Add_value
                    matching_anomaly["similarity_score"] /= 2


            # Find the anomaly with the maximum similarity score
            max_anomaly = max(matching_anomalies, key=lambda x: x["similarity_score"])

            # Print the result
            result = {
                "input": input_text,
                "matching_anomalies": matching_anomalies,
                "max_anomaly": max_anomaly
            }

            return jsonify(result)
        else:
            result = {
                "input": "No Result found"
            }
            return jsonify(result)


if __name__ == "__main__":
    app.config['SECRET_KEY'] = os.urandom(24)
    app.run(debug=True)

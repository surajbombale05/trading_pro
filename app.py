from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
from your_model import predict_market_direction  # Model function for prediction

app = Flask(__name__)

# Folder to store uploaded images
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Homepage route with form to upload an image
@app.route('/')
def index():
    return render_template('index.html')

# Handle image upload and prediction
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Load and process the image using OpenCV
    image = cv2.imread(file_path)
    prediction = predict_market_direction(image)  # Predict market move

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)

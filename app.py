from flask import Flask, render_template, request, redirect, url_for
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load model
model = load_model("model.h5")

# Class labels
class_names = [
    'anadenanthera', 'arecaceae', 'arrabidaea', 'cecropia', 'chromolaena',
    'combretum', 'croton', 'dipteryx', 'eucalipto', 'faramea', 'hyptis',
    'mabea', 'matayba', 'mimosa', 'myrcia', 'protium', 'qualea', 'schinus',
    'senegalia', 'serjania', 'syagrus', 'tridax', 'urochloa'
]

UPLOAD_FOLDER = "static/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Image preprocessing
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Get prediction probabilities
            prediction = model.predict(img_array)
            predicted_index = np.argmax(prediction)
            predicted_class = class_names[predicted_index]
            
            # Calculate confidence score
            confidence = float(prediction[0][predicted_index]) * 100
            confidence = round(confidence, 2)  # Round to 2 decimal places

            # Pass image_url and confidence to template
            return render_template(
                'prediction.html',
                prediction=predicted_class,
                image_url=url_for('static', filename=f'uploads/{file.filename}'),
                confidence=confidence
            )

    return render_template('index.html')

@app.route('/logout')
def logout():
    return render_template('logout.html')

if __name__ == '__main__':
    app.run(debug=True)
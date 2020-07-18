from __future__ import division, print_function
import os
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
app = Flask(__name__)
def model_predict(img_path):
    model = tf.keras.models.load_model('malaria_125_125.h5')
    img = image.load_img(img_path, target_size=(125, 125), color_mode='grayscale')
    data = image.img_to_array(img)
    data = np.expand_dims(data, axis=0)
    preds = model.predict_classes(data)
    if preds == 0:
        preds = "Infected "
    else:
        preds = "Uninfected"

    return preds
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        preds = model_predict(file_path)
        result = preds
        os.remove(file_path)
        return result
    return render_template('index.html', myvar=result)
if __name__ == '__main__':
    app.run(debug=True)
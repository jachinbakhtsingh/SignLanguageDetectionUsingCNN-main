from flask import Flask, request, jsonify
from keras.models import model_from_json
import cv2
import numpy as np
import os
from flask_cors import CORS
import base64

app = Flask(__name__)
CORS(app)

json_file = open("signlanguagedetection48x48.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("signlanguagedetection48x48.h5")

label = ['A', 'M', 'N', 'S', 'T', 'blank']

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0

@app.route('/hi', methods=['GET', 'POST'])
def hi():
    print(" Hi called >>>>>>>>>>>>>>>>>>>>")
    return 'Testing'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image data and time from the request
        image_data = request.json['image']
        image_time = request.json['time']

        # Convert the base64 image data to bytes
        image_bytes = base64.b64decode(image_data.split(',')[1])
        # Write the bytes to a file
        file_path = 'received_image.jpg'
        with open(file_path, 'wb') as image_file:
            image_file.write(image_bytes)

        # Read the saved image file
        img = cv2.imread(file_path)
        
        # Perform image processing and prediction
        cv2.rectangle(img, (0, 40), (300, 300), (0, 165, 255), 1)
        cropframe = img[40:300, 0:300]
        cropframe = cv2.cvtColor(cropframe, cv2.COLOR_BGR2GRAY)
        cropframe = cv2.resize(cropframe, (48, 48))
        cropframe = extract_features(cropframe)
        pred = model.predict(cropframe)
        prediction_label = label[pred.argmax()]
        if prediction_label == 'blank':
            result = " "
        else:
            accu = "{:.2f}".format(np.max(pred)*100)
            result = f'{prediction_label} {accu}%'

        return jsonify(result=result, time=image_time)
    except Exception as e:
        print("Error:", e)
        return jsonify(result="Error processing image")

if __name__ == '__main__':
    # Use Flask's built-in development server with HTTPS support
    # app.run(debug=True, ssl_context=('C:/Users/lenovo/OneDrive/Desktop/cert.pem', 'C:/Users/lenovo/OneDrive/Desktop/New folder/SignLanguageDetectionUsingCNN-main/key.pem'))
    app.run(debug=True)
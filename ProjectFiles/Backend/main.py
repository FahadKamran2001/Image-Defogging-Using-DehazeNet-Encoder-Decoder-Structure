import base64
import numpy as np
import cv2
from flask import Flask, jsonify, request
from flask_cors import CORS
from keras.models import load_model
from keras.applications import VGG16
from keras.losses import mean_squared_error

app = Flask(__name__)
CORS(app)

# Load the trained dehazing model


def defog_image(image, value):
    modelName = ''

    if(value == 1):
        modelName = 'dehaze_250.h5'
    elif (value == 2):
        modelName = 'dehaze_500.h5'
    elif (value == 3):
        modelName = 'dehaze_750.h5'
    elif (value == 4):
        modelName = 'dehaze_1000.h5'


    model = load_model(modelName, custom_objects={'perceptual_loss': mean_squared_error})
    # Preprocess the image (resize and normalize)
    target_height = 480
    target_width = 640
    image = cv2.resize(image, (target_width, target_height))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)

    # Use the model to predict the dehazed image
    dehazed_image = model.predict(image)

    # Rescale the pixel values to the original range
    dehazed_image = dehazed_image.squeeze() * 255.0
    dehazed_image = dehazed_image.astype('uint8')

    # Resize the dehazed image to the original image size
    dehazed_image = cv2.resize(dehazed_image, (image.shape[2], image.shape[1]))  # Resize to (target_width, target_height)

    return dehazed_image



@app.route("/")
def home():
    return 'Image Defogging backend!'

@app.route('/image', methods=['POST'])
def uploaded_image():
    images = []

    value = int(request.form.get('value')) #this is the slider value

    print("This is the key")
    print(value)
    for key, file in request.files.items():
        if file:
            nparr = np.frombuffer(file.read(), np.uint8)
            foggy_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if foggy_image is not None:
                processed_image = defog_image(foggy_image, value)

                _, buffer = cv2.imencode('.jpg', processed_image)
                encoded_image = base64.b64encode(buffer).decode('utf-8')
                images.append(encoded_image)

    return jsonify({'message': 'Images received successfully!', 'images': images})

if __name__ == '__main__':
    app.run(debug=True)

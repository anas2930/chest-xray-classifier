from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)
model = tf.keras.models.load_model('model/vgg16_finetuned.h5')

OPTIMAL_THRESHOLD = 0.9990534
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']

def preprocess_xray(image_bytes, target_size=(224,224), train_mean=0.482, train_std=0.236):
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    h, w = image.shape
    scale = min(target_size[0]/h, target_size[1]/w)
    new_h, new_w = int(h*scale), int(w*scale)
    resized = cv2.resize(image, (new_w, new_h))
    pad_h, pad_w = target_size[0]-new_h, target_size[1]-new_w
    top, bottom = pad_h//2, pad_h-pad_h//2
    left, right = pad_w//2, pad_w-pad_w//2
    image = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    image = image.astype(np.float32) / 255.0
    image = (image - train_mean) / train_std
    image = np.stack([image]*3, axis=-1)
    return np.expand_dims(image, axis=0)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    processed = preprocess_xray(file.read())
    prob = float(model.predict(processed, verbose=0)[0][0])
    prediction = CLASS_NAMES[1] if prob > OPTIMAL_THRESHOLD else CLASS_NAMES[0]
    return jsonify({'prediction': prediction, 'probability': prob})

if __name__ == '__main__':
    app.run(debug=True)
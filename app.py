from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import os
import pickle

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and label binarizer
model = load_model('models/plantmodelcnn4.hdf5')
label_binarizer = pickle.load(open('models/plant_disease_label_transformnew4.pkl', 'rb'))
class_labels = label_binarizer.classes_

# Define image size based on training
image_size = (64, 64)

def process_image(image_path):
    """Preprocess the uploaded image to match the training process."""
    img = cv2.imread(image_path)  # Use cv2 to read the image as in training
    img = cv2.resize(img, image_size)  # Resize to (64, 64)
    img = img_to_array(img) / 255.0  # Convert to array and normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def model_predict(img_path, model):
    """Predict the class of the plant disease from the uploaded image."""
    img = process_image(img_path)
    
    # Debugging: Print shape of the processed image
    print(f"Processed image shape: {img.shape}")
    
    preds = model.predict(img)
    
    # Debugging: Print raw prediction output
    print(f"Raw prediction output: {preds}")
    
    if len(preds) == 0:
        print("Error: No predictions returned by the model")
        return None
    
    # Get the predicted class index
    pred_class = np.argmax(preds, axis=1)
    
    # Debugging: Print the predicted class index
    print(f"Predicted class index: {pred_class}")
    
    return pred_class

# Route for home and image upload
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file:
            # Save the uploaded image
            file_path = os.path.join('static/uploads', file.filename)
            file.save(file_path)

            # Predict using the model
            pred_class = model_predict(file_path, model)

            if pred_class is None or len(pred_class) == 0:
                return "Prediction failed, please try again."

            predicted_label = class_labels[pred_class[0]]  # Get the predicted label

            return render_template('index.html', prediction=predicted_label, img_path=file.filename)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

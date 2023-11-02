from flask import Flask, render_template, request
from skimage import io, color, util
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier  # Import the model (for example)
import joblib
app = Flask(__name__)

with open('model/knn.pkl', 'rb') as file:
    model = pickle.load(file)


def calculate_glcm_properties_and_entropy(image_path):
    # Load the image (grayscale)
    image = io.imread(image_path)
    gray_image_f = color.rgb2gray(image)  # Convert to grayscale if it's a color image
    gray_image = util.img_as_ubyte(gray_image_f)
    # Calculate GLCM with specified parameters
    distances = [1]  # Distance between pixels
    angles = [0]  # Angles for GLCM computation
    glcm = graycomatrix(gray_image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)

    # Calculate GLCM properties
    contrast = graycoprops(glcm, 'contrast').ravel().tolist()
    dissimilarity = graycoprops(glcm, 'dissimilarity').ravel().tolist()
    homogeneity = graycoprops(glcm, 'homogeneity').ravel().tolist()
    energy = graycoprops(glcm, 'energy').ravel().tolist()
    correlation = graycoprops(glcm, 'correlation').ravel().tolist()

    # Calculate entropy
    entropy_value = shannon_entropy(gray_image)

    # Return GLCM properties and entropy as a list
    glcm_properties_and_entropy = contrast + dissimilarity + homogeneity + energy + correlation + [entropy_value]
    return glcm_properties_and_entropy

# Define the homepage route
@app.route('/')
def homepage():
    return render_template('index3.html')

# Define a route for handling image uploads and predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Process the uploaded image and form data
        image = request.files['image']
        patient_name = request.form['patient_name']
        age = request.form['age']
        registration_id = request.form['registration_id']
        doctor_name = request.form['doctor_name']
        image_type = request.form['image_type']
        image_path = "uploaded_image.jpg"
        image.save(image_path)
        
        # Calculate GLCM properties and entropy
        glcm_properties_and_entropy = calculate_glcm_properties_and_entropy(image_path)
        
        # features is a list containing GLCM properties and entropy
        features = [glcm_properties_and_entropy]

        # Make a prediction using the model
        prediction_result = model[0].predict(features)
        return render_template('result2.html',  prediction=prediction_result, patient_name=patient_name, age=age, registration_id=registration_id, doctor_name=doctor_name, image_type=image_type)

if __name__ == '__main__':
    app.run(debug=True)
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input  # Correct import for preprocess_input
import numpy as np
from io import BytesIO
from flask import Flask, request, render_template, jsonify, session

app = Flask(__name__)

# Replace food_classes with the correct list if necessary
food_classes = [
    "burger", "butter_naan", "chai", "chapati", "chole_bhature",
    "dal_makhani", "dhokla", "fried_rice", "idli", "jalebi",
    "kaathi_rolls", "kadai_paneer", "kulfi", "masala_dosa", "momos",
    "paani_puri", "pav_bhaji", "pizza", "samosa"
]

# Load the trained food recognition model
model_path = "Model/model_v1_inceptionV3.h5"
food_recognition_model = load_model(model_path)

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files.get('image')  # Get the uploaded file
    if file:
        try:
            # Read the file and convert it to a suitable format for the model
            file_stream = BytesIO(file.read())
            image = load_img(file_stream, target_size=(299, 299))  # InceptionV3 input size
            image_array = img_to_array(image)  # Convert to numpy array
            image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
            image_array = preprocess_input(image_array)  # Preprocess for InceptionV3

            # Debugging: Check input shape
            print("Input shape:", image_array.shape)

            # Make a prediction using the model
            predictions = food_recognition_model.predict(image_array)
            print("Predictions:", predictions)  # Debugging

            food_index = np.argmax(predictions)
            print("Predicted Index:", food_index)

            # Handle out-of-range predictions
            if food_index >= len(food_classes):
                return jsonify({"error": "Prediction index out of range"}), 400

            food_label = food_classes[food_index]
            print("Predicted Food:", food_label)

            # Check user allergies and provide feedback
            user_allergies = session.get('allergies', [])
            detected_allergens = detect_allergens(food_label)  # Implement allergen detection as needed

            if any(allergen in user_allergies for allergen in detected_allergens):
                message = f"Warning! The food '{food_label}' contains allergens: {', '.join(detected_allergens)}."
            else:
                message = f"No allergens detected in '{food_label}'. Safe to consume."

            return render_template('result.html', food_label=food_label, message=message)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "No file uploaded"}), 400

def detect_allergens(food_label):
    # Placeholder function for detecting allergens based on food label
    # Replace with the actual logic for your allergen detection model
    return ["Gluten", "Nuts"]

if __name__ == '__main__':
    app.run(debug=True)

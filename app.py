import os
import numpy as np
from flask import Flask, request, jsonify, render_template, session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Initialize the Flask app
app = Flask(__name__)
# app.secret_key = 'your_secret_key'  # For session management

#new add
import pandas as pd
allergy_data = pd.read_csv('Food_Allergy_Data.csv')

# Load the pre-trained model
model_path = "Model/model_v1_inceptionV3.h5"  # Update the path to your model
food_recognition_model = load_model(model_path)

# Define the list of food classes (based on your dataset structure)
food_classes = [
    "burger", "butter_naan", "chai", "chapati", "chole_bhature",
    "dal_makhani", "dhokla", "fried_rice", "idli", "jalebi",
    "kaathi_rolls", "kadai_paneer", "kulfi", "masala_dosa", "momos",
    "paani_puri", "pakode", "pav_bhaji", "pizza", "samosa"
]


# Define routes for your Flask app
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/image_upload')
def image_upload():
    return render_template('image_upload.html')

@app.route('/health')
def health():
    return render_template('health.html')

@app.route('/gender')
def gender():
    return render_template('gender.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files.get('image')  # Get the uploaded file
    if file:
        try:
            # Ensure the uploads folder exists
            upload_folder = 'static/uploads/'
            os.makedirs(upload_folder, exist_ok=True)

            # Save the uploaded image
            file_path = os.path.join(upload_folder, file.filename)
            file.save(file_path)

            # Set the image URL relative to 'static'
            image_url = f"uploads/{file.filename}"  # Static-relative path

            # Process the uploaded image for food recognition
            from io import BytesIO
            file_stream = BytesIO(file.read())
            image = load_img(file_path, target_size=(299, 299))  # Load image at required size
            image_array = img_to_array(image) / 255.0  # Normalize image
            image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

            # Predict food label using the model
            predictions = food_recognition_model.predict(image_array)
            food_index = np.argmax(predictions)

            # Get the predicted food label
            if food_index < len(food_classes):
                food_label = food_classes[food_index]
            else:
                return jsonify({'error': 'Prediction index out of range'}), 400

            # Step 1: Normalize user allergies (to lowercase and strip spaces)
            user_allergies = session.get('allergies', [])  # List of user-entered allergies
            user_allergies_normalized = [allergen.strip().lower() for allergen in user_allergies]

            # Step 2: Normalize detected allergens (to lowercase and strip spaces)
            detected_allergens = detect_allergens(food_label)  # Call to your detect_allergens function
            detected_allergens_normalized = [allergen.strip().lower() for allergen in detected_allergens]

            # Step 3: Match detected allergens with user allergies
            matching_allergens = [allergen for allergen in detected_allergens_normalized if allergen in user_allergies_normalized]

            # Debugging step - check the values of allergies for logging purposes
            print("User Allergies (Normalized):", user_allergies_normalized)
            print("Detected Allergens (Normalized):", detected_allergens_normalized)
            print("Matching Allergens:", matching_allergens)

            # Step 4: Prepare allergy message
            if matching_allergens:
                allergy_message = f"Allergy Matched! Do not consume '{food_label}' as it contains: {', '.join(matching_allergens)}."
            else:
                allergy_message = f"You are safe to consume '{food_label}'. No matched allergens detected."

            # Other messages
            message = f"Food '{food_label}' contains the following allergens: {', '.join(detected_allergens)}." if detected_allergens else "No allergens detected."
            health_recommendations = "Please consult a doctor if you are unsure about allergens."

            # Render the result template
            return render_template('result.html',
                                   food_label=food_label,
                                   message=message,
                                   allergy_message=allergy_message,
                                   allergens=detected_allergens,
                                   image_url=image_url,
                                   health_recommendations=health_recommendations)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'No file uploaded'}), 400



#new added
def detect_allergens(food_label):
    matched_row = allergy_data[allergy_data['Food Item'].str.lower() == food_label.lower()]
    
    if not matched_row.empty:
        allergens = matched_row['Common Allergies'].values[0]
        
        # Check if allergens is NaN and handle it
        if pd.isna(allergens):
            return []  # No allergens detected if NaN
        
        # Split the allergen string into a list
        return allergens.split(', ')
    
    return []  # Return empty list if no match found


@app.route('/submit_health_info', methods=['POST'])
def submit_health_info():
    data = request.json
    allergies = data.get('allergies', [])
    session['allergies'] = allergies  # Store allergies in the session
    return jsonify({'message': 'Health information received'}), 200

if __name__ == '__main__':
    app.run(debug=True)

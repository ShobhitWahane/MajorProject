import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pymysql
from werkzeug.security import generate_password_hash, check_password_hash

# --- Initialize Flask App ---
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Use a secure random secret key

# --- Load Pre-Trained Model ---
model_path = "Model/model_v1_inceptionV3.h5"
food_recognition_model = load_model(model_path)
print("Model loaded successfully!")

# --- Load Allergen CSV Data ---
allergy_data = pd.read_csv('Food_Allergy_Data.csv')
print("Allergy data loaded successfully!")

# --- Define Food Classes ---
food_classes = [
    "burger", "butter_naan", "chai", "chapati", "chole_bhature",
    "dal_makhani", "dhokla", "fried_rice", "idli", "jalebi",
    "kaathi_rolls", "kadai_paneer", "kulfi", "masala_dosa", "momos",
    "paani_puri", "pakode", "pav_bhaji", "pizza", "samosa"
]

# --- Database Connection ---
def get_db_connection():
    try:
        connection = pymysql.connect(
            host='localhost',
            user='root',
            password='Sadb@123',  # Update with your MySQL password
            database='foodallergyapp'
        )
        return connection
    except Exception as e:
        print(f"Database connection failed: {str(e)}")
        return None


# --- Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/gender')
def gender():
    return render_template('gender.html')

@app.route('/health')
def health():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('health.html')

@app.route('/image_upload', methods=['GET'])
def image_upload():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('image_upload.html')

@app.route('/register', methods=['POST'])
def register():
    email = request.form.get('email')
    password = request.form.get('password')

    if not email or not password:
        return jsonify({'error': 'Email and password are required!'}), 400

    hashed_password = generate_password_hash(password)

    connection = get_db_connection()
    if not connection:
        return jsonify({'error': 'Database connection failed!'}), 500

    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT user_id FROM users WHERE email = %s", (email,))
            if cursor.fetchone():
                return jsonify({'error': 'User already exists!'}), 400

            cursor.execute("INSERT INTO users (email, password) VALUES (%s, %s)", (email, hashed_password))
            connection.commit()
            return jsonify({'message': 'Registration successful!'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        connection.close()

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        if not email or not password:
            return "Email and password are required!", 400

        connection = get_db_connection()
        if not connection:
            return "Database connection failed!", 500

        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT user_id, password FROM users WHERE email = %s", (email,))
                user = cursor.fetchone()

                if user:
                    stored_hash = user[1]  # Hashed password
                    if check_password_hash(stored_hash, password):
                        session['user_id'] = user[0]
                        session['email'] = email
                        return redirect(url_for('health'))
                    else:
                        return "Invalid email or password!", 401
                else:
                    return "User not found!", 404
        except Exception as e:
            print(f"Login error: {e}")
            return "Error logging in!", 500
        finally:
            connection.close()

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

@app.route('/submit_health_info', methods=['POST'])
def submit_health_info():
    health_issues = request.form.get('health_issues', '').strip()
    allergies = request.form.get('allergies', '').strip().split(',')
    user_id = session.get('user_id')

    print(f"Health Issues: {health_issues}")
    print(f"Allergies Entered: {allergies}")

    if not user_id:
        return jsonify({'error': 'User not logged in!'}), 403

    connection = get_db_connection()
    if not connection:
        return jsonify({'error': 'Database connection failed!'}), 500

    try:
        with connection.cursor() as cursor:
            # Update health issues
            cursor.execute("UPDATE users SET health_issues = %s WHERE user_id = %s", (health_issues, user_id))
            # Insert allergies
            for allergy in allergies:
                print(f"Inserting Allergy: {allergy.strip().lower()} for User: {user_id}")
                cursor.execute(
                    "INSERT INTO allergies (user_id, allergy_name) VALUES (%s, %s)",
                    (user_id, allergy.strip().lower())
                )
            connection.commit()
            print(f"Health information saved for user {user_id}.")
    except Exception as e:
        print(f"Error saving health info: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        connection.close()

    return redirect(url_for('image_upload'))


@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'No file uploaded!'}), 400

    try:
        # Save the uploaded image
        upload_folder = 'static/uploads/'
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)

        # Process the image
        image = load_img(file_path, target_size=(299, 299))
        image_array = img_to_array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Predict food label
        predictions = food_recognition_model.predict(image_array)
        food_index = np.argmax(predictions)
        food_label = food_classes[food_index]

        # Detect allergens for the food
        detected_allergens = detect_allergens(food_label)

        # Retrieve user allergies from the database
        user_allergies = get_user_allergies(session.get('user_id'))

        # Normalize both detected allergens and user allergies
        detected_allergens_normalized = [a.strip().lower() for a in detected_allergens]
        user_allergies_normalized = [a.strip().lower() for a in user_allergies]

        # Match user allergies with detected allergens using substring matching
        matching_allergens = [
            user_allergy for user_allergy in user_allergies_normalized
            if any(user_allergy in allergen for allergen in detected_allergens_normalized)
        ]

        # Prepare the allergy warning message
        if matching_allergens:
            allergy_message = (
                f"⚠️ Allergy Alert! '{food_label}' contains allergens you are allergic to: "
                f"{', '.join(matching_allergens)}."
            )
        else:
            allergy_message = f"✅ Safe to consume '{food_label}'. No matched allergens detected."

        # Render the result page
        return render_template(
            'result.html',
            food_label=food_label,
            allergens=detected_allergens,
            allergy_message=allergy_message,
            image_url=f"uploads/{file.filename}"
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Helper functions
def detect_allergens(food_label):
    # Find the row in the CSV matching the food label
    matched_row = allergy_data[allergy_data['Food Item'].str.strip().str.lower() == food_label.lower()]
    if not matched_row.empty:
        allergens = matched_row['Common Allergies'].values[0]
        # Return allergens as a list if not empty or NaN
        return allergens.split(', ') if not pd.isna(allergens) else []
    return []  # Return an empty list if no allergens found

def get_user_allergies(user_id):
    connection = get_db_connection()
    if not connection:
        return []
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT allergy_name FROM allergies WHERE user_id = %s", (user_id,))
            return [row[0].lower() for row in cursor.fetchall()]
    finally:
        connection.close()

if __name__ == '__main__':
    app.run(debug=True)

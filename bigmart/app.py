from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load the model
model_path = os.path.join(os.path.dirname(__file__), r'C:\Users\vravi\OneDrive\Desktop\bigmart\bigmart\LightGBM_best_model.pkl')
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load training data for dropdowns
try:
    train_data = pd.read_csv('train.csv')
    item_types = sorted(train_data['Item_Type'].unique().tolist())
    outlet_identifiers = sorted(train_data['Outlet_Identifier'].unique().tolist())
    outlet_sizes = sorted(train_data['Outlet_Size'].unique().tolist())
    location_types = sorted(train_data['Outlet_Location_Type'].unique().tolist())
    outlet_types = sorted(train_data['Outlet_Type'].unique().tolist())
    print("Available outlet identifiers:", outlet_identifiers)
except Exception as e:
    print(f"Error loading training data: {e}")
    item_types = []
    outlet_identifiers = []
    outlet_sizes = []
    location_types = []
    outlet_types = []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html', 
                             item_types=item_types,
                             outlet_identifiers=outlet_identifiers,
                             outlet_sizes=outlet_sizes,
                             location_types=location_types,
                             outlet_types=outlet_types)
    
    if request.method == 'POST':
        try:
            # Log received form data
            print("Received form data:", request.form)
            
            # Get form data
            item_identifier = request.form.get('Item_Identifier')
            item_weight = request.form.get('Item_Weight')
            item_visibility = request.form.get('Item_Visibility')
            item_mrp = request.form.get('Item_MRP')
            item_type = request.form.get('Item_Type')
            item_fat_content = request.form.get('Item_Fat_Content')
            outlet_identifier = request.form.get('Outlet_Identifier')
            outlet_establishment_year = request.form.get('Outlet_Establishment_Year')
            outlet_size = request.form.get('Outlet_Size')
            outlet_location_type = request.form.get('Outlet_Location_Type')
            outlet_type = request.form.get('Outlet_Type')

            # Log parsed values
            print("Parsed values:", {
                'item_identifier': item_identifier,
                'item_weight': item_weight,
                'item_visibility': item_visibility,
                'item_mrp': item_mrp,
                'item_type': item_type,
                'item_fat_content': item_fat_content,
                'outlet_identifier': outlet_identifier,
                'outlet_establishment_year': outlet_establishment_year,
                'outlet_size': outlet_size,
                'outlet_location_type': outlet_location_type,
                'outlet_type': outlet_type
            })

            # Validate required fields
            missing_fields = []
            if not item_identifier: missing_fields.append('Item_Identifier')
            if not item_type: missing_fields.append('Item_Type')
            if not item_fat_content: missing_fields.append('Item_Fat_Content')
            if not outlet_identifier: missing_fields.append('Outlet_Identifier')
            if not outlet_size: missing_fields.append('Outlet_Size')
            if not outlet_location_type: missing_fields.append('Outlet_Location_Type')
            if not outlet_type: missing_fields.append('Outlet_Type')

            if missing_fields:
                error_msg = f"Missing required fields: {', '.join(missing_fields)}"
                print(error_msg)
                return jsonify({'error': error_msg}), 400

            # Convert numeric fields
            try:
                item_weight = float(item_weight) if item_weight else 0
                item_visibility = float(item_visibility) if item_visibility else 0
                item_mrp = float(item_mrp) if item_mrp else 0
                outlet_establishment_year = int(outlet_establishment_year) if outlet_establishment_year else 0
            except ValueError as e:
                error_msg = f"Invalid numeric value: {str(e)}"
                print(error_msg)
                return jsonify({'error': error_msg}), 400

            # Standardize Item_Fat_Content
            item_fat_content = item_fat_content.replace('low fat', 'Low Fat').replace('LF', 'Low Fat')

            # Create input DataFrame
            input_data = pd.DataFrame({
                'Item_Identifier': [item_identifier],
                'Item_Weight': [item_weight],
                'Item_Visibility': [item_visibility],
                'Item_MRP': [item_mrp],
                'Item_Type': [item_type],
                'Item_Fat_Content': [item_fat_content],
                'Outlet_Identifier': [outlet_identifier],
                'Outlet_Establishment_Year': [outlet_establishment_year],
                'Outlet_Size': [outlet_size],
                'Outlet_Location_Type': [outlet_location_type],
                'Outlet_Type': [outlet_type]
            })

            # Make prediction
            if model is None:
                return jsonify({'error': 'Model not loaded'}), 500

            prediction = model.predict(input_data)[0]
            return jsonify({'prediction': float(prediction)})

        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            print(error_msg)
            return jsonify({'error': error_msg}), 400

if __name__ == '__main__':
    # Ensure correct directory for Flask
    print(f"Running in directory: {os.getcwd()}")
    app.run(debug=True)

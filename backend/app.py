# app.py
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Initialize your Supabase client with your Supabase URL and API key
supabase_url = 'https://ddakdusfqppdpnmtijtl.supabase.co'
supabase_key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRkYWtkdXNmcXBwZHBubXRpanRsIiwicm9sZSI6ImFub24iLCJpYXQiOjE2OTg4MTIwMzUsImV4cCI6MjAxNDM4ODAzNX0.GM5ndsDBmmvxoQhcQoZgYCsj1LdEMxX-mT6O_FbeSEs'

import supabase

supabase_client = supabase.Client(supabase_url, supabase_key)


def query_similar_images(classification_result):
    # Query the Supabase database to find similar images based on classification_result
    # You may need to customize this logic to match your database structure and classification result
    # Example query:
    similar_images = supabase_client.table('images').select('*').filter('classification_result', 'eq',
                                                                        classification_result).limit(1000).execute()
    return similar_images


@app.route('/get-similar-images', methods=['POST'])
def get_similar_images():
    # Extract the classification result from the request
    classification_result = request.json.get('classification_result')

    # Query similar images from the Supabase database
    similar_images = query_similar_images(classification_result)
    return jsonify(similar_images)


# Define your routes and functionality here
UPLOAD_FOLDER = 'uploads'  # Create a folder for image uploads
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained model
model = torch.load("models\\model_epoch_4_mAP3_0.34.pt", map_location=torch.device('cpu'))
#print(model)

#######################################



###############################


def classify_image(image_path):
    # Load and preprocess the image
    image = Image.open(image_path)
    image = transform_image(image)

    # Perform classification
    with torch.no_grad():
        output = model(image)

    # Process the output to get classification results
    # You'll need to adapt this part based on your model and use case
    return output


def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Classify the uploaded image
        classification_result = classify_image(file_path)

        # Retrieve the first thousand similar images from Supabase
        similar_images = get_similar_images(classification_result)

        return jsonify({
            'classification_result': classification_result,
            'similar_images': similar_images
        })


@app.route('/classify-image', methods=['POST'])
def classify_uploaded_image():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"})

    image = request.files['image']

    if image.filename == '':
        return jsonify({"error": "No selected file"})

    # Handle image classification logic here
    classification_result = classify_image(image)
    return jsonify({"classification_result": classification_result})


if __name__ == "__main__":
    app.run(debug=True)  # For development, set debug=True

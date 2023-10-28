import torch
import os
from flask import Flask, request, jsonify, render_template

# Create flask app
flask_app = Flask(__name__)

# Define the absolute file path to the model
model_file_path = "E:/group 16/MyGit_new/model_deploy/model/model_epoch_4_mAP3_0.34.pt"

# Load the PyTorch model from the specified file path
# Use 'map_location' to specify the device
model = torch.load(model_file_path, map_location=torch.device('cpu'))

# Set the model to evaluation mode
model.eval()

# Define the absolute file path to the HTML file
html_file_path = "E:/group 16/MyGit_new/frontend/index.html"


@flask_app.route("/")
def Home():
    # Load and serve the HTML file
    with open(html_file_path, 'r') as html_file:
        html_content = html_file.read()
    return html_content


@flask_app.route("/predict", methods=["POST"])
def predict():
    # Assuming you have a form with input fields named "feature1", "feature2", etc.
    feature1 = float(request.form['feature1'])
    feature2 = float(request.form['feature2'])
    # Add more feature variables as needed

    # Prepare the input tensor
    input_tensor = torch.tensor([[feature1, feature2]], dtype=torch.float32)

    # Perform the prediction
    with torch.no_grad():
        prediction = model(input_tensor).argmax().item()

    # You should map the prediction to the appropriate class or label

    return render_template("index.html", prediction_text="The flower species is {}".format(prediction))


if __name__ == "__main__":
    flask_app.run(debug=True)

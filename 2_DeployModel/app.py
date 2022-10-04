"""
@author: Mohammad Zarei
"""

import io
import torchvision.transforms as transforms
import torch
from PIL import Image
from flask import Flask, jsonify, request

# Declare a flask app
app = Flask(__name__)

# Some inputs
INPUT_SIZE = 299
MODEL_PATH = "model/model.pt"

# Load the trained model
model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
model.eval()


# Do image preprocessing before prediction on any data
def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(INPUT_SIZE),
                                        transforms.CenterCrop(INPUT_SIZE),
                                        transforms.ToTensor()])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    output = model.forward(tensor)
    pred = output.argmax(1)
    prob = output.max()
    return 'Positive' if pred==0 else f'Negative'


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        result = get_prediction(image_bytes=img_bytes)
        return jsonify({'result': result})


if __name__ == '__main__':
    app.run()
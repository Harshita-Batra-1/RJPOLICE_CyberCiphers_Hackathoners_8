from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import numpy as np

app = Flask(__name__)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Load the model
model = InceptionResnetV1(
    pretrained="vggface2",
    classify=True,
    num_classes=1,
    device=DEVICE
)

model_path = "C:\\kandikits\\deepfake-detection\\deepfake-detection\\resnetinceptionv1_epoch_32.pth"
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

# Load MTCNN
mtcnn = MTCNN(
    select_largest=False,
    post_process=False,
    device=DEVICE
).to(DEVICE).eval()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the uploaded image
        image_file = request.files['image']
        image = Image.open(image_file).convert('RGB')

        # Predict
        face = mtcnn(image)
        if face is None:
            raise Exception('No face detected')
        face = face.unsqueeze(0)  # add the batch dimension
        face = F.interpolate(face, size=(256, 256), mode='bilinear', align_corners=False)
        
        face = face.to(DEVICE)
        face = face.to(torch.float32)
        face = face / 255.0

        with torch.no_grad():
            output = torch.sigmoid(model(face).squeeze(0))
            prediction = "real" if output.item() < 0.5 else "fake"
            
            real_prediction = 1 - output.item()
            fake_prediction = output.item()
            
            confidences = {
                'real': real_prediction,
                'fake': fake_prediction
            }

        return jsonify({'prediction': prediction, 'confidences': confidences})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
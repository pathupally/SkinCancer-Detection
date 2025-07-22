import os
import os
import torch
import torch.nn as nn
from torchvision import transforms
import timm
from PIL import Image
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import io
import base64

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'  

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MODEL_PATH = './models/best_model.pth'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

NUM_CLASSES = 2  
IMG_SIZE = 224   
MODEL_NAME = 'efficientnet_b0'  

def create_model(model_name=MODEL_NAME, num_classes=NUM_CLASSES):
    """Create EfficientNet model matching the training setup"""
    model = timm.create_model(
        model_name, 
        pretrained=False,  
        num_classes=num_classes
    )
    return model


def load_model():
    """Load the trained EfficientNet model"""
    try:
        device = torch.device('mps' if torch.mps.is_available() else 
                            'cuda' if torch.cuda.is_available() else 'cpu')
        
        model = create_model(MODEL_NAME, NUM_CLASSES)
        
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
                if 'metrics' in checkpoint:
                    print(f"Model metrics: {checkpoint['metrics']}")
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        model.to(device)
        print(f"Model loaded successfully on {device}")
        return model, device
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Make sure your model file is at: {MODEL_PATH}")
        print(f"And that MODEL_NAME '{MODEL_NAME}' matches your training setup")
        return None, None

model, device = load_model()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)  
        return image_tensor
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_image(image_path):
    if model is None:
        return None, "Model not loaded"
    
    try:
        image_tensor = preprocess_image(image_path)
        if image_tensor is None:
            return None, "Error preprocessing image"
        
        image_tensor = image_tensor.to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        class_labels = ['Benign', 'Malignant'] 
        prediction = class_labels[predicted_class]
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'probability_benign': probabilities[0][0].item(),
            'probability_malignant': probabilities[0][1].item()
        }, None
        
    except Exception as e:
        return None, f"Prediction error: {str(e)}"

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            result, error = predict_image(filepath)
            
            if error:
                flash(f'Error: {error}')
                return redirect(url_for('index'))
            
            with open(filepath, 'rb') as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode()
            
            os.remove(filepath)
            
            return render_template('result.html', 
                                 result=result, 
                                 image_data=img_base64,
                                 filename=filename)
            
        except Exception as e:
            flash(f'Error processing image: {str(e)}')
            return redirect(url_for('index'))
    
    else:
        flash('Invalid file type. Please upload PNG, JPG, JPEG, or GIF files.')
        return redirect(url_for('index'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result, error = predict_image(filepath)
        
        os.remove(filepath)
        
        if error:
            return jsonify({'error': error}), 500
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    flash('File is too large. Maximum size is 16MB.')
    return redirect(url_for('index'))

if __name__ == '__main__':
    if model is None:
        print("WARNING: Model failed to load. Please check your model file.")
    
    app.run(debug=True, host='0.0.0.0', port=5304)
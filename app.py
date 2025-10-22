from flask import Flask, render_template, request, jsonify
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import io
import os

app = Flask(__name__)

# تعريف نفس نموذج الكلاس من كود التدريب
class PVDefectClassifier(torch.nn.Module):
    def __init__(self, num_classes=5, pretrained=False):
        super(PVDefectClassifier, self).__init__()
        
        # استخدام الطريقة الحديثة لتحميل النموذج
        if pretrained:
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.backbone = resnet50(weights=None)
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_features, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# تحميل النموذج
def load_model(model_path):
    model = PVDefectClassifier(num_classes=5, pretrained=False)
    
    try:
        # تحميل النموذج
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # التعامل مع مختلف تنسيقات حفظ النموذج
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        print(f"✅ Model loaded successfully from: {model_path}")
        return model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

# استخدام مسار raw string صحيح
model_path = r"D:\Deep Learning\solar_defect_detection\models\best_model.pth"

# تحميل النموذج
model = load_model(model_path)

# تعريف التحويلات
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# فئات التصنيف
class_names = ['broken', 'bright_spot', 'black_border', 'scratched', 'non_electricity']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check the model file path.'})
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        if file:
            # تحميل الصورة
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # تطبيق التحويلات
            input_tensor = transform(image).unsqueeze(0)
            
            # التنبؤ
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                predicted_class = torch.argmax(outputs[0]).item()
                confidence = probabilities[predicted_class].item()
            
            # النتائج
            result = {
                'class': class_names[predicted_class],
                'confidence': round(confidence * 100, 2),
                'all_predictions': {
                    class_name: round(prob.item() * 100, 2) 
                    for class_name, prob in zip(class_names, probabilities)
                }
            }
            
            return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    if model is not None:
        print("🚀 Starting Solar Panel Defect Detection System...")
        print(f"📋 Available classes: {class_names}")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to start system - model not loaded")
        
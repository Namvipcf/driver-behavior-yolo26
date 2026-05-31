from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Cấu hình
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi'}
MODEL_PATH = 'best.pt'

# Tạo thư mục
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load model YOLO
try:
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Tên các hành vi (tùy chỉnh theo model của bạn)
BEHAVIOR_NAMES = {
    0: 'Nham mat',
    1: 'Uong nuoc',
    2: 'Soi guong chinh toc',
    3: 'Binh thuong',
    4: 'Voi tay ra sau',
    5: 'Hut thuoc',
    6: 'Noi chuyen dien thoai',
    7: 'Nhan tin dien thoai',
    8: 'Dieu chinh radio',
    9: 'Buon ngu',
    10: 'Ngap',
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_behavior(image_path):
    """Phát hiện hành vi từ ảnh"""
    if model is None:
        return None, "Model không được tải"
    
    try:
        # Detect với YOLOv8
        results = model(image_path, conf=0.25, iou=0.45)
        
        # Format kết quả
        detected_behaviors = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    behavior_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()
                    
                    bbox = {
                        'x1': xyxy[0],
                        'y1': xyxy[1],
                        'x2': xyxy[2],
                        'y2': xyxy[3]
                    }
                    
                    behavior_name = BEHAVIOR_NAMES.get(behavior_id, f'Class {behavior_id}')
                    
                    detected_behaviors.append({
                        'name': behavior_name,
                        'confidence': round(confidence * 100, 2),
                        'bbox': bbox
                    })
        
        # Vẽ bounding boxes lên ảnh
        img_cv2 = cv2.imread(image_path)
        for det in detected_behaviors:
            x1, y1, x2, y2 = int(det['bbox']['x1']), int(det['bbox']['y1']), int(det['bbox']['x2']), int(det['bbox']['y2'])
            color = (0, 255, 0) if 'an toàn' in det['name'].lower() else (0, 0, 255)
            cv2.rectangle(img_cv2, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_cv2, f"{det['name']} {det['confidence']}%", 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Lưu ảnh kết quả
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"result_{timestamp}.jpg"
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        cv2.imwrite(result_path, img_cv2)
        
        return detected_behaviors, result_filename
    
    except Exception as e:
        return None, f"Lỗi khi phát hiện: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'Không có file được chọn'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Không có file được chọn'}), 400
    
    if file and allowed_file(file.filename):
        # Lưu file upload
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"upload_{timestamp}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Kiểm tra loại file
        file_ext = filename.rsplit('.', 1)[1].lower()
        
        if file_ext in ['jpg', 'jpeg', 'png']:
            # Xử lý ảnh
            behaviors, result = detect_behavior(filepath)
            
            if behaviors is None:
                return jsonify({'error': result}), 500
            
            return jsonify({
                'success': True,
                'behaviors': behaviors,
                'result_image': result,
                'original_image': filename,
                'count': len(behaviors)
            })
        
        elif file_ext in ['mp4', 'avi']:
            # Xử lý video (demo cơ bản)
            return jsonify({
                'success': True,
                'message': 'Xử lý video đang được phát triển',
                'original_video': filename
            })
    
    return jsonify({'error': 'File không được hỗ trợ'}), 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(RESULT_FOLDER, filename)

@app.route('/api/stats')
def get_stats():
    """API lấy thống kê demo"""
    return jsonify({
        'total_detections': 0,
        'safe_behaviors': 0,
        'unsafe_behaviors': 0,
        'behavior_types': list(BEHAVIOR_NAMES.values())
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

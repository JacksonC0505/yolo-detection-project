from ultralytics import YOLO
import cv2

class PhoneDetector:
    def __init__(self):
        self.model = None

    def load_model(self):
        # Load the YOLOv5 model pre-trained on COCO dataset
        self.model = YOLO('yolov5s.pt')  # Use 'yolov5s.pt' for speed; replace with 'yolov5m.pt' for more accuracy

    def detect_phone(self, frame):
        """
        Detect phones in the frame and return a list of bounding boxes and confidences.
        """
        # 检查输入格式
        print("frame.shape:", frame.shape, "dtype:", frame.dtype)
    
        # 如果是4通道（如RGBA），转换为3通道（BGR）
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        results = self.model(frame)
        if not results or results[0].boxes.data.numel() == 0:  # Check if tensor is empty
            return []

        detections = results[0].boxes.data.cpu().numpy()
        print("YOLO results:", detections)
        phones = []

        for detection in detections:
            x1, y1, x2, y2, confidence, class_id = detection
            if int(class_id) == 67:  # Class ID 67 corresponds to "cell phone" in COCO
                phone_box = (int(x1), int(y1), int(x2), int(y2))
                phones.append((phone_box, confidence))

        return phones

    def calculate_size(self, phone_box, pixels_per_cm):
        """Calculate actual phone size using calibration data"""
        x1, y1, x2, y2 = phone_box
        pixel_width = x2 - x1
        pixel_height = y2 - y1
        
        # Use inverse square law for distance correction
        # (closer objects appear larger)
        width_cm = pixel_width / pixels_per_cm
        height_cm = pixel_height / pixels_per_cm

        return width_cm, height_cm
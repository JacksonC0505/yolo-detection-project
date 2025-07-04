import cv2
import numpy as np
import time
import sys
import os

# Try to import ncnn
try:
    import ncnn
    HAVE_NCNN = True
except ImportError:
    HAVE_NCNN = False
    print("NCNN Python binding not found. Please install with: pip install ncnn")
    print("Or use pre-converted ONNX model instead")

class YoloDetNCNN:
    def __init__(self):
        if not HAVE_NCNN:
            self.net = None
            return
            
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False  # CPU mode for compatibility
        self.input_size = 320  # Default size for Yolo-Fastest
        
    def init(self, param_path, bin_path):
        """Initialize the YOLO model with NCNN param and bin files"""
        if not HAVE_NCNN:
            print("NCNN Python binding not installed")
            return False
            
        try:
            # Check if model files exist
            if not os.path.exists(param_path):
                print(f"Error: {param_path} not found")
                return False
            if not os.path.exists(bin_path):
                print(f"Error: {bin_path} not found")
                return False
                
            # Load NCNN model
            self.net.load_param(param_path)
            self.net.load_model(bin_path)
            print(f"NCNN model loaded from {param_path} and {bin_path}")
            return True
        except Exception as e:
            print(f"Error loading NCNN model: {e}")
            return False
    
    def detect(self, frame, output_list):
        """Detect objects in the frame using NCNN"""
        output_list.clear()
        
        if self.net is None:
            print("Model not loaded")
            return
        
        try:
            # Prepare input
            h, w = frame.shape[:2]
            input_size = self.input_size
         
            # Create NCNN extractor
            ex = self.net.create_extractor()
            
            # 按照C++代码方式：直接缩放到目标尺寸，不保持宽高比，不填充
            mat_in = ncnn.Mat.from_pixels_resize(frame, ncnn.Mat.PixelType.PIXEL_BGR2RGB, w, h, input_size, input_size)
            
            # Normalize (adjust according to your model requirements)
            mean_vals = [0, 0, 0]
            norm_vals = [1/255.0, 1/255.0, 1/255.0]
            mat_in.substract_mean_normalize(mean_vals, norm_vals)
            
            # Set input
            ex.input("data", mat_in)  # "data" is the input layer name, adjust if needed
            
            # Extract output
            ret, out = ex.extract("output")  # "output" is the output layer name, adjust if needed
            
            # Process output
            # This part needs to be adjusted based on your specific YOLO model output format
            for i in range(out.h):
                values = out.row(i)
                
                # 按照C++代码的格式解析输出
                cate = int(values[0])        # 类别
                score = values[1]            # 分数
                x1_norm = values[2]          # 归一化的左上角x
                y1_norm = values[3]          # 归一化的左上角y  
                x2_norm = values[4]          # 归一化的右下角x
                y2_norm = values[5]          # 归一化的右下角y
                
                
                if score > 0.5:
                    # 直接将归一化坐标转换为原图坐标
                    x1 = int(x1_norm * w)
                    y1 = int(y1_norm * h)
                    x2 = int(x2_norm * w)
                    y2 = int(y2_norm * h)
                    
                    target_box = {
                        'x1': max(0, x1),
                        'y1': max(0, y1),
                        'x2': min(w, x2),
                        'y2': min(h, y2),
                        'cate': cate,
                        'score': float(score)
                    }
                    output_list.append(target_box)
                    print(f"Detection: class={cate}, score={score:.2f}, box=({x1},{y1},{x2},{y2})")
        except Exception as e:
            print(f"NCNN detection error: {e}")
            import traceback
            traceback.print_exc()

def draw_boxes(src_img, boxes):
    """Draw detection boxes on the image"""
    print(f"Detect box num: {len(boxes)}")
    
    for box in boxes:
        # Draw rectangle
        cv2.rectangle(src_img, 
                     (box['x1'], box['y1']), 
                     (box['x2'], box['y2']), 
                     (255, 255, 0), 2)
        
        # Draw category text
        cate_text = f"Category:{box['cate']}"
        cv2.putText(src_img, cate_text, 
                   (box['x1'], box['y1'] - 20), 
                   cv2.FONT_HERSHEY_TRIPLEX, 0.5, 
                   (0, 255, 0), 1, cv2.LINE_AA)
        
        # Draw score text
        score_text = f"Score:{box['score']:.2f}"
        cv2.putText(src_img, score_text, 
                   (box['x1'], box['y1'] - 5), 
                   cv2.FONT_HERSHEY_TRIPLEX, 0.5, 
                   (0, 0, 255), 1, cv2.LINE_AA)
    
    return 0

def test_cam():
    """Test camera detection with NCNN"""
    if not HAVE_NCNN:
        print("Please install NCNN Python binding first: pip install ncnn")
        return -1
        
    api = YoloDetNCNN()
    
    # Initialize model with the same file paths
    success = api.init("model/v5lite-i8e.param", 
                      "model/v5lite-i8e.bin")
    
    if not success:
        print("Failed to load model. Exiting.")
        return -1
    
    output = []
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return -1
    
    try:
        while True:
            print("=========================")
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Failed to read frame")
                break
            
            start_time = time.time()
            api.detect(frame, output)
            end_time = time.time()
            detect_time = (end_time - start_time) * 1000  # Convert to milliseconds
            print(f"Detect Time: {detect_time:7.2f} ms")
            
            draw_boxes(frame, output)
            
            cv2.imshow("NCNN Demo", frame)
            output.clear()
            
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    return 0

def main():
    test_cam()
    return 0

if __name__ == "__main__":
    main()
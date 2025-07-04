import cv2
import numpy as np
import time
import os
import math

# Try to import ncnn
try:
    import ncnn
    HAVE_NCNN = True
except ImportError:
    HAVE_NCNN = False
    print("NCNN Python binding not found. Please install with: pip install ncnn")

class Object:
    def __init__(self):
        self.rect = [0, 0, 0, 0]  # x, y, width, height
        self.label = 0
        self.prob = 0.0

class YoloDetNCNN:
    def __init__(self):
        if not HAVE_NCNN:
            self.net = None
            return
            
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        self.target_size = 320
        self.prob_threshold = 0.40
        self.nms_threshold = 0.40
        
    def init(self, param_path, bin_path):
        """Initialize the YOLO model with NCNN param and bin files"""
        if not HAVE_NCNN:
            print("NCNN Python binding not installed")
            return False
            
        try:
            if not os.path.exists(param_path):
                print(f"Error: {param_path} not found")
                return False
            if not os.path.exists(bin_path):
                print(f"Error: {bin_path} not found")
                return False
                
            self.net.load_param(param_path)
            self.net.load_model(bin_path)
            print(f"NCNN model loaded from {param_path} and {bin_path}")
            return True
        except Exception as e:
            print(f"Error loading NCNN model: {e}")
            return False

    def sigmoid(self, x):
        return 1.0 / (1.0 + math.exp(-x))

    def generate_proposals(self, anchors, stride, in_pad, feat_blob, objects):
        """Generate object proposals from feature map"""
        num_grid = feat_blob.h
        
        # Calculate grid dimensions
        if in_pad.w > in_pad.h:
            num_grid_x = in_pad.w // stride
            num_grid_y = num_grid // num_grid_x
        else:
            num_grid_y = in_pad.h // stride
            num_grid_x = num_grid // num_grid_y

        num_class = feat_blob.w - 5
        num_anchors = len(anchors) // 2

        for q in range(num_anchors):
            anchor_w = anchors[q * 2]
            anchor_h = anchors[q * 2 + 1]

            feat = feat_blob.channel(q)

            for i in range(num_grid_y):
                for j in range(num_grid_x):
                    featptr = feat.row(i * num_grid_x + j)

                    # Find class index with max class score
                    class_index = 0
                    class_score = float('-inf')
                    
                    for k in range(num_class):
                        score = featptr[5 + k]
                        if score > class_score:
                            class_index = k
                            class_score = score

                    box_score = featptr[4]
                    confidence = self.sigmoid(box_score) * self.sigmoid(class_score)

                    if confidence >= self.prob_threshold:
                        dx = self.sigmoid(featptr[0])
                        dy = self.sigmoid(featptr[1])
                        dw = self.sigmoid(featptr[2])
                        dh = self.sigmoid(featptr[3])

                        pb_cx = (dx * 2.0 - 0.5 + j) * stride
                        pb_cy = (dy * 2.0 - 0.5 + i) * stride

                        pb_w = pow(dw * 2.0, 2) * anchor_w
                        pb_h = pow(dh * 2.0, 2) * anchor_h

                        x0 = pb_cx - pb_w * 0.5
                        y0 = pb_cy - pb_h * 0.5
                        x1 = pb_cx + pb_w * 0.5
                        y1 = pb_cy + pb_h * 0.5

                        obj = Object()
                        obj.rect = [x0, y0, x1 - x0, y1 - y0]
                        obj.label = class_index
                        obj.prob = confidence

                        objects.append(obj)

    def nms_sorted_bboxes(self, objects, nms_threshold):
        """Apply Non-Maximum Suppression"""
        if not objects:
            return []

        # Sort by confidence
        objects.sort(key=lambda x: x.prob, reverse=True)

        picked = []
        areas = [obj.rect[2] * obj.rect[3] for obj in objects]

        for i in range(len(objects)):
            keep = True
            
            for j in picked:
                # Calculate intersection
                x1 = max(objects[i].rect[0], objects[j].rect[0])
                y1 = max(objects[i].rect[1], objects[j].rect[1])
                x2 = min(objects[i].rect[0] + objects[i].rect[2], 
                        objects[j].rect[0] + objects[j].rect[2])
                y2 = min(objects[i].rect[1] + objects[i].rect[3], 
                        objects[j].rect[1] + objects[j].rect[3])

                if x2 <= x1 or y2 <= y1:
                    inter_area = 0
                else:
                    inter_area = (x2 - x1) * (y2 - y1)

                union_area = areas[i] + areas[j] - inter_area
                
                if inter_area / union_area > nms_threshold:
                    keep = False
                    break

            if keep:
                picked.append(i)

        return [objects[i] for i in picked]

    def detect(self, frame, output_list):
        """Detect objects using YOLOv5 processing pipeline"""
        output_list.clear()
        
        if self.net is None:
            print("Model not loaded")
            return
        
        try:
            img_w = frame.shape[1]
            img_h = frame.shape[0]

            # Letterbox resize - same as C++ code
            w = img_w
            h = img_h
            scale = 1.0
            
            if w > h:
                scale = self.target_size / w
                w = self.target_size
                h = int(h * scale)
            else:
                scale = self.target_size / h
                h = self.target_size
                w = int(w * scale)

            # Resize image
            mat_in = ncnn.Mat.from_pixels_resize(frame, ncnn.Mat.PixelType.PIXEL_BGR2RGB, 
                                               img_w, img_h, w, h)

            # Pad to target_size rectangle
            wpad = (w + 31) // 32 * 32 - w
            hpad = (h + 31) // 32 * 32 - h
            
            # Create padded matrix
            in_pad = ncnn.Mat()
            ncnn.copy_make_border(mat_in, in_pad, hpad // 2, hpad - hpad // 2, 
                                wpad // 2, wpad - wpad // 2, 
                                ncnn.BorderType.BORDER_CONSTANT, 114.0)

            # Normalize
            mean_vals = [0.0, 0.0, 0.0]  # 修改：确保是float类型
            norm_vals = [1/255.0, 1/255.0, 1/255.0]
            in_pad.substract_mean_normalize(mean_vals, norm_vals)

            # Create extractor
            ex = self.net.create_extractor()
            ex.input("images", in_pad)

            proposals = []

            # Stride 8
            try:
                ret, out = ex.extract("output")
                anchors = [10.0, 13.0, 16.0, 30.0, 33.0, 23.0]
                objects8 = []
                self.generate_proposals(anchors, 8, in_pad, out, objects8)
                proposals.extend(objects8)
            except:
                print("Failed to extract stride 8 output")

            # Stride 16
            try:
                ret, out = ex.extract("1111")
                anchors = [30.0, 61.0, 62.0, 45.0, 59.0, 119.0]
                objects16 = []
                self.generate_proposals(anchors, 16, in_pad, out, objects16)
                proposals.extend(objects16)
            except:
                print("Failed to extract stride 16 output")

            # Stride 32
            try:
                ret, out = ex.extract("2222")
                anchors = [116.0, 90.0, 156.0, 198.0, 373.0, 326.0]
                objects32 = []
                self.generate_proposals(anchors, 32, in_pad, out, objects32)
                proposals.extend(objects32)
            except:
                print("Failed to extract stride 32 output")

            # Apply NMS
            objects = self.nms_sorted_bboxes(proposals, self.nms_threshold)

            # Adjust coordinates back to original image
            for obj in objects:
                # Adjust offset to original unpadded
                x0 = (obj.rect[0] - (wpad / 2)) / scale
                y0 = (obj.rect[1] - (hpad / 2)) / scale
                x1 = (obj.rect[0] + obj.rect[2] - (wpad / 2)) / scale
                y1 = (obj.rect[1] + obj.rect[3] - (hpad / 2)) / scale

                # Clip to image bounds
                x0 = max(min(x0, img_w - 1), 0)
                y0 = max(min(y0, img_h - 1), 0)
                x1 = max(min(x1, img_w - 1), 0)
                y1 = max(min(y1, img_h - 1), 0)

                target_box = {
                    'x1': int(x0),
                    'y1': int(y0),
                    'x2': int(x1),
                    'y2': int(y1),
                    'cate': obj.label,
                    'score': obj.prob
                }
                output_list.append(target_box)

            print(f"Final detections: {len(output_list)}")
            for box in output_list:
                print(f"Detection: class={box['cate']}, score={box['score']:.2f}, box=({box['x1']},{box['y1']},{box['x2']},{box['y2']})")

        except Exception as e:
            print(f"NCNN detection error: {e}")
            import traceback
            traceback.print_exc()

def draw_boxes(src_img, boxes):
    """Draw detection boxes on the image"""
    class_names = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    ]
    
    print(f"Detect box num: {len(boxes)}")
    
    for box in boxes:
        # Draw rectangle
        cv2.rectangle(src_img, 
                     (box['x1'], box['y1']), 
                     (box['x2'], box['y2']), 
                     (0, 255, 0), 2)
        
        # Draw label
        class_name = class_names[box['cate']] if box['cate'] < len(class_names) else f"class_{box['cate']}"
        label = f"{class_name}: {box['score']:.1%}"
        
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        
        # Draw label background
        cv2.rectangle(src_img, (box['x1'], box['y1'] - label_size[1] - 10), 
                     (box['x1'] + label_size[0], box['y1']), (0, 255, 0), -1)
        
        # Draw label text
        cv2.putText(src_img, label, (box['x1'], box['y1'] - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return 0

def test_cam():
    """Test camera detection with NCNN"""
    if not HAVE_NCNN:
        print("Please install NCNN Python binding first: pip install ncnn")
        return -1
        
    api = YoloDetNCNN()
    
    # Initialize model
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
            detect_time = (end_time - start_time) * 1000
            print(f"Detect Time: {detect_time:.2f} ms")
            
            draw_boxes(frame, output)
            output.clear()
            
            cv2.imshow("NCNN Demo", frame)
            
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
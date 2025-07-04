import cv2
from models.phone_detector import PhoneDetector
import time
from picamera2 import Picamera2  # 新增

def calibrate_distance(picam2, detector):
    measurements = []
    for _ in range(10):
        frame = picam2.capture_array()
        print(frame)
        phones = detector.detect_phone(frame)
        if len(phones) > 0 and phones[0] is not None:
            phone_box, confidence = phones[0]
            x1, y1, x2, y2 = phone_box
            pixel_width = x2 - x1
            measurements.append(pixel_width)
        time.sleep(0.1)
    if len(measurements) < 5:
        print("Calibration failed: Could not detect phone consistently")
        return False, None
    reference_pixels = sum(measurements) / len(measurements)
    PHONE_WIDTH_CM = 7.0
    pixels_per_cm = reference_pixels / PHONE_WIDTH_CM
    return True, pixels_per_cm

def main():
    # 初始化 picamera2
    picam2 = Picamera2()
    picam2.start()
    
    detector = PhoneDetector()
    detector.load_model()

    print("Starting camera calibration...")
    is_calibrated = False
    pixels_per_cm = None

    while not is_calibrated:
        frame = picam2.capture_array()
        cv2.putText(frame, "CALIBRATION MODE", (30, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "Place phone at 30cm & press 'c'", (30, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('Camera Feed', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            success, pixels_per_cm = calibrate_distance(picam2, detector)
            if success:
                print("Calibration successful!")
                time.sleep(2)
                is_calibrated = True
            else:
                print("Calibration failed, try again")
                time.sleep(2)
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return

    print("Starting phone detection...")
    while True:
        frame = picam2.capture_array()
        curr_time = time.time()
        if 'prev_time' not in locals():
            prev_time = curr_time - 1e-5
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time

        phones = detector.detect_phone(frame)
        for phone_box, confidence in phones:
            width_cm, height_cm = detector.calculate_size(phone_box, pixels_per_cm)
            x1, y1, x2, y2 = phone_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Width: {width_cm:.2f} cm", (x1, y1 - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f"Height: {height_cm:.2f} cm", (x1, y1 - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Camera Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
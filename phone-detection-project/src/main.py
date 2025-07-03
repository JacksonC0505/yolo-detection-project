import cv2
from models.phone_detector import PhoneDetector
import time

def calibrate_distance(cap, detector):
    """
    Calibrate the camera by measuring phone size at known distance (30cm)
    Returns: (success, reference_pixel_width)
    """
    # Take multiple measurements to get stable reading
    measurements = []
    for _ in range(10):  # Take 10 measurements
        ret, frame = cap.read()
        if not ret:
            return False, None
            
        phones = detector.detect_phone(frame)
        if len(phones) > 0 and phones[0] is not None:
            # Calculate pixel width of phone
            # Unpack the first detected phone's bounding box
            phone_box, confidence = phones[0]
            x1, y1, x2, y2 = phone_box
            pixel_width = x2 - x1
            measurements.append(pixel_width)
            
        time.sleep(0.1)  # Short delay between measurements
        
    if len(measurements) < 5:  # Need at least 5 valid measurements
        print("Calibration failed: Could not detect phone consistently")
        return False, None
        
    # Use median to filter out outliers
    reference_pixels = sum(measurements) / len(measurements)
    
    # Store calibration constant (pixels per cm at 30cm distance)
    # Assuming average phone width is 7cm
    PHONE_WIDTH_CM = 7.0
    pixels_per_cm = reference_pixels / PHONE_WIDTH_CM
    
    return True, pixels_per_cm

def main():
    # Initialize the camera feed
    cap = cv2.VideoCapture(0)
    
    # Load the phone detection model
    detector = PhoneDetector()
    detector.load_model()

    # Calibration phase
    print("Starting camera calibration...")
    is_calibrated = False
    pixels_per_cm = None
    
    while not is_calibrated:
        ret, frame = cap.read()
        if not ret:
            break
            
        cv2.putText(frame, "CALIBRATION MODE", (30, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "Place phone at 30cm & press 'c'", (30, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Camera Feed', frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c'):
            success, pixels_per_cm = calibrate_distance(cap, detector)
            if success:
                print("Calibration successful!")
                time.sleep(2)
                is_calibrated = True
            else:
                print("Calibration failed, try again")
                time.sleep(2)
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

    print("Starting phone detection...")
    # Main detection loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # FPS calculation
        curr_time = time.time()
        if 'prev_time' not in locals():
            prev_time = curr_time - 1e-5  # Initialize to avoid division by zero
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time

        # Detect phone in the frame
        phones = detector.detect_phone(frame)

        # Process each detected phone
        for phone_box, confidence in phones:
            width_cm, height_cm = detector.calculate_size(phone_box, pixels_per_cm)  # Pass pixels_per_cm
            x1, y1, x2, y2 = phone_box

            # Draw bounding box and display size
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Width: {width_cm:.2f} cm", (x1, y1 - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f"Height: {height_cm:.2f} cm", (x1, y1 - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Draw FPS on the frame
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Display the resulting frame
        cv2.imshow('Camera Feed', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
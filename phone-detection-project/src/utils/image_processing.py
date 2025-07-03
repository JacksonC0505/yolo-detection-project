def resize_image(image, width, height):
    return cv2.resize(image, (width, height))

def convert_color_space(image, color_space):
    if color_space == 'gray':
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif color_space == 'hsv':
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return image

def apply_filter(image, filter_type):
    if filter_type == 'blur':
        return cv2.GaussianBlur(image, (5, 5), 0)
    elif filter_type == 'edge':
        return cv2.Canny(image, 100, 200)
    return image
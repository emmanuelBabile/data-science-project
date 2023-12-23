import cv2
import numpy as np
from keras.models import load_model


# Function to convert image to grayscale
def convert_to_grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

# Function to apply Gaussian Blur to the image
def apply_gaussian_blur(img):
    blur = cv2.GaussianBlur(img, (7, 7), cv2.BORDER_DEFAULT)
    return blur

# Function to perform edge detection on the image
def apply_edge_detection(img):
    canny = cv2.Canny(img, 125, 175)
    return canny

# Function to dilate edges in the image
def dilate_edges(img):
    canny = cv2.Canny(img, 125, 175)
    dilated = cv2.dilate(canny, (7, 7), iterations=3)
    return dilated

# Function to perform image cropping
def crop_image(img):
    cropped = img[50:200, 200:400]
    return cropped

# Function to translate (shift) the image
def translate_image(img):
    translated = translate(img, -10, 100)
    return translated

def translate(img, x, y):
    transMat = np.float32([[1, 0, x], [0, 1, y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv2.warpAffine(img, transMat, dimensions)

# Function to rotate the image
def rotate_image(img):
    rotated = rotate(img, -45)
    return rotated

def rotate(img, angle, rotPoint=None):
    (height, width) = img.shape[:2]
    if rotPoint is None:
        rotPoint = (width // 2, height // 2)

    rotMat = cv2.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width, height)
    return cv2.warpAffine(img, rotMat, dimensions)

# Function to resize the image
def resize_image(img):
    resized = cv2.resize(img, (500, 500), interpolation=cv2.INTER_CUBIC)
    return resized

# Function to flip the image
def flip_image(img):
    flip = cv2.flip(img, -1)
    return flip

def apply_convolution(img, conv_matrix):
    # Convert image to grayscale
    gray_img = convert_to_grayscale(img)

    # Perform manual convolution operation
    image_height, image_width = gray_img.shape
    filter_size = len(conv_matrix)
    result_image = np.zeros((image_height - filter_size + 1, image_width - filter_size + 1))

    for i in range(image_height - filter_size + 1):
        for j in range(image_width - filter_size + 1):
            subarray = gray_img[i: i + filter_size, j: j + filter_size]
            convolution_result = np.sum(np.sum(subarray * np.array(conv_matrix)))
            result_image[i, j] = convolution_result

    # Clip values to the valid range [0.0, 1.0]
    result_image = np.clip(result_image, 0.0, 1.0)

    return result_image

def detect_faces(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load the pre-trained Haar Cascade model for face detection
    fname = "haarcascade_frontalface_default.xml"
    haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + fname)

    # Detect faces in the image
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces_rect:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

    # Convert the image to RGB format for compatibility with st.image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img_rgb, len(faces_rect),faces_rect

def classify_emotion(img, emotion_model, faces_rect):
    emotions = []

    for (x, y, w, h) in faces_rect:
        face_roi = img[y:y + h, x:x + w]
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = np.expand_dims(face_roi, axis=-1)

        emotion_prediction = emotion_model.predict(face_roi)
        emotion_label = np.argmax(emotion_prediction)

        emotions.append(emotion_label)

        emotion_text = "Happy" if emotion_label == 3 else "Sad" if emotion_label == 4 else "Neutral"
        cv2.putText(img, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return img, emotions


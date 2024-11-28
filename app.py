from flask import Flask, request, send_file
import cv2
import numpy as np
from skimage import exposure
import io
from flask_cors import CORS
import os
import traceback

print("Starting Flask application...")

app = Flask(__name__)
CORS(app)

print("Flask application initialized.")

# Set maximum content length to handle large file uploads
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB limit

# Enable debug mode if FLASK_ENV is set to development
if os.getenv('FLASK_ENV') == 'development':
    app.config['DEBUG'] = True

print("Configuration set.")

def load_image(image_bytes):
    try:
        print(f"Received image bytes: {len(image_bytes)} bytes")
        print(f"First 100 bytes of data: {image_bytes[:100]}")
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Image could not be loaded.")
        print("Image loaded successfully")
        return image
    except Exception as e:
        print(f"Error in load_image: {e}")
        raise

def contrast_stretching(image):
    try:
        p2, p98 = np.percentile(image, (2, 98))
        result = exposure.rescale_intensity(image, in_range=(p2, p98))
        print("Contrast stretching applied")
        return result
    except Exception as e:
        print(f"Error in contrast_stretching: {e}")
        raise

def gamma_correction(image, gamma=1.5):
    try:
        invGamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
        result = cv2.LUT(image, table)
        print("Gamma correction applied")
        return result
    except Exception as e:
        print(f"Error in gamma_correction: {e}")
        raise

def multi_scale_retinex(image, scales=[15, 80, 250]):
    try:
        image_float = np.float64(image) + 1.0
        retinex = np.zeros_like(image_float)
        for scale in scales:
            blurred = cv2.GaussianBlur(image_float, (0, 0), sigmaX=scale, sigmaY=scale)
            retinex += np.log10(image_float) - np.log10(blurred + 1.0)
        retinex = retinex / len(scales)
        retinex = (retinex - np.min(retinex)) / (np.max(retinex) - np.min(retinex)) * 255
        result = np.uint8(retinex)
        print("Multi-scale retinex applied")
        return result
    except Exception as e:
        print(f"Error in multi_scale_retinex: {e}")
        raise

def adaptive_histogram_equalization(image):
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        result = clahe.apply(image)
        print("Adaptive histogram equalization applied")
        return result
    except Exception as e:
        print(f"Error in adaptive_histogram_equalization: {e}")
        raise

@app.route('/process-image', methods=['POST'])
def process_image():
    print("Received a request to /process-image")
    print("Request Headers:", request.headers)
    print("Request Content-Type:", request.content_type)

    if 'file' not in request.files:
        print("No file part in the request.")
        return 'No file uploaded.', 400

    file = request.files['file']
    if file.filename == '':
        print("No file selected for uploading.")
        return 'No file uploaded.', 400

    try:
        image_bytes = file.read()
        print(f"File received: {file.filename}, size: {len(image_bytes)} bytes")

        # Check if data is received
        data_length = len(image_bytes)
        if data_length == 0:
            print("No data received in the file.")
            return 'No data received.', 400

        # Log the first few bytes of the data to verify it is received correctly
        print(f"First 100 bytes of data: {image_bytes[:100]}")

        # Load image
        image = load_image(image_bytes)
        if image is None:
            return 'Failed to load image.', 400

        # Apply contrast stretching
        contrast_image = contrast_stretching(image)
        if contrast_image is None:
            return 'Failed to apply contrast stretching.', 500

        # Apply gamma correction
        gamma_corrected_image = gamma_correction(contrast_image, gamma=1.8)
        if gamma_corrected_image is None:
            return 'Failed to apply gamma correction.', 500

        # Apply multi-scale retinex
        msr_image = multi_scale_retinex(gamma_corrected_image)
        if msr_image is None:
            return 'Failed to apply multi-scale retinex.', 500

        # Apply adaptive histogram equalization
        equalized_image = adaptive_histogram_equalization(msr_image)
        if equalized_image is None:
            return 'Failed to apply adaptive histogram equalization.', 500

        # Encode image to PNG format
        success, encoded_image = cv2.imencode('.png', equalized_image)
        if not success:
            print("Failed to encode image.")
            return 'Failed to encode image.', 500

        # Send the image directly from memory
        img_byte_arr = io.BytesIO(encoded_image)
        img_byte_arr.seek(0)

        print("Image encoding completed successfully")
        print(f"Encoded image size: {len(encoded_image)} bytes")

        return send_file(img_byte_arr, mimetype='image/png', as_attachment=True, download_name='enhanced_image.png')
    except Exception as e:
        print(f"Error processing image: {e}")
        traceback.print_exc()  # Print the full traceback
        return 'Error processing image.', 500

if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5000, debug=True)
    app.run(host='0.0.0.0', port=8000, debug=True)
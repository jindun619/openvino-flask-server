from flask import Flask, request, jsonify
import subprocess
import os
from PIL import Image

app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']

    text_data = request.form.get('text')
    if not text_data:
        return jsonify({"error": "No text data provided"}), 400

    temp_image_path = "temp_image.jpg"
    image_file.save(temp_image_path)

    try:
        result = subprocess.run(
            ["python", "process.py", temp_image_path, text_data],
            capture_output=True,
            text=True
        )
        output = result.stdout.strip()
        return jsonify({"result": output})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
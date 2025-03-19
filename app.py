from flask import Flask, request, jsonify
from PIL import Image
import io
# from process2 import use_llava
from zzh import use_deepseek

app = Flask(__name__)

@app.route('/test', methods=['POST'])
def test():
    image_file = request.files.get('image')
    if not image_file:
        return jsonify({"error": "No image file provided"}), 400

    text_data = request.form.get('text')
    if not text_data:
        return jsonify({"error": "No text data provided"}), 400

    try:
        return jsonify({"result": f"your face is {text_data}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/process', methods=['POST'])
def process_image():
    image_file = request.files.get('image')
    if not image_file:
        return jsonify({"error": "No image file provided"}), 400

    text_data = request.form.get('text', None)

    try:
        image_bytes = io.BytesIO(image_file.read())
        image = Image.open(image_bytes).convert("RGB")

        output = use_deepseek(image, text_data)
        return jsonify({"result": output})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

from flask import Flask, request, jsonify
from PIL import Image
import io
from process import use_llava

app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process_image():
    image_file = request.files.get('image')
    if not image_file:
        return jsonify({"error": "No image file provided"}), 400

    text_data = request.form.get('text')
    if not text_data:
        return jsonify({"error": "No text data provided"}), 400

    try:
        image_bytes = io.BytesIO(image_file.read())
        image = Image.open(image_bytes).convert("RGB")

        output = use_llava(image, text_data)
        return jsonify({"result": output})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

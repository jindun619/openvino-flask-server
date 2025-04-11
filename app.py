from flask import Flask, request, jsonify
from utils import load_image

import torch
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from transformers import AutoModelForCausalLM

from use_deepseek import use_deepseek

app = Flask(__name__)


def initialize_model():
    model_path = "deepseek-ai/deepseek-vl-7b-chat"
    global vl_chat_processor, tokenizer, vl_gpt
    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer
    vl_gpt = (
        AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        .to(torch.bfloat16)
        .cuda()
        .eval()
    )
    print("model loaded completely!")


initialize_model()


@app.route("/process", methods=["POST"])
def process_image():
    image_file = request.files.get("image")
    objects_list = request.form.getlist("objects")
    if not image_file:
        return jsonify({"error": "No image file provided"}), 400

    try:
        image = load_image(image_file)

        output = use_deepseek(image, objects_list, vl_chat_processor, tokenizer, vl_gpt)

        return jsonify({"result": output})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)

import time
from PIL import Image
import requests
from transformers import AutoModelForCausalLM, AutoProcessor, TextStreamer
from optimum.intel.openvino import OVModelForVisualCausalLM
import torch


def load_image():
    url = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"
    return Image.open(requests.get(url, stream=True).raw)


def run_openvino_model(model_id):
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = OVModelForVisualCausalLM.from_pretrained(model_id, trust_remote_code=True)

    image = load_image()
    prompt = "<|image_1|>\nWhat is unusual on this picture?"

    inputs = model.preprocess_inputs(text=prompt, image=image, processor=processor)

    generation_args = {
        "max_new_tokens": 50,
        "temperature": 0.0,
        "do_sample": False,
        "streamer": TextStreamer(
            processor.tokenizer, skip_prompt=True, skip_special_tokens=True
        ),
    }

    start = time.time()
    generate_ids = model.generate(
        **inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args
    )
    end = time.time()

    generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return end - start, response


def run_hf_model(model_id):
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(
        "cpu"
    )  # CPU only
    model.eval()

    image = load_image()
    prompt = "<|image_1|>\nWhat is unusual on this picture?"

    inputs = processor(prompt, images=image, return_tensors="pt").to("cpu")  # CPU only

    generation_args = {
        "max_new_tokens": 50,
        "temperature": 0.0,
        "do_sample": False,
        "streamer": TextStreamer(
            processor.tokenizer, skip_prompt=True, skip_special_tokens=True
        ),
    }

    start = time.time()
    with torch.no_grad():
        generate_ids = model.generate(
            **inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args
        )
    end = time.time()

    generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return end - start, response


# ===================== 실행 =====================
# OpenVINO (가속화 버전)
ov_time, ov_response = run_openvino_model("OpenVINO/Phi-3.5-vision-instruct-int4-ov")
print(f"[OpenVINO] Time: {ov_time:.2f} sec\nResponse: {ov_response}\n")

# 기본 HuggingFace 모델
hf_time, hf_response = run_hf_model("phi-3.5-vision-instruct")
print(f"[HuggingFace] Time: {hf_time:.2f} sec\nResponse: {hf_response}")

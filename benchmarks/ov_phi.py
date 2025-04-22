from PIL import Image
import requests, time, sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from transformers import AutoModelForCausalLM, AutoProcessor
from optimum.intel.openvino import OVModelForVisualCausalLM

from utils import get_gpu_usage

start_gpu = get_gpu_usage()

model_id = "OpenVINO/Phi-3.5-vision-instruct-int4-ov"

ov_model = OVModelForVisualCausalLM.from_pretrained(model_id, trust_remote_code=True)

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

start_time = time.time()

prompt = "<|image_1|>\nWhat is unusual on this picture?"

url = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"
image = Image.open(requests.get(url, stream=True).raw)

inputs = ov_model.preprocess_inputs(text=prompt, image=image, processor=processor)

generation_args = {"max_new_tokens": 50, "temperature": 0.0, "do_sample": False}

generate_ids = ov_model.generate(
    **inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args
)

generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
response = processor.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

print(response)
print(f"gpu usage: {get_gpu_usage()-start_gpu}")
print(f"time: {time.time()-start_time}")

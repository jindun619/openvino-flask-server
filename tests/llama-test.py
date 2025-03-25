import requests
import torch
import sys
import os
import time
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils import get_gpu_usage

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model = torch.compile(model, backend="inductor")

processor = AutoProcessor.from_pretrained(model_id)

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
image = Image.open(requests.get(url, stream=True).raw)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {
                "type": "text",
                "text": "If I had to write a haiku for this one, it would be: ",
            },
        ],
    }
]

start_time = time.time()
start_gpu = get_gpu_usage()

input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(image, input_text, add_special_tokens=False, return_tensors="pt").to(
    model.device
)

output = model.generate(**inputs, max_new_tokens=30)


print(processor.decode(output[0]))
print(f"time: {time.time()-start_time:.2f}")
print(f"gpu: {get_gpu_usage()-start_gpu:.2f}")

# 1.87 22.48
# Openvino GPU 1.68 22.48
# Openvino CPU 1.64 22.48
# Inductor 1.64 22.48

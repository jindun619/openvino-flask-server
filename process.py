import sys
from PIL import Image
import requests
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

def use_llava(image, text):
    model_id = "bczhou/tiny-llava-v1-hf"
    prompt = "USER: <image>\nDescribe the image briefly\nASSISTANT:"

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(0)

    processor = AutoProcessor.from_pretrained(model_id)

    inputs = processor(prompt, image, return_tensors='pt').to(0, torch.float16)

    output_tensor = model.generate(**inputs, max_new_tokens=200, do_sample=False)

    output_text = processor.batch_decode(output_tensor, skip_special_tokens=True)[0]

    return output_text
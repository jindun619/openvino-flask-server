import sys
from PIL import Image
import requests
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

def extract_assistant_response(output_text):
    return output_text.split("ASSISTANT: ")[1]

model_id = "bczhou/tiny-llava-v1-hf"

#cats
image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#car
# image_url = "https://stimg.cardekho.com/images/carexteriorimages/630x420/BMW/M5-2025/11821/1719462197562/front-left-side-47.jpg?impolicy=resize&imwidth=480"

raw_image = Image.open(requests.get(image_url, stream=True).raw)

# prompt = "USER: <image>\nDescribe the image briefly\nASSISTANT:"
object = "cat"
prompt = (
    "USER: <image>"
    "The image is captured from a first-person perspective of a visually impaired person. "
    f"There is a {object} in the image. "
    f"Tell me the color of the {object}. "
    "Output format: [color]. Do NOT say anything else. "
    "ASSISTANT:"
)

model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(0)

processor = AutoProcessor.from_pretrained(model_id)

inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)

output_tensor = model.generate(
    **inputs,
    max_new_tokens=200,
    do_sample=False,
)

output_text = processor.batch_decode(output_tensor, skip_special_tokens=True)[0]

# print(extract_assistant_response(output_text))
print(output_text)

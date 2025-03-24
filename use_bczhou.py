import sys
from PIL import Image
import requests
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

def use_bczhou(image, object):
    model_id = "bczhou/tiny-llava-v1-hf"

    # prompt = "USER: <image>\nDescribe the image briefly\nASSISTANT:"
    prompt =  (
    "USER: <image> This image belongs to the category: {category}. "
    "You are an AI assistant designed to help visually impaired individuals navigate indoor spaces safely. "
    "Do NOT describe the entire scene. ONLY output the conclusion. "
    "You must respond with exactly TWO sentences and nothing else. "
    "Any extra words or descriptions will be ignored. "
    "Strictly follow this format:\n"
    "Example:\n"
    "Conclusion: The suitcase ahead is a potential hazard, avoid it by moving left.\n"
    "Here is another correct response:\n"
    "Conclusion: The chair in front is blocking the way, move right to avoid it.\n"
)

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(0)

    processor = AutoProcessor.from_pretrained(model_id)

    inputs = processor(prompt, image, return_tensors='pt').to(0, torch.float16)

    output_tensor = model.generate(**inputs, max_new_tokens=200, do_sample=False)

    output_object = processor.batch_decode(output_tensor, skip_special_tokens=True)[0]

    if "Conclusion:" in output_object:
        conclusion_part = output_object.split("Conclusion:")[-1].strip()
        conclusion = conclusion_part.split(".")[0] + ". " + conclusion_part.split(".")[1]
        conclusion = conclusion.split("\n")[0].strip()
    else:
        conclusion = "Conclusion: Unable to determine hazard. Please reprocess."

    return conclusion
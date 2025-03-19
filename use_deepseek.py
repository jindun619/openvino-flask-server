# pip install git+https://github.com/deepseek-ai/DeepSeek-VL
import torch
import requests
import time
from PIL import Image
from io import BytesIO

from transformers import AutoModelForCausalLM
from transformers import CLIPProcessor, CLIPModel
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images

# 정확도가 높은 순서대로 num개의 object 반환
def detect_objects(image, possible_objects, num):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    inputs = processor(images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)  
        image_features = outputs

    scores = []
    for obj in possible_objects:
        text_inputs = processor(text=obj, return_tensors="pt", padding=True)
        text_outputs = model.get_text_features(**text_inputs)

        similarity = torch.cosine_similarity(image_features, text_outputs, dim=-1)
        scores.append(similarity.item())

    score_object_pairs = list(zip(scores, possible_objects))
    score_object_pairs.sort(reverse=True, key=lambda x: x[0])

    return [obj for score, obj in score_object_pairs[:num]]

def use_deepseek(image, object):
    model_path = "deepseek-ai/deepseek-vl-7b-chat"
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    start_time = time.time()

    possible_objects = ["apple", "car", "person", "dog", "bike", "traffic light", "garbage", "trash can", "manhole", "crosswalk","bottle","arm"]
    dangerous_objects = ["apple", "people", "backpack", "trash can","street performance", "manhole", "garbage",  "traffic light","vehicle" ]

    if object is None:
        object = detect_objects(image, possible_objects, 1)[0]

    content_danger = (
        "<image_placeholder>"
        f"USER: <image> This image belongs to the category: {object}. "
        "You are an AI assistant designed to help visually impaired individuals navigate indoor spaces safely. "
        "Do NOT describe the entire scene. ONLY output the conclusion with all relevant obstacles. "
        "Please follow this format: "
        "The potential hazards ahead are: {list of objects}. "
        "The most dangerous object is {most dangerous object}, and the safest action would be to {recommended action}. "
        "Do NOT add any extra words or descriptions. "
    )

    content_safe = (
        "<image_placeholder>"
        f"USER: <image> This image belongs to the category: {object}. "
        "You are an AI assistant designed to help visually impaired individuals navigate indoor spaces safely. "
        "Do NOT describe the entire scene. ONLY output the conclusion with all relevant objects detected. "
        "You must respond with exactly two sentence describing the objects in the environment. "
        "For example: 'There are some computers and a desk in front of you.' "
        "Your response should provide a description of the scene, without any hazard warnings."
    )

    conversation = [
        {
            "role": "User",
            "content": content_danger if object in dangerous_objects else content_safe,  
            "images": [image]
        },
        {
            "role": "Assistant",
            "content": ""
        }
    ]

    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=[image],
        force_batchify=True
    ).to(vl_gpt.device)

    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True
    )

    output_text = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    inference_time = time.time() - start_time
    print(inference_time)

    return output_text

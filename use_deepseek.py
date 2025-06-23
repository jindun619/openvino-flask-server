# pip install git+https://github.com/deepseek-ai/DeepSeek-VL
import torch
import time

from transformers import CLIPProcessor, CLIPModel

from utils import get_gpu_usage


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


def generate_prompt(detected_objects, dangerous_objects):
    is_dangerous = any(obj in dangerous_objects for obj in detected_objects)

    if is_dangerous:
        prompt = (
            f"USER: <image_placeholder> This image belongs to the category: {detected_objects[0]}. "
            "You are an AI assistant designed to help visually impaired individuals navigate indoor spaces safely. "
            "Do NOT describe the entire scene. ONLY output the conclusion with all relevant obstacles. "
            "Please follow this format: "
            "The potential hazards ahead are: {list of objects}. "
            "The most dangerous object is {most dangerous object}, and the safest action would be to {recommended action}. "
            "Do NOT add any extra words or descriptions. "
        )
    else:
        print("safe_prompt used")
        prompt = (
            f"USER: <image_placeholder> This image belongs to the category: {detected_objects[0]}. "
            "You are an AI assistant designed to help visually impaired individuals navigate indoor spaces safely. "
            "Do NOT describe the entire scene. ONLY output the conclusion with all relevant objects detected. "
            "You must respond with exactly two sentence describing the objects in the environment. "
            "For example: 'There are some computers and a desk in front of you.' "
            "Your response should provide a description of the scene, without any hazard warnings."
        )

    return prompt


def use_deepseek(image, detected_objects, vl_chat_processor, tokenizer, vl_gpt):
    start_time = time.time()
    start_gpu = get_gpu_usage()
    print("start infering..")

    possible_objects = [
        "orange",
        "car",
        "person",
        "dog",
        "bike",
        "traffic light",
        "garbage",
        "trash can",
        "manhole",
        "crosswalk",
        "bottle",
        "arm",
        "cell phone"
    ]
    dangerous_objects = [
        "car",
        "bike",
        "dog",
        "manhole",
        "vehicle",
        "cell phone",
        "orange"
    ]

    if not detected_objects:
        detected_objects = detect_objects(image, possible_objects, 1)[0]

    print(detected_objects[0])
    content = generate_prompt(detected_objects, dangerous_objects)

    conversation = [
        {
            "role": "User",
            "content": content,
        },
        {"role": "Assistant", "content": ""},
    ]

    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=[image], force_batchify=True
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
        use_cache=True,
    )

    output_text = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    inference_time = time.time() - start_time
    inference_gpu = get_gpu_usage() - start_gpu
    print("inference finished!")
    print(f"time: {inference_time:.2f} sec")
    print(f"gpu usage: {inference_gpu:.2f} MB")

    return output_text

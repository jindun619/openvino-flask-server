from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import time
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM

model_path = "deepseek-ai/deepseek-vl-7b-chat"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

start_time = time.time()

image_path = "./car.jpg"  # 사용자 이미지 경로
image = Image.open(image_path)
if image.format == "WEBP":
    image = image.convert("RGB")

# 위험한 객체 리스트와 그에 대한 추천 행동
dangerous_objects = {
    "person": "Avoid the person and maintain a safe distance.",
    "trash can": "Avoid getting too close to the trash can.",
    "manhole": "Be cautious and avoid stepping on the manhole.",
    "garbage": "Be cautious of the garbage and avoid stepping on it.",
    "traffic light": "Be cautious when crossing the street.",
    "vehicle": "Stay clear of the vehicle for safety.",
}

try:
    # 이미지 처리
    inputs = processor(images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
        image_features = outputs

    # 첫 번째 프롬프트: "Describe the scene"
    describe_scene_prompt = (
        "<image_placeholder>"
        f"USER: <image> Please describe the scene in this image, including the objects you can identify. "
        "You must provide a description without any hazard warnings."
    )

    # 실제 모델에 입력을 전달하기 위한 처리 (장면 묘사)
    conversation = [
        {
            "role": "User",
            "content": describe_scene_prompt,  # 장면 묘사 요청
            "images": [image],
        },
        {"role": "Assistant", "content": ""},
    ]

    # 이미지 처리 및 모델 입력 준비
    pil_images = [image]
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(vl_gpt.device)

    # 이미지 인코더로 임베딩 계산
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    # 모델을 사용하여 응답 생성
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

    # 출력 결과 디코딩
    description = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    print(f"Scene Description: {description}")

    # 장면 묘사 결과에서 위험한 객체가 포함되어 있는지 확인
    dangerous_detected = [
        obj for obj in dangerous_objects.keys() if obj in description.lower()
    ]
    print(f"Dangerous objects detected in description: {dangerous_detected}")

    # 두 번째 프롬프트 결정: 위험한 객체가 있다면 danger prompt 사용
    if dangerous_detected:
        print("Using danger prompt.")
        # 위험한 객체에 대한 추천 행동을 가져옵니다.
        recommendations = [dangerous_objects[obj] for obj in dangerous_detected]
        content = (
            "<image_placeholder>"
            f"USER: <image> This image contains the following objects: {', '.join(dangerous_detected)}. "
            "You are an AI assistant designed to help visually impaired individuals navigate indoor spaces safely. "
            "Do NOT describe the entire scene. ONLY output the conclusion with all relevant obstacles. "
            f"The potential hazards ahead are: {', '.join(dangerous_detected)}. "
            f"The safest action would be: {', '.join(recommendations)}. "
            "Do NOT add any extra words or descriptions."
            "Conclusion:"
        )
    else:
        print("Using safe prompt.")
        content = (
            "<image_placeholder>"
            f"USER: <image> This image contains the following objects: {', '.join(dangerous_detected)}. "
            "You are an AI assistant designed to help visually impaired individuals navigate indoor spaces safely. "
            "Do NOT describe the entire scene. ONLY output the conclusion with all relevant objects detected. "
            "You must respond with exactly two sentences describing the objects in the environment. "
            "For example: 'There are some computers and a desk in front of you.' "
            "Your response should provide a description of the scene, without any hazard warnings."
        )

    # 실제 모델에 입력을 전달하기 위한 처리
    conversation = [
        {"role": "User", "content": content, "images": [image]},  # 적절한 프롬프트 사용
        {"role": "Assistant", "content": ""},
    ]

    # 모델 응답 생성
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
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

    # 출력 결과 디코딩
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    print(f"Conclusion Output: {answer}")  # 결론만 출력

except Exception as e:
    print(f"An error occurred: {e}")
    detected_objects = []

# Inference time
inference_time = time.time() - start_time
print(f"Inference Time: {inference_time:.2f}sec")

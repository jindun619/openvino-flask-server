import torch
import sys
import os
import time

from transformers import AutoModelForCausalLM
from deepseek_vl.models import VLChatProcessor

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils import load_image, get_gpu_usage

model_path = "deepseek-ai/deepseek-vl-7b-chat"
global vl_chat_processor, tokenizer, vl_gpt

# VLChatProcessor 및 Tokenizer 로드
vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

# 모델 로드 (bfloat16으로 설정 후 GPU로 이동)
vl_gpt = (
    AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    .to(torch.bfloat16)
    .cuda()
    .eval()
)

# OpenVINO 백엔드로 컴파일
# vl_gpt = torch.compile(
#     vl_gpt, backend="openvino", options={"device": "GPU", "model_caching": True}
# )
# Inductor 백엔드로 컴파일
vl_gpt = torch.compile(vl_gpt, backend="inductor")

content = (
    "<image_placeholder>"
    "The image is captured from a first-person perspective of a visually impaired person. "
    f"There is a {object} in the image. "
    f"Tell me the color of the {object}. "
    "You ONLY say the color. Do NOT say anything else. "
    "ASSISTANT:"
)

conversation = [
    {"role": "User", "content": content},
    {"role": "Assistant", "content": ""},
]

start_time = time.time()
start_gpu = get_gpu_usage()

# 입력 준비
prepare_inputs = vl_chat_processor(
    conversations=conversation,
    images=[load_image("../images/car.jpg")],
    force_batchify=True,
).to(vl_gpt.device)

inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

# 텍스트 생성
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

print(output_text)
print(f"time: {time.time()-start_time:.2f}")
print(f"gpu: {get_gpu_usage()-start_gpu:.2f}")

# not using Openvino 0.64 32.43
# using Openvino(CPU) 0.83 32.43
# using Openvino(GPU) 0.86 32.43

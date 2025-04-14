import torch
from transformers import AutoModelForCausalLM
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images
import openvino as ov

import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# 모델 경로 지정
model_path = "deepseek-ai/deepseek-vl-7b-chat"

# 프로세서 및 토크나이저 로드
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

# 모델 로드 및 설정
vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).eval()  # OpenVINO 변환 시에는 .cuda() 제거

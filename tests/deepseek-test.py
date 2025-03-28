# pip install git+https://github.com/deepseek-ai/DeepSeek-VL

import os
import sys
import torch
import time
from transformers import AutoModelForCausalLM

from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils import load_image, get_gpu_usage


# specify the path to the model
model_path = "deepseek-ai/deepseek-vl-7b-chat"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)

tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)

vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

# vl_gpt = torch.compile(
#     vl_gpt, backend="openvino", options={"device": "CPU", "model_caching": True}
# )
vl_gpt = torch.compile(vl_gpt, backend="inductor")

start_time = time.time()
start_gpu = get_gpu_usage()


object = "car"
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

# load images and prepare for inputs
prepare_inputs = vl_chat_processor(
    conversations=conversation,
    images=[load_image("../images/car.jpg")],
    force_batchify=True,
).to(vl_gpt.device)

# run image encoder to get the image embeddings
inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

# run the model to get the response
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

answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
print(f"{prepare_inputs['sft_format'][0]}", answer)

print(f"time: {time.time()-start_time:.2f}")
print(f"gpu: {get_gpu_usage()-start_gpu:.2f}")

# 0.67 32.36
# Openvino GPU 0.81 31.49
# Openvino CPU 0.81 31.49
# Inductor 0.79 32.36

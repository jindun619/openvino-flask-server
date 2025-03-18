import os
import torch
from transformers import AutoModelForCausalLM
from PIL import Image
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images

cache_dir = '/home/featurize/work/openvino-flask-server/model_cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
os.environ['TRANSFORMERS_CACHE'] = cache_dir

def use_llava(image, text):

    model_path = "deepseek-ai/deepseek-vl-7b-chat"
    
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer
    
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
    
    object = "car"
    content = (
        "<image_placeholder>"
        f"There is a {object} in the image. "
        "Explain the image briefly. "
        "ASSISTANT:"
    )
    
    conversation = [
        {
            "role": "User",
            "content": content,
            "images": ["./images/car.jpg"]  # PIL.Image 객체를 직접 전달
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
    
    return output_text
from PIL import Image
import requests, time, sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from transformers import AutoModelForCausalLM, AutoProcessor

from utils import get_gpu_usage

start_gpu = get_gpu_usage()

model_id = "microsoft/Phi-3.5-vision-instruct"

# Note: set _attn_implementation='eager' if you don't have flash_attn installed
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # device_map="cuda",
    device_map="cpu",
    trust_remote_code=True,
    torch_dtype="auto",
    # _attn_implementation="flash_attention_2",
    _attn_implementation="eager",
)

# for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, num_crops=4)

start_time = time.time()

url = "https://img0.baidu.com/it/u=856302126,3599826371&fm=253&fmt=auto&app=120&f=JPEG?w=500&h=889"
images = [Image.open(requests.get(url, stream=True).raw)]
placeholder = "<|image_1|>\n"

messages = [
    {"role": "user", "content": placeholder + "Summarize the deck of slides."},
]

prompt = processor.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

# inputs = processor(prompt, images, return_tensors="pt").to("cuda:0")
inputs = processor(prompt, images, return_tensors="pt")
inputs = {k: v.to("cpu") for k, v in inputs.items()}

generation_args = {
    "max_new_tokens": 50,
    "temperature": 0.0,
    "do_sample": False,
}

generate_ids = model.generate(
    **inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args
)

# remove input tokens
generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
response = processor.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

print(response)
print(f"gpu usage: {get_gpu_usage()-start_gpu}")
print(f"time: {time.time()-start_time}")

# CPU: 11.61
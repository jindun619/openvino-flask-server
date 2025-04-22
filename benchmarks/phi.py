from PIL import Image
import requests, time, sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from transformers import AutoModelForCausalLM, AutoProcessor
from optimum.intel.openvino import OVModelForVisualCausalLM

from utils import get_gpu_usage


def run_original():
    model_id = "microsoft/Phi-3.5-vision-instruct"

    start_gpu = get_gpu_usage()

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
    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=True, num_crops=4
    )

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


def run_openvino():
    model_id = "OpenVINO/Phi-3.5-vision-instruct-int4-ov"

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    ov_model = OVModelForVisualCausalLM.from_pretrained(
        model_id, trust_remote_code=True
    )
    prompt = "<|image_1|>\nWhat is unusual on this picture?"

    url = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"
    image = Image.open(requests.get(url, stream=True).raw)

    inputs = ov_model.preprocess_inputs(text=prompt, image=image, processor=processor)

    generation_args = {"max_new_tokens": 50, "temperature": 0.0, "do_sample": False}

    generate_ids = ov_model.generate(
        **inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args
    )

    generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    print(response)


run_original()
# run_openvino()

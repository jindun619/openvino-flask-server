# pip install git+https://github.com/deepseek-ai/DeepSeek-VL2
from ov_deepseek_vl_helper import OVDeepseekVLV2ForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor

pip_install("-U", "openvino>=2025.0", "nncf>=2.15")
pip_install(
    "-q",
    "torch>=2.1",
    "torchvision",
    "gradio>=4.19",
    "einops",
    "transformers>=4.48.2",
    "timm>=0.9.16",
    "accelerate",
    "sentencepiece",
    "attrdict",
    "mdtex2html",
    "pypinyin",
    "tiktoken",
    "tqdm",
    "colorama",
    "Pygments",
    "markdown",
    "--extra-index-url",
    "https://download.pytorch.org/whl/cpu",
)

model_path = "deepseek-vl2-tiny/INT4"
device = "CPU"

ov_model = OVDeepseekVLV2ForCausalLM(model_path, device.value)
processor = DeepseekVLV2Processor.from_pretrained(model_path)

from PIL import Image
import requests
import io
from urllib.parse import urlparse


def load_image(input_source, timeout=5):
    """
    이미지를 로드하고 RGB로 변환하는 함수

    Args:
        input_source: (파일 경로(str), 파일 객체, BytesIO, 이미지 URL(str))
        timeout: URL 요청 시 타임아웃 (기본값: 5초)

    Returns:
        PIL.Image.Image: RGB 이미지 객체 (실패 시 None)
    """
    try:
        # Case 1: 이미지 URL인 경우
        if isinstance(input_source, str) and urlparse(input_source).scheme in (
            "http",
            "https",
        ):
            response = requests.get(input_source, timeout=timeout)
            response.raise_for_status()  # HTTP 에러 확인
            image_bytes = io.BytesIO(response.content)
            image = Image.open(image_bytes)

        # Case 2: 파일 경로(str) 또는 Path 객체인 경우
        elif isinstance(input_source, str):
            image = Image.open(input_source)

        # Case 3: 파일 객체 (BytesIO, 업로드 파일 등)
        else:
            if hasattr(input_source, "read"):
                image_bytes = io.BytesIO(input_source.read())
            else:
                image_bytes = io.BytesIO(input_source)
            image = Image.open(image_bytes)

        return image.convert("RGB")

    except Exception as e:
        print(f"Error loading image: {e}")
        return None

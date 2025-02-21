import sys
from PIL import Image

def process_image(image_path, text):
    with Image.open(image_path) as img:
        width, height = img.size

    result = f"Text: {text}, Image Size: {width}x{height}"
    return result

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process.py <image_path> <text>")
        sys.exit(1)

    image_path = sys.argv[1]
    text = sys.argv[2]

    output = process_image(image_path, text)
    print(output)
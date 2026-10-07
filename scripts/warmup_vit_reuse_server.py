"""Issue the same non-workload visual request to two CA servers."""

import argparse
import base64
import io

import requests
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ports", nargs="+", type=int, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    image = Image.new("RGB", (128, 128), (127, 127, 127))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    data_url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": "Reply OK."},
        ]}],
        "max_tokens": 1,
        "temperature": 0,
    }
    for port in args.ports:
        response = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 json=payload, timeout=300)
        response.raise_for_status()
        print(f"warmed port {port}", flush=True)


if __name__ == "__main__":
    main()

"""Issue two identical Qwen3-VL API requests and verify persistent ViT reuse."""

import argparse
import base64
import json
import time
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    image = base64.b64encode(args.image.read_bytes()).decode("ascii")
    body = {
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}},
            {"type": "text", "text": "Describe this image in one sentence."},
        ]}],
        "max_tokens": 32,
        "temperature": 0,
    }
    url = f"http://127.0.0.1:{args.port}/v1/chat/completions"
    observations = []
    for _ in range(2):
        start = time.monotonic()
        response = requests.post(url, json=body, timeout=300)
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"]
        print(f"response: {answer!r}", flush=True)
        if not args.metrics.exists():
            raise RuntimeError("The vision cache hook did not write metrics after an image request")
        metrics = json.loads(args.metrics.read_text())
        observations.append({"seconds": time.monotonic() - start,
                             "answer": answer, "metrics": metrics})
    first, second = observations
    assert first["answer"] == second["answer"], observations
    assert first["metrics"]["misses"] >= 1, observations
    assert second["metrics"]["hits"] > first["metrics"]["hits"], observations
    assert second["metrics"]["vit_calls"] == first["metrics"]["vit_calls"], observations
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"observations": observations}, indent=2))
    print(f"Warm request reused {second['metrics']['hits'] - first['metrics']['hits']} visual segment(s)")


if __name__ == "__main__":
    main()

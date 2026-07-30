#!/usr/bin/env python3
"""OpenAI-compatible chat clients for AVA's SA/CA stages, served by vLLM.

The ava conda env's lmdeploy 0.8.0 / transformers 4.49 predate the qwen3_5 and
qwen3_vl architectures, so Qwen3.5 / Qwen3-VL run in a separate `qwen35` env via
`vllm serve` and AVA talks to them over the OpenAI API. These classes duck-type
llms.BaseModel (generate_response / batch_generate_response) so AVA(...) accepts
them directly -- no edits to llms/init_model.py.

- OpenAIChatLM: text-only (SA agentic search + self-consistency). For Qwen3.5
  thinking is DISABLED by default (chat_template_kwargs enable_thinking=false)
  to match the non-thinking Qwen2.5-32B baseline's sampling regime; export
  QWEN35_THINKING=1 to keep it on.
- OpenAIChatVL: multimodal (CA frame recheck); PIL frames go as base64 JPEG
  image_url parts, text last, matching QwenVL's content ordering.

Batching = thread pool over single requests; vLLM continuous-batches server-side.
"""
import base64
import io
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import requests

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


class _OpenAIChatBase:
    def __init__(self, model, base_url="http://127.0.0.1:8035/v1",
                 concurrency=32, timeout=1800, max_retries=3):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.concurrency = concurrency
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()

    def _chat(self, messages, max_new_tokens, temperature, extra=None):
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }
        if extra:
            payload.update(extra)
        last_err = None
        for attempt in range(self.max_retries):
            try:
                r = self.session.post(f"{self.base_url}/chat/completions",
                                      json=payload, timeout=self.timeout)
                r.raise_for_status()
                content = r.json()["choices"][0]["message"]["content"] or ""
                return _THINK_RE.sub("", content).strip()
            except Exception as err:  # noqa: BLE001 - retry then surface
                last_err = err
                time.sleep(min(2 ** attempt * 5, 60))
        raise RuntimeError(f"chat request failed after {self.max_retries} tries: {last_err}")

    def _batch(self, thunks, max_batch_size):
        results = []
        with ThreadPoolExecutor(max_workers=min(self.concurrency, max_batch_size)) as pool:
            for fut in [pool.submit(t) for t in thunks]:
                results.append(fut.result())
        return results


class OpenAIChatLM(_OpenAIChatBase):
    """Text-only client (SA stage)."""

    def __init__(self, model, base_url="http://127.0.0.1:8035/v1", **kw):
        super().__init__(model, base_url, **kw)
        self.extra = None
        if os.environ.get("QWEN35_THINKING", "0") != "1":
            self.extra = {"chat_template_kwargs": {"enable_thinking": False}}

    def generate_response(self, inputs, max_new_tokens=512, temperature=0.5):
        assert "text" in inputs, "Please provide a text prompt."
        messages = [{"role": "user", "content": inputs["text"]}]
        return self._chat(messages, max_new_tokens, temperature, self.extra)

    def batch_generate_response(self, batch_inputs, max_batch_size=64,
                                max_new_tokens=512, temperature=0.5):
        return self._batch(
            [lambda i=i: self.generate_response(i, max_new_tokens, temperature)
             for i in batch_inputs], max_batch_size)


def _to_data_url(img):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class OpenAIChatVL(_OpenAIChatBase):
    """Multimodal client (CA stage): inputs may carry 'video' = list[PIL.Image]."""

    def generate_response(self, inputs, max_new_tokens=512, temperature=0.5):
        assert "text" in inputs, "Please provide a text prompt."
        if inputs.get("video"):
            content = [{"type": "image_url", "image_url": {"url": _to_data_url(im)}}
                       for im in inputs["video"]]
            content.append({"type": "text", "text": inputs["text"]})
        else:
            content = inputs["text"]
        return self._chat([{"role": "user", "content": content}],
                          max_new_tokens, temperature)

    def batch_generate_response(self, batch_inputs, max_batch_size=64,
                                max_new_tokens=512, temperature=0.5):
        return self._batch(
            [lambda i=i: self.generate_response(i, max_new_tokens, temperature)
             for i in batch_inputs], max_batch_size)


def wait_for_server(base_url, model, timeout=2400):
    """Block until the vLLM server is up and serving `model`."""
    deadline = time.time() + timeout
    url = f"{base_url.rstrip('/')}/models"
    while time.time() < deadline:
        try:
            served = [m["id"] for m in requests.get(url, timeout=10).json()["data"]]
            if model in served:
                return True
            raise RuntimeError(f"server up but serving {served}, expected {model}")
        except RuntimeError:
            raise
        except Exception:
            time.sleep(10)
    raise TimeoutError(f"vLLM server at {base_url} not ready within {timeout}s")

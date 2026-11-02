"""Run inference with an extracted vector ID.

Usage: python -m client.inference --vector-id sha256:<id> --layer 10
Set MIA_INFERENCE_URL to the inference deployment root and VLLM_API_KEY.
"""

import argparse
import asyncio
import math
import os
import time
MIA_INFERENCE_URL = "https://noobylub--steer-capture-inference-vllmserver.us-east.modal.direct" 
    
from client.extracting_vectors import connect_HTTP


async def infer(base_url: str, api_key: str, payload: dict) -> dict:
	if not base_url:
		raise RuntimeError("Set MIA_INFERENCE_URL to the Modal inference URL without /v1")
	async with connect_HTTP(base_url.rstrip("/") + "/", api_key) as client:
		response = await client.post("steer", json=payload)
	if not response.is_success:
		raise RuntimeError(
			f"Steered inference failed ({response.status_code}): {response.text[:1000]}"
		)
	return response.json()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--vector-id", required=True,
					 help="vector_id returned by modal_extraction")
	parser.add_argument("--prompt", default="Tell me a bedtime story that involves a cat.")
	parser.add_argument("--layer", type=int, required=True)
	parser.add_argument("--coefficient", type=float, default=1.0)
	parser.add_argument("--max-tokens", type=int, default=128)
	args = parser.parse_args()
	if not args.prompt.strip():
		raise ValueError("prompt must not be empty")
	if args.layer < 0 or not math.isfinite(args.coefficient):
		raise ValueError("Invalid layer or coefficient")
	if not 1 <= args.max_tokens <= 2048:
		raise ValueError("max_tokens must be between 1 and 2048")
	payload = {
		"vector_id": args.vector_id, "prompt": args.prompt, "layer": args.layer,
		"coefficient": args.coefficient, "max_tokens": args.max_tokens,
	}
	print(f"Vector ID: {args.vector_id}; layer: {args.layer}; "
		  f"coefficient: {args.coefficient}")
	started_at = time.perf_counter()
	result = asyncio.run(infer(
		MIA_INFERENCE_URL,
		os.environ.get("VLLM_API_KEY", ""),
		payload,
	))
	elapsed_seconds = time.perf_counter() - started_at
	print(result["choices"][0]["message"]["content"])
	if result.get("usage"):
		usage = result["usage"]
		print("Token usage:", usage)
		completion_tokens = usage.get("completion_tokens")
		if completion_tokens is not None and elapsed_seconds > 0:
			print(f"End-to-end throughput: {completion_tokens / elapsed_seconds:.2f} tokens/sec")
			print(f"Request latency: {elapsed_seconds:.2f} sec")
	else:
		print(f"Request latency: {elapsed_seconds:.2f} sec; token usage unavailable")

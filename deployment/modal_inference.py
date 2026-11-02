"""Steer inference using vectors previously extracted and stored on a Modal Volume.

Deploy: modal deploy deployment/modal_inference.py
POST JSON to /steer with vector_id or vectors, prompt, and max_tokens.
Each saved vector's layer is used unless that vector supplies a layer override.
Authenticate with the VLLM_API_KEY bearer token.
"""

import json
import math
import os
import re
import secrets
import threading
from pathlib import Path

from dotenv import load_dotenv
import httpx
import modal
from openai import AsyncOpenAI


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
MODEL = os.environ.get("MIA_MODEL")
MAX_VECTORS = 16
VECTOR_DIR = Path("/vectors/steering-vectors")
VECTOR_ID_RE = re.compile(r"^[0-9a-f]{64}$")
_VECTOR_RELOAD_LOCK = threading.Lock()


def _build_multi_vector_steer(entries, num_layers):
	"""Build graph routing for one additive vector per requested layer."""
	if not isinstance(entries, list) or not entries or len(entries) > MAX_VECTORS:
		raise ValueError(f"vectors must contain 1 to {MAX_VECTORS} entries")

	steering_vectors = []
	metadata = []
	used_layers = set()
	for entry in entries:
		coefficient = entry["coefficient"]
		layer = entry["requested_layer"]
		if layer is None:
			layer = entry["stored_layer"]
		if layer is None:
			layer = 10
		if (type(coefficient) not in (int, float) or not math.isfinite(coefficient)
				or type(layer) is not int or not 0 <= layer < num_layers):
			raise ValueError("Invalid layer or coefficient in vectors")
		if layer in used_layers:
			raise ValueError("Only one steering vector may target each layer")
		used_layers.add(layer)
		steering_vectors.append({
			"vector_path": entry["vector_path"], "layer": layer,
			"coefficient": coefficient,
		})
		metadata.append({
			"vector_id": entry["vector_id"], "layer": layer,
			"vector_layer_id": entry["stored_layer"],
		})

	return ({
		"method": "add_vector", "vectors": steering_vectors,
		"phase": "both", "positions": "all_tokens",
	}, metadata)


app = modal.App("steer-capture-inference")
image = (
	modal.Image.from_registry(
		"nvidia/cuda:13.0.2-devel-ubuntu22.04", add_python="3.12"
	)
	.entrypoint([])
	.apt_install("git", "ninja-build", "build-essential")
	.add_local_dir(
		str(ROOT), "/opt/vLLM-Hook", copy=True,
		ignore=[".git", ".venv", ".env", "__pycache__", "*.pt", "*.safetensors"],
	)
	.run_commands(
		"which nvcc && nvcc --version",
		"cd /opt/vLLM-Hook && pip install -r requirement.txt",
		"pip uninstall -y torchcodec",
		"cd /opt/vLLM-Hook && pip install -e . --no-deps",
	)
	.env({
		"CUDA_HOME": "/usr/local/cuda",
		"MIA_WORKER": "steer_capture",
		"MIA_ALLOW_CUDAGRAPH": "1",
		"MIA_STEER_VMAX": str(MAX_VECTORS),
		"VLLM_WORKER_MULTIPROC_METHOD": "spawn",
		"HF_HOME": "/cache/huggingface",
		"MIA_MODEL": MODEL,
	})
)

# Retreving from modal volume
cache = modal.Volume.from_name("ibm-model-cache", create_if_missing=True)
vector_store = modal.Volume.from_name("steering-vector-store", create_if_missing=True)


@app.server(
	image=image, gpu="L40S", port=8770,
	volumes={"/cache": cache, "/vectors": vector_store},
	secrets=[modal.Secret.from_name("vllm-auth")],
	startup_timeout=1800, min_containers=0, max_containers=1,
	scaledown_window=600, unauthenticated=True,
)
class VLLMServer:
	@modal.enter()
	def start(self):
		import subprocess
		import threading
		import time

		import uvicorn
		import torch
		from fastapi import Body, FastAPI, Header, HTTPException
		from transformers import AutoConfig

		api_key = os.environ["VLLM_API_KEY"]
		internal_url = "http://127.0.0.1:8771"
		config = AutoConfig.from_pretrained(MODEL)
		text_config = getattr(config, "text_config", config)
		num_layers = int(text_config.num_hidden_layers)
		self.process = subprocess.Popen([
			"vllm", "serve", MODEL,
			"--host", "127.0.0.1", "--port", "8771",
			"--max-model-len", "2048", "--gpu-memory-utilization", "0.7",
			"--no-enable-prefix-caching", "--api-key", api_key,
		], cwd="/opt/vLLM-Hook")
		try:
			deadline = time.monotonic() + 1500
			while time.monotonic() < deadline:
				if self.process.poll() is not None:
					raise RuntimeError("vLLM exited during startup")
				try:
					if httpx.get(f"{internal_url}/health", timeout=2).is_success:
						break
				except httpx.RequestError:
					pass
				time.sleep(2)
			else:
				raise RuntimeError("vLLM startup timed out")

			self.web_app = FastAPI()

			# Find the saved vector file and read the layer it was extracted from.
			def vector_path_for_id(vector_id):
				if not isinstance(vector_id, str) or VECTOR_ID_RE.fullmatch(vector_id) is None:
					raise HTTPException(
						status_code=422,
						detail="vector_id must be a 64-character lowercase SHA-256 hash",
					)
				path = VECTOR_DIR / f"{vector_id}.pt"
				if not path.is_file():
					with _VECTOR_RELOAD_LOCK:
						if not path.is_file():
							vector_store.reload()
					if not path.is_file():
						raise HTTPException(status_code=404, detail="vector_id was not found")
				saved = torch.load(path, map_location="cpu", weights_only=True)
				stored_layer = saved.get("layer_id") if isinstance(saved, dict) else None
				if stored_layer is not None and type(stored_layer) is not int:
					raise HTTPException(status_code=500, detail="Stored vector layer metadata is invalid")
				return str(path), stored_layer

			@self.web_app.get("/health")
			async def health():
				return {"status": "ok"}

			@self.web_app.post("/steer")
			# Method for steering
			async def steer_response(
				payload: dict = Body(...),
				authorization: str = Header(""),
			):
				if not secrets.compare_digest(authorization, f"Bearer {api_key}"):
					raise HTTPException(status_code=401, detail="Invalid API key")
				vector_id = payload.get("vector_id")
				vectors = payload.get("vectors")
				prompt = payload.get("prompt")
				requested_layer = payload.get("layer")
				coefficient = payload.get("coefficient", 1.0)
				max_tokens = payload.get("max_tokens", 128)
				if not isinstance(prompt, str) or not prompt.strip():
					raise HTTPException(status_code=422, detail="prompt must be a non-empty string")
				if type(max_tokens) is not int or not 1 <= max_tokens <= 2048:
					raise HTTPException(status_code=422, detail="max_tokens must be an integer from 1 to 2048")
				if (vector_id is None) == (vectors is None):
					raise HTTPException(
						status_code=422,
						detail="provide exactly one of vector_id or vectors")

				if vectors is not None:
					if not isinstance(vectors, list) or not vectors or len(vectors) > MAX_VECTORS:
						raise HTTPException(
							status_code=422,
							detail=f"vectors must contain 1 to {MAX_VECTORS} entries")
					entries = []
					for item in vectors:
						if not isinstance(item, dict):
							raise HTTPException(status_code=422, detail="each vector must be an object")
						item_id = item.get("vector_id")
						item_coefficient = item.get("coefficient", 1.0)
						if (type(item_coefficient) not in (int, float)
								or not math.isfinite(item_coefficient)):
							raise HTTPException(status_code=422, detail="Invalid vector coefficient")
						vector_path, stored_layer = vector_path_for_id(item_id)
						entries.append({
							"vector_id": item_id, "vector_path": vector_path,
							"stored_layer": stored_layer,
							"requested_layer": item.get("layer"),
							"coefficient": item_coefficient,
						})
					try:
						steer, vector_metadata = _build_multi_vector_steer(entries, num_layers)
					except ValueError as exc:
						raise HTTPException(status_code=422, detail=str(exc)) from exc
				else:
					if not isinstance(vector_id, str):
						raise HTTPException(
							status_code=422,
							detail="vector_id must be a 64-character lowercase SHA-256 hash",
						)
					vector_path, stored_layer = vector_path_for_id(vector_id)
					layer = requested_layer
					if layer is None:
						layer = stored_layer if stored_layer is not None else 10
					if (type(layer) is not int or not 0 <= layer < num_layers
							or type(coefficient) not in (int, float)
							or not math.isfinite(coefficient)):
						raise HTTPException(status_code=422, detail="Invalid layer or coefficient")
					steer = {
						"vector_path": vector_path, "method": "add_vector",
						"optimal_layer": layer, "coefficient": coefficient,
						"phase": "both", "positions": "all_tokens",
					}

				async with AsyncOpenAI(
					base_url=f"{internal_url}/v1", api_key=api_key, timeout=1800.0
				) as client:
					response = await client.chat.completions.create(
						model=MODEL,
						messages=[{"role": "user", "content": prompt}],
						max_tokens=max_tokens,
						temperature=0.0,
						extra_body={"vllm_xargs": {"steer": json.dumps(steer)}},
					)
				result = response.model_dump(mode="json")
				if vectors is not None:
					result["vectors"] = vector_metadata
				else:
					result["vector_id"] = vector_id
					result["layer"] = layer
					result["vector_layer_id"] = stored_layer
				return result

			self.web_server = uvicorn.Server(uvicorn.Config(
				self.web_app, host="0.0.0.0", port=8770, log_level="info",
			))
			self.web_thread = threading.Thread(
				target=self.web_server.run, name="inference-api", daemon=True,
			)
			self.web_thread.start()
			deadline = time.monotonic() + 30
			while time.monotonic() < deadline:
				if self.web_server.started:
					return
				if not self.web_thread.is_alive():
					raise RuntimeError("Inference wrapper failed during startup")
				time.sleep(0.1)
			raise RuntimeError("Inference wrapper startup timed out")
		except BaseException:
			self.stop()
			raise

	@modal.exit()
	def stop(self):
		import subprocess

		web_server = getattr(self, "web_server", None)
		web_thread = getattr(self, "web_thread", None)
		if web_server is not None:
			web_server.should_exit = True
		if web_thread is not None and web_thread.is_alive():
			web_thread.join(timeout=10)
		process = getattr(self, "process", None)
		if process is not None and process.poll() is None:
			process.terminate()
			try:
				process.wait(timeout=10)
			except subprocess.TimeoutExpired:
				process.kill()
				process.wait()

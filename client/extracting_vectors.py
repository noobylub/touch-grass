"""Extract positive-minus-negative steering vectors for explicit prompt pairs.

Usage: ``python -m client.extracting_vectors`` with VLLM_API_KEY set.
"""
from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path

import httpx
import pandas as pd
import torch

BASE_URL = 'https://noobylub--ibm-vllm-hook-vllmserver.us-east.modal.direct'


def layers_from_env(value: str | None) -> list[int] | None:
    """Parse an optional comma-separated layer selection from MIA_LAYERS."""
    if value is None:
        return None

    try:
        layers = [int(layer.strip()) for layer in value.split(",") if layer.strip()]
    except ValueError as exc:
        raise ValueError("MIA_LAYERS must be comma-separated integer indexes") from exc
    if not layers or any(layer < 0 for layer in layers):
        raise ValueError("MIA_LAYERS must contain non-negative layer indexes")
    return layers


# This is specific to the goemotion dataset 
def collect_emotion_batch(
    csv_path: str | Path | None = None,
    *,
    examples_per_group: int = 200,
    seed: int = 42,
    emotion_column: str = "anger",
) -> dict:
    """Sample an equal number of positive and target-negative examples."""
    if examples_per_group < 1:
        raise ValueError("examples_per_group must be positive")

    path = Path(csv_path) if csv_path is not None else Path(__file__).parent / "data" / "eng_emotion.csv"
    data = pd.read_csv(path, usecols=["text", emotion_column])
    data = data.dropna(subset=["text"]).copy()
    data["text"] = data["text"].astype(str).str.strip()
    data = data[data["text"] != ""]
    labels = pd.to_numeric(data[emotion_column], errors="coerce")
    if labels.isna().any() or not set(labels.unique()).issubset({0, 1}):
        raise ValueError(f"CSV column '{emotion_column}' must contain only binary 0/1 labels")

    positive_rows = data[labels == 1]
    if len(positive_rows) < examples_per_group:
        raise ValueError(
            f"Need {examples_per_group} positive examples; found {len(positive_rows)}"
        )

    positive_sample = positive_rows.sample(n=examples_per_group, random_state=seed)
    negative_rows = data[labels == 0]
    if len(negative_rows) < examples_per_group:
        raise ValueError(
            f"Need {examples_per_group} negative examples; found {len(negative_rows)}"
        )
    negative_sample = negative_rows.sample(n=examples_per_group, random_state=seed)
    run_id = uuid.uuid4().hex[:12]
    return {
        "pairs": [{
            "pair_id": f"{emotion_column}-{run_id}",
            "positive": positive_sample["text"].tolist(),
            "negative": negative_sample["text"].tolist(),
        }],
    }


def connect_HTTP(url: str, key:str):
    if not key:
        raise RuntimeError("Set VLLM_API_KEY before connecting")

    return httpx.AsyncClient(
        base_url=url,
        headers={"Authorization": f"Bearer {key}"},
        timeout=httpx.Timeout(1800.0, connect=30.0),
    )


async def post_with_retry(
    client: httpx.AsyncClient,
    path: str,
    payload: dict,
    retry_window_s: float = 300.0,
) -> httpx.Response:
    deadline = asyncio.get_running_loop().time() + retry_window_s
    delay_s = 1.0
    retryable_statuses = {408, 425, 429, 500, 502, 503, 504}
    first_attempt = True
    last_error: httpx.RequestError | None = None
    last_response: httpx.Response | None = None

    while True:
        if not first_attempt and asyncio.get_running_loop().time() >= deadline:
            if last_response is not None:
                return last_response
            assert last_error is not None
            raise RuntimeError(
                f"Vector extraction request failed after retrying for "
                f"{retry_window_s:g} seconds: {last_error}"
            ) from last_error
        first_attempt = False

        try:
            response = await client.post(path, json=payload)
        except httpx.RequestError as exc:
            last_error = exc
            last_response = None
            remaining_s = deadline - asyncio.get_running_loop().time()
            if remaining_s <= 0:
                raise RuntimeError(
                    f"Vector extraction request failed after retrying for "
                    f"{retry_window_s:g} seconds: {exc}"
                ) from exc
        else:
            if response.status_code not in retryable_statuses:
                return response
            last_response = response
            last_error = None
            remaining_s = deadline - asyncio.get_running_loop().time()
            if remaining_s <= 0:
                return response

        await asyncio.sleep(min(delay_s, remaining_s))
        delay_s = min(delay_s * 2, 15.0)

# Saving vectors locally here 
def save_vectors(results: dict, output_dir: str | Path) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for pair_id, result in results.items():
        for layer_id, values in result["layers"].items():
            filename = f"{pair_id}_layer_{layer_id}.pt"
            if Path(filename).name != filename:
                raise ValueError("Pair and layer IDs must not contain path separators")
            path = output_dir / filename
            torch.save(
                {
                    "dir": torch.tensor(values["dir"], dtype=torch.float32),
                    # "positive_mean": torch.tensor(values["positive_mean"], dtype=torch.float32),
                    # "negative_mean": torch.tensor(values["negative_mean"], dtype=torch.float32),
                    # "positive_count": int(values["positive_count"]),
                    # "negative_count": int(values["negative_count"]),
                    "pair_id": pair_id,
                    "model": result["model"],
                    "layer_id": int(layer_id),
                },
                path,
            )
            saved_paths.append(path)
    return saved_paths


if __name__ == "__main__":
    batch = collect_emotion_batch()
    requested_layers = layers_from_env(os.environ.get("MIA_LAYERS"))
    if requested_layers is not None:
        batch["layers"] = requested_layers
    for pair in batch["pairs"]:
        print(
            f"Prepared {pair['pair_id']}: "
            f"{len(pair['positive'])} positive, {len(pair['negative'])} negative examples"
        )
        for item in pair['positive']:
            print(f"Positive example: {item}")
        for item in pair['negative']:
            print(f"Negative example: {item}")

    async def main():
        api_key = os.environ.get("VLLM_API_KEY")
        async with connect_HTTP(BASE_URL, api_key) as client:
            response = await post_with_retry(
                client,
                "/extract-vectors",
                batch,
            )
        if not response.is_success:
            raise RuntimeError(
                f"Vector extraction failed ({response.status_code}): {response.text[:1000]}"
            )

        results = response.json()
        output_dir = Path(__file__).resolve().parents[1] / "steering_vectors" / "extracted"
        for path in save_vectors(results, output_dir):
            print(f"Saved vector: {path}")
        for pair_id, result in results.items():
            for layer_id, values in result["layers"].items():
                print(pair_id, layer_id)
                print("Positive count:", values["positive_count"])
                print("Negative count:", values["negative_count"])
                print("Direction:", values["dir"])


    asyncio.run(main())

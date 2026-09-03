import gc
from typing import Any

import pandas as pd
import torch
from tqdm.auto import tqdm

from mats12_simple.data import process_batch
from mats12_simple.enums import DataSize, ModelVariant


def token_level_analysis(processor, decoded, continuation_ids, eos_token_id):
    outputs = []

    for token_ids, response in zip(continuation_ids, decoded):
        ids = token_ids.tolist()
        first_id = ids[0] if ids else None

        outputs.append({
            "response": response.strip(),
            "generated_token_ids": ids,
            "first_token_id": first_id,
            "first_token": (
                processor.tokenizer.decode(
                    [first_id],
                    skip_special_tokens=False,
                )
                if first_id is not None
                else None
            ),
            "immediate_eos": (
                first_id == eos_token_id
                if isinstance(eos_token_id, int)
                else first_id in set(eos_token_id or [])
            ),
        })
    return outputs


@torch.inference_mode()
def generate_batch(
    batch: list[dict[str, Any]],
    *,
    model,
    processor,
    max_new_tokens: int = 16,
    token_level: bool = False,
) -> list[str]:

    inputs, _ = process_batch(
        batch=batch,
        processor=processor,
    )
    prompt_width = inputs.input_ids.shape[1]

    # This moves input_ids, attention masks, and audio features.
    inputs = inputs.to(model.device)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
    )

    continuation_ids = generated_ids.sequences[:, prompt_width:].cpu()
    eos_token_id = model.generation_config.eos_token_id

    decoded = processor.batch_decode(
        continuation_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    # Move back to cpu
    inputs = inputs.to("cpu")

    if token_level:
        return token_level_analysis(processor, decoded, continuation_ids, eos_token_id)
    
    return decoded


def write_results(
    results: list[dict], 
    model_variant: ModelVariant, 
    data_size: DataSize, 
    n_iter: int
):
    try:
        results_df = pd.DataFrame(results)
        results_df.to_csv(
            f"behavioral_outputs/behavioral_{data_size.value}_iter{n_iter}_{model_variant.value}.csv",
            index=False
        )
    except ValueError as e:
        print(f"could not write results...\n{e}")





def run_batched_inference(
    prepared: list[dict[str, Any]],
    model, 
    processor, 
    *,
    batch_size: int = 8,
    max_new_tokens: int = 16,
    device: str = "cuda",
    token_level: bool = False,
) -> list[dict[str, Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    model.to(device)
    model.eval()
    processor.tokenizer.padding_side = "left"

    results: list[dict[str, Any]] = []

    try:

        progress = tqdm(
            range(0, len(prepared), batch_size),
            total=(len(prepared) + batch_size - 1) // batch_size,
            desc="Qwen2-Audio inference",
        )

        with torch.inference_mode():
            for start in progress:
                batch = prepared[start : start + batch_size]

                responses = generate_batch(
                    batch,
                    model=model,
                    processor=processor,
                    max_new_tokens=max_new_tokens,
                    token_level=token_level,
                )

                if len(responses) != len(batch):
                    raise RuntimeError(
                        f"Expected {len(batch)} responses, "
                        f"received {len(responses)}"
                    )

                for record, response in zip(batch, responses):
                    results.append({
                        "file": record["file"],
                        "label": record["label"],
                        "speaker_id": record["speaker_id"],
                        "utterance_id": record["utterance_id"],
                        "iteration": record["iteration"],
                        "mode": record["mode"],
                        "model_variant": record["model_variant"],
                        "truth": record["truth"],
                        "conflict": record["conflict"],
                        "response": response if token_level else response.strip(),
                    })

    finally:

        model.to("cpu")
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results



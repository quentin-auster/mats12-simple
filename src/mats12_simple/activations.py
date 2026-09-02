from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


class ActivationCollector:
    def __init__(
        self,
        *,
        output_dtype: torch.dtype = torch.float16,
        move_to_cpu: bool = True,
    ):
        self.output_dtype = output_dtype
        self.move_to_cpu = move_to_cpu
        self.values: dict[str, torch.Tensor] = {}
        self.handles = []

    def _hook(self, name: str):
        def capture(module, inputs, output):
            # Some modules return tuples; encoder layers normally return
            # their hidden-state tensor directly.
            value = output[0] if isinstance(output, tuple) else output
            value = value.detach().to(dtype=self.output_dtype)

            if self.move_to_cpu:
                value = value.cpu()

            self.values[name] = value

        return capture

    def register(self, name: str, module: torch.nn.Module) -> None:
        handle = module.register_forward_hook(self._hook(name))
        self.handles.append(handle)

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()

        self.handles.clear()

    def clear(self) -> None:
        self.values.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.remove()


class MeanPooledCollector:
    def __init__(
        self,
        *,
        output_dtype: torch.dtype = torch.float16,
    ):
        self.output_dtype = output_dtype
        self.values: dict[str, torch.Tensor] = {}
        self.handles = []

    def register(self, name: str, module: torch.nn.Module) -> None:
        def capture(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output

            # [batch, audio_position, width] -> [batch, width]
            pooled = hidden.mean(dim=1)

            self.values[name] = (
                pooled.detach()
                .to(dtype=self.output_dtype)
                .cpu()
            )

        self.handles.append(
            module.register_forward_hook(capture)
        )

    def remove(self):
        for handle in self.handles:
            handle.remove()

        self.handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.remove()





@dataclass
class ActivationBatch:
    """
    metadata: information about the test example
    activations: tensors with shape [batch, selected_layers, d_model]
    """
    metadata: list[dict[str, Any]]
    activations: dict[str, torch.Tensor]


def get_output_attention_mask(outputs, inputs) -> torch.Tensor:
    """
    Get the attention mask corresponding to the expanded residual sequence.

    Recent Transformers versions return the post-audio-expansion mask as
    outputs.attention_mask. If the processor already expanded the audio
    placeholders, inputs.attention_mask has the same sequence length.
    """
    hidden_length = outputs.hidden_states[0].shape[1]

    output_mask = getattr(outputs, "attention_mask", None)

    if output_mask is not None and output_mask.shape[1] == hidden_length:
        return output_mask.bool()

    if inputs.attention_mask.shape[1] == hidden_length:
        return inputs.attention_mask.bool()

    raise RuntimeError(
        "Could not align the attention mask with hidden states. "
        f"hidden length={hidden_length}, "
        f"input mask length={inputs.attention_mask.shape[1]}. "
        "Your Transformers version may be using legacy audio expansion."
    )


def get_audio_position_mask(
    *,
    model,
    inputs,
    hidden_length: int,
) -> torch.Tensor:
    """
    Identify audio positions in the expanded residual stream.

    On current Transformers versions, the processor expands <|AUDIO|>
    into the correct number of placeholder token IDs before the forward pass.
    """
    input_ids = inputs.input_ids
    audio_token_id = model.config.audio_token_id

    if input_ids.shape[1] != hidden_length:
        raise RuntimeError(
            "input_ids are not aligned with the expanded residual stream. "
            "Upgrade Transformers or capture the merged input mask with a hook. "
            f"input length={input_ids.shape[1]}, hidden length={hidden_length}"
        )

    return input_ids.eq(audio_token_id)


def masked_mean(
    hidden: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    hidden: [batch, sequence, d_model]
    mask:   [batch, sequence]
    """
    weights = mask.to(hidden.dtype).unsqueeze(-1)
    counts = weights.sum(dim=1).clamp_min(1.0)

    return (hidden * weights).sum(dim=1) / counts


def select_last_position(
    hidden: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Select the final True position for every batch item.

    Works for either left or right padding.
    """
    positions = torch.arange(
        mask.shape[1],
        device=mask.device,
    ).unsqueeze(0)

    last_indices = positions.masked_fill(~mask, -1).max(dim=1).values
    batch_indices = torch.arange(hidden.shape[0], device=hidden.device)

    return hidden[batch_indices, last_indices]


@torch.inference_mode()
def extract_activation_batch(
    batch: list[dict[str, Any]],
    *,
    model,
    processor,
    layers: Sequence[int] | None = None,
    device: str = "cuda",
    output_dtype: torch.dtype = torch.float16,
) -> ActivationBatch:
    """
    Run one prefill forward pass and extract pooled residual activations.

    No generation is needed: the final prompt-position residual is the state
    used to predict the first response token.
    """
    inputs, _ = process_batch(
        batch=batch,
        processor=processor,
    )
    inputs = inputs.to(device)

    outputs = model(
        **inputs,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )

    hidden_states = outputs.hidden_states

    if layers is None:
        selected_layers = list(range(len(hidden_states)))
    else:
        selected_layers = list(layers)

    attention_mask = get_output_attention_mask(outputs, inputs)
    hidden_length = hidden_states[0].shape[1]

    audio_mask = get_audio_position_mask(
        model=model,
        inputs=inputs,
        hidden_length=hidden_length,
    )
    audio_mask &= attention_mask

    if not audio_mask.any(dim=1).all():
        bad_rows = (~audio_mask.any(dim=1)).nonzero().flatten().tolist()
        raise RuntimeError(
            f"Some records have no audio positions: batch rows {bad_rows}. "
            "Process actual text-only conditions separately."
        )

    last_activations = []
    audio_mean_activations = []
    audio_last_activations = []

    for layer_index in selected_layers:
        hidden = hidden_states[layer_index]

        last_activations.append(
            select_last_position(hidden, attention_mask)
        )
        audio_mean_activations.append(
            masked_mean(hidden, audio_mask)
        )
        audio_last_activations.append(
            select_last_position(hidden, audio_mask)
        )

    activations = {
        "last": torch.stack(last_activations, dim=1)
            .to(dtype=output_dtype)
            .cpu(),
        "audio_mean": torch.stack(audio_mean_activations, dim=1)
            .to(dtype=output_dtype)
            .cpu(),
        "audio_last": torch.stack(audio_last_activations, dim=1)
            .to(dtype=output_dtype)
            .cpu(),
    }

    metadata = [
        {
            "file": record["file"],
            "speaker_id": record["speaker_id"],
            "utterance_id": record["utterance_id"],
            "iteration": record["iteration"],
            "mode": record["mode"],
            "model_variant": record["model_variant"],
            "truth": record["truth"],
            "conflict": record["conflict"],
        }
        for record in batch
    ]

    del outputs, hidden_states, inputs
    return ActivationBatch(
        metadata=metadata,
        activations=activations,
    )
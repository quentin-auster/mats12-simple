from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoProcessor,
    Qwen2AudioForConditionalGeneration,
)


@dataclass
class Qwen2AudioActivations:

    # One tensor for LM embeddings plus one per LM transformer layer.
    # Each tensor: [batch, multimodal_sequence_length, 4096]
    lm_hidden_states: tuple[torch.Tensor, ...]

    # Final output of the audio tower, after pooling and final LayerNorm.
    # Shape: [num_audios, max_audio_positions, 1280]
    audio_final: torch.Tensor

    # Audio tower output after projection into LM embedding space.
    # Shape: [num_audios, max_audio_positions, 4096]
    audio_projected: torch.Tensor

    # Number of valid positions for each audio after conv/pooling.
    # Shape: [num_audios]
    audio_lengths: torch.Tensor

    # Final valid audio-tower position for every audio.
    # Shape: [num_audios, 1280]
    audio_last_valid: torch.Tensor

    # Final valid projected audio position for every audio.
    # Shape: [num_audios, 4096]
    audio_projected_last_valid: torch.Tensor

    # Positions occupied by audio inside the multimodal LM sequence.
    # Shape: [batch, multimodal_sequence_length]
    lm_audio_mask: torch.Tensor

    def write_to_disk(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True)

        for layer_index, hidden_state in enumerate(self.lm_hidden_states):
            torch.save(
                hidden_state,
                output_dir / f"lm_hidden_state_{layer_index}.pt",
            )

        for name in (
            "audio_final",
            "audio_projected",
            "audio_lengths",
            "audio_last_valid",
            "audio_projected_last_valid",
            "lm_audio_mask",
        ):
            torch.save(getattr(self, name), output_dir / f"{name}.pt")


def _extract_last_valid(
    sequence: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """
    sequence: [batch, positions, hidden]
    lengths:  [batch]

    Returns sequence[i, lengths[i] - 1] for every i.
    """
    indices = lengths.to(sequence.device) - 1

    if torch.any(indices < 0):
        raise ValueError("Encountered an audio sequence with zero valid positions")

    batch_indices = torch.arange(sequence.shape[0], device=sequence.device)
    return sequence[batch_indices, indices]


@torch.inference_mode()
def extract_activations(
    model: Qwen2AudioForConditionalGeneration,
    inputs: dict[str, torch.Tensor],
    *,
    move_to_cpu: bool = True,
) -> Qwen2AudioActivations:
    captured: dict[str, torch.Tensor] = {}

    def audio_tower_hook(
        module: torch.nn.Module,
        hook_inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        # Qwen2AudioEncoder returns BaseModelOutput.
        captured["audio_final"] = output.last_hidden_state.detach()

    def projector_hook(
        module: torch.nn.Module,
        hook_inputs: tuple[Any, ...],
        output: torch.Tensor,
    ) -> None:
        captured["audio_projected"] = output.detach()

    audio_handle = model.model.audio_tower.register_forward_hook(
        audio_tower_hook
    )
    projector_handle = model.model.multi_modal_projector.register_forward_hook(
        projector_hook
    )

    try:
        outputs = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    finally:
        # Always remove hooks, including when the forward pass raises.
        audio_handle.remove()
        projector_handle.remove()

    audio_final = captured["audio_final"]
    audio_projected = captured["audio_projected"]

    # feature_attention_mask counts valid input log-mel frames.
    input_feature_lengths = inputs["feature_attention_mask"].sum(dim=-1)

    # First return value: length after convolution.
    # Second return value: length after final audio pooling.
    _, audio_lengths = (
        model.model.audio_tower._get_feat_extract_output_lengths(
            input_feature_lengths
        )
    )

    audio_last_valid = _extract_last_valid(
        audio_final,
        audio_lengths,
    )
    audio_projected_last_valid = _extract_last_valid(
        audio_projected,
        audio_lengths,
    )

    # With current processors, the single <|AUDIO|> placeholder is expanded
    # into one placeholder per projected audio position.
    audio_token_id = getattr(
        model.config,
        "audio_token_id",
        getattr(model.config, "audio_token_index", None),
    )

    if audio_token_id is None:
        raise AttributeError(
            "Could not find audio_token_id or audio_token_index in model config"
        )

    lm_audio_mask = inputs["input_ids"].eq(audio_token_id)

    lm_hidden_states = tuple(
        hidden.detach()
        for hidden in outputs.hidden_states
    )

    if move_to_cpu:
        lm_hidden_states = tuple(
            hidden.to(device="cpu", dtype=torch.float32)
            for hidden in lm_hidden_states
        )
        audio_final = audio_final.to(device="cpu", dtype=torch.float32)
        audio_projected = audio_projected.to(
            device="cpu",
            dtype=torch.float32,
        )
        audio_lengths = audio_lengths.cpu()
        audio_last_valid = audio_last_valid.to(
            device="cpu",
            dtype=torch.float32,
        )
        audio_projected_last_valid = audio_projected_last_valid.to(
            device="cpu",
            dtype=torch.float32,
        )
        lm_audio_mask = lm_audio_mask.cpu()

    return Qwen2AudioActivations(
        lm_hidden_states=lm_hidden_states,
        audio_final=audio_final,
        audio_projected=audio_projected,
        audio_lengths=audio_lengths,
        audio_last_valid=audio_last_valid,
        audio_projected_last_valid=audio_projected_last_valid,
        lm_audio_mask=lm_audio_mask,
    )

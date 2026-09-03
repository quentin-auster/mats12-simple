from collections import Counter, defaultdict
from random import Random
from typing import Any

import numpy as np
from datasets import Dataset

from mats12_simple.enums import ModelVariant, Modes


def create_balanced_subset(
    dataset: Dataset,
    target_n: int,
    label_column: str = "label",
    seed: int = 123,
) -> Dataset:
    if target_n > len(dataset):
        raise ValueError(
            f"Requested {target_n} records, but dataset only has {len(dataset)}"
        )

    rng = Random(seed)

    # Collect dataset indices by label.
    indices_by_label = defaultdict(list)

    for index, label in enumerate(dataset[label_column]):
        indices_by_label[label].append(index)

    # Randomize both samples within each label and label traversal order.
    labels = list(indices_by_label)
    rng.shuffle(labels)

    for indices in indices_by_label.values():
        rng.shuffle(indices)

    # Select one example per label per round.
    chosen_indices = []

    while len(chosen_indices) < target_n:
        added_this_round = False

        for label in labels:
            if indices_by_label[label]:
                chosen_indices.append(indices_by_label[label].pop())
                added_this_round = True

                if len(chosen_indices) == target_n:
                    break

        if not added_this_round:
            break

        # Prevent the same labels always occurring first.
        rng.shuffle(labels)

    if len(chosen_indices) < target_n:
        raise ValueError(
            f"Requested {target_n} records, but only selected {len(chosen_indices)}"
        )

    rng.shuffle(chosen_indices)
    return dataset.select(chosen_indices)


def verify_balance(dataset: Dataset, label_column: str = "label") -> None:
    counts = Counter(dataset["label"])
    label_feature = dataset.features["label"]
    for label_id, count in sorted(counts.items()):
        label_name = label_feature.int2str(label_id)
        print(f"{label_name:20s} {count}")


def prepare_audio(
    mode: Modes,
    audio_array: np.ndarray,
    sample_rate: int = 16_000,
    padding_seconds: float = 0.5,
) -> np.ndarray | None:

    audio = np.asarray(audio_array, dtype=np.float32)
    padding_samples = int(sample_rate * padding_seconds)

    padded = np.pad(
        audio,
        pad_width=(padding_samples, padding_samples),
        mode="constant",
        constant_values=0.0,
    )

    if mode in [Modes.TEXT_ONLY_WRONG, Modes.TEXT_ONLY_TRUTH]:
        return np.zeros_like(padded)
    else:
        return padded


def choose_conflict(
    truth: str,
    alternatives: set[str],
    rng: Random,
) -> str:
    candidates = sorted(alternatives - {truth})

    if not candidates:
        raise ValueError(
            f"No conflicting labels available for truth={truth!r}"
        )

    return rng.choice(candidates)


def prepare_mode_text(
    mode: Modes,
    truth: str,
    conflict: str,
) -> str:
    init_instruction = "<task>Transcribe the spoken word yourself for me.</task>"
    final_instruction = "The word was:"

    if mode is Modes.AUDIO_ONLY:
        return f'{init_instruction} {final_instruction}'

    if mode in [Modes.AGREE, Modes.TEXT_ONLY_TRUTH]:
        return f'{init_instruction} I heard {truth!r}. {final_instruction}'

    if mode in [Modes.CONFLICT, Modes.TEXT_ONLY_WRONG]:
        return f'{init_instruction} I heard {conflict!r}. {final_instruction}'

    raise ValueError(f"Unsupported mode: {mode}")


def prepare_pretrained_prompt(
    mode: Modes,
    truth: str,
    conflict: str,
) -> str:

    instruction = prepare_mode_text(
        mode=mode,
        truth=truth,
        conflict=conflict,
    )

    audio_marker = "<|audio_bos|><|AUDIO|><|audio_eos|>"

    return f"{audio_marker}{instruction}"


def prepare_instruct_conversation(
    mode: Modes,
    truth: str,
    conflict: str,
    *,
    system_prompt: str | None = "You are a helpful assistant.",
) -> list[dict[str, Any]]:
    instruction = prepare_mode_text(
        mode=mode,
        truth=truth,
        conflict=conflict,
    )
    user_content = [
        {"type": "audio"},
        {"type": "text", "text": instruction},
    ]

    conversation: list[dict[str, Any]] = []

    if system_prompt is not None:
        conversation.append({
            "role": "system",
            "content": system_prompt,
        })

    conversation.append({
        "role": "user",
        "content": user_content,
    })

    return conversation


def prepare_record(
    example: dict[str, Any],
    *,
    mode: Modes,
    model_variant: ModelVariant,
    truth: str,
    conflict: str,
    processor,
    sample_rate: int,
    padding_seconds: float = 0.5,
    system_prompt: str | None = "You are a helpful assistant.",
) -> dict[str, Any]:
    audio = prepare_audio(
        mode=mode,
        audio_array=example["audio"]["array"],
        sample_rate=sample_rate,
        padding_seconds=padding_seconds,
    )

    if model_variant is ModelVariant.PRETRAINED:
        prompt = prepare_pretrained_prompt(
            mode=mode,
            truth=truth,
            conflict=conflict,
        )
        conversation = None

    elif model_variant is ModelVariant.INSTRUCT:
        conversation = prepare_instruct_conversation(
            mode=mode,
            truth=truth,
            conflict=conflict,
            system_prompt=system_prompt,
        )

        prompt = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )

    else:
        raise ValueError(f"Unsupported model_variant: {model_variant}")

    return {
        "file": example["file"],
        "label": example["label"],
        "speaker_id": example["speaker_id"],
        "utterance_id": example["utterance_id"],
        "model_variant": model_variant.value,
        "mode": mode.value,
        "truth": truth,
        "conflict": conflict,
        "prompt": prompt,
        "conversation": conversation,
        "audio": audio,
    }


def prepare_experiment(
    dataset,
    processor,
    *,
    model_variant: ModelVariant,
    alternatives: set[str],
    n_iterations: int = 10,
    seed: int = 123,
    padding_seconds: float = 0.5,
    system_prompt: str | None = "You are a helpful assistant.",
) -> list[dict[str, Any]]:
    rng = Random(seed)
    label_feature = dataset.features["label"]
    sample_rate = processor.feature_extractor.sampling_rate

    prepared = []
    for example in dataset:
        truth = label_feature.int2str(example["label"])

        for iteration in range(n_iterations):
            conflict = choose_conflict(
                truth=truth,
                alternatives=alternatives,
                rng=rng,
            )

            for mode in Modes:
                record = prepare_record(
                    example,
                    mode=mode,
                    model_variant=model_variant,
                    truth=truth,
                    conflict=conflict,
                    processor=processor,
                    sample_rate=sample_rate,
                    padding_seconds=padding_seconds,
                    system_prompt=system_prompt,
                )

                record["iteration"] = iteration
                prepared.append(record)

    return prepared


def process_batch(
    batch: list[dict[str, Any]],
    processor,
):
    prompts = [record["prompt"] for record in batch]

    # Flatten the audio inputs while skipping TEXT_ONLY records.
    audios = [
        record["audio"]
        for record in batch
        if record["audio"] is not None
    ]

    kwargs = {
        "text": prompts,
        "return_tensors": "pt",
        "padding": True,
    }

    if audios:
        kwargs["audio"] = audios
        kwargs["sampling_rate"] = (
            processor.feature_extractor.sampling_rate
        )

    return processor(**kwargs), kwargs


def get_alternatives(data):
    return (
        set(data.features["label"].names)
        - {"_silence_"}
    )
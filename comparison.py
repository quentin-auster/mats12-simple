import pandas as pd
import torch


def compare_parameter_groups(base_model, instruct_model):
    groups = {
        "audio_tower": (
            base_model.model.audio_tower,
            instruct_model.model.audio_tower,
        ),
        "projector": (
            base_model.model.multi_modal_projector,
            instruct_model.model.multi_modal_projector,
        ),
        "language_model": (
            base_model.model.language_model,
            instruct_model.model.language_model,
        ),
        "lm_head": (
            base_model.lm_head,
            instruct_model.lm_head,
        ),
    }

    rows = []

    for group_name, (base_group, instruct_group) in groups.items():
        base_parameters = dict(base_group.named_parameters())
        instruct_parameters = dict(instruct_group.named_parameters())

        squared_difference = 0.0
        squared_base_norm = 0.0
        max_absolute_difference = 0.0
        changed_tensors = 0
        total_tensors = 0

        for name, base_parameter in base_parameters.items():
            instruct_parameter = instruct_parameters[name]

            base_float = base_parameter.detach().float().cpu()
            instruct_float = instruct_parameter.detach().float().cpu()
            difference = instruct_float - base_float

            squared_difference += difference.square().sum().item()
            squared_base_norm += base_float.square().sum().item()
            max_absolute_difference = max(
                max_absolute_difference,
                difference.abs().max().item(),
            )

            changed_tensors += int(
                not torch.equal(base_float, instruct_float)
            )
            total_tensors += 1

        rows.append({
            "group": group_name,
            "relative_l2_change": (
                squared_difference**0.5
                / max(squared_base_norm**0.5, 1e-12)
            ),
            "max_absolute_change": max_absolute_difference,
            "changed_tensors": changed_tensors,
            "total_tensors": total_tensors,
        })

    return pd.DataFrame(rows)
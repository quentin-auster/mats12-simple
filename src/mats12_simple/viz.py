import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_activation_heatmaps(
    activation_top: torch.Tensor,
    activation_bottom: torch.Tensor,
    *,
    batch_index: int = 0,
    normalize: str | None = None,
    max_dimensions: int | None = None,
    token_labels: list[str] | None = None,
    titles: tuple[str, str] = (
        "Activation 1",
        "Activation 2",
    ),
    figsize: tuple[int, int] = (14, 12),
):
    """
    Plot two activation tensors as vertically stacked heatmaps.

    Both tensors must have shape:

        [batch, sequence, model_dim]

    or:

        [sequence, model_dim]

    The plots share:

    - The sequence-position x-axis
    - The selected model dimensions
    - The activation color scale

    normalize:
        None
            Plot raw activation values.

        "dimension"
            Independently z-score each model dimension across sequence
            positions within each activation tensor.

        "position"
            Independently z-score each sequence position across model
            dimensions within each activation tensor.

    max_dimensions:
        If provided, display the dimensions with the highest average
        variance across the two activation tensors.
    """

    def prepare_activation(
        activation: torch.Tensor,
    ) -> np.ndarray:
        activation = activation.detach().float().cpu()

        if activation.ndim == 3:
            if not 0 <= batch_index < activation.shape[0]:
                raise IndexError(
                    f"batch_index={batch_index} is invalid for a batch "
                    f"of size {activation.shape[0]}"
                )

            activation = activation[batch_index]

        if activation.ndim != 2:
            raise ValueError(
                "Expected [batch, sequence, model_dim] or "
                f"[sequence, model_dim], got {tuple(activation.shape)}"
            )

        values = activation.numpy()

        if normalize == "dimension":
            mean = values.mean(axis=0, keepdims=True)
            std = values.std(axis=0, keepdims=True)
            values = (values - mean) / np.maximum(std, 1e-8)

        elif normalize == "position":
            mean = values.mean(axis=1, keepdims=True)
            std = values.std(axis=1, keepdims=True)
            values = (values - mean) / np.maximum(std, 1e-8)

        elif normalize is not None:
            raise ValueError(
                "normalize must be None, 'dimension', or 'position'"
            )

        return values

    top_values = prepare_activation(activation_top)
    bottom_values = prepare_activation(activation_bottom)

    if top_values.shape != bottom_values.shape:
        raise ValueError(
            "The two activations must have the same sequence length "
            "and model dimension for direct comparison. Got "
            f"{top_values.shape} and {bottom_values.shape}."
        )

    num_positions, model_dim = top_values.shape

    if token_labels is not None and len(token_labels) != num_positions:
        raise ValueError(
            f"Received {len(token_labels)} token labels for "
            f"{num_positions} sequence positions"
        )

    selected_dimensions = np.arange(model_dim)

    if max_dimensions is not None:
        if max_dimensions <= 0:
            raise ValueError("max_dimensions must be positive")

        if max_dimensions < model_dim:
            # Select dimensions that vary most, on average, across
            # the two activation tensors.
            mean_variance = (
                top_values.var(axis=0)
                + bottom_values.var(axis=0)
            ) / 2

            selected_dimensions = np.argsort(mean_variance)[
                -max_dimensions:
            ]

            # Keep dimensions in their original model order.
            selected_dimensions = np.sort(selected_dimensions)

    top_heatmap = top_values[:, selected_dimensions].T
    bottom_heatmap = bottom_values[:, selected_dimensions].T

    # Use a shared, symmetric color scale.
    combined_absolute_values = np.concatenate(
        [
            np.abs(top_heatmap).ravel(),
            np.abs(bottom_heatmap).ravel(),
        ]
    )

    color_limit = np.quantile(combined_absolute_values, 0.99)

    # Avoid an invalid color range if all activations are zero.
    color_limit = max(float(color_limit), 1e-8)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=figsize,
        sharex=True,
        constrained_layout=True,
    )

    images = []

    for ax, heatmap, title in zip(
        axes,
        [top_heatmap, bottom_heatmap],
        titles,
    ):
        image = ax.imshow(
            heatmap,
            aspect="auto",
            interpolation="nearest",
            cmap="RdBu_r",
            vmin=-color_limit,
            vmax=color_limit,
            origin="lower",
        )

        images.append(image)

        ax.set_title(title, fontsize=12)

        if max_dimensions is not None:
            ax.set_ylabel(
                f"Top {len(selected_dimensions)} varying dimensions"
            )
        else:
            ax.set_ylabel("Model dimension")

    axes[-1].set_xlabel("Sequence position")

    if token_labels is not None:
        ticks = np.arange(num_positions)

        axes[-1].set_xticks(ticks)
        axes[-1].set_xticklabels(
            token_labels,
            rotation=90,
            fontsize=7,
        )

    else:
        tick_step = max(1, num_positions // 15)
        ticks = np.arange(0, num_positions, tick_step)

        axes[-1].set_xticks(ticks)
        axes[-1].set_xticklabels(ticks)

    colorbar = fig.colorbar(
        images[0],
        ax=axes,
        pad=0.02,
    )

    colorbar.set_label(
        "Standardized activation"
        if normalize is not None
        else "Activation value"
    )

    audio_cutoff_index = 51

    for ax in axes:
        ax.axvline(
            x=audio_cutoff_index - 0.5,
            color="black",
            linestyle="-",
            linewidth=3,
            alpha=0.9,
        )

    axes[0].annotate(
        "Audio cutoff",
        xy=(audio_cutoff_index - 0.5, 1),
        xycoords=("data", "axes fraction"),
        xytext=(4, 6),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color="black",
    )

    return fig, axes
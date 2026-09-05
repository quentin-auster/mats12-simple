
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from mats12_simple.activations import PositionStrategy, activation_to_vector

SplitStrategy = Literal["stratified", "group"]


def prepare_probe_metadata(
    metadata: pd.DataFrame,
    *,
    modes: Sequence[str] | None,
    layers: Sequence[int],
    activation_template: str = "lm_hidden_state_{layer}.pt",
    output_dir_column: str = "output_dir",
    label_column: str = "label",
    group_column: str | None = "speaker_id",
) -> pd.DataFrame:
    required_columns = {
        output_dir_column,
        label_column,
    }

    if modes is not None:
        required_columns.add("mode")

    if group_column is not None:
        required_columns.add(group_column)

    missing_columns = required_columns - set(metadata.columns)

    if missing_columns:
        raise ValueError(
            f"Metadata is missing columns: {sorted(missing_columns)}"
        )

    rows = metadata.copy()

    if modes is not None:
        rows = rows.loc[rows["mode"].isin(modes)]

    rows = rows.reset_index(drop=True)

    if rows.empty:
        raise ValueError("No metadata rows matched the requested modes")

    activation_names = [
        activation_template.format(layer=layer)
        for layer in layers
    ]

    has_all_files = rows[output_dir_column].map(
        lambda output_dir: False if pd.isna(output_dir) else all(
            (Path(str(output_dir)) / name).exists()
            for name in activation_names
        )
    )

    missing_count = int((~has_all_files).sum())

    if missing_count:
        print(
            f"Dropping {missing_count} examples that do not have "
            "all requested activation files"
        )
        rows = rows.loc[has_all_files].reset_index(drop=True)

    if rows.empty:
        raise ValueError(
            "No examples have activation files for all requested layers"
        )

    return rows


def create_probe_split(
    metadata: pd.DataFrame,
    *,
    label_column: str = "label",
    strategy: SplitStrategy = "group",
    group_column: str = "speaker_id",
    test_size: float = 0.2,
    random_state: int = 123,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(metadata))
    y = metadata[label_column].to_numpy()

    if strategy == "stratified":
        train_indices, test_indices = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

    elif strategy == "group":
        if group_column not in metadata.columns:
            raise ValueError(
                f"Group column {group_column!r} is not in metadata"
            )

        groups = metadata[group_column].to_numpy()

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state,
        )

        train_indices, test_indices = next(
            splitter.split(
                indices,
                y,
                groups=groups,
            )
        )

        train_groups = set(groups[train_indices])
        test_groups = set(groups[test_indices])

        if not train_groups.isdisjoint(test_groups):
            raise RuntimeError(
                "Group leakage detected between train and test"
            )

    else:
        raise ValueError(f"Unknown split strategy: {strategy}")

    return train_indices, test_indices


def load_layer_activations(
    metadata: pd.DataFrame,
    *,
    layer: int,
    activation_template: str = "lm_hidden_state_{layer}.pt",
    output_dir_column: str = "output_dir",
    label_column: str = "label",
    position_strategy: PositionStrategy = "already_vector",
    tensor_key: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    activation_name = activation_template.format(layer=layer)

    vectors = []
    labels = []

    iterator = metadata.itertuples(index=False)

    for row in tqdm(
        iterator,
        total=len(metadata),
        desc=f"Loading layer {layer}",
        leave=False,
    ):
        output_dir = getattr(row, output_dir_column)
        label = getattr(row, label_column)

        activation_path = Path(output_dir) / activation_name

        activation = torch.load(
            activation_path,
            map_location="cpu",
            weights_only=True,
        )

        # Useful if each .pt file contains a dictionary.
        if tensor_key is not None:
            if not isinstance(activation, dict):
                raise TypeError(
                    f"{activation_path} is not a dictionary"
                )
            activation = activation[tensor_key]

        if not isinstance(activation, torch.Tensor):
            raise TypeError(
                f"Expected a tensor in {activation_path}, "
                f"received {type(activation)}"
            )

        vector = activation_to_vector(
            activation,
            strategy=position_strategy,
        )

        vectors.append(vector.numpy())
        labels.append(label)

    X = np.stack(vectors).astype(np.float32)
    y = np.asarray(labels)

    return X, y


def fit_linear_probe(
    X: np.ndarray,
    y: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    *,
    C: float = 1.0,
    max_iter: int = 2_000,
):
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver="lbfgs",
        ),
    )

    probe.fit(X_train, y_train)

    train_predictions = probe.predict(X_train)
    test_predictions = probe.predict(X_test)

    metrics = {
        "train_accuracy": accuracy_score(
            y_train,
            train_predictions,
        ),
        "test_accuracy": accuracy_score(
            y_test,
            test_predictions,
        ),
        "test_balanced_accuracy": balanced_accuracy_score(
            y_test,
            test_predictions,
        ),
        "test_macro_f1": f1_score(
            y_test,
            test_predictions,
            average="macro",
            zero_division=0,
        ),
        "num_train": len(train_indices),
        "num_test": len(test_indices),
        "num_features": X.shape[1],
    }

    return {
        "probe": probe,
        "metrics": metrics,
        "train_predictions": train_predictions,
        "test_predictions": test_predictions,
        "y_train": y_train,
        "y_test": y_test,
    }


def probe_layers(
    metadata: pd.DataFrame,
    *,
    layers: Sequence[int],
    modes: Sequence[str] | None = None,
    activation_template: str = "lm_hidden_state_{layer}.pt",
    output_dir_column: str = "output_dir",
    label_column: str = "label",
    position_strategy: PositionStrategy = "already_vector",
    split_strategy: SplitStrategy = "group",
    group_column: str = "speaker_id",
    test_size: float = 0.2,
    random_state: int = 123,
    C: float = 1.0,
    tensor_key: str | None = None,
):
    layers = list(layers)

    if not layers:
        raise ValueError("At least one layer must be provided")

    probe_metadata = prepare_probe_metadata(
        metadata,
        modes=modes,
        layers=layers,
        activation_template=activation_template,
        output_dir_column=output_dir_column,
        label_column=label_column,
        group_column=(
            group_column
            if split_strategy == "group"
            else None
        ),
    )

    # These exact indices are reused for every layer.
    train_indices, test_indices = create_probe_split(
        probe_metadata,
        label_column=label_column,
        strategy=split_strategy,
        group_column=group_column,
        test_size=test_size,
        random_state=random_state,
    )

    result_rows = []
    probes = {}
    predictions = {}

    for layer in tqdm(layers, desc="Probing layers"):
        X, y = load_layer_activations(
            probe_metadata,
            layer=layer,
            activation_template=activation_template,
            output_dir_column=output_dir_column,
            label_column=label_column,
            position_strategy=position_strategy,
            tensor_key=tensor_key,
        )

        result = fit_linear_probe(
            X,
            y,
            train_indices,
            test_indices,
            C=C,
        )

        result_rows.append({
            "layer": layer,
            **result["metrics"],
        })

        probes[layer] = result["probe"]
        predictions[layer] = result["test_predictions"]

    results = pd.DataFrame(result_rows)

    split_metadata = {
        "train": probe_metadata.iloc[train_indices].copy(),
        "test": probe_metadata.iloc[test_indices].copy(),
    }

    return {
        "results": results,
        "probes": probes,
        "predictions": predictions,
        "metadata": probe_metadata,
        "split_metadata": split_metadata,
        "train_indices": train_indices,
        "test_indices": test_indices,
    }
"""Read a formatted artifact (H5/JSON/Parquet) back into PLAP-ready dicts."""

from __future__ import annotations

import json
from pathlib import Path


def load_formatted(path: str | Path) -> dict:
    """Read a formatted artifact written by :func:`write` back into a dict.

    Parameters
    ----------
    path : str or Path
        Path to the artifact.  If it is a directory, the Parquet layout is
        assumed (``train_images.parquet``, ``train_labels.parquet``,
        ``test_images.parquet``, ``test_labels.parquet`` +
        ``metadata.json``).  Otherwise the file extension is used to
        determine the format (``.h5`` / ``.hdf5`` → HDF5; ``.json`` → JSON).

    Returns
    -------
    dict
        Keys ``train_images``, ``train_labels``, ``test_images``,
        ``test_labels``, ``input_dim``, ``output_dim``, ``n_train``,
        ``n_test``, ``label_map``, and optionally ``means`` / ``stds``.
        The four data values are plain Python lists (or nested lists)
        suitable for serializing into a composite document.
    """
    p = Path(path)

    if p.is_dir():
        return _load_formatted_parquet(p)

    ext = p.suffix.lower()
    loaders = {".h5": _load_formatted_h5, ".hdf5": _load_formatted_h5,
               ".json": _load_formatted_json}
    loader = loaders.get(ext)
    if loader is None:
        raise ValueError(
            f"Unrecognised formatted-artifact path '{p}'. "
            f"Expected a .h5 / .hdf5 / .json file or a Parquet directory."
        )
    return loader(p)


# ---------------------------------------------------------------------------
# Per-format readers
# ---------------------------------------------------------------------------

def _load_formatted_h5(path: Path) -> dict:
    import h5py
    import numpy as np

    data: dict = {}
    with h5py.File(path, "r") as f:
        data["train_images"] = np.array(f["train_images"]).tolist()
        data["train_labels"] = np.array(f["train_labels"]).tolist()
        data["test_images"] = np.array(f["test_images"]).tolist()
        data["test_labels"] = np.array(f["test_labels"]).tolist()
        data["input_dim"] = f.attrs["input_dim"]
        data["output_dim"] = f.attrs["output_dim"]
        data["n_train"] = f.attrs["n_train"]
        data["n_test"] = f.attrs["n_test"]
        data["label_map"] = json.loads(f.attrs["label_map"])
        if "means" in f.attrs:
            data["means"] = json.loads(f.attrs["means"])
        if "stds" in f.attrs:
            data["stds"] = json.loads(f.attrs["stds"])
    return data


def _load_formatted_json(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)
    return data


def _load_formatted_parquet(path: Path) -> dict:
    import polars as pl
    import numpy as np

    train_img = pl.read_parquet(path / "train_images.parquet")
    train_lbl = pl.read_parquet(path / "train_labels.parquet")
    test_img = pl.read_parquet(path / "test_images.parquet")
    test_lbl = pl.read_parquet(path / "test_labels.parquet")

    with open(path / "metadata.json") as f:
        meta = json.load(f)

    return {
        "train_images": [np.array(r).tolist() for r in train_img["feature_vector"]],
        "train_labels": train_lbl["label"].to_list(),
        "test_images": [np.array(r).tolist() for r in test_img["feature_vector"]],
        "test_labels": test_lbl["label"].to_list(),
        **meta,
    }

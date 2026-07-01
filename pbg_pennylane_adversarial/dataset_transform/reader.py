"""Auto-detect file format and read any classification dataset into a polars DataFrame."""

from __future__ import annotations

from pathlib import Path

_SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".csv": "CSV",
    ".tsv": "TSV (tab-separated)",
    ".tab": "TSV (tab-separated)",
    ".parquet": "Parquet",
    ".pq": "Parquet",
    ".json": "JSON",
    ".jsonl": "JSON Lines",
    ".ndjson": "JSON Lines",
    ".xlsx": "Excel (.xlsx)",
    ".xls": "Excel (.xls)",
    ".feather": "Feather / Arrow IPC",
    ".arrow": "Feather / Arrow IPC",
    ".h5": "HDF5",
    ".hdf5": "HDF5",
    ".npy": "NumPy (.npy)",
    ".npz": "NumPy (.npz)",
    ".pkl": "Pickle",
    ".pickle": "Pickle",
}


def supported_extensions() -> list[str]:
    return sorted(_SUPPORTED_EXTENSIONS)


SUPPORTED_EXTENSIONS = set(_SUPPORTED_EXTENSIONS)


def read_dataset(path: str | Path) -> "polars.DataFrame":
    """Read a dataset file into a polars DataFrame.

    The file format is detected from the file extension.

    Parameters
    ----------
    path : str or Path
        Path to the dataset file.

    Returns
    -------
    polars.DataFrame
        The dataset contents as a DataFrame.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file extension is not supported.
    """
    import polars as pl

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    ext = p.suffix.lower()
    if ext not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{ext}'. "
            f"Supported formats: {', '.join(_SUPPORTED_EXTENSIONS)}"
        )

    if ext in (".csv",):
        return pl.read_csv(p)
    elif ext in (".tsv", ".tab"):
        return pl.read_csv(p, separator="\t")
    elif ext in (".parquet", ".pq"):
        return pl.read_parquet(p)
    elif ext in (".json",):
        return pl.read_json(p)
    elif ext in (".jsonl", ".ndjson"):
        return pl.read_ndjson(p)
    elif ext in (".xlsx", ".xls"):
        return pl.read_excel(p)
    elif ext in (".feather", ".arrow"):
        return pl.read_ipc(p)
    elif ext in (".h5", ".hdf5"):
        return _read_hdf5(p)
    elif ext in (".npy",):
        return _read_npy(p)
    elif ext in (".npz",):
        return _read_npz(p)
    elif ext in (".pkl", ".pickle"):
        return _read_pickle(p)
    else:
        raise ValueError(f"Unsupported file extension '{ext}'")


def _read_hdf5(path: Path) -> "polars.DataFrame":
    import h5py
    import numpy as np

    data: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            data[key] = f[key][:]

    import polars as pl
    return pl.DataFrame(data)


def _read_npy(path: Path) -> "polars.DataFrame":
    import numpy as np
    import polars as pl

    arr = np.load(str(path))
    if arr.ndim == 1:
        return pl.DataFrame({"column_0": arr})
    columns = [f"column_{i}" for i in range(arr.shape[1])]
    return pl.DataFrame(arr, schema=columns)


def _read_npz(path: Path) -> "polars.DataFrame":
    import numpy as np
    import polars as pl

    data = np.load(str(path))
    if len(data.files) == 0:
        return pl.DataFrame()
    first = data.files[0]
    arr = data[first]
    if arr.ndim <= 1:
        return pl.DataFrame({first: arr})
    columns = [f"{first}_{i}" for i in range(arr.shape[1])]
    return pl.DataFrame(arr, schema=columns)


def _read_pickle(path: Path) -> "polars.DataFrame":
    import pandas as pd
    import polars as pl

    pdf = pd.read_pickle(str(path))
    return pl.from_pandas(pdf)

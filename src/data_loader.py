"""
data_loader.py
--------------
Reusable functions to load the Amazon ML Entity Resolution Challenge datasets.
Uses pathlib for robust path handling and pandas for tab-separated reading.
"""

from pathlib import Path
from typing import Dict, Optional, Union
import pandas as pd


def get_dataset_dir(base_dir: Optional[Union[str, Path]] = None) -> Path:
    """
    Locates the dataset directory automatically or from a given base_dir.
    Checks candidate locations:
    1. base_dir / dataset (if base_dir supplied)
    2. ./dataset
    3. ./student_resource/dataset
    4. ../student_resource/dataset
    """
    if base_dir is not None:
        path = Path(base_dir)
        if (path / "dataset").exists():
            return path / "dataset"
        if path.name == "dataset" and path.exists():
            return path
        if path.exists():
            return path

    cwd = Path.cwd()
    candidates = [
        cwd / "dataset",
        cwd / "student_resource" / "dataset",
        cwd.parent / "student_resource" / "dataset",
        Path(__file__).resolve().parent.parent / "student_resource" / "dataset",
        Path(__file__).resolve().parent.parent / "dataset",
    ]

    for candidate in candidates:
        if candidate.exists() and (candidate / "train").exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Could not locate 'dataset' directory in any expected location: {[str(c) for c in candidates]}"
    )


def load_tsv(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Loads a tab-separated (.tsv) file into a pandas DataFrame cleanly.
    Enforces sep='\\t', dtype=str, and keep_default_na=False to avoid
    unintended type casting or turning empty strings into NaNs.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path}")

    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )


def load_train_source1(data_dir: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Loads dataset/train/train_source1.tsv"""
    ds_dir = get_dataset_dir(data_dir)
    return load_tsv(ds_dir / "train" / "train_source1.tsv")


def load_train_source2(data_dir: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Loads dataset/train/train_source2.tsv"""
    ds_dir = get_dataset_dir(data_dir)
    return load_tsv(ds_dir / "train" / "train_source2.tsv")


def load_train_source3(data_dir: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Loads dataset/train/train_source3.tsv"""
    ds_dir = get_dataset_dir(data_dir)
    return load_tsv(ds_dir / "train" / "train_source3.tsv")


def load_train_ground_truth(data_dir: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Loads dataset/train/train_ground_truth.tsv"""
    ds_dir = get_dataset_dir(data_dir)
    return load_tsv(ds_dir / "train" / "train_ground_truth.tsv")


def load_test_source1(data_dir: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Loads dataset/test/test_source1.tsv"""
    ds_dir = get_dataset_dir(data_dir)
    return load_tsv(ds_dir / "test" / "test_source1.tsv")


def load_test_source2(data_dir: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Loads dataset/test/test_source2.tsv"""
    ds_dir = get_dataset_dir(data_dir)
    return load_tsv(ds_dir / "test" / "test_source2.tsv")


def load_test_source3(data_dir: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Loads dataset/test/test_source3.tsv"""
    ds_dir = get_dataset_dir(data_dir)
    return load_tsv(ds_dir / "test" / "test_source3.tsv")


def load_all_datasets(data_dir: Optional[Union[str, Path]] = None) -> Dict[str, pd.DataFrame]:
    """
    Loads all 7 train and test datasets into a dictionary.
    Keys:
      - train_s1, train_s2, train_s3, train_gt
      - test_s1, test_s2, test_s3
    """
    ds_dir = get_dataset_dir(data_dir)
    return {
        "train_s1": load_train_source1(ds_dir),
        "train_s2": load_train_source2(ds_dir),
        "train_s3": load_train_source3(ds_dir),
        "train_gt": load_train_ground_truth(ds_dir),
        "test_s1": load_test_source1(ds_dir),
        "test_s2": load_test_source2(ds_dir),
        "test_s3": load_test_source3(ds_dir),
    }

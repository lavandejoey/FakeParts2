"""
Dataset structure
Relative paths has 2 patterns, SUBSET in {fake_frames, fake_videos, real_frames, real_videos}:

```text
<TASK>/<METHOD>/<SUBSET>/
<TASK>/<METHOD>/<S_METHOD>/<SUBSET>/
```

After that, the files are organized as:

```text
<v_name>/frame_<%06d>.jpg
<v_name>.mp4
```
"""
from __future__ import annotations

import cv2
import numpy as np
import os
import pandas as pd
import torch
from PIL import Image, ImageFile
from dataclasses import dataclass, asdict
from enum import Enum
from io import BytesIO
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Mapping, Sequence, Tuple
from typing import Optional, Union, Dict, Any, List
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

ImageFile.LOAD_TRUNCATED_IMAGES = True

# VIDEOS_DIR = Path("/home/zliu/FakeParts2/datasets/FakeParts_V2")
# FRAMES_ROOT = Path("/home/zliu/FakeParts2/datasets/FakeParts_V2_Frame")
VIDEOS_DIR = Path("/projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_videos_only")
FRAMES_ROOT = Path("/projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_frames_only")
VIDEOS_CSV = Path("/projects/hi-paris/DeepFakeDataset/videos_index.csv")
FRAMES_CSV = Path("/projects/hi-paris/DeepFakeDataset/frames_index.csv")
VID_EXTS: Tuple[str, ...] = (".mp4", ".avi", ".mkv", ".mov")
IMG_EXTS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
EXCLUDE = ["Annotations"]  # keywords for folder / file to be excluded during indexing


class Subset(str, Enum):
    FAKE_VIDEOS = "fake_videos"
    REAL_VIDEOS = "real_videos"
    FAKE_FRAMES = "fake_frames"
    REAL_FRAMES = "real_frames"


@dataclass(frozen=True)
class IndexEntry:
    root: Path  # Root path
    rel_path: Path  # Path to file
    task: str  # task name, metadata
    method: str  # method name, metadata
    subset: Subset  # help defining the label
    name: str  # video name
    label: int  # 0=real, 1=fake for gt value
    mode: str  # 'video' or 'frame' data file type

    def __getitem__(self, item: str) -> object:
        return getattr(self, item)


ENTRY_COL = ["rel_path", "task", "method", "subset", "name", "label", "mode"]


def collate_skip_none(batch: List[Optional[Tuple[torch.Tensor, int, Dict[str, Any]]]]):
    """Filter out None samples produced by the dataset (e.g., corrupt files)."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None  # signal to skip this batch
    imgs, labels, metas = zip(*batch)  # type: ignore[misc]
    imgs = torch.stack(imgs, dim=0)
    labels = torch.as_tensor(labels, dtype=torch.long)
    merged_meta = {k: [m[k] for m in metas] for k in metas[0].keys()}
    return imgs, labels, merged_meta


class FakePartsV2DatasetBase(Dataset):
    def __init__(self,
                 data_root: Union[str, Path] = FRAMES_ROOT,
                 mode: str = "frame",  # "frame" or "video"
                 csv_path: Optional[Union[str, Path]] = FRAMES_CSV,
                 done_csv_list: Optional[List[Union[str, Path]]] = None,
                 model_name: str = "unknown_model",
                 transform=None,
                 on_corrupt: str = "raise",  # "raise" | "warn" | "skip"
                 ):
        super().__init__()

        if mode not in ("frame", "video"):
            raise ValueError(f"mode must be 'frame' or 'video', got: {mode}")

        self.model_name = model_name
        self.data_root = Path(data_root)
        self.mode = mode
        self.exts = IMG_EXTS if mode == "frame" else VID_EXTS
        self.transform = transform
        self.on_corrupt = on_corrupt

        # Build/ingest index once
        df = index_dataframe(self.data_root, file_exts=self.exts, csv_path=csv_path)

        # Remove done files if provided
        # safe use csv -> df
        safe_done_csv = []
        for done_csv in done_csv_list:
            # if dir, scan for csv files inside
            if os.path.isdir(done_csv):
                # walk dir for csv files
                for dirpath, _, filenames in os.walk(done_csv):
                    for fn in filenames:
                        if fn.lower().endswith(".csv"):
                            safe_done_csv.append(os.path.join(dirpath, fn))
            elif os.path.isfile(done_csv) and done_csv.lower().endswith(".csv"):
                safe_done_csv.append(done_csv)

        if len(safe_done_csv) > 0:
            done_rel_paths = set()
            for safe_done_csv in done_csv_list:
                if os.path.exists(safe_done_csv):
                    done_df = pd.read_csv(safe_done_csv, usecols=["sample_id"])
                    done_rel_paths.update(done_df["sample_id"].astype(str).tolist())
            if done_rel_paths:
                initial_count = len(df)
                df = df[~df["rel_path"].isin(done_rel_paths)]
                log.info(f"Filtered {initial_count - len(df)} done entries from index using {done_csv_list}.")

        # Keep only required cols & filter by mode once
        df = df[ENTRY_COL]
        df = df[df["mode"] == mode]
        if df.empty:
            raise ValueError(f"No data found for mode={mode} under {data_root} with extensions {self.exts}.")

        # Precompute lightweight arrays/lists to avoid df.iloc in __getitem__
        self._rel_paths: List[str] = df["rel_path"].tolist()
        self._abs_paths: List[Path] = [self.data_root / p for p in self._rel_paths]
        # int32/64 for labels to be collate-friendly
        self._labels: np.ndarray = df["label"].to_numpy(dtype=np.int64, copy=False)
        self._tasks: List[str] = df["task"].tolist()
        self._methods: List[str] = df["method"].tolist()
        self._subsets: List[str] = df["subset"].tolist()
        self._modes: List[str] = df["mode"].tolist()

        # (Optional) free pandas memory early
        del df

    def __len__(self) -> int:
        return len(self._rel_paths)

    def _make_meta(self, idx: int, label: int) -> Dict[str, Any]:
        # minimal metadata needed to populate REQUIRED_COLS
        #     "sample_id",  # unique id per row (string or int) you use to join with index
        #     "task",  # from dataset
        #     "method",  # from dataset
        #     "subset",  # real_videos/fake_videos/... from dataset
        #     "label",  # 0=real, 1=fake  (ground truth)
        #     "model",  # model name or identifier
        #     "mode",  # 'video' or 'frame'
        #     "score",  # real-valued score, higher => more likely fake, -1 indicating unavailable
        #     "pred",  # hard prediction in {0,1} produced by the model
        return {
            "sample_id": self._rel_paths[idx],
            "task": self._tasks[idx],
            "method": self._methods[idx],
            "subset": self._subsets[idx],
            "label": label,  # 0=real, 1=fake
            "model": self.model_name,
            "mode": self._modes[idx],  # 'frame' or 'video'
            "score": None,
            "pred": None,
        }

    def __getitem__(self, idx: int):
        # Support negative indices like Python sequences
        if idx < 0:
            idx += len(self)
        abs_path = self._abs_paths[idx]
        label = int(self._labels[idx])

        if self.mode == "frame":
            try:
                # Ensure file handle is closed even if transform reads lazily
                with open(abs_path, 'rb') as f:
                    b = f.read()
                with Image.open(BytesIO(b)) as im:
                    im = im.convert("RGB")
                    im.load()  # materialise, so file can close before transform
                image = self.transform(im) if self.transform is not None else im
                meta = self._make_meta(idx, label)
                return image, label, meta
            except Exception as e:
                if self.on_corrupt == "raise":
                    raise
                elif self.on_corrupt == "warn":
                    print(f"[warn] Failed to load image: {abs_path} ({e})")
                # "skip" and "warn" both try to return a sentinel; DataLoader can filter None in custom collate_fn
                return None
        else:  # video
            # Keep behaviour: return a handle (consider decoding to tensors with PyAV/decord for batching)
            cap = cv2.VideoCapture(str(abs_path))
            if not cap.isOpened():
                if self.on_corrupt == "raise":
                    raise RuntimeError(f"Failed to open video: {abs_path}")
                elif self.on_corrupt == "warn":
                    print(f"[warn] Failed to open video: {abs_path}")
                return None
            meta = self._make_meta(idx, label)
            return cap, label, meta

    def __repr__(self) -> str:
        n = len(self)
        return (f"{self.__class__.__name__}(n={n}, mode='{self.mode}', "
                f"root='{self.data_root}', model='{self.model_name}')")


def data_parse(file_path: Path, root: Path) -> IndexEntry:
    """
    Parse a file path into IndexEntry components.
    Expected structures:
      <root>/<Task>/<Method>/{real_videos|fake_videos|real_frames|fake_frames}/file
      <root>/<Task>/<Method>/<Method2>/{real_videos|fake_videos|real_frames|fake_frames}/file
      <root>/{0_real|1_fake}/<Task>/<Method>/{real_videos|fake_videos|real_frames|fake_frames}/file
      <root>/{0_real|1_fake}/<Task>/<Method>/<Method2>/{real_videos|fake_videos|real_frames|fake_frames}/file
    - file could be <v_name>.mp4 or <v_name>/frame_%06d.jpg
    - 0_real/1_fake is an optional flag that can be used for quick filtering,
    """
    parts = file_path.relative_to(root).parts
    if len(parts) < 4:        raise ValueError(f"Unexpected path structure: {file_path} (relative: {parts})")

    # Remove the filename; only directories remain
    dir_parts = list(parts[:-1])
    subset_values = {s.value for s in Subset}

    # 1) Find the subset by scanning from right to left (to allow extra dirs after subset, e.g., video_name/)
    subset_idx = -1
    for i in range(len(dir_parts) - 1, -1, -1):
        if dir_parts[i] in subset_values:
            subset_idx = i
            break
    if subset_idx == -1:
        raise ValueError(
            f"Could not locate subset in path: {file_path}. "
            f"Expected one of {sorted(subset_values)}."
        )

    subset = dir_parts[subset_idx]

    # 2) Everything before the subset contains: [optional flag]/Task/Method[/Method2/...]
    head = dir_parts[:subset_idx]

    # Optional leading flag
    flag = None
    if head and head[0] in {"0_real", "1_fake"}:
        flag = head.pop(0)

    # We now require at least <Task>/<Method>
    if len(head) < 2:
        raise ValueError(
            f"Cannot parse <Task>/<Method> from: {dir_parts}. "
            f"Got head={head}, subset={subset}."
        )

    task = head[0]
    method_parts = head[1:]
    method = "/".join(method_parts)

    # 3) Label & mode
    label_from_subset = 0 if subset.startswith("real_") else 1
    if flag is not None:
        label_from_flag = 0 if flag == "0_real" else 1
        if label_from_flag != label_from_subset:
            print(f"[warn] flag {flag} != subset {subset}; using subset-derived label.")

    mode = "video" if "videos" in subset else ("frame" if "frames" in subset else "unknown")

    return IndexEntry(
        root=Path(root),
        rel_path=file_path.relative_to(root),
        task=str(task),
        method=str(method),
        subset=Subset(subset),
        label=label_from_subset,
        mode=mode,
    )


def index_list(root_path: Path, file_exts: Tuple[str, ...]) -> List[IndexEntry]:
    entries: List[IndexEntry] = []
    root_path = Path(root_path)
    for dirpath, _, filenames in tqdm(os.walk(root_path), desc=f"Indexing {root_path}"):
        if any(excl in dirpath for excl in EXCLUDE):  # skip excluded folders
            continue
        dirpath = Path(dirpath)
        for fn in filenames:
            if fn.lower().endswith(file_exts):
                p = dirpath / fn
                try:
                    entries.append(data_parse(p, root_path))
                except Exception as e:
                    tqdm.write(f"[warn] Skipping file {p}: {e}")
                    continue
    return entries


def index_dataframe(root_path: Path,
                    file_exts: Tuple[str, ...] = IMG_EXTS,
                    csv_path: str | Path = FRAMES_CSV,
                    ) -> pd.DataFrame:
    """
    Build a DataFrame with one row per media file (video or frame).
    Columns: ['task','method','subset','label','mode','rel_path','abs_path','root']
    """
    # Check csv first, if provided -> load and return
    if csv_path is not None and os.path.exists(csv_path):
        read_kwargs = dict(usecols=ENTRY_COL, memory_map=True)
        try:
            # dtype_backend="pyarrow" keeps columns as Arrow types (less memory, faster ops)
            df = pd.read_csv(csv_path, engine="pyarrow", dtype_backend="pyarrow", **read_kwargs)
        except Exception:
            df = pd.read_csv(csv_path, **read_kwargs)

        # Vectorised path build (no lambda per row)
        root_str = os.fspath(Path(root_path))
        df["abs_path"] = root_str + os.sep + df["rel_path"].astype(str)
        return df

    root_path = Path(root_path)
    all_entries: List[IndexEntry] = []
    all_entries.extend(index_list(root_path, file_exts=file_exts))

    if not all_entries:
        return pd.DataFrame(columns=[
            "task", "method", "subset", "label", "mode", "rel_path", "abs_path", "root"
        ])

    df = pd.DataFrame([asdict(e) for e in all_entries])
    df["abs_path"] = df["rel_path"].map(lambda p: str(root_path / p))
    # Cast types
    df["task"] = df["task"].astype("string")
    df["method"] = df["method"].astype("string")
    df["subset"] = df["subset"].astype("string")
    df["mode"] = df["mode"].astype("string")
    df["label"] = df["label"].astype(np.int32)
    df["root"] = df["root"].astype("string")
    df["rel_path"] = df["rel_path"].astype("string")
    df["abs_path"] = df["abs_path"].astype("string")
    return df


# ==================== Fixed column schema for model outputs ====================

# Required columns every model output should provide.
REQUIRED_COLS: Tuple[str, ...] = (
    "sample_id",  # unique id per row (string or int) you use to join with index
    "task",  # from dataset
    "method",  # from dataset
    "subset",  # real_videos/fake_videos/... from dataset
    "label",  # 0=real, 1=fake  (ground truth)
    "model",  # model name or identifier
    "mode",  # 'video' or 'frame'
    "score",  # real-valued score, higher => more likely fake, -1 indicating unavailable
    "pred",  # hard prediction in {0,1} produced by the model
)


def standardise_predictions(
        df_like: Union[pd.DataFrame, Sequence[Mapping[str, object]]],
        column_map: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    if not isinstance(df_like, pd.DataFrame):
        df = pd.DataFrame(df_like)
    else:
        df = df_like.copy()

    if column_map:
        df = df.rename(columns=dict(column_map))

    # Check required
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Provided columns: {list(df.columns)}")

    # Minimal dtype hygiene (no value transforms)
    # Leave 'pred' and 'label' as ints; 'score' as float.
    df["label"] = df["label"].astype(int)
    df["pred"] = df["pred"].astype(int)
    df["score"] = df["score"].astype(float)

    # Ensure string-ish cols are strings
    for c in ("sample_id", "task", "method", "subset", "model", "mode"):
        df[c] = df[c].astype("string")

    return df


# ========================= Metrics (overall & grouped) =========================

def find_best_threshold(y_true: Sequence[int], y_pred: Sequence[float]) -> float:
    """
    Robust threshold search:
    - No assumption on ordering or class balance.
    - If only one class is present, return 0.5 as a safe default.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(float)

    # single-class set: no meaningful ROC; return neutral threshold
    if len(np.unique(y_true)) < 2:
        return 0.5

    # Youden's J: argmax(tpr - fpr)
    fpr, tpr, thr = roc_curve(y_true, y_pred)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def _ensure_binary_labels(df: pd.DataFrame) -> None:
    uniq = set(pd.unique(df["label"]))
    if not uniq.issubset({0, 1}):
        raise ValueError(f"`label` must be binary in {{0,1}}, got: {uniq}")


def overall_accuracy(df: pd.DataFrame) -> float:
    _ensure_binary_labels(df)
    return float(accuracy_score(df["label"].to_numpy(), df["pred"].to_numpy()))


def accuracy_by(df: pd.DataFrame, by: Union[str, Sequence[str]]) -> pd.DataFrame:
    _ensure_binary_labels(df)
    if isinstance(by, str):
        by = [by]
    g = df.groupby(list(by), dropna=False)
    acc = g.apply(lambda x: accuracy_score(x["label"].to_numpy(), x["pred"].to_numpy()))
    return acc.reset_index(name="accuracy")


def roc_auc_overall(df: pd.DataFrame) -> Optional[float]:
    _ensure_binary_labels(df)
    y_true = df["label"].to_numpy()
    y_score = df["score"].to_numpy()
    # Need both classes present for AUC
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def roc_auc_by(df: pd.DataFrame, by: Union[str, Sequence[str]]) -> pd.DataFrame:
    _ensure_binary_labels(df)
    if isinstance(by, str):
        by = [by]
    rows = []
    for keys, x in df.groupby(list(by), dropna=False):
        y_true = x["label"].to_numpy()
        y_score = x["score"].to_numpy()
        if len(np.unique(y_true)) < 2:
            auc = None
        else:
            auc = float(roc_auc_score(y_true, y_score))
        key_vals = keys if isinstance(keys, tuple) else (keys,)
        rows.append(tuple(key_vals) + (auc,))
    cols = list(by) + ["roc_auc"]
    return pd.DataFrame(rows, columns=cols)


def tpr_at_fpr(df: pd.DataFrame, target_fpr: float = 1e-2) -> Optional[float]:
    """
    Compute TPR at the smallest threshold where FPR <= target_fpr.
    Returns None if ROC cannot be computed.
    """
    _ensure_binary_labels(df)
    y_true = df["label"].to_numpy()
    y_score = df["score"].to_numpy()
    if len(np.unique(y_true)) < 2:
        return None

    fpr, tpr, thresh = roc_curve(y_true, y_score, pos_label=1)
    # Find all positions with FPR <= target; pick the *max* TPR among them
    mask = fpr <= target_fpr
    if not np.any(mask):
        # No point on ROC meets the target FPR; return best-effort (closest over)
        idx = int(np.argmin(np.abs(fpr - target_fpr)))
        return float(tpr[idx])
    return float(np.max(tpr[mask]))


def tpr_at_fpr_by(df: pd.DataFrame, by: Union[str, Sequence[str]], target_fpr: float = 1e-2) -> pd.DataFrame:
    _ensure_binary_labels(df)
    if isinstance(by, str):
        by = [by]
    rows = []
    for keys, x in df.groupby(list(by), dropna=False):
        val = tpr_at_fpr(x, target_fpr=target_fpr)
        key_vals = keys if isinstance(keys, tuple) else (keys,)
        rows.append(tuple(key_vals) + (val,))
    cols = list(by) + [f"tpr@fpr={target_fpr:g}"]
    return pd.DataFrame(rows, columns=cols)


def _balance_by_group(df: pd.DataFrame, by: Union[str, Sequence[str]],
                      random_state: Optional[int] = None) -> pd.DataFrame:
    """
    For each group in `by`, undersample the real class (label==0) to match the number of fakes (label==1).
    If a group has no fakes, the group is returned unchanged (metrics for fake-only will be None).
    """
    if isinstance(by, str):
        by = [by]
    pieces = []
    for keys, g in df.groupby(list(by), dropna=False):
        n_fake = int((g["label"] == 1).sum())
        if n_fake == 0:
            pieces.append(g)  # nothing to balance here
            continue
        fake_part = g[g["label"] == 1]
        real_part = g[g["label"] == 0]
        if len(real_part) <= n_fake:
            # already <= balanced
            pieces.append(g)
        else:
            real_sampled = real_part.sample(n=n_fake, random_state=random_state, replace=False)
            pieces.append(pd.concat([fake_part, real_sampled], axis=0, ignore_index=True))
    if not pieces:
        return df.iloc[0:0].copy()
    return pd.concat(pieces, axis=0, ignore_index=True)


def accuracy_by_fake(df: pd.DataFrame, by: Union[str, Sequence[str]]) -> pd.DataFrame:
    """
    Accuracy computed **only on the fake class** per group (equivalent to recall/TPR for label==1).
    """
    if isinstance(by, str):
        by = [by]
    _ensure_binary_labels(df)
    rows = []
    for keys, x in df.groupby(list(by), dropna=False):
        x_fake = x[x["label"] == 1]
        if len(x_fake) == 0:
            acc = None
        else:
            acc = float(accuracy_score(x_fake["label"].to_numpy(), x_fake["pred"].to_numpy()))
        key_vals = keys if isinstance(keys, tuple) else (keys,)
        rows.append(tuple(key_vals) + (acc,))
    cols = list(by) + ["accuracy_fake_only"]
    return pd.DataFrame(rows, columns=cols)


# ============================== Convenience report ==============================
def quick_report(df: pd.DataFrame, *, balance_for_groups: bool = True, group_fake_only: bool = True,
                 random_state: Optional[int] = None) -> Mapping[str, object]:
    """
    Produce a small dictionary of core numbers. Suitable for logging.

    DF input need cols: ['sample_id', 'task', 'method', 'subset', 'label', 'model', 'mode', 'score', 'pred']
    "sample_id",  # unique id per row (string or int) you use to join with index
    "task",  # from dataset
    "method",  # from dataset
    "subset",  # real_videos/fake_videos/... from dataset
    "label",  # 0=real, 1=fake  (ground truth)
    "model",  # model name or identifier
    "mode",  # 'video' or 'frame'
    "score",  # real-valued score, higher => more likely fake
    "pred",  # hard prediction in {0,1} produced by the model

    Overall Accuracy
    Accuracy by Task
    Accuracy by Methods
    ROC-AUC by Task
    ROC-AUC by Methods
    TPR@FPR=1e-2 by Task
    TPR@FPR=1e-2 by Methods
    """
    # Prepare grouped DataFrames
    df_task = df
    df_method = df
    if balance_for_groups:
        df_task = _balance_by_group(df, by="task", random_state=random_state)
        df_method = _balance_by_group(df, by="method", random_state=random_state)
    # Grouped accuracy: fake-only (recall) if requested; else legacy accuracy
    acc_task = accuracy_by_fake(df_task, "task") if group_fake_only else accuracy_by(df_task, "task")
    acc_method = accuracy_by_fake(df_method, "method") if group_fake_only else accuracy_by(df_method, "method")
    # Grouped ROC-AUC / TPR: computed on the (optionally) balanced sets including both classes
    roc_task = roc_auc_by(df_task, "task")
    roc_method = roc_auc_by(df_method, "method")
    tpr_task = tpr_at_fpr_by(df_task, "task", target_fpr=1e-2)
    tpr_method = tpr_at_fpr_by(df_method, "method", target_fpr=1e-2)
    return {
        "overall": {
            "accuracy": overall_accuracy(df),
            "roc_auc": roc_auc_overall(df),
            "tpr@1e-2": tpr_at_fpr(df, target_fpr=1e-2),
        },
        "by_task": {
            "accuracy": acc_task,
            "roc_auc": roc_task,
            "tpr@1e-2": tpr_task,
        },
        "by_method": {
            "accuracy": acc_method,
            "roc_auc": roc_method,
            "tpr@1e-2": tpr_method,
        },
    }


__all__ = [
    # indexing
    "Subset", "IndexEntry", "data_parse", "index_list", "index_dataframe",
    # schema helpers
    "REQUIRED_COLS", "standardise_predictions",
    # metrics
    "overall_accuracy", "accuracy_by", "roc_auc_overall", "roc_auc_by",
    "tpr_at_fpr", "tpr_at_fpr_by", "quick_report",
]

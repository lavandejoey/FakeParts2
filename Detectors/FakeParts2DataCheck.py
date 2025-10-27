"""
@File: FakeParts2DataCheck.py

Purpose:
  - Index videos/frames
  - Summarise coverage
  - Extract frames for videos that have no corresponding frame folders
    (with special-case splitting rules)

Subsets:
  - fake_videos / real_videos  -> source videos
  - fake_frames / real_frames  -> extracted frames

Path patterns (both supported):
  - <Task>/<Method>/{fake_videos|real_videos}
  - <Task>/<Method>/<Method>/{fake_videos|real_videos}

Special cases (applied on extraction for fake_videos):
  - Extrapolation/<Any>/fake_videos:   first 40 frames -> real_frames, rest -> fake_frames
  - Interpolation/<Any>/fake_videos:   16 real + 14 fake + 16 real (by order)
  - Inpainting/ROVI/Annotations:       skip entirely

Outputs:
  - CSV: videos_index.csv (all videos)
  - CSV: frames_index.csv (all frames)
  - Console: concise report + extraction summary

Notes:
  - Uses OpenCV for decoding (no external ffmpeg dependency). If OpenCV is not available,
    the extractor will be skipped gracefully.
"""
from __future__ import annotations

import sys

import enum
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from typing import Iterable, List, Optional, Tuple

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

# Config
# VIDEOS_ROOT = Path("/home/zliu/FakeParts2/datasets/FakeParts_V2")
# FRAMES_ROOT = Path("/home/zliu/FakeParts2/datasets/FakeParts_V2_Frame")
# VIDEOS_CSV = Path("/home/zliu/FakeParts2/datasets/videos_index.csv")
# FRAMES_CSV = Path("/home/zliu/FakeParts2/datasets/frames_index.csv")
VIDEOS_ROOT = Path("/projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_videos_only")
FRAMES_ROOT = Path("/projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_frames_only")
VIDEOS_CSV = Path("/projects/hi-paris/FakeParts2Tests/Detectors/videos_index.csv")
FRAMES_CSV = Path("/projects/hi-paris/FakeParts2Tests/Detectors/frames_index.csv")

VID_EXTS: Tuple[str, ...] = (".mp4", ".avi", ".mkv", ".mov")
IMG_EXTS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
SKIP_FOLDER_KEYWORDS = ["Annotations", "annotations", "README", "readme"]

# Behaviour flags
DRY_RUN = False
OVERWRITE_EXISTING = False


class Subset(str, enum.Enum):
    FAKE_VIDEOS = "fake_videos"
    REAL_VIDEOS = "real_videos"
    FAKE_FRAMES = "fake_frames"
    REAL_FRAMES = "real_frames"


@dataclass(frozen=True)
class IndexEntry:
    root: Path
    rel_path: Path
    task: str
    method: str
    subset: Subset
    name: str
    label: int  # 0=real, 1=fake
    mode: str  # 'video' or 'frame'

    def as_dict(self) -> dict:
        return {"root": str(self.root), "rel_path": str(self.rel_path), "task": self.task, "method": self.method,
                "subset": self.subset.value, "name": self.name, "label": self.label, "mode": self.mode, }


def _is_subset_name(part: str) -> bool:
    return part in {s.value for s in Subset}


def relative_path_parser(rel_path: Path) -> Tuple[str, str, str]:
    parts = rel_path.parts
    if len(parts) < 2:
        raise ValueError(f"Unexpected path structure: {rel_path}")

    subset_idx: Optional[int] = None
    for i, p in enumerate(parts):
        if _is_subset_name(p):
            subset_idx = i
            break
    if subset_idx is None:
        raise ValueError(f"No subset folder found in: {rel_path}")

    subset = parts[subset_idx]
    task = parts[0]
    core = parts[1:subset_idx]
    if not core:
        raise ValueError(f"No method segment found in: {rel_path}")
    method = "-".join(core)  # normalise multi-segment method path
    return task, method, subset


def _label_from_subset(subset: Subset) -> int:
    return 1 if subset in {Subset.FAKE_VIDEOS, Subset.FAKE_FRAMES} else 0


def _iter_files(root: Path, exts: Tuple[str, ...]) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_dir():
            if any(k in p.parts for k in SKIP_FOLDER_KEYWORDS):
                continue
            continue
        if p.suffix.lower() in exts:
            if any(k in p.parts for k in SKIP_FOLDER_KEYWORDS):
                continue
            yield p


def _to_index_entry(root: Path, file_path: Path, mode: str) -> IndexEntry:
    rel = file_path.relative_to(root)
    task, method, subset = relative_path_parser(rel)
    subset_enum = Subset(subset)
    name = file_path.parent.name if mode == "frame" else file_path.stem
    return IndexEntry(root=root, rel_path=rel, task=task, method=method, subset=subset_enum, name=name,
                      label=_label_from_subset(subset_enum), mode=mode, )


# Indexers
def build_videos_index() -> pd.DataFrame:
    rows: List[dict] = []
    for f in _iter_files(VIDEOS_ROOT, VID_EXTS):
        if any(k in f.parts for k in SKIP_FOLDER_KEYWORDS):
            continue
        rel = f.relative_to(VIDEOS_ROOT)
        if any(k in rel.parts for k in SKIP_FOLDER_KEYWORDS):
            continue
        try:
            entry = _to_index_entry(VIDEOS_ROOT, f, "video")
        except Exception as e:
            print(f"Skip video: {f} | {e}", file=sys.stderr)
            continue
        rows.append(entry.as_dict())
    return pd.DataFrame(rows)


def build_frames_index() -> pd.DataFrame:
    rows: List[dict] = []
    for f in _iter_files(FRAMES_ROOT, IMG_EXTS):
        rel = f.relative_to(FRAMES_ROOT)
        if any(k in rel.parts for k in SKIP_FOLDER_KEYWORDS):
            continue
        try:
            entry = _to_index_entry(FRAMES_ROOT, f, "frame")
        except Exception as e:
            print(f"Skip frame: {f} | {e}", file=sys.stderr)
            continue
        rows.append(entry.as_dict())
    return pd.DataFrame(rows)


# Extraction helpers
def _ensure_dir(d: Path) -> None:
    if d.exists():
        return
    if not DRY_RUN:
        d.mkdir(parents=True, exist_ok=True)


def _count_images(dir_path: Path) -> int:
    return sum(1 for f in dir_path.glob("*") if f.is_file() and f.suffix.lower() in IMG_EXTS)


def _extract_with_cv2(video_path: Path, out_dir: Path, frame_selector: Optional[Tuple[int, int]] = None) -> int:
    """Extract frames to out_dir.
    Args:
        frame_selector: (start_inclusive, end_exclusive) in 0-based indices.
                        If None, export all frames.
    Returns:
        count of written frames
    """
    if cv2 is None:
        print("OpenCV not available: cannot extract", file=sys.stderr)
        return 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}", file=sys.stderr)
        return 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if frame_selector is None else frame_selector[1]
    start = 0 if frame_selector is None else max(0, frame_selector[0])
    end = total if frame_selector is None else max(start, frame_selector[1])

    idx = 0
    written = 0
    _ensure_dir(out_dir)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx >= end:
            break
        if idx >= start:
            if not DRY_RUN:
                out_path = out_dir / f"frame_{idx:06d}.jpg"
                cv2.imwrite(str(out_path), frame)
            written += 1
        idx += 1

    cap.release()
    return written


# Special-case splitting policies
def _split_policy(task: str, subset: Subset, TotalFrames: int) -> Tuple[Tuple[int, int], Tuple[int, int]] | None:
    """Return (real_range, fake_range) if a split is required for this (task, subset).
    Ranges are [start, end) in frame indices. Return None for the default policy.
    """
    if subset != Subset.FAKE_VIDEOS:
        return None

    if task == "Extrapolation":
        # first 40 -> real, rest -> fake
        real_end = min(40, max(0, TotalFrames))
        fake_start = real_end
        fake_end = TotalFrames
        if real_end == 0 or fake_end <= fake_start:
            return None
        return (0, real_end), (fake_start, fake_end)

    if task == "Interpolation":
        # 16 real + 14 fake + 16 real; if not enough frames, clip safely
        first_real = 16
        fake_mid = 14
        last_real = 16
        if TotalFrames < first_real + fake_mid:  # not enough, skip special split
            return None
        r1 = (0, min(first_real, TotalFrames))
        f = (r1[1], min(r1[1] + fake_mid, TotalFrames))
        r2 = (f[1], min(f[1] + last_real, TotalFrames))
        return r1, f  # We'll handle r2 by merging with real as (r1 union r2) in extraction call(s)
    return None


def _target_frame_dirs(rel_video: Path, video_stem: str, subset: Subset) -> Tuple[Path, Path]:
    """Return (real_dir, fake_dir) destination dirs for this video."""
    parts = list(rel_video.parts)
    subset_idx = parts.index(subset.value)
    frames_base = FRAMES_ROOT.joinpath(*parts[:subset_idx])
    real_dir = frames_base / Subset.REAL_FRAMES.value / video_stem
    fake_dir = frames_base / Subset.FAKE_FRAMES.value / video_stem
    return real_dir, fake_dir


def extract_missing_frames(videos_df: pd.DataFrame) -> pd.DataFrame:
    """Attempt extraction for any video lacking its corresponding frame folder/content.
    Policy:
      - real_videos -> extract all frames into real_frames/<video_stem>
      - fake_videos (default) -> extract all into fake_frames/<video_stem>
      - Special splits applied for (task, fake_videos) under Interpolation/Extrapolation:
            * Interpolation: 16 real + 14 fake + 16 real by order
            * Extrapolation: first 40 real, rest fake
    Returns a DataFrame summarising extraction actions.
    """
    actions = []
    for _, row in tqdm(videos_df.iterrows(), total=len(videos_df), desc="Extracting missing frames"):
        task: str = row["task"]
        method: str = row["method"]
        subset = Subset(row["subset"])
        rel_path = Path(row["rel_path"])  # relative to VIDEOS_ROOT
        video_path = VIDEOS_ROOT / rel_path
        video_stem = video_path.stem

        # Decide target dirs
        real_dir, fake_dir = _target_frame_dirs(rel_path, video_stem, subset)

        # Determine whether extraction is needed
        if subset == Subset.REAL_VIDEOS:
            need = OVERWRITE_EXISTING or (_count_images(real_dir) == 0)
        elif subset == Subset.FAKE_VIDEOS:
            # For split policies we require checks on both dirs; otherwise just fake_dir
            need = OVERWRITE_EXISTING or ((_count_images(fake_dir) == 0) and (_count_images(real_dir) == 0))
        else:
            continue  # shouldn't occur for videos

        if not need:
            actions.append({
                "task": task, "method": method, "subset": subset.value, "video": video_stem,
                "action": "skip_already_present",
            })
            continue

        if cv2 is None:
            actions.append({
                "task": task, "method": method, "subset": subset.value, "video": video_stem,
                "action": "cannot_extract_no_opencv",
            })
            continue

        cap = cv2.VideoCapture(str(video_path)) if cv2 is not None else None
        if cap is None or not cap.isOpened():
            actions.append({
                "task": task, "method": method, "subset": subset.value, "video": video_stem,
                "action": "cannot_open_video",
            })
            if cap is not None:
                cap.release()
            continue
        TotalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Apply policy
        if subset == Subset.REAL_VIDEOS:
            # all to real
            written = _extract_with_cv2(video_path, real_dir, None)
            actions.append({
                "task": task, "method": method, "subset": subset.value, "video": video_stem,
                "action": "extract_real_all", "written": written,
            })
            continue

        # fake_videos
        policy = _split_policy(task, subset, TotalFrames)
        if policy is None:
            # default: all to fake
            written = _extract_with_cv2(video_path, fake_dir, None)
            actions.append({
                "task": task, "method": method, "subset": subset.value, "video": video_stem,
                "action": "extract_fake_all", "written": written,
            })
        else:
            # Extrapolation: (real_range, fake_range)
            # Interpolation: returns (r1, f); we also compute r2 if space remains
            r_range, f_range = policy
            # For Interpolation, add trailing real block if any
            if task == "Interpolation":
                r2_start = f_range[1]
                r2_end = min(r2_start + 16, TotalFrames)
                # Merge both real ranges by extracting twice into the same real_dir
                w1 = _extract_with_cv2(video_path, real_dir, r_range)
                w2 = _extract_with_cv2(video_path, real_dir, (r2_start, r2_end))
                wf = _extract_with_cv2(video_path, fake_dir, f_range)
                actions.append({
                    "task": task, "method": method, "subset": subset.value, "video": video_stem,
                    "action": "extract_interpolation_split",
                    "written_real": w1 + w2, "written_fake": wf,
                })
            else:
                # Extrapolation or other future split
                wr = _extract_with_cv2(video_path, real_dir, r_range)
                wf = _extract_with_cv2(video_path, fake_dir, f_range)
                actions.append({
                    "task": task, "method": method, "subset": subset.value, "video": video_stem,
                    "action": "extract_split", "written_real": wr, "written_fake": wf,
                })

    return pd.DataFrame(actions)


# Reports
def differential_report(videos_df: pd.DataFrame, frames_df: pd.DataFrame) -> pd.DataFrame:
    # Normalize
    v = videos_df.copy()
    f = frames_df.copy()
    v["subset"] = v["subset"].astype(str)
    f["subset"] = f["subset"].astype(str)

    # Map each video subset to the frame subsets it should check
    mapping = {
        "real_videos": ["real_frames"],
        "fake_videos": ["fake_frames", "real_frames"],  # split policies write to both
    }

    # Aggregate frames by (task, method, name) across the mapped subsets
    f["is_frame"] = 1
    f_group = f.groupby(["task", "method", "subset", "name"], as_index=False)["is_frame"].sum()

    rows = []
    for _, r in v.drop_duplicates(subset=["task", "method", "subset", "name"]).iterrows():
        task, method, subset, name = r["task"], r["method"], r["subset"], r["name"]
        target_subsets = mapping.get(subset, ["fake_frames", "real_frames"])

        frames_count = (
            f_group[(f_group["task"] == task) &
                    (f_group["method"] == method) &
                    (f_group["name"] == name) &
                    (f_group["subset"].isin(target_subsets))]
            ["is_frame"].sum()
        )

        rows.append({
            "task": task,
            "method": method,
            "subset": subset,
            "name": name,
            "frames_count": int(frames_count),
            "has_frames": int(frames_count) > 0,
        })

    rep = pd.DataFrame(rows).sort_values(["task", "method", "subset", "name"]).reset_index(drop=True)
    return rep


# Method-level summary
def method_summary_table(videos_df: pd.DataFrame, frames_df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    Aggregate counts per (task, method):
      - RealVideos, FakeVideos
      - RealFrames, FakeFrames
      - TotalVideos, TotalFrames
      - AvgFramePerVideo (over VFrame)
    """
    v = videos_df.copy()
    f = frames_df.copy()
    v["subset"] = v["subset"].astype(str)
    f["subset"] = f["subset"].astype(str)

    # Video counts per (task, method)
    vv = (
        v.groupby(["task", "method", "subset"])["name"]
        .nunique()
        .reset_index(name="video_cnt")
    )
    real_videos = vv[vv["subset"] == "real_videos"].rename(columns={"video_cnt": "RealVideos"})[
        ["task", "method", "RealVideos"]]
    fake_videos = vv[vv["subset"] == "fake_videos"].rename(columns={"video_cnt": "FakeVideos"})[
        ["task", "method", "FakeVideos"]]

    # Frame counts per (task, method)
    ff = (
        f.groupby(["task", "method", "subset"])["rel_path"]
        .size()
        .reset_index(name="frame_cnt")
    )
    real_frames = ff[ff["subset"] == "real_frames"].rename(columns={"frame_cnt": "RealFrames"})[
        ["task", "method", "RealFrames"]]
    fake_frames = ff[ff["subset"] == "fake_frames"].rename(columns={"frame_cnt": "FakeFrames"})[
        ["task", "method", "FakeFrames"]]

    # Coverage
    f2 = f.copy()
    f2["has_frame"] = 1
    any_frames_by_name = (
        f2.groupby(["task", "method", "name"])["has_frame"].sum().reset_index()
    )
    any_frames_by_name["has_frame"] = any_frames_by_name["has_frame"] > 0

    videos_unique = v.drop_duplicates(subset=["task", "method", "name", "subset"])

    merged_cov = videos_unique.merge(
        any_frames_by_name, on=["task", "method", "name"], how="left"
    )
    merged_cov["has_frame"] = merged_cov["has_frame"].fillna(False)

    cov_by_tm = (
        merged_cov.groupby(["task", "method"])
        .agg(TotalVideos=("name", "nunique"),
             VFrame=("has_frame", "sum"))
        .reset_index()
    )

    # Merge all pieces
    out = (
        cov_by_tm
        .merge(real_videos, on=["task", "method"], how="left")
        .merge(fake_videos, on=["task", "method"], how="left")
        .merge(real_frames, on=["task", "method"], how="left")
        .merge(fake_frames, on=["task", "method"], how="left")
        .fillna(0)
        .sort_values(["task", "method"])
        .reset_index(drop=True)
    )

    out["RealVideos"] = out["RealVideos"].astype(int)
    out["FakeVideos"] = out["FakeVideos"].astype(int)
    out["RealFrames"] = out["RealFrames"].astype(int)
    out["FakeFrames"] = out["FakeFrames"].astype(int)
    out["TotalVideos"] = out["RealVideos"] + out["FakeVideos"]
    out["TotalFrames"] = out["RealFrames"] + out["FakeFrames"]
    # out["AvgFramePerVideo"] = (out["TotalFrames"] / out["VFrame"].replace({0: pd.NA})).round(1)
    denom = out["VFrame"].mask(out["VFrame"] == 0)
    out["AvgFramePerVideo"] = (out["TotalFrames"] / denom).round(1)

    cols = [
        "task", "method",
        "RealVideos", "FakeVideos", "TotalVideos",
        "RealFrames", "FakeFrames", "TotalFrames",
        "VFrame", "AvgFramePerVideo"
    ]
    for c in out.columns:
        if c not in cols:
            cols.append(c)
    total_denom = out["VFrame"].sum()
    if total_denom == 0:
        avg_total = float("nan")
    else:
        avg_total = out["TotalFrames"].sum() / total_denom
    # Calculate total row
    total_data = {
        "task": "ALL",
        "method": "ALL",
        "RealVideos": out["RealVideos"].sum(),
        "FakeVideos": out["FakeVideos"].sum(),
        "TotalVideos": out["TotalVideos"].sum(),
        "RealFrames": out["RealFrames"].sum(),
        "FakeFrames": out["FakeFrames"].sum(),
        "TotalFrames": out["TotalFrames"].sum(),
        "VFrame": total_denom,
        "AvgFramePerVideo": round(avg_total, 1) if total_denom != 0 else float("nan"),
    }
    total_df = pd.DataFrame([total_data])
    return out[cols], total_df


def print_method_summary_table(videos_df: pd.DataFrame, frames_df: pd.DataFrame,
                               save_dir: Optional[Path] = None) -> None:
    df_methods, df_total = method_summary_table(videos_df, frames_df)
    df = pd.concat([df_methods, df_total], ignore_index=True)
    import sys
    title = "Method-level summary"
    dash_len = 58
    print(f"\n{'=' * dash_len} {title} {'=' * dash_len}", file=sys.stderr)
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 200):
        print(df.to_string(index=False), file=sys.stderr)
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / "method_summary.csv"
        df.to_csv(path, index=False)
        print(f"\n  -> {path}", file=sys.stderr)


# Main
def main(argv: Optional[List[str]] = None) -> int:
    # 1) Index
    print("[1/6] Indexing videos ...", file=sys.stderr)
    videos_df = build_videos_index()
    print(f"  -> videos: {len(videos_df):,}", file=sys.stderr)

    print("[2/6] Indexing frames ...", file=sys.stderr)
    frames_df = build_frames_index()
    print(f"  -> frames: {len(frames_df):,}", file=sys.stderr)

    if videos_df.empty:
        print("No videos found: exiting.", file=sys.stderr)
        return 0
    if frames_df.empty:
        print("No frames found.", file=sys.stderr)

    # 2) Coverage & write CSVs
    print("[3/6] Writing CSVs ...", file=sys.stderr)
    VIDEOS_CSV.parent.mkdir(parents=True, exist_ok=True)
    videos_df.to_csv(VIDEOS_CSV, index=False)
    print(f"  -> {VIDEOS_CSV}", file=sys.stderr)
    if not frames_df.empty:
        frames_df.to_csv(FRAMES_CSV, index=False)
        print(f"  -> {FRAMES_CSV}", file=sys.stderr)
    else:
        print("  -> no frames to write", file=sys.stderr)

    print("[4/6] Coverage report ...", file=sys.stderr)
    if frames_df.empty:
        print("  -> all videos missing frames", file=sys.stderr)
        before_rep = videos_df.copy()
        before_rep["has_frames"] = False
        missing_before = before_rep
    else:
        before_rep = differential_report(videos_df, frames_df)
        missing_before = before_rep[~before_rep["has_frames"]]
    print(f"  -> videos without frames: {len(missing_before):,}", file=sys.stderr)

    # 3) Extraction for missing
    print("[5/6] Extracting missing frames ...", file=sys.stderr)
    extract_df = extract_missing_frames(videos_df if OVERWRITE_EXISTING else videos_df.merge(
        before_rep[~before_rep["has_frames"]][["task", "method", "subset", "name"]],
        on=["task", "method", "subset", "name"], how="inner"
    ))
    # Show action summary
    if not extract_df.empty:
        act_counts = extract_df["action"].value_counts().to_dict()
    else:
        act_counts = {}
    print(f"  -> actions: {act_counts}", file=sys.stderr)
    # collect "cannot_open_video" videos
    bug_video_list = [row["video"] for _, row in extract_df.iterrows() if row["action"] == "cannot_open_video"]
    if len(bug_video_list) > 0:
        print("Videos that cannot be opened:", file=sys.stderr)
        for bv in bug_video_list:
            print(f"  - {bv}", file=sys.stderr)

    # 4) Re-index frames after extraction
    print("[6/6] Re-index frames ...", file=sys.stderr)
    frames_after_df = build_frames_index()
    after_rep = differential_report(videos_df, frames_after_df)
    remaining = after_rep[~after_rep["has_frames"]]
    print(f"  -> remaining videos without frames: {len(remaining):,}", file=sys.stderr)
    VIDEOS_CSV.parent.mkdir(parents=True, exist_ok=True)
    videos_df.to_csv(VIDEOS_CSV, index=False)
    frames_after_df.to_csv(FRAMES_CSV, index=False)
    print(f"  -> {VIDEOS_CSV}", file=sys.stderr)
    print(f"  -> {FRAMES_CSV}", file=sys.stderr)

    # 5) Method-level summary
    print_method_summary_table(videos_df, frames_after_df, save_dir=VIDEOS_CSV.parent)

    # Minimal console summaries
    ok_rate = 1.0 - (len(remaining) / max(1, len(after_rep)))
    print(f"Extraction success ratio (by video key): {ok_rate:.3f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

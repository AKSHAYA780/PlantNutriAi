"""Discover leaf images, normalise class-folder names, and write a stratified split.

Expected layout: data/raw/<crop>/<class_folder>/*.(jpg|png|...).
Folder names are matched case-insensitively. Short symbols such as "K" or "Mg"
match only as a whole folder name. Anything unmapped is printed and skipped.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.common import (
    CLASS_ORDER,
    DUMMY_BANNER,
    IMAGE_EXTENSIONS,
    ensure_dir,
    imread_bgr,
    load_config,
    path_from_key,
    set_seed,
    write_data_source,
)


def _alias_at_boundary(compact: str, pattern: str) -> bool:
    """True when pattern occurs on underscore boundaries, not inside another token.

    "n_def" is an alias for nitrogen, but it must not match the tail of "iron_def".
    """
    if not pattern:
        return False
    start = 0
    while True:
        index = compact.find(pattern, start)
        if index < 0:
            return False
        end = index + len(pattern)
        left_ok = index == 0 or compact[index - 1] == "_"
        right_ok = end == len(compact) or compact[end] == "_"
        if left_ok and right_ok:
            return True
        start = index + 1


def canonical_class(folder_name: str, cfg: dict) -> str | None:
    """Map a raw class-folder name to a canonical deficiency label, or None."""
    compact = re.sub(r"[\s\-]+", "_", folder_name.strip().lower())
    exact = {str(key).lower(): value for key, value in cfg.get("exact_aliases", {}).items()}
    if compact in exact:
        return exact[compact]

    pairs: list[tuple[str, str]] = []
    for label, patterns in cfg.get("class_aliases", {}).items():
        for pattern in patterns:
            pairs.append((str(pattern).lower(), label))
    # Longest alias wins so "magnesium" beats a shorter overlapping token.
    pairs.sort(key=lambda item: len(item[0]), reverse=True)
    for pattern, label in pairs:
        if _alias_at_boundary(compact, pattern):
            return label
    return None


def raw_has_images(raw_dir: Path) -> bool:
    if not raw_dir.exists():
        return False
    for path in raw_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            return True
    return False


def scan_raw(raw_dir: Path, cfg: dict) -> tuple[list[dict], list[str]]:
    """Walk crop/class folders. Unmapped class folders are skipped, not fatal."""
    records: list[dict] = []
    unmapped: list[str] = []
    if not raw_dir.exists():
        return records, unmapped

    for crop_dir in sorted(path for path in raw_dir.iterdir() if path.is_dir()):
        loose = [
            path
            for path in crop_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if loose:
            print(
                f"WARNING: {len(loose)} image(s) directly under {crop_dir.name}/ "
                "have no class folder and were skipped."
            )
        for class_dir in sorted(path for path in crop_dir.iterdir() if path.is_dir()):
            label = canonical_class(class_dir.name, cfg)
            if label is None:
                rel = f"{crop_dir.name}/{class_dir.name}"
                unmapped.append(rel)
                print(f"WARNING: unmapped class folder skipped: {rel}")
                continue
            for image_path in sorted(class_dir.rglob("*")):
                if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                records.append(
                    {
                        "path": image_path.relative_to(cfg["_root"]).as_posix(),
                        "abs_path": image_path,
                        "crop": crop_dir.name.lower(),
                        "label": label,
                        "class_folder": class_dir.name,
                    }
                )
    return records, unmapped


def drop_unreadable(records: list[dict], cfg: dict) -> tuple[list[dict], list[dict]]:
    kept: list[dict] = []
    skipped: list[dict] = []
    for record in records:
        image = imread_bgr(record["abs_path"])
        if image is None:
            print(f"WARNING: unreadable image skipped: {record['path']}")
            skipped.append({"path": record["path"], "reason": "unreadable"})
            continue
        kept.append(record)
    skipped_path = path_from_key(cfg, "skipped_csv")
    ensure_dir(skipped_path.parent)
    pd.DataFrame(skipped, columns=["path", "reason"]).to_csv(skipped_path, index=False)
    return kept, skipped


def print_count_table(frame: pd.DataFrame) -> None:
    """Print image counts for every crop x class combination that was kept."""
    if frame.empty:
        print("No usable images.")
        return
    table = (
        frame.pivot_table(index="crop", columns="label", values="path", aggfunc="count", fill_value=0)
        .reindex(columns=[label for label in CLASS_ORDER if label in set(frame["label"])], fill_value=0)
        .astype(int)
    )
    print("\nImage counts (crop x class):")
    print(table.to_string())
    print(f"\nTotal usable images: {len(frame)}")

    expected_crops = [crop.lower() for crop in frame.attrs.get("expected_crops", [])]
    present_crops = sorted(frame["crop"].unique())
    for crop in expected_crops:
        if crop not in present_crops:
            print(f"NOTE: no usable images for crop '{crop}'. Continuing with: {present_crops}.")
    for crop in present_crops:
        present = set(frame.loc[frame["crop"] == crop, "label"])
        missing = [label for label in CLASS_ORDER if label not in present]
        if missing:
            have = [label for label in CLASS_ORDER if label in present]
            print(
                f"NOTE: {crop} is missing class(es) {missing}. "
                f"Continuing with the classes that exist: {have}."
            )


def stratified_split(frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """80/20 split stratified on deficiency class. Seed comes from config."""
    seed = int(cfg["split"]["seed"])
    test_size = float(cfg["split"]["test_size"])
    set_seed(seed)
    ordered = frame.sort_values(["crop", "label", "path"]).reset_index(drop=True)
    counts = ordered["label"].value_counts()
    too_small = sorted(counts[counts < 2].index.tolist())
    if too_small:
        print(
            f"WARNING: classes with fewer than 2 images {too_small} cannot be stratified. "
            "Falling back to an unstratified split."
        )
        stratify = None
    else:
        stratify = ordered["label"]
    train_df, test_df = train_test_split(
        ordered,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    test_df["split"] = "test"
    split = pd.concat([train_df, test_df], ignore_index=True)
    split = split.sort_values(["split", "crop", "label", "path"]).reset_index(drop=True)
    print("\nSplit sizes (seed %s, test_size %s):" % (seed, test_size))
    print(split.groupby(["split", "label"]).size().unstack(fill_value=0).to_string())
    return split


def prepare_dataset(cfg: dict | None = None, allow_dummy: bool = True) -> pd.DataFrame:
    """Scan data/raw, optionally synthesise a dummy set, and write outputs/split.csv."""
    cfg = cfg or load_config()
    set_seed(int(cfg["seed"]))
    raw_dir = path_from_key(cfg, "raw_dir")
    marker = raw_dir / "DUMMY_MARKER.txt"
    source = "real"
    detail = f"Images loaded from {raw_dir.as_posix()}."

    if marker.is_file():
        source = "dummy"
        detail = (
            "DUMMY_MARKER.txt is present, so these images are synthetic. "
            "DUMMY DATA - DO NOT REPORT."
        )
        print(DUMMY_BANNER)
    elif not raw_has_images(raw_dir):
        if not allow_dummy:
            raise FileNotFoundError(f"No images under {raw_dir} and dummy generation is disabled.")
        print(DUMMY_BANNER)
        from scripts.make_dummy_data import generate_dummy_dataset

        generate_dummy_dataset(cfg)
        source = "dummy"
        detail = "data/raw was empty. Synthetic images were generated. DUMMY DATA - DO NOT REPORT."
        print(DUMMY_BANNER)

    write_data_source(cfg, source, detail)
    records, _unmapped = scan_raw(raw_dir, cfg)
    kept, skipped = drop_unreadable(records, cfg)
    if not kept:
        raise RuntimeError(
            "No usable labelled images after alias mapping and readability checks. "
            "Add class folders under data/raw/<crop>/<class>/."
        )
    frame = pd.DataFrame(kept)
    frame.attrs["expected_crops"] = cfg.get("crops", [])
    print_count_table(frame)
    if skipped:
        print(f"Skipped {len(skipped)} unreadable file(s). See {path_from_key(cfg, 'skipped_csv')}.")

    split = stratified_split(frame, cfg)
    split_path = path_from_key(cfg, "split_csv")
    ensure_dir(split_path.parent)
    columns = ["path", "crop", "label", "split", "class_folder"]
    split[columns].to_csv(split_path, index=False)
    print(f"\nWrote {split_path.as_posix()} ({len(split)} rows, source={source}).")
    return split


def main() -> None:
    prepare_dataset(load_config(), allow_dummy=True)


if __name__ == "__main__":
    main()

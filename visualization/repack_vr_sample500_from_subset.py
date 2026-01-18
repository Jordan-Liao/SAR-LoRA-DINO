#!/usr/bin/env python3
import argparse
import hashlib
import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class PackedZip:
    tag: str
    split: str
    timestamp: str
    subset_json: Path
    source_vis_dir: Path
    zip_path: Path
    list_path: Path
    image_count: int
    zip_bytes: int
    zip_md5: str


DEFAULT_TAGS: Sequence[str] = (
    "E0003_linear-probe",
    "E0019_full-ft",
    "E0019_full-ft_ep3",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repack VR visualizations into 500-image zips based on a fixed COCO subset JSON "
            "(to align different methods on the exact same images)."
        )
    )
    parser.add_argument("--pkg-dir", required=True, help="Output package directory (zips/manifest/readme)")
    parser.add_argument(
        "--vr-root",
        default="artifacts/visualizations/VR",
        help="Root directory that contains per-tag VR exports",
    )
    parser.add_argument("--val-subset", required=True, help="COCO JSON containing exactly 500 val images")
    parser.add_argument("--test-subset", required=True, help="COCO JSON containing exactly 500 test images")
    parser.add_argument(
        "--tags",
        nargs="*",
        default=list(DEFAULT_TAGS),
        help=f"Model tags under --vr-root (default: {', '.join(DEFAULT_TAGS)})",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing VR_SAMPLE500_*.{zip,txt}, manifest.tsv, md5sum.txt, subsets/ before rebuilding.",
    )
    return parser.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _md5_file(path: Path) -> str:
    md5 = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            md5.update(chunk)
    return md5.hexdigest()


def _load_subset_filenames(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    images = data.get("images")
    if not isinstance(images, list):
        raise ValueError(f"Invalid COCO JSON (missing 'images' list): {path}")
    filenames = []
    for img in images:
        name = img.get("file_name")
        if not name:
            raise ValueError(f"Invalid COCO JSON (missing image.file_name): {path}")
        filenames.append(str(name))
    if len(filenames) != 500:
        raise ValueError(f"Expected 500 images in subset JSON, got {len(filenames)}: {path}")
    if len(set(filenames)) != 500:
        raise ValueError(f"Duplicate file_name found in subset JSON: {path}")
    return sorted(filenames)


def _find_latest_timestamp_dir(split_out: Path) -> Path:
    candidates = [p for p in split_out.iterdir() if p.is_dir() and p.name[:8].isdigit() and "_" in p.name]
    if not candidates:
        raise RuntimeError(f"No timestamp dir found under {split_out}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _build_vis_file_map(vis_dir: Path) -> Dict[str, Path]:
    m: Dict[str, Path] = {}
    for p in vis_dir.iterdir():
        if not p.is_file():
            continue
        key = p.name.lower()
        if key not in m:
            m[key] = p
    return m


def _zip_selected(
    repo_root: Path,
    selected_files: Sequence[Path],
    zip_path: Path,
) -> Tuple[int, int]:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for p in selected_files:
            arcname = p.resolve().relative_to(repo_root.resolve()).as_posix()
            zf.write(p, arcname)

    return len(selected_files), zip_path.stat().st_size


def _write_list(repo_root: Path, selected_files: Sequence[Path], list_path: Path) -> None:
    lines = [p.resolve().relative_to(repo_root.resolve()).as_posix() for p in selected_files]
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _clean_pkg_dir(pkg_dir: Path) -> None:
    for p in pkg_dir.glob("VR_SAMPLE500_*.zip"):
        if p.is_file():
            p.unlink()
    for p in pkg_dir.glob("VR_SAMPLE500_*.txt"):
        if p.is_file():
            p.unlink()
    for name in ("manifest.tsv", "md5sum.txt", "README.md"):
        p = pkg_dir / name
        if p.is_file():
            p.unlink()
    subsets = pkg_dir / "subsets"
    if subsets.is_dir():
        shutil.rmtree(subsets)


def _write_manifest(pkg_dir: Path, rows: Sequence[PackedZip]) -> None:
    header = [
        "tag",
        "split",
        "timestamp",
        "subset_json",
        "source_vis_dir",
        "zip_file",
        "zip_md5",
        "zip_bytes",
        "image_count",
        "list_file",
    ]
    out = pkg_dir / "manifest.tsv"
    lines = ["\t".join(header)]
    for r in rows:
        lines.append(
            "\t".join(
                [
                    r.tag,
                    r.split,
                    r.timestamp,
                    r.subset_json.as_posix(),
                    r.source_vis_dir.as_posix(),
                    r.zip_path.name,
                    r.zip_md5,
                    str(r.zip_bytes),
                    str(r.image_count),
                    r.list_path.name,
                ]
            )
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_md5sum(pkg_dir: Path) -> None:
    files = sorted([p for p in pkg_dir.rglob("*") if p.is_file() and p.name != "md5sum.txt"])
    lines = []
    for p in files:
        md5 = _md5_file(p)
        rel = p.relative_to(pkg_dir).as_posix()
        lines.append(f"{md5}  {rel}")
    (pkg_dir / "md5sum.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_readme(pkg_dir: Path, val_subset: Path, test_subset: Path, rows: Sequence[PackedZip]) -> None:
    lines: List[str] = []
    lines.append("# VR 可视化抽样包（固定同一套 500 张，便于对比）")
    lines.append("")
    lines.append("本目录用于把 **全量 VR 导出** 中的 `vis/` 图片，按一份固定的 COCO 子集 JSON（500 张）重新打包成 ZIP。")
    lines.append("这样 **不同方法/不同版本** 的 500 张图片是同一批，便于逐图对比。")
    lines.append("")
    lines.append("## 1) 抽样子集（固定）")
    lines.append("")
    lines.append(f"- Val: `{val_subset.as_posix()}`")
    lines.append(f"- Test: `{test_subset.as_posix()}`")
    lines.append("")
    lines.append("ZIP 内只包含这两份 JSON 中声明的 `images[*].file_name` 对应的 500 张可视化图片。")
    lines.append("")
    lines.append("## 2) 包含哪些方法/ZIP")
    lines.append("")
    for r in rows:
        lines.append(
            f"- `{r.source_vis_dir.as_posix()}` → `{r.zip_path.name}`（{r.split} 500 张）"
        )
    lines.append("")
    lines.append("每个 ZIP 内路径保持为仓库相对路径（repo-relative），例如：")
    lines.append("")
    if rows:
        example = f"{rows[0].source_vis_dir.as_posix()}/0000004.jpg"
        lines.append(f"- `{example}`")
    lines.append("")
    lines.append("## 3) 清单与校验")
    lines.append("")
    lines.append("- `VR_SAMPLE500_*.txt`：每个 ZIP 内文件列表（repo-relative 路径，与 ZIP 内路径一致）。")
    lines.append("- `manifest.tsv`：每个 ZIP 的来源目录、所用子集 JSON、zip md5/大小等信息。")
    lines.append("- `md5sum.txt`：对本目录全部文件的 MD5 校验，可用下面命令检查：")
    lines.append("")
    lines.append("```bash")
    lines.append(f"cd {pkg_dir.as_posix()}")
    lines.append("md5sum -c md5sum.txt")
    lines.append("```")
    lines.append("")
    lines.append("## 4) 解压方式")
    lines.append("")
    lines.append("```bash")
    lines.append("mkdir -p unpack")
    lines.append("unzip -q 'VR_SAMPLE500_*.zip' -d unpack")
    lines.append("```")
    lines.append("")
    lines.append("## 5) 分享/分发")
    lines.append("")
    lines.append("你可以把该目录上传到 GitHub Releases / Kaggle / 云盘等作为补充材料；`md5sum.txt` 可用于校验文件一致性。")
    (pkg_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = _parse_args()
    repo_root = _repo_root()

    pkg_dir = Path(args.pkg_dir)
    vr_root = Path(args.vr_root)
    val_subset = Path(args.val_subset)
    test_subset = Path(args.test_subset)

    pkg_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        _clean_pkg_dir(pkg_dir)

    subsets_dir = pkg_dir / "subsets"
    subsets_dir.mkdir(parents=True, exist_ok=True)
    val_subset_dst = subsets_dir / val_subset.name
    test_subset_dst = subsets_dir / test_subset.name
    shutil.copy2(val_subset, val_subset_dst)
    shutil.copy2(test_subset, test_subset_dst)

    subset_files = {
        "val": _load_subset_filenames(val_subset_dst),
        "test": _load_subset_filenames(test_subset_dst),
    }

    rows: List[PackedZip] = []
    for tag in args.tags:
        for split in ("val", "test"):
            subset_json = val_subset_dst if split == "val" else test_subset_dst
            split_out = vr_root / tag / split
            if not split_out.is_dir():
                raise FileNotFoundError(f"Missing VR split dir: {split_out}")
            ts_dir = _find_latest_timestamp_dir(split_out)
            timestamp = ts_dir.name
            vis_dir = ts_dir / "vis"
            if not vis_dir.is_dir():
                raise FileNotFoundError(f"Missing vis dir: {vis_dir}")

            vis_map = _build_vis_file_map(vis_dir)
            selected: List[Path] = []
            missing: List[str] = []
            for name in subset_files[split]:
                p = vis_map.get(name.lower())
                if p is None:
                    missing.append(name)
                    continue
                selected.append(p)
            if missing:
                raise FileNotFoundError(f"{tag} {split} missing {len(missing)}/500 files in {vis_dir}: {missing[:10]}")

            zip_stem = f"VR_SAMPLE500_{tag}_{split}_{timestamp}"
            zip_path = pkg_dir / f"{zip_stem}.zip"
            list_path = pkg_dir / f"{zip_stem}.txt"

            image_count, zip_bytes = _zip_selected(repo_root, selected, zip_path)
            _write_list(repo_root, selected, list_path)
            zip_md5 = _md5_file(zip_path)
            rows.append(
                PackedZip(
                    tag=tag,
                    split=split,
                    timestamp=timestamp,
                    subset_json=subset_json,
                    source_vis_dir=vis_dir,
                    zip_path=zip_path,
                    list_path=list_path,
                    image_count=image_count,
                    zip_bytes=zip_bytes,
                    zip_md5=zip_md5,
                )
            )

    rows.sort(key=lambda r: (r.tag, r.split))
    _write_manifest(pkg_dir, rows)
    _write_readme(pkg_dir, val_subset_dst, test_subset_dst, rows)
    _write_md5sum(pkg_dir)
    print(f"OK: {pkg_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup


_HERE = Path(__file__).resolve().parent


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_version() -> str:
    version_path = _HERE / "sar_lora_dino" / "version.py"
    ns = {}
    exec(_read_text(version_path), ns)  # noqa: S102
    return str(ns["__version__"])


def _read_requirements() -> list[str]:
    req_path = _HERE / "requirements.txt"
    if not req_path.exists():
        return []
    reqs: list[str] = []
    for line in _read_text(req_path).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        reqs.append(line)
    return reqs


if __name__ == "__main__":
    setup(
        name="sar_lora_dino",
        version=_read_version(),
        description="SAR-LoRA-DINO toolkit extensions",
        long_description=_read_text(_HERE / "README.md"),
        long_description_content_type="text/markdown",
        author="shatianming5",
        author_email="",
        keywords="computer vision, object detection",
        url="https://github.com/shatianming5/dino_sar",
        packages=find_packages(exclude=("configs",)),
        include_package_data=True,
        python_requires=">=3.8",
        install_requires=_read_requirements(),
        license="Apache-2.0",
        zip_safe=False,
    )

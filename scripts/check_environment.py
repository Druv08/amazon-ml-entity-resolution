"""Read-only fresh-clone preflight for the production Hybrid50 pipeline."""

from __future__ import annotations

import hashlib
import importlib.metadata
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BUNDLE_SHA256 = "f9bb2c379e95a6157ea04535c33bc7b2b82c343af6a93e49a9363cea0b3e1a89"
RAW_FILES = (
    "data/raw/train/train_source1.tsv",
    "data/raw/train/train_source2.tsv",
    "data/raw/train/train_source3.tsv",
    "data/raw/train/train_ground_truth.tsv",
    "data/raw/test/test_source1.tsv",
    "data/raw/test/test_source2.tsv",
    "data/raw/test/test_source3.tsv",
)
SOURCE_HEADER = "entity_id\tbusiness_name\tbusiness_address\tcountry"
TRUTH_HEADER = "source1_entity_id\tmatched_entity_ids"
BUNDLE = "output/candidates_p3/hybrid_meta.pkl"
CANDIDATE_RUNS = (
    "output/candidates_k20/test/run.json",
    "output/candidates_k50/test/run.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def requirements() -> list[tuple[str, str]]:
    pins = []
    for raw in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "==" in line:
            pins.append(tuple(line.split("==", 1)))
    return pins


def main() -> int:
    blockers = 0
    print(f"Repository: {ROOT}")
    print(f"Python: {sys.version.split()[0]}")
    if sys.version_info < (3, 12):
        print("  [MISSING] Python 3.12+ is required; Python 3.13 is the documented environment.")
        blockers += 1
    elif sys.version_info[:2] not in ((3, 13), (3, 14)):
        print("  [NOTE] Python 3.13/3.14 is documented; verify tests on this interpreter.")

    print("Packages:")
    for name, expected in requirements():
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            print(f"  [MISSING] {name}=={expected}")
            blockers += 1
            continue
        label = "OK" if installed == expected else "MISMATCH"
        print(f"  [{label}] {name} {installed} (expected {expected})")
        blockers += installed != expected

    print("Official dataset:")
    for rel in RAW_FILES:
        path = ROOT / rel
        if not path.is_file():
            print(f"  [MISSING] {rel}")
            blockers += 1
            continue
        expected = TRUTH_HEADER if rel.endswith("ground_truth.tsv") else SOURCE_HEADER
        with path.open(encoding="utf-8") as fh:
            header = fh.readline().rstrip("\r\n")
        if header != expected:
            print(f"  [INVALID] {rel}: unexpected TSV header")
            blockers += 1
        else:
            print(f"  [OK] {rel} ({path.stat().st_size:,} bytes)")

    print("Production bundle:")
    bundle = ROOT / BUNDLE
    if not bundle.is_file():
        print(f"  [MISSING] {BUNDLE}")
        blockers += 1
    else:
        actual = sha256(bundle)
        print(f"  [OK] {BUNDLE} ({bundle.stat().st_size:,} bytes)")
        if actual == EXPECTED_BUNDLE_SHA256:
            print(f"  [OK] SHA-256 {actual}")
        else:
            print(f"  [NOTE] SHA-256 differs from the audited bundle: {actual}")
            print("         This is expected only if the bundle was deliberately retrained.")

    print("Generated candidate runs:")
    for rel in CANDIDATE_RUNS:
        state = "READY" if (ROOT / rel).is_file() else "PENDING"
        print(f"  [{state}] {rel}")

    free = shutil.disk_usage(ROOT).free
    print(f"Disk free: {free / 2**30:.1f} GiB")
    if free < 10 * 2**30:
        print("  [NOTE] Less than 10 GiB is free; candidate caches and shards need substantial space.")

    if blockers:
        print(f"\nPreflight: BLOCKED ({blockers} required item(s) missing or mismatched)")
        return 1
    print("\nPreflight: READY for candidate generation/inference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

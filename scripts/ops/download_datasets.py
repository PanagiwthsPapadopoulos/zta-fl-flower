#!/usr/bin/env python3
"""
ZTA-FL dataset download and numeric CSV preparation (NO browser needed)
=============================================================================
Downloads the three IDS benchmarks used in:
  "Zero-Trust Agentic Federated Learning for Secure IIoT Defense Systems"
  (Edge-IIoTset, CIC-IDS2017, UNSW-NB15)

Sources (all verified working, no manual registration):
  Edge-IIoTset : Kaggle, author's official upload (Ferrag)
                 -> kagglehub anonymous access, selective file fetch
                 (only the 1.2 GB DNN csv + 82 MB ML csv are pulled,
                  NOT the full 11 GB archive with pcaps)
  CIC-IDS2017  : HuggingFace mirror of the official MachineLearningCVE CSVs
                 (bypasses UNB's registration form; direct UNB downloads
                  are blocked as of 2026 - verified)
                 -> optional Kaggle fallback if HF is unreachable
  UNSW-NB15    : Kaggle full mirror (mrwellsdavid/unsw-nb15)
                 -> kagglehub anonymous access

Requirements:  Python 3.9+, internet.
  pip install kagglehub requests
Optional (only if Kaggle rate-limits anonymous downloads):
  export KAGGLE_USERNAME=... KAGGLE_KEY=...   (free account -> kaggle.com/settings)

Usage:
  python download_datasets.py                      # all three datasets
  python download_datasets.py --only edge unsw     # subset
  python download_datasets.py --smoke              # downloads only; no cleaning
  python download_datasets.py --verify             # + row-count integrity checks
  python download_datasets.py --preprocess-only    # reuse completed downloads
  python download_datasets.py --out ./data         # custom data root

Outputs: data/<dataset>/full_dataset/<existing CSV basename>.csv, with original
downloads in full_dataset/downloads/ and cleaning counts in preprocessing_report.json.
Cleaned outputs are replaced only after cleaning succeeds; snippets remain in raw/.
Point dataset_path at the cleaned CSV before running the existing artifact builder.
The cleaner does not scale, run PCA/SMOTE, or split the data. UNSW's four full
files supply the observations; its prepared training/testing files are references.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import time
import urllib.parse
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ops.preprocess_datasets import CIC_FILES, DATASETS, download_directory, preprocess_dataset

# --------------------------------------------------------------------------
# Constants - dataset handles and exact in-archive paths (verified 2026-10)
# --------------------------------------------------------------------------
EDGE_SLUG = "mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot"
EDGE_PREFIX = "Edge-IIoTset dataset/Selected dataset for ML and DL"
EDGE_DNN = f"{EDGE_PREFIX}/DNN-EdgeIIoT-dataset.csv"      # 1,217.4 MB, 2,219,201 rows
EDGE_ML = f"{EDGE_PREFIX}/ML-EdgeIIoT-dataset.csv"        #    82.2 MB
EDGE_PDF = "Edge_IIoTset__DatasetFL.pdf"                  # dataset documentation

UNSW_SLUG = "mrwellsdavid/unsw-nb15"
UNSW_FILES = [  # exact names in archive (note: 'NUSW' typo is in the archive)
    "UNSW-NB15_1.csv",            #   169.0 MB } full dataset,
    "UNSW-NB15_2.csv",            #   165.2 MB } 2,540,044 records
    "UNSW-NB15_3.csv",            #   154.6 MB } across 4 parts
    "UNSW-NB15_4.csv",            #    97.6 MB }
    "UNSW_NB15_training-set.csv", #    15.4 MB, 175,341 rows
    "UNSW_NB15_testing-set.csv",  #    32.3 MB,  82,332 rows
    # NOTE: in the mrwellsdavid mirror these two files are SWAPPED on disk
    # (verified 2026-10). The script auto-corrects them after download.
    "NUSW-NB15_features.csv",     #   feature definitions (49 features)
    "UNSW-NB15_LIST_EVENTS.csv",
]

CIC_HF_BASE = "https://huggingface.co/datasets/c01dsnap/CIC-IDS2017/resolve/main"
CIC_KAGGLE_FALLBACK = "dhoogla/cicids2017"
# Expected integrity numbers (from the ZTA-FL paper + official dataset docs)
EXPECTED = {
    "edge_dnn_rows": 2_219_201,
    "cic_total_rows": 2_830_743,
    "unsw_full_rows": 2_540_044,
    "unsw_train_rows": 175_341,
    "unsw_test_rows": 82_332,
}

RETRIES = 3


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# --------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------
def ensure_kagglehub() -> None:
    try:
        import kagglehub  # noqa: F401
    except ImportError:
        log("ERROR: kagglehub is not installed. Run:  pip install kagglehub")
        sys.exit(1)


def ensure_requests() -> None:
    try:
        import requests  # noqa: F401
    except ImportError:
        log("ERROR: requests is not installed. Run:  pip install requests")
        sys.exit(1)


def kaggle_fetch_file(slug: str, rel_path: str, dest_dir: Path) -> Path:
    """Download ONE file from a public Kaggle dataset (anonymous ok), move it
    into dest_dir. Uses kagglehub cache so re-runs are instant."""
    import kagglehub

    log(f"  kagglehub fetching: {rel_path}")
    src = Path(kagglehub.dataset_download(slug, path=rel_path))
    dest = dest_dir / Path(rel_path).name
    dest_dir.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dest.resolve():
        # Cross-filesystem moves may copy first. Never expose that partial copy
        # as a completed source that skip_existing would accept on the next run.
        staged = dest.with_suffix(dest.suffix + ".kaggle.part")
        shutil.move(str(src), str(staged))
        staged.replace(dest)
    return dest


def kaggle_list_files(slug: str) -> list[tuple[str, int]]:
    """List files of a public Kaggle dataset anonymously (name, bytes)."""
    from kagglehub.clients import build_kaggle_client
    from kagglesdk.datasets.types.dataset_api_service import ApiListDatasetFilesRequest

    owner, dataset = slug.split("/")
    req = ApiListDatasetFilesRequest()
    req.owner_slug = owner
    req.dataset_slug = dataset
    req.page_size = 100
    out: list[tuple[str, int]] = []
    with build_kaggle_client() as client:
        api = client.datasets.dataset_api_client
        token = None
        for _ in range(20):
            if token:
                req.page_token = token
            resp = api.list_dataset_files(req)
            out.extend((f.name, getattr(f, "total_bytes", 0) or 0) for f in resp.files)
            token = getattr(resp, "next_page_token", None)
            if not token:
                break
    return out


def hf_download(fname: str, dest_dir: Path) -> Path:
    """Streamed, resumable download of one file from the HF mirror."""
    import requests

    dest = dest_dir / fname
    dest_dir.mkdir(parents=True, exist_ok=True)
    url = f"{CIC_HF_BASE}/{urllib.parse.quote(fname)}"
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, RETRIES + 1):
        # Recalculate after each failed transfer: a retry must not append an
        # earlier byte range to bytes already written by the previous attempt.
        offset = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": "ztafl-dataset-downloader/1.0", "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
            log(f"  resuming {fname} from {human(offset)}")
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60, allow_redirects=True) as r:
                if r.status_code == 416 and offset:
                    complete = re.fullmatch(r"bytes \*/(\d+)", r.headers.get("content-range", ""))
                    if complete and int(complete[1]) == offset:
                        part.replace(dest)
                        return dest
                if r.status_code not in (200, 206):
                    r.raise_for_status()
                    raise RuntimeError(f"Unexpected HTTP status {r.status_code}")
                expected_bytes = int(r.headers.get("content-length") or 0)
                if r.status_code == 206:
                    span = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", r.headers.get("content-range", ""))
                    if not span or int(span[1]) != offset or int(span[2]) < offset:
                        raise ValueError("Server returned an inconsistent resume range")
                    total = int(span[3])
                    expected_bytes = int(span[2]) - offset + 1
                    mode, done = "ab", offset
                else:
                    total = int(r.headers.get("x-linked-size") or expected_bytes)
                    mode, done = "wb", 0
                received = 0
                with open(part, mode) as f:
                    last = done
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        received += len(chunk)
                        if total and done - last >= 50 * (1 << 20):
                            last = done
                            log(f"  {fname}: {human(done)} / {human(total)}")
                if (expected_bytes and received != expected_bytes) or (total and done != total):
                    raise ValueError("Incomplete download; retaining .part file for resume")
            part.replace(dest)
            return dest
        except Exception as e:  # noqa: BLE001
            log(f"  attempt {attempt}/{RETRIES} failed: {e}")
            time.sleep(3 * attempt)
    raise RuntimeError(f"Failed to download {fname} from HuggingFace")


def count_data_rows(path: Path) -> int:
    """Count non-header lines (buffered, fast)."""
    rows = 0
    with open(path, "rb") as f:
        first = True
        while True:
            block = f.read(1 << 22)
            if not block:
                break
            rows += block.count(b"\n") - (1 if first else 0)
            first = False
    # handle files not ending with newline
    with open(path, "rb") as f:
        f.seek(max(0, path.stat().st_size - 2))
        tail = f.read()
    if tail and not tail.endswith(b"\n"):
        rows += 1
    return max(0, rows)


def finalize(dest: Path) -> None:
    log(f"  saved -> {dest}  ({human(dest.stat().st_size)})")


# --------------------------------------------------------------------------
# Dataset 1: Edge-IIoTset
# --------------------------------------------------------------------------
def download_edge(out_root: Path, args) -> None:
    log("== Edge-IIoTset (Kaggle, anonymous) ==")
    dest_dir = download_directory(out_root, "edge")
    # sanity: confirm the two key files exist in the archive
    listing = kaggle_list_files(EDGE_SLUG)
    names = {n for n, _ in listing}
    for required in (EDGE_DNN, EDGE_ML):
        if required not in names:
            log(f"  WARNING: {required} not found in archive listing; will try anyway")
    files = [EDGE_DNN] + ([] if args.no_ml else [EDGE_ML])
    if args.with_pdf:
        files.append(EDGE_PDF)
    for relative in files:
        target = dest_dir / Path(relative).name
        if args.skip_existing and target.exists() and target.stat().st_size > 0:
            log(f"  skip existing: {target.name}")
        else:
            finalize(kaggle_fetch_file(EDGE_SLUG, relative, dest_dir))
    if args.verify:
        rows = count_data_rows(dest_dir / Path(EDGE_DNN).name)
        exp = EXPECTED["edge_dnn_rows"]
        status = "OK" if rows == exp else "MISMATCH"
        log(f"  verify DNN rows: {rows:,} (expected {exp:,}) -> {status}")


# --------------------------------------------------------------------------
# Dataset 2: CIC-IDS2017
# --------------------------------------------------------------------------
def hf_file_sizes() -> dict[str, int]:
    import requests

    sizes = {}
    for f in CIC_FILES:
        url = f"{CIC_HF_BASE}/{urllib.parse.quote(f)}"
        try:
            r = requests.head(url, allow_redirects=True, timeout=30,
                              headers={"User-Agent": "ztafl-dataset-downloader/1.0"})
            size = int(r.headers.get("x-linked-size") or r.headers.get("content-length") or 0)
            sizes[f] = size
        except Exception:  # noqa: BLE001
            sizes[f] = 0
    return sizes


def download_cic(out_root: Path, args) -> None:
    log("== CIC-IDS2017 (HuggingFace mirror, no auth) ==")
    dest_dir = download_directory(out_root, "cic")
    dest_dir.mkdir(parents=True, exist_ok=True)
    files = CIC_FILES
    if args.smoke:
        sizes = hf_file_sizes()
        files = [min((f for f in CIC_FILES if sizes.get(f)), key=lambda f: sizes[f], default=CIC_FILES[0])]
        log(f"  smoke mode: downloading smallest file only -> {files[0]}")
    done = []
    for f in files:
        if args.skip_existing and (dest_dir / f).exists() and (dest_dir / f).stat().st_size > 0:
            log(f"  skip existing: {f}")
            done.append(dest_dir / f)
            continue
        hf_download(f, dest_dir)
        finalize(dest_dir / f)
        done.append(dest_dir / f)
    if args.verify and not args.smoke:
        total = sum(count_data_rows(p) for p in done)
        exp = EXPECTED["cic_total_rows"]
        status = "OK" if abs(total - exp) < 5000 else "MISMATCH"
        log(f"  verify total rows: {total:,} (expected ~{exp:,}) -> {status}")


def download_cic_fallback(out_root: Path, args) -> bool:
    """Kaggle fallback if the HF mirror is unreachable."""
    log("== CIC-IDS2017 fallback (Kaggle dhoogla/cicids2017) ==")
    try:
        ensure_kagglehub()
        listing = kaggle_list_files(CIC_KAGGLE_FALLBACK)
        names = {n for n, _ in listing}
        dest_dir = download_directory(out_root, "cic")
        for f in CIC_FILES:
            target = dest_dir / f
            if args.skip_existing and target.exists() and target.stat().st_size > 0:
                continue
            match = next((n for n in names if Path(n).name == f), None)
            if match is None:
                raise ValueError(f"Required original CSV {f} not in fallback archive")
            p = kaggle_fetch_file(CIC_KAGGLE_FALLBACK, match, dest_dir)
            finalize(p)
        return True
    except Exception as e:  # noqa: BLE001
        log(f"  fallback failed: {e}")
        return False


# --------------------------------------------------------------------------
# Dataset 3: UNSW-NB15
# --------------------------------------------------------------------------
def fix_unsw_partition_swap(dest_dir: Path) -> None:
    """The mrwellsdavid Kaggle mirror ships the partition files with their
    contents swapped. Detect by row count and rename so files match their
    official meaning (training=175,341 rows, testing=82,332 rows)."""
    tr = dest_dir / "UNSW_NB15_training-set.csv"
    te = dest_dir / "UNSW_NB15_testing-set.csv"
    if not (tr.exists() and te.exists()):
        return
    r_tr, r_te = count_data_rows(tr), count_data_rows(te)
    if r_tr == EXPECTED["unsw_test_rows"] and r_te == EXPECTED["unsw_train_rows"]:
        log("  detected swapped partition files in mirror -> auto-correcting names")
        tmp = dest_dir / "__swap_tmp.csv"
        tr.rename(tmp)
        te.rename(tr)
        tmp.rename(te)
    elif r_tr != EXPECTED["unsw_train_rows"] or r_te != EXPECTED["unsw_test_rows"]:
        log(f"  WARNING: partition row counts unexpected: train={r_tr:,}, test={r_te:,}")


def download_unsw(out_root: Path, args) -> None:
    log("== UNSW-NB15 (Kaggle, anonymous) ==")
    dest_dir = download_directory(out_root, "unsw")
    listing = kaggle_list_files(UNSW_SLUG)
    names = {n for n, _ in listing}
    files = UNSW_FILES
    if args.smoke:
        files = ["UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"]
        log("  smoke mode: downloading partition files only")
    for rel in files:
        if rel not in names:
            raise ValueError(f"Required UNSW download {rel} not in archive")
        if args.skip_existing and (dest_dir / rel).exists() and (dest_dir / rel).stat().st_size > 0:
            log(f"  skip existing: {rel}")
            continue
        p = kaggle_fetch_file(UNSW_SLUG, rel, dest_dir)
        finalize(p)
    if not args.smoke:
        fix_unsw_partition_swap(dest_dir)
    if args.verify and not args.smoke:
        tr = dest_dir / "UNSW_NB15_training-set.csv"
        te = dest_dir / "UNSW_NB15_testing-set.csv"
        if tr.exists() and te.exists():
            r_tr, r_te = count_data_rows(tr), count_data_rows(te)
            ok_tr = r_tr == EXPECTED["unsw_train_rows"]
            ok_te = r_te == EXPECTED["unsw_test_rows"]
            log(f"  verify training rows: {r_tr:,} (expected {EXPECTED['unsw_train_rows']:,}) -> {'OK' if ok_tr else 'MISMATCH'}")
            log(f"  verify testing  rows: {r_te:,} (expected {EXPECTED['unsw_test_rows']:,}) -> {'OK' if ok_te else 'MISMATCH'}")
            full = sum(count_data_rows(dest_dir / f) for f in UNSW_FILES[:4] if (dest_dir / f).exists())
            if full:
                # These four files have no header, unlike the reference splits.
                full += sum((dest_dir / f).exists() for f in UNSW_FILES[:4])
                status = "OK" if abs(full - EXPECTED["unsw_full_rows"]) <= 100 else "MISMATCH"
                log(f"  verify full-dataset rows (4 parts): {full:,} (expected ~{EXPECTED['unsw_full_rows']:,}) -> {status}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Download and clean Edge-IIoTset, CIC-IDS2017, UNSW-NB15")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "data"),
                    help="data root (default: repository data/)")
    ap.add_argument("--only", nargs="+", choices=["edge", "cic", "unsw"], help="download only these datasets")
    ap.add_argument("--verify", action="store_true", help="verify row counts after download")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="download connectivity check only; do not publish cleaned inputs")
    mode.add_argument("--preprocess-only", action="store_true", help="clean full_dataset/downloads/ without network access or download dependencies")
    ap.add_argument("--no-ml", action="store_true", help="skip Edge ML download in smoke mode only; full cleaning requires both CSVs")
    ap.add_argument("--with-pdf", action="store_true", help="also fetch the Edge-IIoTset documentation PDF")
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                    help="re-download files even if present")
    ap.add_argument("--kaggle-user", default=os.environ.get("KAGGLE_USERNAME"), help=argparse.SUPPRESS)
    ap.add_argument("--kaggle-key", default=os.environ.get("KAGGLE_KEY"), help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.no_ml and not args.smoke:
        ap.error("--no-ml is only supported with --smoke; full Edge cleaning uses both DNN and ML files")

    if args.kaggle_user and args.kaggle_key:
        os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"] = args.kaggle_user, args.kaggle_key
        log("Using provided Kaggle credentials")

    if not args.preprocess_only:
        ensure_kagglehub()
        ensure_requests()
    out_root = Path(args.out).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    only = args.only or ["edge", "cic", "unsw"]

    t0 = time.time()
    results = {}
    downloaders = {"edge": download_edge, "cic": download_cic, "unsw": download_unsw}
    for dataset in dict.fromkeys(only):
        name = DATASETS[dataset][0]
        try:
            if not args.preprocess_only:
                try:
                    downloaders[dataset](out_root, args)
                except Exception as exc:
                    if dataset != "cic":
                        raise
                    log(f"HF mirror failed ({str(exc)[:60]}), trying Kaggle fallback...")
                    if not download_cic_fallback(out_root, args):
                        raise RuntimeError("CIC download incomplete") from exc
            if args.smoke:
                results[name] = "DOWNLOAD CHECK ONLY (no cleaned CSV published)"
            else:
                report = preprocess_dataset(out_root, dataset, log=log)
                results[name] = (f"OK: {report['kept_rows']:,} retained; "
                                 f"{report['duplicate_rows']:,} identical duplicates removed; "
                                 f"{report['invalid_numeric_rows']:,} invalid numeric rows excluded")
                if dataset == "unsw":
                    results[name] += (f"; {report['zero_filled_values']:,} values zero-filled "
                                      f"across {report['zero_filled_rows']:,} retained rows")
        except Exception as e:  # noqa: BLE001
            log(f"{name} FAILED: {e}")
            results[name] = f"FAILED: {str(e)[:120]}"

    log("------------ SUMMARY ------------")
    for k, v in results.items():
        log(f"  {k:<14} {v}")
    log(f"Output directory: {out_root}")
    log(f"Elapsed: {time.time() - t0:.0f}s")
    if not args.smoke:
        log("Next: select the cleaned CSV as dataset_path, then run scripts/setup/build_artifacts.py with the intended experiment configuration.")
    return 1 if any(value.startswith("FAILED") for value in results.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())

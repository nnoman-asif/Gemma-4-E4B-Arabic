"""Package LoRA Adapters, Evaluation Reports, vLLM Telemetry, Plots & Logs for Local Machine Sync.

When training on a remote Vast.ai `2x A100 80 GB` instance:
- The heavy standalone 16-bit merged models (`~60 GB`) and raw dataset shards (`~14 GB`) are
  pushed directly from Vast.ai's high-speed network to Hugging Face Hub (`noman-asif/...`).
- Everything else that you want preserved on your **local laptop** (`D:\\PERSONAL\\Projects\\Gemma-4-E4B-Arabic`)
  is packaged by this script into a single lightweight archive:
  `outputs/gemma4_arabic_local_sync_bundle.tar.gz` (~850 MB with all 4 LoRA adapters, <5 MB without).

Included in the Local Sync Bundle:
1. All 4 LoRA Adapters + Training Telemetry (`outputs/sft_e4b_a100`, `outputs/dpo_e4b_a100`,
   `outputs/sft_26b_a4b_a100`, `outputs/dpo_26b_a4b_a100`, `outputs/sft_stage1`, `outputs/dpo_stage2`)
2. All Benchmark, Latency & vLLM 30-50 User Concurrency Reports (`outputs/eval_reports/*`)
3. All Generated Hugging Face Model & Dataset Cards (`outputs/hub_cards/*`)
4. All 5 Publication Plots in PNG & SVG (`assets/plots/*`)
5. Dual-GPU Training Logs (`logs/*`)
"""

import argparse
import json
import logging
import os
import pathlib
import sys
import tarfile
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import DEFAULT_HF_NAMESPACE

logger = logging.getLogger(__name__)

ADAPTER_DIR_CANDIDATES: Tuple[str, ...] = (
    "outputs/sft_e4b_a100",
    "outputs/dpo_e4b_a100",
    "outputs/sft_26b_a4b_a100",
    "outputs/dpo_26b_a4b_a100",
    "outputs/sft_stage1",
    "outputs/dpo_stage2",
)

REPORT_AND_ASSET_DIRS: Tuple[str, ...] = (
    "outputs/eval_reports",
    "outputs/hub_cards",
    "assets/plots",
    "logs",
)


def collect_local_sync_files(project_root: pathlib.Path) -> List[pathlib.Path]:
    """Collect all adapter checkpoints, telemetry JSONs, eval reports, plots, and logs."""
    collected: List[pathlib.Path] = []

    for rel_dir in ADAPTER_DIR_CANDIDATES + REPORT_AND_ASSET_DIRS:
        abs_dir = project_root / rel_dir
        if not abs_dir.exists():
            continue
        for file_path in sorted(abs_dir.rglob("*")):
            if not file_path.is_file():
                continue
            # Skip intermediate optimizer states (optimizer.pt / rng_state.pth) inside checkpoint-* subfolders
            # while keeping final LoRA adapter weights (`adapter_model.safetensors`, `adapter_config.json`, etc.)
            if "checkpoint-" in str(file_path) and file_path.name in ("optimizer.pt", "scheduler.pt"):
                continue
            collected.append(file_path)

    return collected


def build_local_sync_bundle(
    project_root: pathlib.Path,
    bundle_path: pathlib.Path,
    manifest_path: pathlib.Path,
) -> Dict[str, Any]:
    """Create `gemma4_arabic_local_sync_bundle.tar.gz` and a JSON manifest of included files."""
    files = collect_local_sync_files(project_root)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    included_entries: List[Dict[str, Any]] = []
    total_uncompressed_bytes = 0

    with tarfile.open(bundle_path, "w:gz") as tar:
        for file_path in files:
            rel_path = file_path.relative_to(project_root).as_posix()
            size_bytes = file_path.stat().st_size
            total_uncompressed_bytes += size_bytes
            tar.add(str(file_path), arcname=rel_path)
            included_entries.append(
                {
                    "relative_path": rel_path,
                    "size_bytes": size_bytes,
                    "size_mb": round(size_bytes / (1024.0 * 1024.0), 3),
                }
            )

    bundle_size_bytes = bundle_path.stat().st_size if bundle_path.exists() else 0
    manifest = {
        "status": "packaged",
        "bundle_archive_path": str(bundle_path),
        "bundle_size_mb": round(bundle_size_bytes / (1024.0 * 1024.0), 3),
        "total_uncompressed_mb": round(total_uncompressed_bytes / (1024.0 * 1024.0), 3),
        "files_count": len(included_entries),
        "excluded_heavy_directories": [
            "outputs/merged_gemma4_e4b_arabic (~9.2 GB -> uploaded directly to HF Hub)",
            "outputs/merged_gemma4_26b_a4b_arabic (~50.8 GB -> uploaded directly to HF Hub)",
            "data/sft_mixed & data/dpo_mixed (~14 GB -> uploaded directly to HF Hub)",
        ],
        "local_restore_commands": {
            "option_1_scp_from_windows_powershell": (
                "scp -P <VAST_SSH_PORT> root@<VAST_SSH_IP>:/workspace/Gemma-4-E4B-Arabic/"
                "outputs/gemma4_arabic_local_sync_bundle.tar.gz .\n"
                "tar -xzf gemma4_arabic_local_sync_bundle.tar.gz"
            ),
            "option_2_private_hf_repo_download": (
                f"huggingface-cli download {DEFAULT_HF_NAMESPACE}/Gemma-4-Arabic-Adapters-Vault "
                "--repo-type model --local-dir ."
            ),
        },
        "files": included_entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def push_private_backup_to_hub(
    project_root: pathlib.Path,
    bundle_path: pathlib.Path,
    manifest_path: pathlib.Path,
    hf_namespace: str,
    private_repo_name: str = "Gemma-4-Arabic-Adapters-Vault",
    token: Optional[str] = None,
) -> str:
    """Optionally push all local sync artifacts + bundle to a PRIVATE Hugging Face repo for effortless local pull."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    repo_id = f"{hf_namespace}/{private_repo_name}"
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)

    # Upload the single compressed bundle + manifest + individual folders so `hf download` restores everything
    api.upload_file(
        path_or_fileobj=str(bundle_path),
        path_in_repo="outputs/gemma4_arabic_local_sync_bundle.tar.gz",
        repo_id=repo_id,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj=str(manifest_path),
        path_in_repo="outputs/local_sync_manifest.json",
        repo_id=repo_id,
        repo_type="model",
    )
    for rel_dir in ADAPTER_DIR_CANDIDATES + REPORT_AND_ASSET_DIRS:
        abs_dir = project_root / rel_dir
        if abs_dir.exists() and any(abs_dir.iterdir()):
            api.upload_folder(
                folder_path=str(abs_dir),
                path_in_repo=rel_dir,
                repo_id=repo_id,
                repo_type="model",
            )
    return repo_id


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI options for packaging local sync artifacts."""
    parser = argparse.ArgumentParser(
        description="Package LoRA adapters, evaluation reports, vLLM profiles, and plots for local sync."
    )
    parser.add_argument(
        "--bundle-path",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "gemma4_arabic_local_sync_bundle.tar.gz"),
        help="Output path for the compressed `.tar.gz` local sync bundle.",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "local_sync_manifest.json"),
        help="Output path for the JSON manifest listing all packaged files.",
    )
    parser.add_argument(
        "--push-private-backup",
        action="store_true",
        help="Upload the adapters, reports, and bundle to a PRIVATE Hugging Face repository for fast local pull.",
    )
    parser.add_argument(
        "--hf-namespace",
        type=str,
        default=DEFAULT_HF_NAMESPACE,
        help="Hugging Face namespace for the optional private backup repository.",
    )
    parser.add_argument(
        "--private-repo-name",
        type=str,
        default="Gemma-4-Arabic-Adapters-Vault",
        help="Private repository name on Hugging Face Hub.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for building the local sync bundle."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args(argv)
    bundle_path = pathlib.Path(args.bundle_path)
    manifest_path = pathlib.Path(args.manifest_path)

    manifest = build_local_sync_bundle(
        project_root=PROJECT_ROOT,
        bundle_path=bundle_path,
        manifest_path=manifest_path,
    )

    if args.push_private_backup:
        token = os.environ.get("HF_TOKEN")
        if token:
            private_repo = push_private_backup_to_hub(
                project_root=PROJECT_ROOT,
                bundle_path=bundle_path,
                manifest_path=manifest_path,
                hf_namespace=args.hf_namespace,
                private_repo_name=args.private_repo_name,
                token=token,
            )
            manifest["private_backup_repo"] = private_repo
        else:
            logger.warning("HF_TOKEN not set; skipping --push-private-backup upload.")

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

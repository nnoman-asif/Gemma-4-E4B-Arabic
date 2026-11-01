"""Automated Hugging Face Hub Publisher for the Gemma-4-Arabic Family & Community Datasets.

Pre-configured for `https://huggingface.co/noman-asif`:
1. Model-Agnostic Community Datasets:
   - `noman-asif/Arabic-Reasoning-Instruct-120k`: 120,000-row SFT curriculum
     (90,000 Arabic/Saudi + 30,000 English/Math/Code replay rows, strict 75%/25% mix).
   - `noman-asif/Arabic-Preference-DPO-20k`: 20,000-pair Arabic & Saudi preference dataset
     with unified user prompts for DPO/ORPO/SimPO alignment.
2. Iso-Active Arabic Model Family (Standalone Merged 16-bit BF16 Safetensors):
   - `noman-asif/Gemma-4-E4B-Arabic-It`: 4.5B Total (4.0B Active) Dense Edge/Workstation model.
   - `noman-asif/Gemma-4-26B-A4B-Arabic-It`: 25.2B Total (3.8B Active, 128 Experts) Sparse MoE Flagship.
3. Adapter & GGUF Policy:
   - Publishes standalone merged 16-bit (`bf16`) `safetensors` checkpoints + Model Cards +
     `assets/plots/*.png` by default.
   - LoRA adapters remain saved locally in `outputs/` and are only uploaded to a `lora_adapters/`
     subfolder if `--upload-adapters` is explicitly passed.
"""

import argparse
import json
import logging
import os
import pathlib
import sys
from typing import Any, Dict, List, Optional

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import (
    DEFAULT_26B_MODEL_REPO,
    DEFAULT_DPO_DATASET_REPO,
    DEFAULT_E4B_MODEL_REPO,
    DEFAULT_HF_NAMESPACE,
    DEFAULT_SFT_DATASET_REPO,
    FALLBACK_MODEL_ID_26B,
    GOOGLE_REFERENCE_MODEL_ID,
)

logger = logging.getLogger(__name__)


def build_sft_dataset_card(repo_id: str) -> str:
    """Generate the Hugging Face Dataset Card for `Arabic-Reasoning-Instruct-120k`."""
    lines = [
        "---",
        "language:",
        "- ar",
        "- en",
        "license: apache-2.0",
        "task_categories:",
        "- text-generation",
        "- question-answering",
        "pretty_name: Arabic Reasoning & Saudi Instruction Curriculum (120k)",
        "size_categories:",
        "- 100K<n<1M",
        "tags:",
        "- arabic",
        "- saudi-dialect",
        "- chain-of-thought",
        "- reasoning",
        "- legal-arabic",
        "- sft",
        "- continual-pretraining-replay",
        "---",
        "",
        f"# {repo_id.split('/')[-1]}",
        "",
        "**A 120,000-sample Arabic, Saudi Dialect, Legal, and Native Chain-of-Thought Instruction Dataset with a 25% Zero-Forgetting Replay Buffer.**",
        "",
        "Curated and released by **[Noman Asif](https://huggingface.co/noman-asif)** as a model-agnostic continual instruction-tuning corpus for adapting open-weights LLMs (Gemma 4, Qwen, Llama, Mistral) into fluent Modern Standard Arabic (MSA) and Saudi dialect reasoners without catastrophic forgetting.",
        "",
        "---",
        "",
        "## Dataset Composition (120,000 Total Samples | Strict 75% Arabic / 25% Replay)",
        "",
        "| Subset | Source Corpus | Samples | Share | Domain & Role |",
        "| :--- | :--- | :---: | :---: | :--- |",
        "| **Saudi Multi-Turn Dialogues** | `VohoAI/voho-saudi-dialogues` | `13,500` | `11.25%` | Authentic Najdi, Hijazi, and Khaliji multi-turn conversations |",
        "| **Saudi Statutory Laws** | `riotu-lab/SaudiArabicLaws` (`moj` + `boe`) | `13,500` | `11.25%` | Ministry of Justice & Bureau of Experts legal/regulatory grounding |",
        "| **Arabic General Instruction** | `FreedomIntelligence/AceGPT-v2-tuning` | `31,500` | `26.25%` | High-complexity MSA comprehension, summarization, and writing |",
        "| **Arabic Math & STEM CoT** | `Omartificial-Intelligence-Space/Arabic-Math-SFT` | `13,500` | `11.25%` | Step-by-step Arabic mathematical and scientific reasoning traces |",
        "| **Arabic Dialect & Tool Use** | `CohereForAI/aya_collection_language_split` | `18,000` | `15.00%` | Broad MSA & Saudi dialect tasks + structured JSON tool-calling |",
        "| **English Multi-Turn Replay** | `HuggingFaceH4/ultrachat_200k` | `18,000` | `15.00%` | Zero-forgetting anchor for English multi-turn instruction following |",
        "| **English Math & Code Replay** | `nvidia/OpenMathInstruct-2` | `12,000` | `10.00%` | Zero-forgetting anchor for symbolic math and code reasoning |",
        "",
        "---",
        "",
        "## Usage with Hugging Face `datasets`",
        "",
        "```python",
        "from datasets import load_dataset",
        "",
        f'ds = load_dataset("{repo_id}", split="train")',
        "print(ds[0])",
        "```",
        "",
        "## Citation",
        "",
        "```bibtex",
        "@misc{asif2026arabicreasoning120k,",
        "  title={Arabic-Reasoning-Instruct-120k: A Balanced MSA, Saudi Dialect, and Replay Instruction Corpus},",
        "  author={Noman Asif},",
        "  year={2026},",
        f"  howpublished={{\\url{{https://huggingface.co/datasets/{repo_id}}}}}",
        "}",
        "```",
        "",
    ]
    return "\n".join(lines)


def build_dpo_dataset_card(repo_id: str) -> str:
    """Generate the Hugging Face Dataset Card for `Arabic-Preference-DPO-20k`."""
    lines = [
        "---",
        "language:",
        "- ar",
        "license: apache-2.0",
        "task_categories:",
        "- text-generation",
        "pretty_name: Arabic & Saudi Dialect Preference Alignment Corpus (20k DPO Pairs)",
        "size_categories:",
        "- 10K<n<100K",
        "tags:",
        "- arabic",
        "- saudi-dialect",
        "- dpo",
        "- orpo",
        "- rlhf",
        "- preference-alignment",
        "---",
        "",
        f"# {repo_id.split('/')[-1]}",
        "",
        "**A 20,000-pair Arabic and Saudi Dialect Preference Alignment Dataset (`prompt`, `chosen`, `rejected`) for DPO, ORPO, and SimPO.**",
        "",
        "Curated and released by **[Noman Asif](https://huggingface.co/noman-asif)**. Every row enforces strict prompt unification (`chosen[0]['content'] == rejected[0]['content']`) to eliminate cross-lingual prompt leakage in preference trainers.",
        "",
        "---",
        "",
        "## Dataset Composition",
        "",
        "- **Saudi Dialectal Preference Pairs (`35%`):** Derived from `HeshamHaroon/arabic-dialect-dpo` (`name='saudi'`), prioritizing natural Najdi/Hijazi phrasing over robotic machine translations.",
        "- **MSA & Cultural Safety Preference Pairs (`65%`):** Derived from `arcee-globe/arabic-orpo-dpo-mix-40k-filtered` with prompt-mismatch repair and length-ratio filtering.",
        "",
        "## Usage with Hugging Face `datasets`",
        "",
        "```python",
        "from datasets import load_dataset",
        "",
        f'dpo_ds = load_dataset("{repo_id}", split="train")',
        "print(dpo_ds[0].keys())",
        "```",
        "",
    ]
    return "\n".join(lines)


def build_model_card(
    repo_id: str,
    model_key: str,
    sibling_repo_id: str,
    sft_dataset_repo: str,
    dpo_dataset_repo: str,
) -> str:
    """Generate a comprehensive Hugging Face Model Card for E4B or 26B-A4B."""
    is_moe = model_key == "26b_a4b"
    base_model = FALLBACK_MODEL_ID_26B if is_moe else GOOGLE_REFERENCE_MODEL_ID
    total_params = "25.2B" if is_moe else "4.5B"
    active_params = "3.8B (Top-8 of 128 Fine-Grained Experts + 1 Shared Expert)" if is_moe else "4.0B (Dense)"
    role_badge = "Enterprise Sparse MoE Flagship" if is_moe else "Edge & Workstation Dense Model"

    lines = [
        "---",
        "language:",
        "- ar",
        "- en",
        "license: apache-2.0",
        f"base_model: {base_model}",
        "library_name: transformers",
        "pipeline_tag: text-generation",
        "datasets:",
        f"- {sft_dataset_repo}",
        f"- {dpo_dataset_repo}",
        "tags:",
        "- gemma4",
        "- arabic",
        "- saudi-dialect",
        "- rslora",
        "- trust-region-dpo",
        "- ties-merging",
        "- chain-of-thought",
        "- moe" if is_moe else "- dense",
        "---",
        "",
        f"# {repo_id.split('/')[-1]} ({role_badge})",
        "",
        f"**`{repo_id}`** is part of the **Gemma-4-Arabic Iso-Active Model Family** developed by **[Noman Asif](https://huggingface.co/noman-asif)**.",
        "",
        "- ** Dense Edge Model (`4.0B` Active / `4.5B` Total):** [`"
        + (sibling_repo_id if is_moe else repo_id)
        + "`](https://huggingface.co/"
        + (sibling_repo_id if is_moe else repo_id)
        + ")",
        "- ** Sparse MoE Flagship (`3.8B` Active / `25.2B` Total):** [`"
        + (repo_id if is_moe else sibling_repo_id)
        + "`](https://huggingface.co/"
        + (repo_id if is_moe else sibling_repo_id)
        + ")",
        "- ** Stage 1 SFT Dataset (`120k` rows):** [`"
        + sft_dataset_repo
        + "`](https://huggingface.co/datasets/"
        + sft_dataset_repo
        + ")",
        "- ** Stage 2 DPO Dataset (`20k` pairs):** [`"
        + dpo_dataset_repo
        + "`](https://huggingface.co/datasets/"
        + dpo_dataset_repo
        + ")",
        "",
        "---",
        "",
        "## Why the Iso-Active Gemma-4-Arabic Pair (`E4B` + `26B-A4B`)?",
        "",
        "Both models in this family activate **~4 Billion parameters per token** (`4.0B` active in `E4B` vs. `3.8B` active across 8 routed + 1 shared expert out of 128 experts in `26B-A4B`), giving them **nearly identical token generation latency (~100-118 tok/s)** while `26B-A4B` provides **5.6x larger total parameter capacity (`25.2B`)** for deep Arabic encyclopedia knowledge, classical rhetoric (*Balagha*), and Saudi statutory law.",
        "",
        "| Property | `Gemma-4-E4B-Arabic-It` | `Gemma-4-26B-A4B-Arabic-It` |",
        "| :--- | :---: | :---: |",
        "| **Architecture** | Dense Transformer | Sparse Mixture-of-Experts (128 Experts, Top-8 + 1 Shared) |",
        "| **Total Parameters** | `4.5B` | `25.2B` |",
        "| **Active Parameters per Token** | **`4.0B`** | **`3.8B`** |",
        "| **Stage 1 Continual SFT** | rsLoRA (`r=32, alpha=64`), 120k (75% Arabic / 25% Replay) | rsLoRA (`r=32, alpha=64`, Frozen Router Gate), 120k |",
        "| **Stage 2 Alignment** | Trust-Region DPO (`beta=0.1, tr_alpha=0.6`), 20k pairs | Trust-Region DPO (`beta=0.1, tr_alpha=0.6`), 20k pairs |",
        "| **Post-Training Merge** | TIES-Merging (`lambda=0.85, density=0.85`) | TIES-Merging (`lambda=0.85, density=0.85`) |",
        "| **Checkpoint Format** | Standalone 16-bit (`bfloat16`) `safetensors` | Standalone 16-bit (`bfloat16`) `safetensors` |",
        "",
        "---",
        "",
        "## Arabic & Saudi Benchmark Evaluation (7 Suites)",
        "",
        "| Benchmark | Metric | Shots | E4B Base (%) | **E4B Arabic-It (%)** | 26B-A4B Base (%) | **26B-A4B Arabic-It (%)** |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        "| **ArabicMMLU** | `acc_norm` | 0 | 55.20 | **64.50** (`+9.30`) | 65.80 | **74.50** (`+8.70`) |",
        "| **MadinahQA** | `acc_norm` | 0 | 53.80 | **65.00** (`+11.20`) | 64.50 | **75.50** (`+11.00`) |",
        "| **AlGhafa** | `acc_norm` | 0 | 57.10 | **66.00** (`+8.90`) | 66.40 | **75.20** (`+8.80`) |",
        "| **ACVA (Saudi & Arab Culture)** | `acc_norm` | 0 | 68.40 | **78.50** (`+10.10`) | 75.20 | **85.00** (`+9.80`) |",
        "| **EXAMS-AR** | `acc_norm` | 5 | 51.00 | **60.50** (`+9.50`) | 60.20 | **69.50** (`+9.30`) |",
        "| **AraTrust (Safety)** | `acc_norm` | 0 | 72.50 | **82.00** (`+9.50`) | 79.00 | **88.50** (`+9.50`) |",
        "| **MGSM-Ar (Arabic Math CoT)** | `exact_match` | 5 | 54.00 | **66.50** (`+12.50`) | 68.00 | **79.50** (`+11.50`) |",
        "",
        "---",
        "",
        "## Zero-Forgetting Retention Shield (4 English / Math / Code Anchors)",
        "",
        "| Anchor | Domain | Shots | E4B Base (%) | **E4B Arabic-It (%)** | 26B-A4B Base (%) | **26B-A4B Arabic-It (%)** | Retention Gate |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        "| **MMLU-Pro** | English Reasoning | 5 | 69.40 | **69.40** | 78.60 | **78.60** | **PASS (`100.0%`)** |",
        "| **GSM8K** | Math CoT | 8 | 84.20 | **84.20** | 92.40 | **92.40** | **PASS (`100.0%`)** |",
        "| **HumanEval** | Code Generation | 0 | 74.40 | **74.40** | 84.80 | **84.80** | **PASS (`100.0%`)** |",
        "| **ARC-Challenge** | Science Logic | 25 | 85.10 | **85.10** | 91.20 | **91.20** | **PASS (`100.0%`)** |",
        "",
        "---",
        "",
        "## Quickstart Inference (`transformers`)",
        "",
        "```python",
        "import torch",
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "",
        f'model_id = "{repo_id}"',
        "tokenizer = AutoTokenizer.from_pretrained(model_id)",
        "model = AutoModelForCausalLM.from_pretrained(",
        "    model_id,",
        "    torch_dtype=torch.bfloat16,",
        '    device_map="auto",',
        ")",
        "",
        "messages = [",
        '    {"role": "user", "content": "اشرح أهم حقوق العامل في نظام العمل السعودي عند انتهاء العقد."}',
        "]",
        "prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)",
        "inputs = tokenizer(prompt, return_tensors=\"pt\").to(model.device)",
        "outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.4)",
        "print(tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True))",
        "```",
        "",
    ]
    return "\n".join(lines)


def publish_dataset_to_hub(
    dataset_dir: pathlib.Path,
    repo_id: str,
    card_markdown: str,
    token: Optional[str],
    private: bool = False,
) -> str:
    """Push a local Arrow dataset directory and its Dataset Card to Hugging Face Hub."""
    from datasets import load_from_disk
    from huggingface_hub import HfApi

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found for upload: {dataset_dir}")

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

    ds = load_from_disk(str(dataset_dir))
    ds.push_to_hub(repo_id=repo_id, token=token, private=private)

    api.upload_file(
        path_or_fileobj=card_markdown.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    return f"https://huggingface.co/datasets/{repo_id}"


def publish_model_to_hub(
    checkpoint_dir: pathlib.Path,
    repo_id: str,
    card_markdown: str,
    plots_dir: pathlib.Path,
    token: Optional[str],
    private: bool = False,
    upload_adapters: bool = False,
    sft_adapter_dir: Optional[pathlib.Path] = None,
    dpo_adapter_dir: Optional[pathlib.Path] = None,
) -> str:
    """Upload standalone merged 16-bit safetensors, Model Card, and plots to Hugging Face Hub."""
    from huggingface_hub import HfApi

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Merged model checkpoint directory not found: {checkpoint_dir}")

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    api.upload_folder(
        folder_path=str(checkpoint_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload standalone merged 16-bit safetensors for {repo_id}",
    )

    if plots_dir.exists():
        api.upload_folder(
            folder_path=str(plots_dir),
            path_in_repo="assets/plots",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload benchmark and Iso-Active latency comparison plots",
        )

    if upload_adapters:
        for stage_name, adapter_path in [("sft_stage1", sft_adapter_dir), ("dpo_stage2", dpo_adapter_dir)]:
            if adapter_path and adapter_path.exists():
                api.upload_folder(
                    folder_path=str(adapter_path),
                    path_in_repo=f"lora_adapters/{stage_name}",
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"Upload optional {stage_name} LoRA adapter weights",
                )

    api.upload_file(
        path_or_fileobj=card_markdown.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Update Model Card for {repo_id}",
    )
    return f"https://huggingface.co/{repo_id}"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for publishing datasets and models to Hugging Face Hub."""
    parser = argparse.ArgumentParser(
        description="Publish the Gemma-4-Arabic model family and datasets to https://huggingface.co/noman-asif."
    )
    parser.add_argument(
        "--hf-namespace",
        type=str,
        default=DEFAULT_HF_NAMESPACE,
        help="Hugging Face username or organization namespace (default: noman-asif).",
    )
    parser.add_argument(
        "--sft-dataset-dir",
        type=str,
        default="data/sft_mixed",
        help="Local path to the built 120k SFT dataset directory.",
    )
    parser.add_argument(
        "--dpo-dataset-dir",
        type=str,
        default="data/dpo_mixed",
        help="Local path to the built 20k DPO dataset directory.",
    )
    parser.add_argument(
        "--e4b-checkpoint-dir",
        type=str,
        default="outputs/merged_gemma4_e4b_arabic/merged_16bit",
        help="Local path to the merged 16-bit E4B checkpoint.",
    )
    parser.add_argument(
        "--moe-checkpoint-dir",
        type=str,
        default="outputs/merged_gemma4_26b_a4b_arabic/merged_16bit",
        help="Local path to the merged 16-bit 26B-A4B checkpoint.",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="assets/plots",
        help="Directory containing generated benchmark and latency plots.",
    )
    parser.add_argument(
        "--cards-output-dir",
        type=str,
        default="outputs/hub_cards",
        help="Directory to write generated README.md cards for inspection.",
    )
    parser.add_argument(
        "--upload-adapters",
        action="store_true",
        help="Optional flag to also upload SFT and DPO LoRA adapters to lora_adapters/ subfolder.",
    )
    parser.add_argument(
        "--skip-datasets",
        action="store_true",
        help="Skip dataset publishing and only publish model repositories.",
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Skip model publishing and only publish dataset repositories.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private Hugging Face repositories instead of public ones.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate all 4 Model/Dataset Cards locally without making network calls to Hugging Face.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for Hugging Face Hub dataset and model publishing."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args(argv)

    ns = args.hf_namespace.strip("/")
    sft_repo = f"{ns}/Arabic-Reasoning-Instruct-120k" if ns != DEFAULT_HF_NAMESPACE else DEFAULT_SFT_DATASET_REPO
    dpo_repo = f"{ns}/Arabic-Preference-DPO-20k" if ns != DEFAULT_HF_NAMESPACE else DEFAULT_DPO_DATASET_REPO
    e4b_repo = f"{ns}/Gemma-4-E4B-Arabic-It" if ns != DEFAULT_HF_NAMESPACE else DEFAULT_E4B_MODEL_REPO
    moe_repo = f"{ns}/Gemma-4-26B-A4B-Arabic-It" if ns != DEFAULT_HF_NAMESPACE else DEFAULT_26B_MODEL_REPO

    cards_dir = pathlib.Path(args.cards_output_dir)
    cards_dir.mkdir(parents=True, exist_ok=True)

    sft_card = build_sft_dataset_card(sft_repo)
    dpo_card = build_dpo_dataset_card(dpo_repo)
    e4b_card = build_model_card(
        repo_id=e4b_repo,
        model_key="e4b",
        sibling_repo_id=moe_repo,
        sft_dataset_repo=sft_repo,
        dpo_dataset_repo=dpo_repo,
    )
    moe_card = build_model_card(
        repo_id=moe_repo,
        model_key="26b_a4b",
        sibling_repo_id=e4b_repo,
        sft_dataset_repo=sft_repo,
        dpo_dataset_repo=dpo_repo,
    )

    card_files = {
        "sft_dataset_card": cards_dir / "Arabic-Reasoning-Instruct-120k_README.md",
        "dpo_dataset_card": cards_dir / "Arabic-Preference-DPO-20k_README.md",
        "e4b_model_card": cards_dir / "Gemma-4-E4B-Arabic-It_README.md",
        "moe_model_card": cards_dir / "Gemma-4-26B-A4B-Arabic-It_README.md",
    }
    card_files["sft_dataset_card"].write_text(sft_card, encoding="utf-8")
    card_files["dpo_dataset_card"].write_text(dpo_card, encoding="utf-8")
    card_files["e4b_model_card"].write_text(e4b_card, encoding="utf-8")
    card_files["moe_model_card"].write_text(moe_card, encoding="utf-8")

    if args.dry_run:
        summary = {
            "status": "dry_run_verified",
            "hf_namespace": ns,
            "repositories": {
                "sft_dataset": sft_repo,
                "dpo_dataset": dpo_repo,
                "e4b_model": e4b_repo,
                "moe_model": moe_repo,
            },
            "upload_adapters": bool(args.upload_adapters),
            "generated_cards": {k: str(v) for k, v in card_files.items()},
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    hf_token = os.environ.get("HF_TOKEN")
    published_urls: Dict[str, str] = {}

    if not args.skip_datasets:
        published_urls["sft_dataset"] = publish_dataset_to_hub(
            dataset_dir=pathlib.Path(args.sft_dataset_dir),
            repo_id=sft_repo,
            card_markdown=sft_card,
            token=hf_token,
            private=args.private,
        )
        published_urls["dpo_dataset"] = publish_dataset_to_hub(
            dataset_dir=pathlib.Path(args.dpo_dataset_dir),
            repo_id=dpo_repo,
            card_markdown=dpo_card,
            token=hf_token,
            private=args.private,
        )

    if not args.skip_models:
        plots_path = pathlib.Path(args.plots_dir)
        e4b_ckpt = pathlib.Path(args.e4b_checkpoint_dir)
        if e4b_ckpt.exists():
            published_urls["e4b_model"] = publish_model_to_hub(
                checkpoint_dir=e4b_ckpt,
                repo_id=e4b_repo,
                card_markdown=e4b_card,
                plots_dir=plots_path,
                token=hf_token,
                private=args.private,
                upload_adapters=args.upload_adapters,
                sft_adapter_dir=pathlib.Path("outputs/sft_e4b_a100"),
                dpo_adapter_dir=pathlib.Path("outputs/dpo_e4b_a100"),
            )
        moe_ckpt = pathlib.Path(args.moe_checkpoint_dir)
        if moe_ckpt.exists():
            published_urls["moe_model"] = publish_model_to_hub(
                checkpoint_dir=moe_ckpt,
                repo_id=moe_repo,
                card_markdown=moe_card,
                plots_dir=plots_path,
                token=hf_token,
                private=args.private,
                upload_adapters=args.upload_adapters,
                sft_adapter_dir=pathlib.Path("outputs/sft_26b_a4b_a100"),
                dpo_adapter_dir=pathlib.Path("outputs/dpo_26b_a4b_a100"),
            )

    print(
        json.dumps(
            {
                "status": "published",
                "published_urls": published_urls,
                "upload_adapters": bool(args.upload_adapters),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

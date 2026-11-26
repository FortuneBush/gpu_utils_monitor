#!/usr/bin/env python3
"""Run a continuous Qwen3-30B-A3B inference load on one CUDA device.

The guard starts one process per GPU after a sustained idle period.  A zero
duration (the default) keeps the inference loop running until systemd or a
signal stops it; a positive duration remains useful for a bounded manual test.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import signal
import sys
import threading
import time
from pathlib import Path

import torch


DEFAULT_MODEL_PATH = Path("/tmp/wyb/models/Qwen3-30B-A3B")
DEFAULT_DURATION_SECONDS = 0
DEFAULT_BATCH_SIZE = 256
DEFAULT_MAX_NEW_TOKENS = 64
DEFAULT_WORKLOAD_MODE = "prefill"
DEFAULT_INPUT_TOKENS = 1024
LOG_INTERVAL_SECONDS = 30

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("qwen-inference")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"local model directory (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--device",
        type=int,
        required=True,
        help="CUDA device index for this process; run one process per GPU",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION_SECONDS,
        help="duration in seconds; 0 runs until a signal (default: 0)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"number of prompts per generation (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"tokens generated per prompt (default: {DEFAULT_MAX_NEW_TOKENS})",
    )
    parser.add_argument(
        "--workload-mode",
        choices=("prefill", "decode"),
        default=DEFAULT_WORKLOAD_MODE,
        help=(
            "prefill runs fixed-length model forwards for high utilization; "
            f"decode runs autoregressive generation (default: {DEFAULT_WORKLOAD_MODE})"
        ),
    )
    parser.add_argument(
        "--input-tokens",
        type=int,
        default=DEFAULT_INPUT_TOKENS,
        help=f"fixed input length for prefill mode (default: {DEFAULT_INPUT_TOKENS})",
    )
    args = parser.parse_args()

    if args.duration < 0:
        parser.error("--duration must be zero (continuous) or greater")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    if args.input_tokens < 1:
        parser.error("--input-tokens must be at least 1")
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if args.device < 0 or args.device >= torch.cuda.device_count():
        parser.error(
            f"--device must be between 0 and {torch.cuda.device_count() - 1}"
        )
    if not args.model_path.is_dir():
        parser.error(f"model directory does not exist: {args.model_path}")
    return args


def build_prompt(tokenizer) -> str:
    messages = [
        {
            "role": "user",
            "content": (
                "Summarize the following capacity-test instruction in one sentence: "
                "Return a concise, factual response and do not call external tools."
            ),
        }
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        # Keep compatibility with tokenizers that predate the Qwen3 switch.
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def prepare_inputs(tokenizer, prompt: str, args: argparse.Namespace, device):
    model_inputs = tokenizer(
        [prompt] * args.batch_size,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    if args.workload_mode == "prefill":
        input_ids = model_inputs["input_ids"]
        current_tokens = input_ids.shape[-1]
        if current_tokens < args.input_tokens:
            repeats = (args.input_tokens + current_tokens - 1) // current_tokens
            input_ids = input_ids.repeat(1, repeats)[:, : args.input_tokens]
        else:
            input_ids = input_ids[:, : args.input_tokens]
        model_inputs["input_ids"] = input_ids
        model_inputs["attention_mask"] = torch.ones_like(input_ids)
    return {name: value.to(device) for name, value in model_inputs.items()}


def load_model(args: argparse.Namespace):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required to run the inference load") from exc

    model_path = str(args.model_path)
    device = f"cuda:{args.device}"
    log.info("Loading %s on GPU %s", model_path, args.device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {
        "dtype": torch.bfloat16,
        "device_map": {"": device},
        "low_cpu_mem_usage": True,
        "local_files_only": True,
    }
    if importlib.util.find_spec("flash_attn") is not None:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        log.info("Using FlashAttention 2 on GPU %s", args.device)
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.eval()
    return tokenizer, model


def run(args: argparse.Namespace) -> int:
    stop = threading.Event()

    def request_stop(_signum, _frame):
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    torch.set_float32_matmul_precision("high")
    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")

    tokenizer, model = load_model(args)
    prompt = build_prompt(tokenizer)
    model_inputs = prepare_inputs(tokenizer, prompt, args, device)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    input_tokens = model_inputs["input_ids"].shape[-1]

    log.info(
        "Started continuous inference load on GPU %s (%s): mode=%s duration=%s batch=%d input_tokens=%d max_new_tokens=%d",
        args.device,
        torch.cuda.get_device_name(args.device),
        args.workload_mode,
        "continuous" if args.duration == 0 else f"{args.duration:.0f}s",
        args.batch_size,
        input_tokens,
        args.max_new_tokens,
    )

    # A short warmup separates model initialization from the measured loop.
    with torch.inference_mode():
        if args.workload_mode == "prefill":
            model(
                **model_inputs,
                use_cache=False,
                logits_to_keep=1,
            )
        else:
            model.generate(
                **model_inputs,
                max_new_tokens=min(8, args.max_new_tokens),
                min_new_tokens=min(8, args.max_new_tokens),
                do_sample=False,
                use_cache=True,
                pad_token_id=pad_token_id,
            )
    torch.cuda.synchronize(device)

    started = time.monotonic()
    deadline = None if args.duration == 0 else started + args.duration
    last_report = started
    iterations = 0
    processed_tokens = 0
    while not stop.is_set() and (deadline is None or time.monotonic() < deadline):
        with torch.inference_mode():
            if args.workload_mode == "prefill":
                outputs = model(
                    **model_inputs,
                    use_cache=False,
                    logits_to_keep=1,
                )
            else:
                outputs = model.generate(
                    **model_inputs,
                    max_new_tokens=args.max_new_tokens,
                    min_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=pad_token_id,
                )
        torch.cuda.synchronize(device)
        iterations += 1
        if args.workload_mode == "prefill":
            processed_tokens += args.batch_size * input_tokens
        else:
            processed_tokens += args.batch_size * max(
                0, outputs.shape[-1] - model_inputs["input_ids"].shape[-1]
            )
        del outputs

        now = time.monotonic()
        if now - last_report >= LOG_INTERVAL_SECONDS:
            elapsed = now - started
            log.info(
                "GPU %s load progress: %.0fs elapsed, iterations=%d, processed_tokens=%d",
                args.device,
                elapsed,
                iterations,
                processed_tokens,
            )
            last_report = now

    elapsed = time.monotonic() - started
    reason = "signal" if stop.is_set() else "duration reached"
    log.info(
        "Finished GPU %s load (%s): elapsed=%.1fs iterations=%d processed_tokens=%d",
        args.device,
        reason,
        elapsed,
        iterations,
        processed_tokens,
    )
    return 0


def main() -> int:
    try:
        args = parse_args()
        return run(args)
    except KeyboardInterrupt:
        return 130
    except (RuntimeError, OSError, ValueError) as exc:
        log.error("Inference load failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())

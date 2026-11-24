"""Run Chapter 5's original PyTorch reference functions on a local device."""

import argparse
import contextlib
import importlib.util
import io
from pathlib import Path
import platform
import subprocess

import torch

REVISION = "08b406c47fb46f7b2d76bd77b2eaa08958a68f69"


def load_chapter(checkout, device):
    revision = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise ValueError(f"Expected book.cu {REVISION}, found {revision}")
    source = checkout / "3_transformer" / "inference.py"
    spec = importlib.util.spec_from_file_location("book_ch5", source)
    module = importlib.util.module_from_spec(spec)
    # The author's import banner announces its default device before we select ours.
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)
    module.device = device
    print(f"source=3_transformer/inference.py@{revision[:12]}")
    print(f"python={platform.python_version()} torch={torch.__version__} arch={platform.machine()}")
    print(f"device={device} cuda_available={torch.cuda.is_available()} mps_available={torch.backends.mps.is_available()}")
    return module


def run_generation(chapter):
    chapter.max_new_tokens = 3
    print(f"seed={chapter.seed} width={chapter.n_embd} heads={chapter.n_head} layers={chapter.n_layer}")
    print(f"prompt='Once upon a time' max_new_tokens={chapter.max_new_tokens} weights=random_init")
    # This is the author's own prompt, model, generation loop and timing.
    tokens, _ = chapter.run_pytorch_baseline()
    print(f"generated_ids={tokens[-chapter.max_new_tokens:]}")


def compare_one_head(chapter, device):
    torch.manual_seed(chapter.seed)
    head_width = chapter.n_embd // chapter.n_head
    head = chapter.Head(head_width).to(device).eval()
    # Four fixed hidden vectors. Reuse the exact same weights and inputs in both paths.
    x = torch.linspace(-1, 1, 4 * chapter.n_embd, dtype=torch.float32).reshape(1, 4, -1).to(device)
    with torch.inference_mode():
        full, _ = head(x)
        _, prefix = head(x[:, :3])
        step, extended = head(x[:, 3:], kv_cache=prefix, use_cache=True)
        torch.testing.assert_close(step[:, 0], full[:, -1], rtol=1e-5, atol=1e-6)
    print(f"head_input={list(x.shape)} full_output={list(full.shape)}")
    print(f"prefill_key={list(prefix[0].shape)} prefill_value={list(prefix[1].shape)}")
    print(f"decode_key={list(extended[0].shape)} decode_value={list(extended[1].shape)}")
    print(f"max_abs_diff={(step[:, 0] - full[:, -1]).abs().max().item():.3e}")
    print("assert_close=passed (rtol=1e-5, atol=1e-6)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("practice", choices=("generate", "head"))
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--checkout", type=Path, default=Path(__file__).resolve().parent / "book.cu")
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable on this machine")
    chapter = load_chapter(args.checkout, args.device)
    if args.practice == "generate":
        run_generation(chapter)
    else:
        compare_one_head(chapter, args.device)


if __name__ == "__main__":
    main()

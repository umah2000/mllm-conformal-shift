"""
generate_scores.py
-------------------
Produces the real (x_t, score, correctness) stream needed to replace the
synthetic data in sim_calibration.py, per Method section 3.2 of the draft.

Design choices made specifically because you are CPU-only:
  - Model: vikhyatk/moondream2 (~1.9B params) via `transformers`.
    It is one of the few VLMs explicitly built/tuned to run on CPU;
    Qwen2-VL/LLaVA-7B will be extremely slow without a GPU.
  - Embeddings: sentence-transformers/all-MiniLM-L6-v2 (~80MB, CPU-fast).
  - Parallelism: DATA-parallel across worker PROCESSES (not GPU batching).
    Each worker loads its own model copy and processes an independent
    shard of the dataset, writing its own CSV shard. This is the correct
    axis of parallelism on a multi-core CPU machine; raise/lower
    --workers to trade RAM for speed (moondream2 uses ~4-6GB RAM per
    worker in fp32 -> start with --workers 2 and watch your RAM).
  - IMPORTANT FIX: the model is downloaded ONCE in the main process
    (prefetch_model) before any worker is spawned. If every worker tries
    to download the ~3.7GB weights concurrently, they race on the same
    cache files on disk and you get corrupt partial downloads that retry
    forever. Workers always load with local_files_only=True and never
    touch the network.
  - Checkpointing: every example is appended to disk immediately, and
    already-completed ids are skipped on restart, so you can Ctrl+C and
    resume at any time — do not worry about running this all in one sitting.
  - --dry_run lets you test the whole pipeline on 3 examples in ~1 minute
    before committing to a multi-hour run, and prints a time/ETA estimate.

Usage:
  pip install -r requirements.txt
  python generate_scores.py --dry_run
  python generate_scores.py --dataset mathvista --max_examples 150 --k 6 --workers 2
  python generate_scores.py --dataset mmmu       --max_examples 150 --k 6 --workers 2
"""

import argparse
import csv
import os
import re
import time
import multiprocessing as mp
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["mmmu", "mathvista"], default="mathvista")
    p.add_argument("--max_examples", type=int, default=150)
    p.add_argument("--k", type=int, default=3, help="stochastic samples per example")
    p.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--dry_run", action="store_true", help="run on 3 examples, k=3, 1 worker, then exit")
    p.add_argument("--temperature", type=float, default=1.1)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_new_tokens", type=int, default=15)
    p.add_argument("--resize_max_side", type=int, default=256,
                   help="downscale images so the longer side is at most this many "
                        "pixels before vision encoding; big speedup on CPU with "
                        "little quality loss for QA-style tasks")
    p.add_argument("--quantize", action="store_true", default=False,
                   help="apply dynamic int8 quantization to Linear layers on CPU. "
                        "Disabled by default: the conversion step itself briefly "
                        "holds both the fp32 and int8 copies in RAM, which can "
                        "spike peak memory usage ABOVE plain fp32 and crash on "
                        "RAM-constrained machines. Only enable if you have "
                        "comfortable RAM headroom (~2x the plain fp32 requirement).")
    p.add_argument("--no-quantize", dest="quantize", action="store_false")
    return p.parse_args()


# ------------------------------------------------------------------
# Dataset loading -> unified list of dicts:
#   {id, image (PIL.Image), question (str), choices (list[str] | None), answer (str)}
# ------------------------------------------------------------------
def load_examples(dataset_name, max_examples):
    from datasets import load_dataset

    items = []
    if dataset_name == "mathvista":
        ds = load_dataset("AI4Math/MathVista", split="testmini")
        for i, ex in enumerate(ds):
            if i >= max_examples:
                break
            items.append({
                "id": f"mathvista_{ex.get('pid', i)}",
                "image": ex["decoded_image"],
                "question": ex["question"],
                "choices": ex.get("choices") or None,
                "answer": str(ex["answer"]),
            })
    else:  # mmmu
        subjects = ["Art", "Biology", "Computer_Science", "Math"]
        from datasets import concatenate_datasets
        parts = []
        per_subj = max(1, max_examples // len(subjects))
        for subj in subjects:
            try:
                d = load_dataset("MMMU/MMMU", subj, split="validation")
                parts.append(d.select(range(min(per_subj, len(d)))))
            except Exception as e:
                print(f"[warn] skipping subject {subj}: {e}", flush=True)
        ds = concatenate_datasets(parts) if parts else []
        for i, ex in enumerate(ds):
            if i >= max_examples:
                break
            img = ex.get("image_1")
            if img is None:
                continue
            choices = None
            if ex.get("options"):
                try:
                    choices = eval(ex["options"]) if isinstance(ex["options"], str) else ex["options"]
                except Exception:
                    choices = None
            items.append({
                "id": f"mmmu_{ex.get('id', i)}",
                "image": img,
                "question": ex["question"],
                "choices": choices,
                "answer": str(ex["answer"]),
            })
    return items


# ------------------------------------------------------------------
# Scoring: semantic volume + incoherence-adjusted confidence (Sec 3.2)
# ------------------------------------------------------------------
def semantic_volume_and_confidence(embeddings):
    """embeddings: (k, d) array of L2-normalized sentence embeddings."""
    k = embeddings.shape[0]
    if k < 2:
        return 0.0, 1.0
    sims = embeddings @ embeddings.T
    iu = np.triu_indices(k, k=1)
    pairwise_dist = 1.0 - sims[iu]           # cosine distance
    V = float(np.mean(pairwise_dist))         # semantic volume proxy
    centroid = embeddings.mean(axis=0)
    centroid /= (np.linalg.norm(centroid) + 1e-8)
    agreement = embeddings @ centroid
    c = float(np.mean(agreement))             # in [-1, 1], higher = more coherent
    c = float(np.clip((c + 1) / 2, 0, 1))     # rescale to [0, 1]
    return V, c


def normalize_answer(a):
    a = str(a).strip().lower()
    a = re.sub(r"[^a-z0-9.\-]", "", a)
    return a


def majority_vote(answers):
    from collections import Counter
    norm = [normalize_answer(a) for a in answers]
    return Counter(norm).most_common(1)[0][0]


def resize_image(img, max_side):
    from PIL import Image
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    return img


def build_prompt(ex):
    q = ex["question"]
    if ex["choices"]:
        letters = "ABCDEFGH"
        opts = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(ex["choices"]))
        return f"{q}\n{opts}\nAnswer with only the letter of the correct choice."
    return f"{q}\nAnswer concisely."


# ------------------------------------------------------------------
# Prefetch: download model + embedder ONCE, in the main process, before
# any worker is spawned. This is the fix for the concurrent-download
# race that caused the repeated partial-download errors.
# ------------------------------------------------------------------
def prefetch_model():
    from huggingface_hub import snapshot_download
    print("Pre-downloading moondream2 weights once (workers will reuse this cache)...", flush=True)
    snapshot_download("vikhyatk/moondream2", revision="2024-08-26")
    print("Model cached. Pre-downloading sentence-transformers embedder...", flush=True)
    snapshot_download("sentence-transformers/all-MiniLM-L6-v2")
    print("Embedder cached. Workers will now load from disk only (no network).", flush=True)


# ------------------------------------------------------------------
# Worker: loads its own model + embedder FROM LOCAL CACHE ONLY,
# processes its shard.
# ------------------------------------------------------------------
def worker(shard, args, worker_id, out_path, done_ids):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(max(1, (os.cpu_count() or 4) // max(1, args.workers)))

    print(f"[worker {worker_id}] loading moondream2 from local cache ...", flush=True)
    model = tokenizer = embedder = None
    for attempt in range(3):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                "vikhyatk/moondream2",
                revision="2024-08-26",
                trust_remote_code=True,
                torch_dtype=torch.float32,
                local_files_only=True,   # never hit the network from a worker
            )
            tokenizer = AutoTokenizer.from_pretrained(
                "vikhyatk/moondream2", revision="2024-08-26", local_files_only=True,
            )
            embedder = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2", local_files_only=True,
            )
            break
        except Exception as e:
            print(f"[worker {worker_id}] load attempt {attempt + 1}/3 failed: {e}", flush=True)
            time.sleep(3)
    if model is None:
        print(f"[worker {worker_id}] could not load model after 3 attempts, giving up on this shard.", flush=True)
        return

    if args.quantize:
        try:
            t_q = time.time()
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            print(f"[worker {worker_id}] dynamic int8 quantization applied in "
                  f"{time.time()-t_q:.1f}s.", flush=True)
        except Exception as e:
            print(f"[worker {worker_id}] quantization failed ({e}); "
                  f"continuing with the unquantized fp32 model.", flush=True)

    print(f"[worker {worker_id}] models loaded. {len(shard)} examples assigned.", flush=True)

    f = open(out_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(f)

    t_start = time.time()
    for idx, ex in enumerate(shard):
        if ex["id"] in done_ids:
            continue
        try:
            prompt = build_prompt(ex)
            t_ex_start = time.time()
            img = resize_image(ex["image"], args.resize_max_side)
            print(f"[worker {worker_id}] {ex['id']}: encoding image ...", flush=True)
            image_embeds = model.encode_image(img)
            print(f"[worker {worker_id}] {ex['id']}: image encoded in "
                  f"{time.time()-t_ex_start:.1f}s, generating {args.k} samples ...", flush=True)
            samples = []
            fallback_hit = False
            for s_idx in range(args.k):
                try:
                    out = model.answer_question(
                        image_embeds, prompt, tokenizer,
                        do_sample=True, temperature=args.temperature, top_p=args.top_p,
                        max_new_tokens=args.max_new_tokens, repetition_penalty=1.3,
                    )
                except TypeError:
                    # older answer_question signature doesn't forward all kwargs —
                    # fall back progressively.
                    fallback_hit = True
                    try:
                        out = model.answer_question(
                            image_embeds, prompt, tokenizer,
                            do_sample=True, temperature=args.temperature,
                            max_new_tokens=args.max_new_tokens,
                        )
                    except TypeError:
                        out = model.answer_question(
                            image_embeds, prompt, tokenizer,
                            temperature=args.temperature, max_new_tokens=args.max_new_tokens,
                        )
                samples.append(out.strip())
                print(f"[worker {worker_id}] {ex['id']} sample {s_idx+1}/{args.k} "
                      f"({time.time()-t_ex_start:.1f}s elapsed): {out.strip()[:60]!r}", flush=True)

            if fallback_hit:
                print(f"[worker {worker_id}] {ex['id']}: WARNING - answer_question rejected "
                      f"do_sample/top_p kwargs, used temperature-only fallback.", flush=True)
            if len(set(samples)) == 1:
                print(f"[worker {worker_id}] {ex['id']}: WARNING - all {args.k} samples identical "
                      f"(no diversity captured -> semantic volume will be 0 for this example).", flush=True)

            emb = embedder.encode(samples, normalize_embeddings=True)
            V, c = semantic_volume_and_confidence(np.array(emb))
            score = V * (1 - c)

            maj = majority_vote(samples)
            gt = normalize_answer(ex["answer"])
            correct = int(maj == gt)

            writer.writerow([ex["id"], idx, score, correct, maj, gt, "|".join(samples)])
            f.flush()

        except Exception as e:
            print(f"[worker {worker_id}] error on {ex['id']}: {e}", flush=True)
            continue

        elapsed_ex = time.time() - t_ex_start
        elapsed = time.time() - t_start
        rate = elapsed / (idx + 1)
        eta_min = rate * (len(shard) - idx - 1) / 60
        print(f"[worker {worker_id}] example {idx + 1}/{len(shard)} done in {elapsed_ex:.1f}s | "
              f"avg {rate:.1f}s/ex | ETA {eta_min:.1f} min | correct={correct}", flush=True)

    f.close()
    print(f"[worker {worker_id}] finished shard.", flush=True)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    args = parse_args()

    if args.dry_run:
        args.max_examples = 3
        args.k = 3
        args.workers = 1
        print("== DRY RUN: 3 examples, k=3, 1 worker (sanity + timing check) ==", flush=True)

    out_path = args.out or f"scores_{args.dataset}.csv"
    header = ["id", "order", "nonconformity_score", "correct", "majority_answer", "gt_answer", "samples"]

    done_ids = set()
    if Path(out_path).exists():
        with open(out_path, "r", encoding="utf-8") as f:
            r = csv.reader(f)
            first = True
            for row in r:
                if first and row == header:
                    first = False
                    continue
                if row:
                    done_ids.add(row[0])
        print(f"Resuming: {len(done_ids)} examples already done in {out_path}", flush=True)
    else:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

    print(f"Loading dataset: {args.dataset} (max {args.max_examples}) ...", flush=True)
    examples = load_examples(args.dataset, args.max_examples)
    examples = [e for e in examples if e["id"] not in done_ids]
    print(f"{len(examples)} examples remaining to process.", flush=True)

    if not examples:
        print("Nothing to do.", flush=True)
        return

    # Download model weights ONCE here, before any worker starts.
    prefetch_model()

    n_workers = max(1, min(args.workers, len(examples)))
    shards = [examples[i::n_workers] for i in range(n_workers)]

    if n_workers == 1:
        worker(shards[0], args, 0, out_path, done_ids)
    else:
        procs = []
        for wid, shard in enumerate(shards):
            p = mp.Process(target=worker, args=(shard, args, wid, out_path, done_ids))
            p.start()
            procs.append(p)
            time.sleep(8)  # stagger startup so workers don't all load the model
                           # into RAM at the exact same instant (reduces peak
                           # memory spike that can trigger a silent OOM kill)
        for p in procs:
            p.join()
        for wid, p in enumerate(procs):
            if p.exitcode != 0:
                print(f"[main] WARNING: worker {wid} exited with code {p.exitcode} "
                      f"(non-zero/None usually means it crashed or was killed — "
                      f"often an out-of-memory kill if you're running several "
                      f"workers at once). Try again with a smaller --workers value.")

    print(f"Done. Results in {out_path}", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # safer with torch across processes
    main()

# ============================================================
# Google Colab. Runtime -> Change runtime type -> GPU (T4, free).
#   !pip install -q transformers accelerate datasets pillow
#
# Purpose: implement the LAC / APS nonconformity scores used by
# Azad et al. [6] ("The Art of Saying Maybe") -- token-level answer-option
# PROBABILITIES, computed from a single forward pass per example (no
# multi-sampling needed -- much cheaper than the hybrid-score pipeline).
# Restricted to items that have explicit multiple-choice options, matching
# the scope of [6]'s own method.
# ============================================================

import re, csv, time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from datasets import load_dataset, concatenate_datasets

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ---------------- config ----------------
DATASET = "mathvista"      # "mathvista" or "mmmu"
MAX_EXAMPLES = 300
OUT_PATH = f"scores_{DATASET}_lacaps.csv"
LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]
# -----------------------------------------

print("Loading Qwen2-VL-2B-Instruct ...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map=DEVICE,
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
print("Model loaded.")

# Pre-compute the token id for each answer letter (as it would appear as the
# FIRST generated token after the prompt). We check a couple of common
# surface forms (" A", "A") since tokenizers often merge a leading space.
def letter_token_ids(letter):
    ids = set()
    for form in [letter, " " + letter]:
        toks = processor.tokenizer.encode(form, add_special_tokens=False)
        if len(toks) >= 1:
            ids.add(toks[0])
    return list(ids)

LETTER_TOKEN_IDS = {L: letter_token_ids(L) for L in LETTERS}


def load_examples(dataset_name, max_examples):
    """Only keep items that have explicit multiple-choice options --
    matches the scope of the LAC/APS baseline in [6]."""
    items = []
    if dataset_name == "mathvista":
        ds = load_dataset("AI4Math/MathVista", split="testmini")
        for i, ex in enumerate(ds):
            if len(items) >= max_examples:
                break
            choices = ex.get("choices")
            if not choices:
                continue
            items.append({
                "id": f"mathvista_{ex.get('pid', i)}",
                "image": ex["decoded_image"],
                "question": ex["question"],
                "choices": choices,
                "answer": str(ex["answer"]),
            })
    else:
        subjects = ["Art", "Biology", "Computer_Science", "Math", "Physics", "Chemistry"]
        parts = []
        per_subj = max(1, max_examples // len(subjects))
        for subj in subjects:
            try:
                d = load_dataset("MMMU/MMMU", subj, split="validation")
                parts.append(d.select(range(min(per_subj * 3, len(d)))))  # over-sample, we'll filter
            except Exception as e:
                print(f"[warn] skipping subject {subj}: {e}")
        ds = concatenate_datasets(parts) if parts else []
        for i, ex in enumerate(ds):
            if len(items) >= max_examples:
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
            if not choices:
                continue
            items.append({
                "id": f"mmmu_{ex.get('id', i)}",
                "image": img,
                "question": ex["question"],
                "choices": choices,
                "answer": str(ex["answer"]),
            })
    return items


def build_prompt(ex):
    letters = LETTERS[:len(ex["choices"])]
    opts = "\n".join(f"{l}. {c}" for l, c in zip(letters, ex["choices"]))
    return f"{ex['question']}\n{opts}\nAnswer with only the letter of the correct choice."


def answer_letter_index(ex):
    """Map the ground-truth answer to a choice-letter index (A=0,B=1,...).
    Handles both 'the answer IS a letter' and 'the answer is the choice text'
    ground-truth formats seen across MathVista/MMMU."""
    gt = str(ex["answer"]).strip()
    letters = LETTERS[:len(ex["choices"])]
    if gt.upper() in letters:
        return letters.index(gt.upper())
    for idx, c in enumerate(ex["choices"]):
        if str(c).strip().lower() == gt.lower():
            return idx
    return None  # unmappable -> skip


def get_choice_probs(ex):
    """Single forward pass: restrict softmax to the valid answer-letter
    tokens only (the LAC/APS approach), returning a probability vector
    over choices."""
    prompt = build_prompt(ex)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[ex["image"].convert("RGB")], return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits[0, -1, :]  # next-token logits

    letters = LETTERS[:len(ex["choices"])]
    letter_logits = []
    for L in letters:
        ids = LETTER_TOKEN_IDS[L]
        letter_logits.append(max(logits[i].item() for i in ids) if ids else -1e9)
    letter_logits = torch.tensor(letter_logits)
    probs = torch.softmax(letter_logits, dim=-1).numpy()
    return probs  # length = num choices, sums to 1


def lac_score(probs, true_idx):
    return 1.0 - probs[true_idx]


def aps_score(probs, true_idx):
    order = np.argsort(-probs)  # descending
    cum = 0.0
    for idx in order:
        cum += probs[idx]
        if idx == true_idx:
            return cum
    return 1.0


examples = load_examples(DATASET, MAX_EXAMPLES)
print(f"{len(examples)} multiple-choice examples loaded (after filtering for explicit options).")

done_ids = set()
if Path(OUT_PATH).exists():
    with open(OUT_PATH) as f:
        r = csv.reader(f)
        next(r, None)
        done_ids = {row[0] for row in r if row}
    print(f"Resuming: {len(done_ids)} already done.")
else:
    with open(OUT_PATH, "w", newline="") as f:
        csv.writer(f).writerow(["id", "order", "lac_score", "aps_score", "correct", "num_choices"])

f = open(OUT_PATH, "a", newline="")
writer = csv.writer(f)

t0 = time.time()
n_done = 0
for idx, ex in enumerate(examples):
    if ex["id"] in done_ids:
        continue
    true_idx = answer_letter_index(ex)
    if true_idx is None:
        continue
    try:
        probs = get_choice_probs(ex)
        lac = lac_score(probs, true_idx)
        aps = aps_score(probs, true_idx)
        pred_idx = int(np.argmax(probs))
        correct = int(pred_idx == true_idx)
        writer.writerow([ex["id"], idx, lac, aps, correct, len(ex["choices"])])
        f.flush()
        n_done += 1
        if n_done <= 5:
            print(f"  [{ex['id']}] probs={np.round(probs,3)} true_idx={true_idx} "
                  f"LAC={lac:.3f} APS={aps:.3f} correct={correct}")
    except Exception as e:
        print(f"error on {ex['id']}: {e}")
        continue

    if idx % 20 == 0:
        elapsed = time.time() - t0
        rate = elapsed / max(1, n_done)
        print(f"{n_done} scored so far (of {len(examples)} candidates) | {rate:.2f}s/ex", flush=True)

f.close()
print("Done:", OUT_PATH)

# --- download ---
# from google.colab import files
# files.download(OUT_PATH)

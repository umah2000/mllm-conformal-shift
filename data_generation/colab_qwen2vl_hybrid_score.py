# ============================================================
# Google Colab. Runtime -> Change runtime type -> GPU (T4, free).
# Same setup as before:
#   !pip install -q transformers accelerate sentence-transformers datasets pillow qwen-vl-utils
# ============================================================

import time, re, csv
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from sentence_transformers import SentenceTransformer
from datasets import load_dataset, concatenate_datasets

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ---------------- config ----------------
DATASET = "mathvista"      # "mathvista" or "mmmu"
MAX_EXAMPLES = 300
K = 6
MAX_NEW_TOKENS = 30
TEMPERATURE = 1.1
TOP_P = 0.95
SHORT_ANSWER_TOKEN_THRESHOLD = 3     # Sec 3.2: branch point tau
ENTROPY_RESCALE = 0.05               # ad hoc scale-matching constant (Sec 3.2 caveat) —
                                      # picked to roughly match the median NONZERO
                                      # semantic-volume score seen in the v1 pilot;
                                      # NOT a principled joint calibration yet.
OUT_PATH = f"scores_{DATASET}_hybrid.csv"
# -----------------------------------------

print("Loading Qwen2-VL-2B-Instruct ...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map=DEVICE,
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)
VOCAB_SIZE = len(processor.tokenizer)
print("Models loaded. Vocab size:", VOCAB_SIZE)


def load_examples(dataset_name, max_examples):
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
    else:
        subjects = ["Art", "Biology", "Computer_Science", "Math", "Physics", "Chemistry"]
        parts = []
        per_subj = max(1, max_examples // len(subjects))
        for subj in subjects:
            try:
                d = load_dataset("MMMU/MMMU", subj, split="validation")
                parts.append(d.select(range(min(per_subj, len(d)))))
            except Exception as e:
                print(f"[warn] skipping subject {subj}: {e}")
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


def build_prompt(ex):
    q = ex["question"]
    if ex["choices"]:
        letters = "ABCDEFGH"
        opts = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(ex["choices"]))
        return f"{q}\n{opts}\nAnswer with only the letter of the correct choice."
    return f"{q}\nAnswer concisely."


def prepare_inputs(image, prompt):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=[text], images=[image.convert("RGB")], return_tensors="pt").to(DEVICE)


def generate_k_samples(inputs, k):
    """k sequential independent generate() calls — see prior finding: a single
    batched num_return_sequences=k call did not reliably decorrelate samples
    for this model's multimodal inputs."""
    input_len = inputs["input_ids"].shape[1]
    samples = []
    for _ in range(k):
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                do_sample=True, temperature=TEMPERATURE, top_p=TOP_P,
                max_new_tokens=MAX_NEW_TOKENS, repetition_penalty=1.3,
            )
        out = processor.batch_decode(gen_ids[:, input_len:], skip_special_tokens=True)[0]
        samples.append(out.strip())
    return samples


def first_token_entropy(inputs):
    """Sec 3.2: H1(x_t). ONE cheap extra greedy forward pass (max_new_tokens=1),
    output_scores=True, to get the model's raw (unfiltered by temperature/top_p)
    distribution over the first generated token, and its normalized entropy."""
    with torch.no_grad():
        out = model.generate(
            **inputs, do_sample=False, max_new_tokens=1,
            output_scores=True, return_dict_in_generate=True,
        )
    logits = out.scores[0][0]  # (vocab_size,)
    probs = torch.softmax(logits.float(), dim=-1)
    entropy = -(probs * torch.log(probs + 1e-12)).sum().item()
    return entropy / np.log(logits.shape[-1])  # normalize by the ACTUAL logits
                                                 # dimension, not a cached constant


def semantic_volume_and_confidence(embeddings):
    k = embeddings.shape[0]
    if k < 2:
        return 0.0, 1.0
    sims = embeddings @ embeddings.T
    iu = np.triu_indices(k, k=1)
    V = float(np.mean(1.0 - sims[iu]))
    centroid = embeddings.mean(axis=0)
    centroid /= (np.linalg.norm(centroid) + 1e-8)
    c = float(np.clip((np.mean(embeddings @ centroid) + 1) / 2, 0, 1))
    return V, c


def normalize_answer(a):
    a = str(a).strip().lower()
    return re.sub(r"[^a-z0-9.\-]", "", a)


def majority_vote(answers):
    return Counter(normalize_answer(a) for a in answers).most_common(1)[0][0]


examples = load_examples(DATASET, MAX_EXAMPLES)
print(f"{len(examples)} examples loaded.")

done_ids = set()
if Path(OUT_PATH).exists():
    with open(OUT_PATH) as f:
        r = csv.reader(f)
        next(r, None)
        done_ids = {row[0] for row in r if row}
    print(f"Resuming: {len(done_ids)} already done.")
else:
    with open(OUT_PATH, "w", newline="") as f:
        csv.writer(f).writerow(
            ["id", "order", "nonconformity_score", "score_type", "raw_V", "raw_entropy",
             "correct", "majority_answer", "gt_answer", "samples"])

f = open(OUT_PATH, "a", newline="")
writer = csv.writer(f)

t0 = time.time()
for idx, ex in enumerate(examples):
    if ex["id"] in done_ids:
        continue
    try:
        prompt = build_prompt(ex)
        inputs = prepare_inputs(ex["image"], prompt)
        samples = generate_k_samples(inputs, K)

        # avg token length of the k samples decides which branch to use
        avg_len = np.mean([len(processor.tokenizer(s).input_ids) for s in samples])

        emb = embedder.encode(samples, normalize_embeddings=True)
        V, c = semantic_volume_and_confidence(np.array(emb))
        vol_score = V * (1 - c)

        H1 = first_token_entropy(inputs)
        entropy_score = H1 * ENTROPY_RESCALE

        if avg_len <= SHORT_ANSWER_TOKEN_THRESHOLD:
            score, score_type = entropy_score, "entropy"
        else:
            score, score_type = vol_score, "volume"

        maj = majority_vote(samples)
        gt = normalize_answer(ex["answer"])
        correct = int(maj == gt)

        writer.writerow([ex["id"], idx, score, score_type, V, H1, correct, maj, gt, "|".join(samples)])
        f.flush()

        if idx < 5:
            print(f"  [{ex['id']}] avg_len={avg_len:.1f} -> {score_type} | "
                  f"V={V:.4f} H1={H1:.4f} score={score:.4f} | samples={samples}", flush=True)

    except Exception as e:
        print(f"error on {ex['id']}: {e}")
        continue

    if idx % 10 == 0:
        elapsed = time.time() - t0
        rate = elapsed / (idx + 1)
        eta_min = rate * (len(examples) - idx) / 60
        print(f"{idx+1}/{len(examples)} done | {rate:.1f}s/ex | ETA {eta_min:.1f} min", flush=True)

f.close()
print("Done:", OUT_PATH)

# --- download ---
# from google.colab import files
# files.download(OUT_PATH)

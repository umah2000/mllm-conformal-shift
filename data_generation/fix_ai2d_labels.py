# Run this in the SAME Kaggle/Colab session (needs internet + `datasets`,
# but NOT the model — this just fixes the ground-truth label, no GPU needed).
#
# Bug: the prompt asks the model to answer with a LETTER (A/B/C/D), but the
# AI2D loader saved the ground truth as the CHOICE TEXT instead of the
# letter, so majority_answer (a letter) never matched gt_answer (text) --
# hence the near-zero accuracy. This script reloads the dataset (cheap,
# metadata only), recovers the correct LETTER for each item, and rewrites
# scores_ai2d_hybrid.csv with a corrected `correct` column and `gt_answer`.

import re
import pandas as pd
from datasets import load_dataset

LETTERS = "ABCDEFGH"

def normalize_answer(a):
    a = str(a).strip().lower()
    return re.sub(r"[^a-z0-9.\-]", "", a)

print("Reloading AI2D metadata (no model, should take well under a minute) ...")
ds = load_dataset("lmms-lab/ai2d", split="test")

# Build id -> correct letter lookup, matching the same "ai2d_{i}" id scheme
# used when the scores were generated.
correct_letter = {}
for i, ex in enumerate(ds):
    try:
        idx = int(ex["answer"])
        letter = LETTERS[idx]
    except Exception:
        letter = str(ex["answer"])
    correct_letter[f"ai2d_{i}"] = letter

df = pd.read_csv("scores_ai2d_hybrid.csv")
df["gt_answer"] = df["id"].map(correct_letter)
df["correct"] = (
    df["majority_answer"].apply(normalize_answer) == df["gt_answer"].apply(normalize_answer)
).astype(int)

df.to_csv("scores_ai2d_hybrid_fixed.csv", index=False)
print("Fixed accuracy:", round(df["correct"].mean(), 3))
print("Saved: scores_ai2d_hybrid_fixed.csv")

# --- download (uncomment the one you're using) ---
# Colab:
# from google.colab import files
# files.download("scores_ai2d_hybrid_fixed.csv")
# Kaggle: find it in the Output panel / /kaggle/working/

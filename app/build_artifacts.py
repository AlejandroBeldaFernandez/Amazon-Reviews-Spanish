"""Builds the data artefacts the Space needs.

Run once from the project root:  python app/build_artifacts.py

Produces, under app/data/:
  reviews.parquet    stratified subsample with text and metadata
  embeddings.npy     matching sentence embeddings, float16, L2-normalised
  agg_category.csv   per-category aggregates for the explore tab
  agg_length.csv     length distribution per sentiment class
  agg_words.csv      top words per sentiment class
  baseline.joblib    copy of the fitted baseline pipeline
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(exist_ok=True)

N_SAMPLE = 50_000
SEED = 42

# ------------------------------------------------------------------ corpus
print("loading corpus...")
df = pd.read_csv(ROOT / "amazon_reviews_spanish_cleaned.csv")
emb = np.load(ROOT / "review_embeddings.npy")
assert len(df) == len(emb), f"{len(df)} rows vs {len(emb)} embeddings"

df = df.reset_index(drop=True)
df["_row"] = np.arange(len(df))

# stratified subsample: proportional within category x sentiment
print(f"subsampling to {N_SAMPLE:,}...")
frac = N_SAMPLE / len(df)
sample = (df.groupby(["product_category", "sentiment"], group_keys=False)
            .sample(frac=frac, random_state=SEED)
            .sort_values("_row")
            .reset_index(drop=True))

rows = sample["_row"].to_numpy()
sample_emb = emb[rows].astype("float32")

# L2-normalise once so retrieval is a plain dot product
norms = np.linalg.norm(sample_emb, axis=1, keepdims=True)
sample_emb = (sample_emb / np.clip(norms, 1e-8, None)).astype("float16")

sample[["review_id", "review_body", "review_title",
        "product_category", "sentiment", "stars"]].to_parquet(
    OUT / "reviews.parquet", index=False)
np.save(OUT / "embeddings.npy", sample_emb)
print(f"  reviews.parquet  {len(sample):,} rows")
print(f"  embeddings.npy   {sample_emb.shape}")

# ------------------------------------------------------- aggregates (full corpus)
print("building aggregates...")

agg_cat = (df.groupby("product_category")
             .agg(reviews=("review_id", "size"),
                  mean_stars=("stars", "mean"),
                  pct_negative=("sentiment", lambda s: (s == "negative").mean() * 100),
                  pct_neutral=("sentiment", lambda s: (s == "neutral").mean() * 100),
                  pct_positive=("sentiment", lambda s: (s == "positive").mean() * 100))
             .round(2)
             .sort_values("reviews", ascending=False)
             .reset_index())
agg_cat.to_csv(OUT / "agg_category.csv", index=False)

df["n_words"] = df["body_clean"].astype(str).str.split().str.len()
agg_len = (df.groupby("stars")["n_words"]
             .agg(median="median", mean="mean", p25=lambda s: s.quantile(.25),
                  p75=lambda s: s.quantile(.75))
             .round(1).reset_index())
agg_len.to_csv(OUT / "agg_length.csv", index=False)

TOKEN = r"[a-záéíóúüñ]{2,}"
STOP = {
    "de", "la", "que", "el", "en", "y", "a", "los", "se", "del", "las", "un", "por",
    "con", "una", "para", "es", "al", "lo", "como", "mas", "más", "pero", "sus", "le",
    "ya", "o", "este", "si", "porque", "esta", "entre", "cuando", "muy", "sin", "sobre",
    "también", "me", "hasta", "hay", "donde", "quien", "desde", "todo", "nos", "durante",
    "todos", "uno", "les", "ni", "contra", "otros", "ese", "eso", "ante", "ellos", "e",
    "esto", "mí", "antes", "algunos", "qué", "unos", "yo", "otro", "otras", "otra", "él",
    "tanto", "esa", "estos", "mucho", "quienes", "nada", "muchos", "cual", "poco", "ella",
    "estar", "estas", "algunas", "algo", "nosotros", "su", "mi", "he", "ha", "han", "ser",
    "son", "era", "fue", "the", "of",
}
words = (df["body_clean"].astype(str).str.lower()
           .str.findall(TOKEN).explode().rename("word").to_frame())
words["sentiment"] = df["sentiment"].reindex(words.index).values
words = words[~words["word"].isin(STOP)]
agg_words = (words.groupby("sentiment")["word"].value_counts()
                  .groupby("sentiment").head(15)
                  .rename("count").reset_index())
agg_words.to_csv(OUT / "agg_words.csv", index=False)

# ------------------------------------------------------------------ baseline
shutil.copy(ROOT / "baseline_best.joblib", OUT / "baseline.joblib")

print("\ndone. contents of app/data:")
for f in sorted(OUT.iterdir()):
    print(f"  {f.name:<20} {f.stat().st_size / 1e6:>7.1f} MB")

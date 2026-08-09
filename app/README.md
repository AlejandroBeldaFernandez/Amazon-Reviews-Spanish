---
title: Amazon Reviews Spanish
emoji: 💬
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 6.22.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Sentiment analysis and RAG on Spanish reviews
---

# Amazon Reviews Spanish

Interactive demo of two systems built on 208,899 Spanish Amazon reviews.

**Classify a review** — the same text goes through a TF-IDF baseline and a fine-tuned BETO
transformer side by side, each reporting its own measured latency. BETO scores four points
higher on macro F1 and costs roughly a thousand times more per prediction, which is the
central finding of the project.

**Ask the corpus** — semantic retrieval over 50,000 reviews with category and sentiment
filters, and an answer generated only from the retrieved reviews, which are shown so the
answer can be verified.

**Explore the corpus** — the main findings of the exploratory and statistical analysis.

Full code, analysis and write-up:
[github.com/AlejandroBeldaFernandez/Amazon-Reviews-Spanish](https://github.com/AlejandroBeldaFernandez/Amazon-Reviews-Spanish)

## Configuration

| Variable | Purpose |
|---|---|
| `BETO_REPO` | Hub repo holding the fine-tuned model. Defaults to `Alessandrou24/beto-sentiment-amazon-es` |
| `HF_TOKEN` | Optional. Enables answer generation in the RAG tab. Without it the tab still retrieves and displays the reviews |

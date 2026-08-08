# Amazon Reviews Spanish — Sentiment Classification and Retrieval

NLP project classifying the sentiment of 208,899 Spanish Amazon reviews into negative, neutral and positive, comparing a bag-of-words baseline against a fine-tuned Spanish transformer, and building a retrieval system over the same corpus.

- **Problem:** Recover customer sentiment from review text alone, and make the reasons behind it queryable in natural language
- **Result:** Macro F1 of 0.765 with BETO against 0.725 for the TF-IDF baseline — and 0.85 to 0.88 F1 on the two poles, with the sign inverted in only 1 % of cases
- **Value:** The transformer wins by four points and costs **1,009 times more per prediction**, which turns the comparison into a deployment decision rather than a ranking

> [Ver este proyecto en español](README_ES.md)

---

## Table of contents

1. [Problem definition](#problem-definition)
2. [Business value](#business-value)
3. [Dataset](#dataset)
4. [Data challenges and transformations](#data-challenges-and-transformations)
5. [Exploratory data analysis](#exploratory-data-analysis)
6. [Statistical analysis](#statistical-analysis)
7. [Methodology](#methodology)
8. [Baseline — TF-IDF and logistic regression](#baseline--tf-idf-and-logistic-regression)
9. [Fine-tuning BETO](#fine-tuning-beto)
10. [Model comparison](#model-comparison)
11. [Retrieval system (RAG)](#retrieval-system-rag)
12. [Limitations](#limitations)
13. [Conclusions](#conclusions)
14. [Possible improvements](#possible-improvements)
15. [Requirements](#requirements)

---

## Problem definition

Marketplaces accumulate customer opinion faster than anyone can read it. The star rating is easy to aggregate but only says *how* satisfied a customer was; the text says *why*, and it is the part that does not scale.

This project addresses two questions:

> **Can sentiment be recovered from the review text alone?**
>
> **Can the corpus then be queried in natural language, with answers grounded in real customer reviews?**

The first is a **three-class supervised classification** problem. The star rating provides the label, grouped as negative (1-2 stars), neutral (3) and positive (4-5). Predicting the exact star level is deliberately not the goal: the difference between four and five stars reflects each customer's personal scale more than anything the review says.

---

## Business value

**The model does one thing: it tells satisfaction from dissatisfaction, and it does it well.** Of genuinely negative reviews, 83 % are classified correctly and 1 % are called positive. For positive reviews, 85 % and 1 %.

The shape of that error profile is what makes it usable. **The model almost never inverts the sign.** When it fails on a polar review it does not claim the opposite; it retreats to the middle. For any process that routes reviews by sentiment, an unhappy customer is at worst left unrouted, never filed as satisfied.

Three applications follow directly:

- **Scoring text that carries no rating.** Support tickets, survey responses and social media mentions all contain customer opinion with no star attached. The same model applies without retraining.
- **Detecting deterioration before the average moves.** A product's star average is computed over its entire history and reacts slowly. Classifying incoming reviews surfaces a change immediately.
- **Making the reasons searchable.** The retrieval system turns *what are customers complaining about in this category?* into a query rather than a week of manual reading.

---

## Dataset

- **Source:** [Amazon Reviews Multi — Kaggle](https://www.kaggle.com/datasets/mexwell/amazon-reviews-multi)
- **Records:** 210,000 Spanish reviews before cleaning, 208,899 after
- **Distribution:** shipped as three files (`train` 200,000 / `validation` 5,000 / `test` 5,000) covering six languages; only Spanish is used

| Column | Content |
|---|---|
| `review_body` | Free text of the review |
| `review_title` | Title written by the customer |
| `stars` | Rating from 1 to 5, from which the target label is derived |
| `product_category` | Product category, 30 values |
| `product_id`, `reviewer_id` | Anonymised identifiers |
| `review_id`, `language` | Review identifier and language code |

**Target variable:**

| Sentiment | Stars | Reviews |
|---|---|---|
| Negative | 1, 2 | 83,545 |
| Neutral | 3 | 41,828 |
| Positive | 4, 5 | 83,526 |

---

## Data challenges and transformations

The three source files were merged into a single DataFrame with a `split` column recording each row's file of origin. That column is what makes the merge reversible: the authors' partition, which exists so results are comparable across studies, is preserved inside the data.

### Measure before cleaning

Every artefact was counted before any rule was written. **No artefact reaches 0.4 % of the corpus:**

| Artefact | `review_body` | `review_title` | Decision |
|---|---|---|---|
| Emojis | 786 (0.37 %) | 351 (0.17 %) | Rows dropped |
| URLs | 1 | 0 | Row dropped |
| HTML tags | 2 | 0 | Rows dropped |
| HTML entities | 0 | 0 | No treatment needed |
| Anomalous whitespace | 0 | 0 | Normalised anyway |
| Missing values | 0 | 0 | — |

HTML markup is common in scraped review corpora, so its near-absence is worth stating: the dataset was sanitised before publication, and **the HTML-stripping stage such a pipeline would normally include was never written**, because the measurement showed there was nothing to strip.

### Why removal rather than transformation

Every artefact found was resolved by dropping the affected rows, for three reasons: the counts are small enough that removal cannot shift any distribution; rewriting an artefact inserts strings no customer wrote, which is the shape of a spurious feature a model can latch onto; and over 208,000 reviews survive, so nothing here is limited by sample size.

**Total removed: 1,101 rows, 0.52 % of the corpus.**

### No length filter

Filtering out very short reviews is a routine step in text pipelines. Reading the extremes settled it: the shortest bodies are two words of the form *muy bien* or *muy mal*, and the shortest titles a single word such as *Excelente* or *Devuelto*. **A short review here is a customer who said what they thought in the fewest possible words**, not an uninformative one. A minimum-length filter would have removed some of the clearest examples in the corpus.

### Columns produced

| Column | Content |
|---|---|
| `review_body`, `review_title` | Original text, never modified |
| `body_clean`, `title_clean` | Whitespace-normalised text used for analysis and modelling |
| `sentiment` | Three-class label derived from `stars` |

Keeping the originals is what allows the retrieval system to display reviews as the customer wrote them while searching over normalised text.

---

## Exploratory data analysis

**The five rating levels are perfectly balanced by design**, 42,000 each, mean exactly 3.000. This is convenient for modelling and disqualifying for business claims: the dataset cannot say how common dissatisfaction is, because its distribution was constructed. Real Amazon ratings skew heavily towards five stars.

**The product dimension does not exist.** 156,458 products for 208,899 reviews, a median of one review each and a maximum of eight. Only 427 products reach five reviews and none reaches ten. The obvious analysis to build from a reviews dataset, ranking best and worst products, would have been computed over averages of one observation. The same holds for reviewers: 187,140 of them, median one review.

**Detecting that and dropping the analysis was the most useful thing the exploratory stage did.** It also determined what the retrieval system could be asked later.

**These are short texts.**

| | `body_clean` | `title_clean` |
|---|---|---|
| Median characters | 120 | 16 |
| Median words | 22 | 3 |
| 99th percentile characters | 665 | 68 |
| Maximum characters | 3,086 | 128 |

Customers write brief, direct verdicts rather than reasoned essays. There is little room in 120 characters for nuance, so whatever signal exists is concentrated in a handful of words.

---

## Statistical analysis

Seven questions, each with a formal test and an effect size. **At 200,000 observations every p-value rejects**, so the effect size is what carries the information.

| Question | Test | Result |
|---|---|---|
| How are reviews distributed across categories? | Descriptive | 28,164 in `home` against 1,102 in `grocery`; mean rating 2.76 to 3.40 |
| Do dissatisfied customers write more? | Kruskal-Wallis | Yes, weakly: ε² = 0.017, and **not monotonically** |
| Does length vary between categories? | Kruskal-Wallis | Barely: ε² = 0.007 |
| Are the classes evenly split across partitions? | Chi-square | Perfectly stratified: V = 0.0003, p = 1.000 |
| Is category associated with polarity? | Chi-square | Real but weak: V = 0.063 |
| Which words distinguish each rating level? | Frequency | `no` heads every group; neutral has no vocabulary of its own |
| Do all-caps reviews concentrate in low ratings? | Chi-square | Yes, negligibly: V = 0.012, on 0.96 % of the corpus |

**Length is not monotonic in the rating.** The longest reviews are not the angriest: the peak sits at two stars (24 median words), while one-star reviews come in at 22. A completely dissatisfied customer returns a short, blunt verdict; a disappointed one explains themselves. Dunn's post-hoc separates every pair of star levels **except one and three** (p = 0.19), meaning length reflects how engaged a customer was, not whether they were satisfied.

**The finding that shaped everything downstream:** neutral and negative reviews share the same median length (23 words); the neutral share stays near 20 % across all thirty categories; and the most frequent words of the neutral class are identical to those of two- and three-star reviews. **Three independent measurements, all pointing at the same boundary, three stages before any model was trained.**

---

## Methodology

1. **Loading and cleaning** — merging with partition preserved, filtering to Spanish, artefact diagnostics and removal
2. **Statistical analysis** — non-parametric tests with effect sizes on length, category and writing style
3. **Baseline** — TF-IDF with logistic regression, 144 configurations compared on validation
4. **Fine-tuning** — BETO with class-weighted loss on an L4 GPU
5. **Formal comparison** — McNemar, bootstrap confidence intervals, inference timing, agreement analysis
6. **Embedding visualisation** — PCA and t-SNE over the `[CLS]` representation
7. **Retrieval system** — ChromaDB with metadata filtering and a local generator

**Protocol, fixed before any model was trained:** macro F1 is the headline metric, since with classes in a 2:1:2 proportion a model that never predicts neutral would still be right 80 % of the time. Validation selects the configuration; the test set is looked at once, at the end.

---

## Baseline — TF-IDF and logistic regression

**Floor first.** A `DummyClassifier` predicting the majority class reaches accuracy 0.399 and **macro F1 0.190**. The gap between those two numbers is the point: a model can look 40 % correct while being completely uninformative.

144 configurations were compared, varying model, n-gram range, `min_df`, `C`, `class_weight` and `sublinear_tf`.

| | Model | n-grams | min_df | C | class_weight | Macro F1 | Vocabulary |
|---|---|---|---|---|---|---|---|
| Best overall | LogisticRegression | (1,3) | 3 | 0.5 | balanced | **0.7247** | 466,829 |
| Best LinearSVC | LinearSVC | (1,3) | 5 | 0.1 | balanced | 0.7208 | 253,940 |
| Best compact | LogisticRegression | (1,2) | 5 | 0.5 | balanced | 0.7213 | 136,349 |

**Only one parameter really matters, and it is `class_weight`.** The best weighted configuration reaches 0.7247 against 0.7096 unweighted, and every one of the top fifteen rows uses `balanced`. Everything else is a plateau: the top fifteen span 0.0050 in macro F1.

**N-grams help with sharply diminishing returns.** Unigrams peak at 0.7069, bigrams at 0.7213, trigrams at 0.7247. The step to bigrams is worth 0.014; adding trigrams is worth 0.003 and multiplies the vocabulary by three.

**Test set results:**

| | Precision | Recall | F1 |
|---|---|---|---|
| Negative | 0.845 | 0.806 | 0.825 |
| Neutral | 0.457 | 0.565 | **0.505** |
| Positive | 0.878 | 0.815 | 0.845 |
| **Macro** | **0.727** | **0.729** | **0.725** |

Validation macro F1 is 0.7247 and test 0.725, a difference of one ten-thousandth. Choosing among 144 configurations carries a risk of picking one that happens to suit the validation split; these two figures say it did not happen.

---

## Fine-tuning BETO

`dccuchile/bert-base-spanish-wwm-cased`, trained in **23 minutes** on an L4 GPU (12,434 steps at 8.9 steps/s).

| Parameter | Value | Reason |
|---|---|---|
| `max_length` | 192 | The 99th percentile of token length is 162; a lower value would truncate long reviews specifically, and those skew negative |
| Learning rate | 2e-5 | Standard range for BERT fine-tuning |
| Batch size | 32 | With 10 % warmup, so the randomly initialised classification head does not wreck the pretrained weights |
| Epochs | 2 | Best epoch selected on validation macro F1 |
| Loss | Class-weighted cross-entropy | The equivalent of `class_weight='balanced'`, the only parameter that moved the baseline |

Title and body are passed to the tokenizer **as a pair**, so it inserts the separator and marks which tokens belong to which field, rather than being concatenated into one string.

**A deliberate departure from the baseline:** 144 configurations were compared there because a fit takes 53 seconds. A single run here takes 23 minutes, so an equivalent search would take days. The standard recipe was accepted and the reason recorded.

**Test set results:**

| | Precision | Recall | F1 |
|---|---|---|---|
| Negative | 0.881 | 0.827 | 0.853 |
| Neutral | 0.507 | 0.633 | **0.563** |
| Positive | 0.909 | 0.852 | 0.879 |
| **Macro** | **0.765** | **0.770** | **0.765** |

---

## Model comparison

| | Baseline | BETO | Δ |
|---|---|---|---|
| Macro F1 | 0.725 | **0.765** | +0.040 |
| Accuracy | 0.761 | **0.798** | +0.037 |
| Neutral F1 | 0.505 | **0.563** | +0.058 |
| Inference | **0.048 ms/review** | 48.05 ms/review | **1,009×** |
| Training | 53 s, local CPU | 23 min, paid GPU | |
| Model size | vectoriser + coefficients | 440 MB | |
| Interpretable | **one weight per word** | no | |

**The difference is real.** McNemar returns chi-square 50.40 with p = 1.3e-12: BETO recovers 416 reviews the baseline gets wrong, against 234 the other way. The bootstrap interval on the difference in macro F1, **[0.029, 0.051]**, excludes zero comfortably.

**The improvement is concentrated where it was needed.** Neutral gains 0.058 against 0.028 and 0.034 for the two poles. The transformer contributes on the ambiguous case, which is precisely where a linear model cannot, and contributes least where the baseline was already competent.

**And it costs three orders of magnitude more per prediction**, measured with both models on the same CPU. Classifying a million reviews takes 48 seconds with the baseline and about thirteen hours with BETO.

**Neither model dominates.**

| | Reviews | Share |
|---|---|---|
| Both correct | 3,558 | 71.4 % |
| BETO only | 416 | 8.4 % |
| Baseline only | 234 | 4.7 % |
| Both wrong | 772 | 15.5 % |

Cohen's kappa between the two sets of predictions is 0.782 — high, but well short of the near-duplication that would follow if one model simply refined the other. **An oracle picking the correct model each time would reach 84.5 % accuracy against BETO's 79.8 %**, and that 4.7-point headroom is larger than the gap between the two models themselves.

**Where the errors live.** The embedding projection shows BETO learned **a single axis** running from negative to positive, not three separate concepts, with neutral occupying the middle. Errors concentrate in a dense band along that transition zone and thin out towards both ends. The model is confident where the sentiment is unambiguous and confused where the text itself is.

---

## Retrieval system (RAG)

Reviews indexed in ChromaDB with sentence embeddings, retrieved by semantic similarity with metadata filtering, and summarised by a language model constrained to the retrieved text.

**Design decisions:**

- **One review is one chunk.** With a median of 22 words, a review is already the right unit. The corpus removes a whole tuning problem.
- **Retrieval runs on cleaned text; the original is what gets returned.** Only possible because the original columns were preserved during cleaning.
- **Sentiment is an indexed, filterable field**, which is what connects the two halves of the project. On unlabelled text the label would come from the classifier — and the cost analysis says which: 10 seconds with the baseline against 2.8 hours with BETO.
- **Cosine distance, not Chroma's default**, so vector magnitude does not let review length interfere with the ranking.
- **Deduplication is not optional.** *Buena relación calidad precio* appears 112 times verbatim; without a diversity filter a query returns fifteen passages that all say the same thing.
- **Three rules in the prompt:** answer only from the supplied reviews, cite the number of each supporting review, and say explicitly when the answer is not there.

**What the retrieval revealed.** Asking *what do customers complain about?* in the `wireless` category returned fifteen reviews of which almost none discussed the product: sellers who do not reply, orders that never arrived, refunds that took weeks, warranties nobody honoured. **In the negative reviews of that category, dissatisfaction is predominantly logistical rather than about the product itself** — which points the lever at seller management and delivery rather than manufacturing.

Asking specifically about battery life returned fifteen reviews all on topic, clustering into short runtime, failure to charge, and mismatch with the advertised specification. **The index was working in both cases; the first question was simply too generic.**

---

## Limitations

- **The neutral class is not reliable for automated decisions.** At a precision of 0.507, half of what the model labels neutral is not.
- **The corpus cannot support absolute business claims.** Its five rating levels were balanced by design, so it says nothing about how common each rating is in reality. Comparisons between categories are valid; levels are not.
- **No product-level analysis is possible.** A median of one review per product rules it out, for both the statistical analysis and the retrieval system.
- **15.5 % of the test set is beyond both models.** A three-star review reading *"está bien pero se rompió a los dos meses"* carries signal for two classes at once. This is the practical ceiling of the problem, not a defect of the approach.
- **BETO's hyperparameters were not tuned**, only accepted from the standard recipe, because each run costs 23 minutes.
- **The retrieval system has not been evaluated quantitatively.** There is no measurement of whether the retrieved reviews are the right ones.
- **RAG retrieves and summarises evidence; it does not count.** *What percentage of customers complain about price?* is outside its scope and belongs to the statistical analysis.

---

## Conclusions

**The corpus decides more than the models do.** Three findings from the exploratory stages shaped everything downstream: the artificial class balance, which disqualifies absolute business claims; the absence of a product dimension, which cancelled an entire planned analysis; and the fact that no measured feature separates neutral from negative.

**The models behaved as the data predicted.** Both classifiers fail exactly where the exploration said they would. Length did not separate neutral from negative, nor did category, nor did vocabulary — and neutral F1 is 0.505 for the baseline and 0.563 for BETO, against 0.83 to 0.88 for the two poles.

**What the models are good at is the distinction that matters commercially.** The sign is almost never inverted, and when the model fails on a polar review it retreats to the middle rather than claiming the opposite.

**The comparison is a decision, not a ranking.** BETO wins by four points and the result is statistically solid. It also costs 1,009 times more per prediction. Which turns the question from *which is better* into *what are four points worth here*, and the answer depends on volume, latency budget and whether the decision has to be auditable. For most uses, the baseline.

**The method is the part worth reusing.** Nothing was cleaned before it was measured, which is why the HTML-stripping stage was never written. No test was reported without an effect size, because at this sample size the p-value carries no information. And the rigour was scaled to the cost: 144 configurations where a fit takes 53 seconds, one standard recipe where it takes 23 minutes.

---

## Possible improvements

**Baseline**

- Vectorise title and body separately with a `ColumnTransformer` instead of concatenating them, so a word in the title is a distinct feature from the same word in the body
- Test `ComplementNB` and `SGDClassifier`, neither included in the search
- Refit on train plus validation once the configuration is fixed

**BETO**

- **Tune the decision thresholds** on validation instead of taking the `argmax`. The cheapest available improvement, requiring no retraining, and aimed directly at the neutral class
- **A larger model.** `roberta-large-bne` could not be loaded because of a tokenizer dependency issue; `xlm-roberta-large` works but needs an hour and a half of training for an expected gain of one or two points
- **Hyperparameter search**, constrained by the 23-minute cost per run

**The task itself**

- **Collapse the problem to two classes.** This would push macro F1 into the low nineties, but as a redefinition of the task rather than an improvement to the model: the gain comes from deleting the hard part, not from solving it

**Both models**

- **An ensemble.** 8.4 % of reviews are recovered only by BETO and 4.7 % only by the baseline; the oracle gap of 4.7 points is the single largest opportunity identified in this work
- **Quantify the label ceiling.** The corpus contains identical review texts carrying different star ratings; measuring how much of the residual 15.5 % is irreducible would establish how much of the remaining error is worth pursuing

**Retrieval**

- A stronger embedding model, query rewriting, hybrid retrieval combining vectors with keyword search, cross-encoder reranking, and a labelled set of questions to evaluate retrieval quality

---

## Requirements

```bash
pip install kagglehub pandas numpy matplotlib seaborn scikit-learn scikit-posthocs statsmodels nltk emoji
pip install torch transformers datasets accelerate
pip install chromadb sentence-transformers
```

Fine-tuning requires a GPU. The training runs were executed on Google Colab with an L4; everything else runs on CPU.

---

*Data source: https://www.kaggle.com/datasets/mexwell/amazon-reviews-multi*

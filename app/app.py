"""Amazon Reviews Spanish — interactive demo.

Three tabs:
  1. Classify a review with both models side by side, with measured latency
  2. Ask the corpus (retrieval-augmented question answering)
  3. Explore the corpus

Run locally:  python app/app.py
"""

import os
import time
from pathlib import Path

import gradio as gr
import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path(__file__).resolve().parent / "data"

# Hugging Face ZeroGPU Spaces require at least one @spaces.GPU function to exist at startup.
# Both models are nonetheless kept on CPU on purpose: the point of this demo is the latency
# comparison, and it is only meaningful if both run on the same hardware.
try:
    import spaces

    GPU = spaces.GPU
except Exception:  # noqa: BLE001 - not running on a Space
    def GPU(fn=None, **_kwargs):
        return fn if fn is not None else (lambda f: f)

# Model repo on the Hugging Face Hub holding the fine-tuned BETO.
BETO_REPO = os.environ.get("BETO_REPO", "Alessandrou24/beto-sentiment-amazon-es")
ENCODER_REPO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# tried in order; the first one the account can reach is used
GENERATORS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-1.5B-Instruct",
]

LABELS = ["negative", "neutral", "positive"]
LABEL_ES = {"negative": "Negativa", "neutral": "Neutra", "positive": "Positiva"}
COLORS = {"negative": "#c0392b", "neutral": "#e67e22", "positive": "#27ae60"}

# --------------------------------------------------------------------- assets

print("loading artefacts...")
reviews = pd.read_parquet(DATA / "reviews.parquet")
embeddings = np.load(DATA / "embeddings.npy").astype("float32")  # already L2-normalised
agg_category = pd.read_csv(DATA / "agg_category.csv")
agg_length = pd.read_csv(DATA / "agg_length.csv")
agg_words = pd.read_csv(DATA / "agg_words.csv")
baseline = joblib.load(DATA / "baseline.joblib")
baseline.predict_proba(["texto de calentamiento"])  # first call carries fixed overhead

CATEGORIES = sorted(reviews["product_category"].unique().tolist())

_beto = {"model": None, "tokenizer": None, "error": None}
_encoder = {"model": None}


def get_beto():
    """Loads BETO on first use so the Space starts fast."""
    if _beto["model"] is None and _beto["error"] is None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            _beto["tokenizer"] = AutoTokenizer.from_pretrained(BETO_REPO)
            model = AutoModelForSequenceClassification.from_pretrained(BETO_REPO)
            model.to("cpu")  # explicit: ZeroGPU emulates CUDA availability outside GPU
            model.eval()     # functions, and any implicit .cuda() call fails there
            _beto["model"] = model
            _beto["torch"] = torch
            with torch.no_grad():  # first pass carries fixed overhead
                model(**_beto["tokenizer"]("calentamiento", "texto", return_tensors="pt"))
        except Exception as exc:  # noqa: BLE001
            _beto["error"] = str(exc)
    return _beto


def get_encoder():
    if _encoder["model"] is None:
        from sentence_transformers import SentenceTransformer

        # device must be given explicitly: the default auto-detection sees ZeroGPU's
        # emulated CUDA and tries to move the model to a GPU that is not there
        _encoder["model"] = SentenceTransformer(ENCODER_REPO, device="cpu")
    return _encoder["model"]


# ------------------------------------------------------------ tab 1: classify

def _bar(probs, title):
    fig, ax = plt.subplots(figsize=(4, 2.2))
    ax.barh(LABELS[::-1], [probs[l] for l in LABELS[::-1]],
            color=[COLORS[l] for l in LABELS[::-1]])
    ax.set_xlim(0, 1)
    ax.set_title(title, fontsize=10)
    for i, l in enumerate(LABELS[::-1]):
        ax.text(min(probs[l] + 0.02, 0.85), i, f"{probs[l]:.2f}", va="center", fontsize=9)
    ax.set_xticks([])
    fig.tight_layout()
    return fig


@GPU(duration=5)
def _zerogpu_probe():
    """Exists only so ZeroGPU Spaces pass their startup check.

    Nothing in this app runs on GPU. Both models are kept on CPU on purpose: the demo is a
    latency comparison, and it is only meaningful if both run on the same hardware.
    """
    return "ok"


def beto_predict(title, body):
    """Runs on CPU deliberately, so its timing is comparable with the baseline."""
    b = get_beto()
    torch = b["torch"]
    t0 = time.perf_counter()
    enc = b["tokenizer"](title or "", body or "", truncation=True,
                         max_length=192, return_tensors="pt")
    with torch.no_grad():
        logits = b["model"](**enc).logits
    probs = dict(zip(LABELS, torch.softmax(logits, dim=-1)[0].tolist()))
    return probs, (time.perf_counter() - t0) * 1000


def classify(title, body):
    text = f"{title or ''} {body or ''}".strip()
    if not text:
        return "Escribe una reseña.", None, "", None, ""

    # --- baseline
    t0 = time.perf_counter()
    proba = baseline.predict_proba([text])[0]
    t_base = (time.perf_counter() - t0) * 1000
    base_probs = dict(zip(baseline.classes_, proba))
    base_label = max(base_probs, key=base_probs.get)

    base_md = (f"### {LABEL_ES[base_label]}\n"
               f"**{t_base:.2f} ms**  ·  TF-IDF + regresión logística")

    # --- BETO
    b = get_beto()
    if b["error"]:
        beto_md = (f"### No disponible\n"
                   f"El modelo no se pudo cargar desde `{BETO_REPO}`.\n\n"
                   f"```\n{b['error'][:300]}\n```")
        return base_md, _bar(base_probs, "Baseline"), beto_md, None, ""

    beto_probs, t_beto = beto_predict(title, body)
    beto_label = max(beto_probs, key=beto_probs.get)

    beto_md = (f"### {LABEL_ES[beto_label]}\n"
               f"**{t_beto:.2f} ms**  ·  BETO ajustado")

    agree = "coinciden" if base_label == beto_label else "**no coinciden**"
    verdict = (f"Los dos modelos {agree}. En esta predicción BETO ha tardado "
               f"**{t_beto / max(t_base, 1e-9):.0f} veces más** "
               f"({t_beto:.1f} ms frente a {t_base:.2f} ms). "
               f"Los tiempos varían entre ejecuciones según la carga de la máquina.")

    return base_md, _bar(base_probs, "Baseline"), beto_md, _bar(beto_probs, "BETO"), verdict


# ----------------------------------------------------------------- tab 2: RAG

def retrieve(question, category, sentiment, k=30, keep=12, threshold=0.9):
    mask = np.ones(len(reviews), dtype=bool)
    if category and category != "Todas":
        mask &= (reviews["product_category"] == category).to_numpy()
    if sentiment and sentiment != "Todas":
        inv = {v: k_ for k_, v in LABEL_ES.items()}
        mask &= (reviews["sentiment"] == inv[sentiment]).to_numpy()

    idx_pool = np.flatnonzero(mask)
    if idx_pool.size == 0:
        return pd.DataFrame(), np.empty((0, embeddings.shape[1]), dtype="float32")

    q = get_encoder().encode([question], convert_to_numpy=True).astype("float32")[0]
    q /= max(np.linalg.norm(q), 1e-8)

    sims = embeddings[idx_pool] @ q
    top = idx_pool[np.argsort(-sims)[:k]]

    # diversity filter
    kept, kept_emb = [], []
    for i in top:
        e = embeddings[i]
        if all(float(e @ ke) < threshold for ke in kept_emb):
            kept.append(i)
            kept_emb.append(e)
        if len(kept) == keep:
            break

    return reviews.iloc[kept].copy(), np.stack(kept_emb) if kept_emb else None


PROMPT = """Eres un analista de opiniones de clientes. Responde a la pregunta basándote \
ÚNICAMENTE en las reseñas que se te proporcionan.

Reglas:
- No uses conocimiento externo ni supongas nada que no esté en las reseñas.
- Cita el número de las reseñas que respaldan cada afirmación.
- Si las reseñas no contienen la respuesta, dilo explícitamente.

Reseñas:
{context}

Pregunta: {question}

Respuesta:"""


def generate(question, docs):
    token = os.environ.get("HF_TOKEN")
    if not token:
        return None

    from huggingface_hub import InferenceClient

    context = "\n".join(f"{i}. {d}" for i, d in enumerate(docs, 1))
    messages = [{"role": "user",
                 "content": PROMPT.format(context=context, question=question)}]
    client = InferenceClient(api_key=token)

    errors = []
    for model in GENERATORS:
        try:
            out = client.chat_completion(model=model, messages=messages,
                                         max_tokens=500, temperature=0.1)
            return out.choices[0].message.content
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{model}: {type(exc).__name__} — {str(exc)[:200]}")

    detail = "\n".join(f"- `{e}`" for e in errors)
    return ("_No se pudo generar el resumen con ninguno de los modelos disponibles. "
            "Las reseñas recuperadas se muestran abajo, que es exactamente el contexto "
            "que recibiría el modelo._\n\n"
            f"<details><summary>Detalle del error</summary>\n\n{detail}\n\n</details>")


def ask(question, category, sentiment):
    if not question or not question.strip():
        return "Escribe una pregunta.", pd.DataFrame()

    hits, _ = retrieve(question, category, sentiment)
    if hits.empty:
        return "No hay reseñas que cumplan esos filtros.", pd.DataFrame()

    docs = hits["review_body"].tolist()
    answer = generate(question, docs)

    if answer is None:
        answer = ("_Generación desactivada: no hay token de Hugging Face configurado en el "
                  "Space. Se muestran las reseñas recuperadas, que es lo que el modelo "
                  "recibiría como contexto._")

    table = hits[["review_body", "product_category", "sentiment", "stars"]].copy()
    table["sentiment"] = table["sentiment"].map(LABEL_ES)
    table.columns = ["Reseña", "Categoría", "Sentimiento", "Estrellas"]
    table.insert(0, "#", range(1, len(table) + 1))
    return answer, table


# ------------------------------------------------------------- tab 3: explore

def plot_categories():
    d = agg_category.sort_values("reviews")
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(d["product_category"], d["reviews"], color="#34495e")
    ax.set_xlabel("Reseñas")
    ax.set_title("Volumen por categoría (corpus completo)")
    fig.tight_layout()
    return fig


def plot_polarity():
    d = agg_category.sort_values("pct_negative")
    fig, ax = plt.subplots(figsize=(7, 8))
    left = np.zeros(len(d))
    for lab in LABELS:
        vals = d[f"pct_{lab}"].to_numpy()
        ax.barh(d["product_category"], vals, left=left,
                color=COLORS[lab], label=LABEL_ES[lab])
        left += vals
    ax.set_xlim(0, 100)
    ax.set_xlabel("% de reseñas")
    ax.set_title("Reparto de sentimiento por categoría")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_length():
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(agg_length["stars"], agg_length["median"], marker="o", color="#34495e")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("Estrellas")
    ax.set_ylabel("Palabras (mediana)")
    ax.set_title("La relación no es monótona: el máximo está en 2 estrellas")
    fig.tight_layout()
    return fig


def plot_words():
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, lab in zip(axes, LABELS):
        d = agg_words[agg_words["sentiment"] == lab].head(10).iloc[::-1]
        ax.barh(d["word"], d["count"], color=COLORS[lab])
        ax.set_title(LABEL_ES[lab], fontsize=10)
        ax.tick_params(labelsize=8)
    fig.suptitle("Palabras más frecuentes por clase", fontsize=11)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------- interface

EXAMPLES = [
    ["Perfecto", "Llegó antes de lo previsto y funciona exactamente como esperaba."],
    ["Una decepción", "El producto no se parece a las fotos y el vendedor no contesta a los mensajes."],
    ["Cumple", "Está bien por el precio, aunque el material se nota algo justo y la batería dura poco."],
]

with gr.Blocks(title="Amazon Reviews Spanish") as demo:
    gr.Markdown(
        "# Amazon Reviews Spanish\n"
        "Clasificación de sentimiento y respuesta a preguntas sobre 208.899 reseñas de "
        "Amazon en español. [Código y análisis completo]"
        "(https://github.com/AlejandroBeldaFernandez/Amazon-Reviews-Spanish)."
    )

    with gr.Tab("Clasificar una reseña"):
        gr.Markdown(
            "Los dos modelos clasifican el mismo texto. **Fíjate en los tiempos**: es el "
            "hallazgo central del proyecto. BETO acierta cuatro puntos más de F1 macro "
            "(0,765 frente a 0,725) y cuesta **decenas de veces más** por reseña.\n\n"
            "Esa cifra es la latencia de una petición individual, que es el caso de "
            "producción. Procesando el corpus entero por lotes la diferencia sube a mil "
            "veces, porque vectorizar miles de documentos de golpe amortiza el coste fijo "
            "del baseline. Las dos mediciones están en el análisis completo.\n\n"
            "Los dos modelos se ejecutan **en la misma CPU**, que es lo que hace la "
            "comparación válida."
        )
        with gr.Row():
            in_title = gr.Textbox(label="Título", placeholder="Muy buen producto")
            in_body = gr.Textbox(label="Cuerpo de la reseña", lines=3,
                                 placeholder="Llegó a tiempo y funciona perfectamente.")
        btn = gr.Button("Clasificar", variant="primary")

        with gr.Row():
            with gr.Column():
                out_base_md = gr.Markdown()
                out_base_plot = gr.Plot()
            with gr.Column():
                out_beto_md = gr.Markdown()
                out_beto_plot = gr.Plot()
        out_verdict = gr.Markdown()

        gr.Examples(EXAMPLES, inputs=[in_title, in_body])
        btn.click(classify, [in_title, in_body],
                  [out_base_md, out_base_plot, out_beto_md, out_beto_plot, out_verdict])

    with gr.Tab("Preguntar al corpus"):
        gr.Markdown(
            "Recuperación semántica sobre 50.000 reseñas, con filtrado previo por categoría "
            "y sentimiento. La respuesta se genera **solo** a partir de las reseñas "
            "recuperadas, que se muestran debajo para que puedas verificarla.\n\n"
            "Funciona a nivel de categoría. Preguntas sobre un producto concreto no tienen "
            "respuesta: el corpus tiene una mediana de una reseña por producto."
        )
        q = gr.Textbox(label="Pregunta",
                       placeholder="¿De qué se quejan los clientes de productos wireless?")
        with gr.Row():
            f_cat = gr.Dropdown(["Todas"] + CATEGORIES, value="Todas", label="Categoría")
            f_sent = gr.Dropdown(["Todas"] + list(LABEL_ES.values()),
                                 value="Negativa", label="Sentimiento")
        ask_btn = gr.Button("Preguntar", variant="primary")
        out_answer = gr.Markdown(label="Respuesta")
        out_docs = gr.Dataframe(label="Reseñas recuperadas", wrap=True)

        gr.Examples(
            [["¿De qué se quejan los clientes de productos wireless?", "wireless", "Negativa"],
             ["¿Qué problemas hay con la batería?", "wireless", "Negativa"],
             ["¿Qué destacan los clientes satisfechos con los juguetes?", "toy", "Positiva"]],
            inputs=[q, f_cat, f_sent],
        )
        ask_btn.click(ask, [q, f_cat, f_sent], [out_answer, out_docs])

    with gr.Tab("Explorar el corpus"):
        gr.Markdown(
            "Hallazgos del análisis exploratorio y estadístico sobre el corpus completo."
        )
        with gr.Row():
            gr.Plot(plot_categories, label="Volumen")
            gr.Plot(plot_polarity, label="Polaridad")
        gr.Markdown(
            "El reparto de sentimiento varía entre categorías, pero la asociación es débil "
            "(V de Cramér = 0,063). Y **la proporción de neutras se mantiene cerca del 20 % "
            "en las treinta**: lo que cambia es el equilibrio entre satisfacción e "
            "insatisfacción."
        )
        gr.Plot(plot_length, label="Longitud")
        gr.Plot(plot_words, label="Vocabulario")
        gr.Markdown(
            "`no` encabeza las tres clases, incluidas las positivas. Contar palabras sueltas "
            "no distingue *no funciona* de *no está mal*, que es el argumento para incluir "
            "bigramas en el baseline."
        )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())

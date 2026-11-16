# Semantic Book Recommender

A book recommender that understands **what a book is about**, not just keywords.
Describe the book you want ("*a story about forgiveness*", "*a book to teach children about nature*"), pick a category and an emotional tone, and get matching books with their covers.

Everything runs **locally and for free**: open-source Hugging Face models, no paid API key needed.

## How it works

| Step | Notebook | What it does | Output |
|---|---|---|---|
| 1 | [`data-exploration.ipynb`](data-exploration.ipynb) | Downloads the [7k Books with Metadata](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata) dataset, analyses missing values, keeps books with a description of at least 25 words, adds an ISBN-tagged description | `books_cleaned.csv` |
| 2 | [`vector-search.ipynb`](vector-search.ipynb) | Turns every description into an embedding and stores them in a Chroma vector database, then finds the books closest in meaning to a query | `tagged_description.txt` |
| 3 | [`text-classification.ipynb`](text-classification.ipynb) | Simplifies hundreds of categories into *Fiction / Nonfiction / Children's Fiction / Children's Nonfiction*; a zero-shot model predicts the missing ones | `books_with_categories.csv` |
| 4 | [`sentiment-analysis.ipynb`](sentiment-analysis.ipynb) | Scores each description on 7 emotions (anger, disgust, fear, joy, sadness, surprise, neutral) | `books_with_emotions.csv` |
| 5 | [`gradio-dashboard.py`](gradio-dashboard.py) | Web app combining everything: semantic search + category filter + tone sorting | web UI |

```
Kaggle dataset ──► 1. clean ──► books_cleaned.csv ──► 2. embeddings + Chroma (search)
                                       │
                                       └──► 3. categories ──► 4. emotions ──► books_with_emotions.csv
                                                                                    │
                                                    5. Gradio dashboard ◄───────────┘
```

### Models

| Task | Model | Size |
|---|---|---|
| Embeddings (semantic search) | [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | ~90 MB |
| Zero-shot classification | [`typeform/distilbert-base-uncased-mnli`](https://huggingface.co/typeform/distilbert-base-uncased-mnli) | ~270 MB |
| Emotion classification | [`j-hartmann/emotion-english-distilroberta-base`](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) | ~330 MB |

All models run on CPU and are downloaded automatically the first time they are used (then cached in `~/.cache/huggingface`).

### Key ideas

- **Embeddings:** a model converts text into a vector of numbers; texts with similar meaning get vectors that are close together. A query is embedded the same way, and the nearest book vectors are the recommendations.
- **ISBN tagging:** each description is stored as `"<isbn13> <description>"`, so a search result (plain text) can be linked back to the full book row.
- **Zero-shot classification:** the model scores how well a text matches labels it was never trained on (here *Fiction* vs *Nonfiction*).
- **Emotion per sentence:** descriptions are split into sentences and, for each emotion, the highest sentence score is kept, so one very sad sentence makes the book "sad".

## Getting started

Requirements: Python 3.11+ (developed with 3.14).

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. Install the dependencies
pip install -r requirements.txt
```

### Run the pipeline

Run the notebooks **in order** (1 → 4); each one reads the CSV produced by the previous one.
The generated CSV files are already included in the repository, so you can skip directly to the dashboard.

### Launch the dashboard

```bash
python gradio-dashboard.py
```

Then open the local URL printed in the terminal (usually http://127.0.0.1:7860).

The **first launch** embeds all ~5,200 descriptions (a couple of minutes on CPU) and saves the vector database to `chroma_db/`. The next launches reuse it and start in seconds. Delete `chroma_db/` to rebuild it after changing `tagged_description.txt`.

## Project structure

```
book-recommender/
├── data-exploration.ipynb      # Step 1: cleaning
├── vector-search.ipynb         # Step 2: embeddings + vector search
├── text-classification.ipynb   # Step 3: Fiction / Nonfiction
├── sentiment-analysis.ipynb    # Step 4: emotion scores
├── gradio-dashboard.py         # Step 5: web app
├── books_cleaned.csv           # output of step 1
├── tagged_description.txt      # "<isbn> <description>" per line (step 2)
├── books_with_categories.csv   # output of step 3
├── books_with_emotions.csv     # output of step 4 (used by the dashboard)
└── requirements.txt
```

## Troubleshooting

- **Hugging Face download stuck on "reconstructing file":** set `HF_HUB_DISABLE_XET=1` before importing `transformers` (already done in `text-classification.ipynb`).
- **`device="mps"` error:** `mps` only exists on Apple Silicon Macs; use `device=-1` (CPU) or `device=0` (NVIDIA GPU).
- **Books without a cover** use `cover-not-found.jpg`. Add an image with that name to the project folder to display a placeholder.

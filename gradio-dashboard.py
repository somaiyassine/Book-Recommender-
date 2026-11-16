"""Gradio web app: semantic book recommender.

Type a description of the book you want, optionally pick a category and an emotional tone,
and get matching books with their covers.

Run:  python gradio-dashboard.py
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr

# ---------------------------------------------------------------------------
# Data: books with categories (step 3) and emotion scores (step 4)
# ---------------------------------------------------------------------------
books = pd.read_csv("books_with_emotions.csv")

# Ask Google Books for a bigger cover image; use a placeholder when there is no thumbnail
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

# ---------------------------------------------------------------------------
# Vector database (step 2)
# ---------------------------------------------------------------------------
# Free local embedding model — no API key needed (replaces OpenAIEmbeddings)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

CHROMA_DIR = "chroma_db"

if Path(CHROMA_DIR).exists():
    # Reuse the vectors computed on a previous launch (fast start)
    db_books = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
else:
    # First launch: one document per line of tagged_description.txt (= one per book), then embed them all
    logging.getLogger("langchain_text_splitters").setLevel(logging.ERROR)  # silence "chunk longer than 1" warnings
    raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)
    db_books = Chroma.from_documents(documents, embedding=embeddings, persist_directory=CHROMA_DIR)


# Tone choice in the UI -> emotion score column used for sorting
TONE_TO_EMOTION = {
    "Happy": "joy",
    "Surprising": "surprise",
    "Angry": "anger",
    "Suspenseful": "fear",
    "Sad": "sadness",
}


def retrieve_semantic_recommendations(
        query: str,
        category: str = "All",
        tone: str = "All",
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:
    # 1. Vector search: the descriptions closest in meaning to the query, best match first
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]

    # 2. ISBN -> full book rows, keeping the similarity order
    book_recs = (
        books.set_index("isbn13")
        .reindex(books_list)
        .dropna(subset=["title"])
        .reset_index()
    )

    # 3. Optional category filter
    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category]
    book_recs = book_recs.head(final_top_k)

    # 4. Optional tone: put the books with the strongest matching emotion first
    if tone in TONE_TO_EMOTION:
        book_recs = book_recs.sort_values(by=TONE_TO_EMOTION[tone], ascending=False)

    return book_recs


def recommend_books(query: str, category: str, tone: str):
    """Format the recommendations for the Gradio gallery: (image, caption) pairs."""
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        # Keep only the first 30 words of the description
        truncated_description = " ".join(row["description"].split()[:30]) + "..."

        # "A;B;C" -> "A, B, and C"
        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results


# ---------------------------------------------------------------------------
# User interface
# ---------------------------------------------------------------------------
categories = ["All"] + sorted(books["simple_categories"].dropna().unique())
tones = ["All"] + list(TONE_TO_EMOTION)

with gr.Blocks() as dashboard:
    gr.Markdown("# Semantic book recommender")

    with gr.Row():
        user_query = gr.Textbox(label="Please enter a description of a book:",
                                placeholder="e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices=categories, label="Select a category:", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Select an emotional tone:", value="All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommended books", columns=8, rows=2)

    submit_button.click(fn=recommend_books,
                        inputs=[user_query, category_dropdown, tone_dropdown],
                        outputs=output)


if __name__ == "__main__":
    dashboard.launch(theme=gr.themes.Glass())  # Gradio 6: theme is passed to launch()

<<<<<<< HEAD
# Recipe & Nutrition RAG (Dietary Restrictions)

A minimal, Python-only RAG system that retrieves recipes with nutrition-aware filtering and optional LLM generation.

## Features
- Ingest recipes (CSV) with nutrition & allergens into Chroma vector DB
- Local embeddings via `sentence-transformers` (default: `all-MiniLM-L6-v2`)
- Rule-based filters for common health conditions (e.g., diabetes/hypertension) and dietary restrictions (vegan, gluten-free, etc.)
- Streamlit UI for interactive queries
- Optional OpenAI generation (fallback to extractive summarization if not provided)
- Basic hooks for evaluation (RAGAS)

## Quickstart

```bash
git clone <your-repo>  # or unzip this folder
cd recipe-nutrition-rag
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
# (Optionally) set OPENAI_API_KEY in .env for generative answers

# 1) Ingest sample data
python src/ingest.py --csv data/sample_recipes.csv

# 2) Run the app
streamlit run app.py
```

## Data Format
CSV headers:
- `id`: unique id
- `title`: recipe title
- `ingredients`: semicolon `;` separated list
- `instructions`: free text
- `calories, protein_g, carbs_g, fat_g, fiber_g, sodium_mg, sugar_g`: numeric nutrition fields
- `tags`: semicolon separated coarse labels (e.g., `vegan;gluten_free;low_sodium`)
- `allergens`: semicolon separated list (e.g., `nuts;dairy;gluten`)
- `cuisine`: optional, e.g., `Indian` / `Italian`

## Evaluation (Optional)
- Add your own Q&A pairs under `data/eval/*.json`
- Use RAGAS to compute retrieval precision/recall and answer faithfulness.
- See `notebooks/01_evaluate_ragas.ipynb` (placeholder).

## Notes
- This starter is intentionally small and opinionated to help you move fast in 3 days.
- You can replace `Streamlit` with `Gradio` or deploy to HuggingFace Spaces.
- For larger datasets, consider chunking strategies & more advanced retrievers.
- Always validate nutrition & allergy data integrity before use.
=======
# RecipeRag
>>>>>>> 15c70c0b8f2dcd92fb0bbc1a204a494f07530261

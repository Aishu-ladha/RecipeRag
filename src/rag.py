# ===== MUST be at very top, before importing transformers/torch =====
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"         # avoid torch.compile
os.environ["TORCH_DISABLE_META_KERNELS"] = "1"  # avoid meta device kernels

import pandas as pd

# Patch sqlite3 with pysqlite3 to fix ChromaDB sqlite version on some hosts
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

import chromadb
from chromadb.utils import embedding_functions
from transformers import pipeline


class RAGEngine:
    def __init__(self, top_k=4):
        self.top_k = top_k
        self.db = None
        self.hf_llm = None

        # pick a small, CPU-friendly model; allow override via env if needed
        model_name = os.getenv("HF_TEXT2TEXT_MODEL", "google/flan-t5-small")

        # Try to build the pipeline, but don’t crash the app if it fails
        try:
            self.hf_llm = pipeline(
                "text2text-generation",
                model=model_name,
                device=-1,                   # force CPU
                # keep generation predictable & lighter
                model_kwargs={"torch_dtype": "float32"},
            )
            print(f"[INFO] Loaded HF model: {model_name} (CPU)")
        except Exception as e:
            self.hf_llm = None
            print(f"[WARN] Could not load HF model '{model_name}'. "
                  f"Falling back to rule-based output. Error: {e}")

    def load(self, data_path="data/recipes.csv"):
        print("[INFO] Loading recipe data...")
        ext = os.path.splitext(data_path)[1].lower()
        if ext == ".csv":
            df = pd.read_csv(data_path)
        elif ext == ".parquet":
            df = pd.read_parquet(data_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or Parquet.")

        print(f"[INFO] Loaded {len(df)} recipes.")

        # Embeddings
        embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        # Chroma persistent (new client API)
        client = chromadb.PersistentClient(path=".chromadb")

        # Create or reuse collection
        names = [c.name for c in client.list_collections()]
        if "recipes" in names:
            self.db = client.get_collection(
                name="recipes",
                embedding_function=embedding_func
            )
            print("[INFO] Reusing existing ChromaDB collection.")
        else:
            self.db = client.create_collection(
                name="recipes",
                embedding_function=embedding_func
            )
            print("[INFO] Created new ChromaDB collection and adding data...")
            # Add rows
            for idx, row in df.iterrows():
                instr = str(row.get("instructions", "") or "")
                title = str(row.get("title", "") or "")
                ingred = str(row.get("ingredients", "") or "")
                if not instr.strip():
                    continue
                self.db.add(
                    ids=[str(idx)],
                    documents=[instr],
                    metadatas=[{
                        "title": title,
                        "ingredients": ingred
                    }]
                )
            print("[INFO] Recipes added successfully.")

    def retrieve(self, query, diets=None, allergens=None, conditions=None):
        if not self.db:
            raise ValueError("Database not loaded. Run load() first.")

        # Build a soft filter by appending terms to the query
        filter_terms = []
        if diets:
            filter_terms.extend(diets)
        if allergens:
            filter_terms.extend([f"no {a}" for a in allergens])
        if conditions:
            filter_terms.extend(conditions)

        search_text = query.strip()
        if filter_terms:
            search_text += " " + " ".join(filter_terms)

        results = self.db.query(query_texts=[search_text], n_results=self.top_k)
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        candidates = []
        for doc, meta in zip(docs, metas):
            candidates.append({
                "title": meta.get("title", "") or "",
                "ingredients": meta.get("ingredients", "") or "",
                "instructions": doc or ""
            })
        print(f"[INFO] Retrieved {len(candidates)} candidates.")
        return candidates

    def _fallback_generate(self, query, contexts):
        """Simple, deterministic explanation without HF model."""
        if not contexts:
            return ("I couldn't find exact matches for your request. "
                    "Try being more specific with cuisine, main ingredient, or dietary needs.")
        lines = [f"Request: {query}", "", "Here are some recipe options I found:"]
        for i, c in enumerate(contexts[:4], 1):
            lines.append(
                f"{i}. {c['title'] or 'Untitled Recipe'}\n"
                f"   - Key ingredients: {', '.join([x.strip() for x in (c['ingredients'] or '').split(',')][:6])}\n"
                f"   - Why it may fit: matches your query terms in instructions/ingredients."
            )
        lines.append("")
        lines.append("Tip: You can adjust filters (diet, allergens, conditions) in the sidebar to refine results.")
        return "\n".join(lines)

    def generate(self, query):
        contexts = self.retrieve(query)

        # If no matching recipes
        if not contexts:
            return ("I couldn't find exact matches for your request. "
                    "Try being more specific with cuisine, main ingredient, or dietary needs.")

        # If HF pipeline failed to initialize, use fallback
        if self.hf_llm is None:
            return self._fallback_generate(query, contexts)

        # Build prompt for the model
        joined = "\n\n".join(
            f"Title: {c['title']}\nIngredients: {c['ingredients']}\nInstructions: {c['instructions']}"
            for c in contexts
        )
        prompt = f"""
User request: {query}

Here are some recipe options:
{joined}

Select 2-4 that fit dietary restrictions and health conditions.
Explain why each is suitable and give 1-2 safe substitutions.
Avoid any medical claims; focus only on food choices.
"""

        try:
            print("[DEBUG] Sending prompt to Hugging Face model...")
            result = self.hf_llm(prompt, max_length=512, do_sample=False)
            text = (
                result[0].get("generated_text")
                or result[0].get("summary_text")
                or result[0].get("text")
            )
            return text.strip() if text else self._fallback_generate(query, contexts)
        except Exception as e:
            print(f"[WARN] HF generation failed, using fallback. Error: {e}")
            return self._fallback_generate(query, contexts)

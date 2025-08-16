import pandas as pd

# Patch sqlite3 with pysqlite3 to fix ChromaDB error
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from transformers import pipeline
import os


class RAGEngine:
    def __init__(self, top_k=4):
        self.top_k = top_k
        self.db = None
        self.hf_llm = pipeline(
            "text2text-generation",
            model="google/flan-t5-large"
        )

       def load(self, data_path="data/recipes.csv"):
        print("[INFO] Loading recipe data...")
        ext = os.path.splitext(data_path)[1].lower()
        df = pd.read_csv(data_path) if ext == ".csv" else pd.read_parquet(data_path)

        print(f"[INFO] Loaded {len(df)} recipes.")

        embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        client = chromadb.PersistentClient(path=".chromadb")

        if "recipes" in [c.name for c in client.list_collections()]:
            self.db = client.get_collection(name="recipes", embedding_function=embedding_func)
            print("[INFO] Reusing existing ChromaDB collection.")
        else:
            self.db = client.create_collection(name="recipes", embedding_function=embedding_func)
            print("[INFO] Created new ChromaDB collection and adding data...")
            for idx, row in df.iterrows():
                self.db.add(
                    ids=[str(idx)],
                    documents=[row["instructions"]],
                    metadatas=[{
                        "title": row.get("title", ""),
                        "ingredients": row.get("ingredients", "")
                    }]
                )
            print("[INFO] Recipes added successfully.")

    def retrieve(self, query, diets=None, allergens=None, conditions=None):
        if not self.db:
            raise ValueError("Database not loaded. Run load() first.")

        # Build a filter string from diets, allergens, conditions
        filter_query = []
        if diets:
            filter_query.extend(diets)
        if allergens:
            filter_query.extend([f"no {a}" for a in allergens])
        if conditions:
            filter_query.extend(conditions)

        search_text = query
        if filter_query:
            search_text += " " + " ".join(filter_query)

        results = self.db.query(query_texts=[search_text], n_results=self.top_k)
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        candidates = []
        for doc, meta in zip(docs, metas):
            candidates.append({
                "title": meta.get("title", ""),
                "ingredients": meta.get("ingredients", ""),
                "instructions": doc
            })
        print(f"[INFO] Retrieved {len(candidates)} candidates.")
        return candidates

    def generate(self, query):
        contexts = self.retrieve(query)

        if not contexts:
            return (
                "I couldn't find exact matches for your request. "
                "Try being more specific with cuisine, main ingredient, or dietary needs."
            )

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

        print("[DEBUG] Sending prompt to Hugging Face model...")
        result = self.hf_llm(prompt, max_length=512, do_sample=False)

        text = (
            result[0].get("generated_text")
            or result[0].get("summary_text")
            or result[0].get("text")
        )

        return text.strip() if text else "[No response generated]"

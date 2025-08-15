import argparse
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from src.config import settings

def build_text(row: pd.Series) -> str:
    fields = [
        f"Title: {row['title']}",
        f"Ingredients: {row['ingredients']}",
        f"Instructions: {row['instructions']}",
        f"Nutrition (per serving): calories={row['calories']}, protein_g={row['protein_g']}, carbs_g={row['carbs_g']}, fat_g={row['fat_g']}, fiber_g={row['fiber_g']}, sodium_mg={row['sodium_mg']}, sugar_g={row['sugar_g']}",
        f"Tags: {row.get('tags','')}",
        f"Allergens: {row.get('allergens','')}",
        f"Cuisine: {row.get('cuisine','')}",
    ]
    return "\n".join(fields)

def main(csv_path: str):
    df = pd.read_csv(csv_path)
    df.fillna("", inplace=True)

    # Initialize DB & model
    client = chromadb.PersistentClient(path=settings.chroma_dir)
    collection = client.get_or_create_collection(name="recipes")

    model = SentenceTransformer(settings.embedding_model)

    ids, docs, metas, embeddings = [], [], [], []
    for _, row in df.iterrows():
        rid = str(row["id"])
        text = build_text(row)
        emb = model.encode(text)
        meta = {
            "title": row["title"],
            "ingredients": row["ingredients"],
            "instructions": row["instructions"],
            "calories": float(row["calories"]),
            "protein_g": float(row["protein_g"]),
            "carbs_g": float(row["carbs_g"]),
            "fat_g": float(row["fat_g"]),
            "fiber_g": float(row["fiber_g"]),
            "sodium_mg": float(row["sodium_mg"]),
            "sugar_g": float(row["sugar_g"]),
            "tags": row.get("tags",""),
            "allergens": row.get("allergens",""),
            "cuisine": row.get("cuisine",""),
        }
        ids.append(rid)
        docs.append(text)
        metas.append(meta)
        embeddings.append(emb)

    collection.upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)
    print(f"Ingested {len(ids)} recipes into collection 'recipes' at {settings.chroma_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to recipes CSV")
    args = parser.parse_args()
    main(args.csv)

import streamlit as st
from src.rag import RAGEngine

st.set_page_config(page_title="Recipe & Nutrition RAG", page_icon="🍲", layout="wide")
st.title("🍲 Recipe & Nutrition RAG (Diet & Health Aware)")

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    diets = st.multiselect(
        "Dietary preferences", ["vegan", "vegetarian", "gluten_free", "low_carb", "high_protein"]
    )
    allergens = st.multiselect("Avoid allergens", ["nuts", "dairy", "soy", "gluten"])
    conditions = st.multiselect(
        "Health conditions", ["diabetes", "hypertension", "celiac", "high_cholesterol"]
    )
    top_k = st.slider("Results (top_k)", 1, 10, 4)

query = st.text_input(
    "What would you like to eat or optimize? (e.g., 'low sodium Indian dinner, high protein')", ""
)

# Initialize engine
if "engine" not in st.session_state:
    st.session_state.engine = RAGEngine(top_k=top_k)

engine = st.session_state.engine
engine.top_k = top_k

# Load database if not loaded
if not engine.db:
    with st.spinner("Loading recipe database..."):
        try:
            engine.load(data_path="data/sample_recipes.csv")  # CSV or Parquet
            st.success("Recipe database loaded!")
        except Exception as e:
            st.error(f"Failed to load database: {e}")
            st.stop()

# Search and generate
if st.button("Search") and query:
    with st.spinner("Retrieving matching recipes..."):
        results = engine.retrieve(query, diets=diets, allergens=allergens, conditions=conditions)

    if not results:
        st.warning("No exact matches found — generating alternative suggestions...")
        with st.spinner("Generating recommendations..."):
            answer = engine.generate(query)
        st.subheader("Recommendations")
        st.write(answer)
    else:
        with st.spinner("Generating recommendations..."):
            answer = engine.generate(query)
        st.subheader("Recommendations")
        st.write(answer)

        st.divider()
        st.subheader("Matched Recipes")
        for r in results:
            with st.expander(r["title"]):
                st.write(f"**Ingredients**: {r['ingredients']}")
                st.write(f"**Instructions**: {r['instructions']}")

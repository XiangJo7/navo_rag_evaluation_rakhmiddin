import yaml
import re
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from utils.test_synthetic_with_SHAP import generate_prediction_insights

# === Configuration ===
project_root = Path(__file__).resolve().parents[1]
vector_space_dir = project_root / "Academic Paper Storage"
feature_mapping_dir = project_root / "configs" / "feature_medical_term_mapping.yaml"
embed_model = "sentence-transformers/all-MiniLM-L6-v2"
no_chunks = 10  # Number of most relevant chunks to retrieve

def get_top_feature_query():
    insights = generate_prediction_insights()
    top_features = insights["top_features"]

    # Load yaml mapping
    with open(feature_mapping_dir, "r") as f:
        config = yaml.safe_load(f)
    feature_mapping = config["feature_medical_term_mapping"]

    # Convert them into a single query string
    translated_top_features = [feature_mapping.get(feat, feat) for feat in top_features]
    query = " ".join(translated_top_features)
    print(f"🔍 Query based on top features: {query}\n")
    return query

def is_structured_text(text: str, min_words: int = 10, min_alpha_ratio: float = 0.5) -> bool:
    # Remove excess whitespace
    cleaned = text.strip()

    # Reject very short texts
    if len(cleaned.split()) < min_words:
        return False

    # Count alphabetic vs total characters
    alpha_chars = len(re.findall(r'[a-zA-Z]', cleaned))
    total_chars = len(cleaned)

    # Reject if too many non-alphabetic characters (e.g., symbols, numbers)
    if total_chars == 0 or alpha_chars / total_chars < min_alpha_ratio:
        return False

    return True

def retrieve_relevant_chunks(query: str, index_path=vector_space_dir, top_k=no_chunks, min_score=0.7):
    embedding = HuggingFaceEmbeddings(model_name=embed_model)
    vectorstore = FAISS.load_local(str(index_path), embedding, allow_dangerous_deserialization=True)

    relevant_chunk_documentation = vectorstore.similarity_search_with_score(query, k=top_k)

    # Filter by relevance score and structured quality
    filtered_chunks = [
        doc for doc, score in relevant_chunk_documentation
        if score >= min_score and is_structured_text(doc.page_content)
    ]

    return filtered_chunks

if __name__ == "__main__":
    query = get_top_feature_query()

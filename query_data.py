import argparse
from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from transformers import pipeline

# --- Config (keep these identical to your create_database.py) ---
BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = (BASE_DIR / "chroma").resolve()  # absolute path
COLLECTION = "webpages"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Use the SAME embedding model as when you built the DB
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Load the persisted Chroma store
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION,
    )

    # --- Debug info to see what's happening ---
    print("DB path:", CHROMA_PATH)
    try:
        print("Docs in collection:", db._collection.count())
    except Exception as e:
        print("Could not read collection count:", repr(e))

    # Retrieve top documents (with scores)
    results = db.similarity_search_with_relevance_scores(query_text, k=5)
    print("Raw results:", len(results))
    for i, (doc, score) in enumerate(results):
        print(f"{i}: score={score:.4f} source={doc.metadata.get('source')}")

    # If scores are low, try a more forgiving threshold or MMR fallback
    used_docs = None
    if results and results[0][1] >= 0.3:  # lowered threshold to 0.3
        used_docs = [d for d, _ in results]
    else:
        print("Unable to find strong matches with similarity; trying MMR fallback...")
        retriever = db.as_retriever(
            search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20}
        )
        used_docs = retriever.get_relevant_documents(query_text)
        print("MMR docs:", len(used_docs))

        if not used_docs:
            print("No documents found. Check path/collection/embeddings.")
            return

    # Build the prompt with retrieved context
    context_text = "\n\n---\n\n".join(d.page_content for d in used_docs)
    prompt_value = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context_text, question=query_text
    )
    prompt_str = prompt_value.to_string() if hasattr(prompt_value, "to_string") else str(prompt_value)

    print("----- PROMPT -----\n", prompt_str, "\n------------------")

    # Lightweight local generator (CPU-friendly)
    gen = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=256)
    response_text = gen(prompt_str)[0]["generated_text"]

    sources = [d.metadata.get("source") for d in used_docs]
    print(f"RESPONSE: \n \n {response_text} \n \n Sources: {sources}  \n \n")

if __name__ == "__main__":
    main()

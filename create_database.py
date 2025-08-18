# create_database.py

from pathlib import Path
import os
import shutil

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- Config (keep in sync with query_data.py) ---
BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = str(BASE_DIR / "chroma")
DATA_PATH = BASE_DIR / "data" / "webpages"   # <- reads from data/webpages
COLLECTION = "webpages"                      # <- define the collection name
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"

def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents():
    # Recursively load all Markdown files under data/webpages
    loader = DirectoryLoader(str(DATA_PATH), glob="**/*.md")
    docs = loader.load()
    print(f"Loaded {len(docs)} documents from {DATA_PATH}")
    return docs

def split_text(documents: list[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    if chunks:
        print("Sample chunk content:\n", chunks[0].page_content[:300], "...\n")
        print("Sample metadata:", chunks[0].metadata)
    return chunks

def save_to_chroma(chunks: list[Document]):
    # Clear out the database first (optional).
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # Create a new DB from the documents. (Auto-persist with persist_directory)
    Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION,
        collection_metadata={"hnsw:space": "cosine"},  # ensure cosine space
    )
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH} (collection='{COLLECTION}').")

if __name__ == "__main__":
    main()

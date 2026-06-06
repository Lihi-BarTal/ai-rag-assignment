import os
import uuid
import pandas as pd
from pinecone import Pinecone
from langchain_text_splitters import TokenTextSplitter
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

# ==========================================
# 1. Configuration & API Keys
# ==========================================
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
LLMOD_API_KEY = os.getenv("LLMOD_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
EMBEDDING_MODEL = "4UHRUIN-text-embedding-3-small"

# ==========================================
# 2. Clients Initialization
# ==========================================
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)
embeddings_model = OpenAIEmbeddings(
    api_key=LLMOD_API_KEY,
    base_url="https://api.llmod.ai", 
    model=EMBEDDING_MODEL,
)

# ==========================================
# 3. Helper Functions
# ==========================================
def load_csv_subset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.fillna("", inplace=True)
    return df

def chunk_texts(texts: list[str], chunk_size: int = 500, chunk_overlap_ratio: float = 0.2) -> list[list[str]]:
    chunk_overlap = int(chunk_size * chunk_overlap_ratio)
    
    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        encoding_name="cl100k_base",
    )
    
    return [splitter.split_text(text) for text in texts]

# ==========================================
# 4. Main Processing Flow
# ==========================================
def main() -> None:
    csv_path = "medium-english-50mb.csv" 

    print(f"Loading data from {csv_path}...")
    subset = load_csv_subset(csv_path)
    print(f"Loaded {len(subset)} rows for testing.")
    texts = subset["text"].astype(str).tolist() 
    print("\nChunking texts...")
    chunked_texts = chunk_texts(texts)
    chunk_texts_to_embed = []
    metadatas = []
    ids = []
    
    for index_num, (chunks, row) in enumerate(zip(chunked_texts, subset.itertuples()), start=1):
        print(f"Article {index_num}: {len(chunks)} chunk(s)")
        
        for chunk_idx, chunk in enumerate(chunks, start=1):
            chunk_texts_to_embed.append(chunk)
            unique_id = f"art_{index_num}_chunk_{chunk_idx}_{uuid.uuid4().hex[:6]}"
            ids.append(unique_id)
            metadatas.append({
                "article_id": str(index_num), 
                "title": str(row.title),
                "url": str(row.url),
                "authors": str(row.authors),
                "timestamp": str(row.timestamp),
                "tags": str(row.tags),
                "chunk": chunk 
            })
            
    print(f"\nTotal chunks to embed: {len(chunk_texts_to_embed)}")
    
    if not chunk_texts_to_embed:
        print("No chunks to process. Exiting.")
        return

    print("Generating embeddings using LangChain (automatic batching)...")
    try:
        vectors = embeddings_model.embed_documents(chunk_texts_to_embed)
        print(f"Successfully generated {len(vectors)} vectors.")
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return

    print("\nPreparing vectors for Pinecone...")
    vectors_to_upsert = list(zip(ids, vectors, metadatas))
    
    # --- New batching logic to prevent timeouts ---
    batch_size = 200
    total_vectors = len(vectors_to_upsert)
    print(f"Upserting {total_vectors} vectors to Pinecone index '{PINECONE_INDEX_NAME}' in batches of {batch_size}...")
    
    for i in range(0, total_vectors, batch_size):
        batch = vectors_to_upsert[i : i + batch_size]
        try:
            index.upsert(vectors=batch)
            print(f"  Successfully uploaded vectors {i} to {min(i + batch_size, total_vectors)}")
        except Exception as e:
            print(f"  Error uploading batch starting at index {i}: {e}")
            print("  Retrying this batch...")
            # Network fallback retry
            index.upsert(vectors=batch)
            
    print("Success! All vectors uploaded to Pinecone.")

if __name__ == "__main__":
    main()
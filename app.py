import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from mangum import Mangum

app = FastAPI()
handler = Mangum(app) # שורה זו עוזרת ל-Vercel לנהל את הבקשות ל-FastAPI


# onfiguration & API Keys
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
LLMOD_API_KEY = os.getenv("LLMOD_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
EMBEDDING_MODEL = "4UHRUIN-text-embedding-3-small"
CHAT_MODEL = "4UHRUIN-gpt-5-mini"

print("Init clients...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

embeddings_model = OpenAIEmbeddings(
    api_key=LLMOD_API_KEY,
    base_url="https://api.llmod.ai/v1",
    model=EMBEDDING_MODEL
)

chat_model = ChatOpenAI(
    api_key=LLMOD_API_KEY,
    base_url="https://api.llmod.ai/v1",
    model=CHAT_MODEL
)

# Request schema for the API
class PromptRequest(BaseModel):
    question: str

# GET endpoint for stats
@app.get("/api/stats")
def get_stats():
    print("-> GET /api/stats called")
    return {
        "chunk_size": 500,
        "overlap_ratio": 0.2,
        "top_k": 5
    }

# POST endpoint for queries
@app.post("/api/prompt")
def handle_prompt(request: PromptRequest):
    print(f"\n=== New Request: '{request.question}' ===")
    
    # Step 1: Embed question
    print("1. Generating embedding...")
    query_vector = embeddings_model.embed_query(request.question)
    
    # Step 2: Search Pinecone
    print("2. Searching Pinecone...")
    search_response = index.query(
        vector=query_vector,
        top_k=5,
        include_metadata=True
    )
    
    # Step 3: Build context
    print("3. Building context from retrieved chunks...")
    retrieved_chunks = []
    context_blocks = []
    
    for match in search_response.get("matches", []):
        meta = match.get("metadata", {})
        
        # Save chunk data for final JSON
        retrieved_chunks.append({
            "article_id": str(meta.get("article_id", "unknown")),
            "title": str(meta.get("title", "")),
            "chunk": str(meta.get("chunk", "")),
            "score": float(match.get("score", 0.0))
        })
        
        # Format chunk for the LLM prompt
        context_blocks.append(
            f"Title: {meta.get('title')}\n"
            f"Author(s): {meta.get('authors')}\n"
            f"URL: {meta.get('url')}\n"
            f"Tags: {meta.get('tags')}\n"
            f"Passage: {meta.get('chunk')}"
        )
        
    context_for_llm = "\n---\n".join(context_blocks)
    
    # Step 4: Prepare prompts
    print("4. Preparing prompts...")
    system_prompt = (
        "You are a Medium-article assistant that answers questions strictly and only "
        "based on the Medium articles dataset context provided to you (metadata and article passages). "
        "You must not use any external knowledge, the open internet, or information that is not explicitly "
        "contained in the retrieved context. If the answer cannot be determined from the provided context, "
        "respond: \"I don't know based on the provided Medium articles data.\"\n"
        "Always explain your answer using the given context, quoting or paraphrasing the relevant article passage "
        "or metadata when helpful."
    )
    
    user_prompt = f"Retrieved Context:\n{context_for_llm}\n\nQuestion: {request.question}"
    
    # Step 5: Call LLM
    print("5. Calling Chat Model...")
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    ai_response = chat_model.invoke(messages)
    
    # Step 6: Return JSON
    print("6. Returning final JSON response.")
    return {
        "response": ai_response.content,
        "context": retrieved_chunks,
        "Augmented_prompt": {
            "System": system_prompt,
            "User": user_prompt
        }
    }
handler = app
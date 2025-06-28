from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
import cohere
import httpx
import requests
from qdrant_client import QdrantClient
from langsmith import traceable

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "lhc_judgments")

co = cohere.Client(COHERE_API_KEY)
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data model
class QueryRequest(BaseModel):
    question: str

# Reformulate query
def reformulate_query(question: str) -> str:
    return f"What are similar legal interpretations or procedural considerations for: {question}"

# Rerank using Cohere
def rerank_with_cohere(query, docs, co):
    res = co.rerank(
        query=query,
        documents=docs,
        model="rerank-english-v3.0",
        top_n=5
    )
    return [r.document for r in res.results if r.document is not None]


# Vector search
@traceable(name="Vector Search")
def search_query(query: str, collection_name: str, co, top_k: int = 6) -> List[str]:
    query_embed = co.embed(
        texts=[query],
        model="embed-english-light-v3.0",
        input_type="search_query"
    ).embeddings[0]

    payload = {
        "vector": query_embed,
        "top": top_k,
        "with_payload": True
    }
    headers = {"Content-Type": "application/json", "api-key": QDRANT_API_KEY}
    url = f"{QDRANT_URL}/collections/{collection_name}/points/search"

    response = httpx.post(url, headers=headers, json=payload)
    response.raise_for_status()

    results = response.json()["result"]
    contexts = []
    for hit in results:
        payload = hit.get("payload", {})
        text = payload.get("text", "")
        url = payload.get("url", "N/A")
        updated_at = payload.get("updated_at", "Unknown")
        source = payload.get("source", "Unknown")
        if text:
            chunk = f"{text}\n\n(Source: {source} | URL: {url} | Updated: {updated_at})"
            contexts.append(chunk)
    return contexts

# Answer generation using Groq
@traceable(name="Answer Generation")
def generate_answer_groq(context: str, question: str) -> str:
    prompt = f"""Use the following legal context chunks to answer the question.
Each chunk ends with source info (PDF name, URL, update date). Cite sources in your answer.

Context:
{context}

Question: {question}
Answer:"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=data
    )
    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]

@traceable(name="Query Endpoint")
@app.post("/query")
def query_api(request: QueryRequest):
    try:
        original_query = request.question
        rewritten_query = reformulate_query(original_query)

        context1 = search_query(original_query, COLLECTION_NAME, co, top_k=6)
        context2 = search_query(rewritten_query, COLLECTION_NAME, co, top_k=6)

        combined = list(set(context1 + context2))
        reranked = rerank_with_cohere(original_query, combined, co)

        formatted_context = "\n\n".join(reranked)
        answer = generate_answer_groq(formatted_context, original_query)

        return {
            "question": original_query,
            "rewritten_query": rewritten_query,
            "sources_used": len(reranked),
            "answer": answer
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

user_sessions = {}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = id(websocket)
    user_sessions[session_id] = []

    try:
        while True:
            data = await websocket.receive_text()
            question = data.strip()
            user_sessions[session_id].append({"role": "user", "content": question})

            # Dual search
            rewritten = reformulate_query(question)
            ctx1 = search_query(question, COLLECTION_NAME, co)
            ctx2 = search_query(rewritten, COLLECTION_NAME, co)
            combined = list(set(ctx1 + ctx2))
            reranked = rerank_with_cohere(question, combined, co)
            context = "\n\n".join(reranked)

            prompt = f"""Use the following context to answer the question.
Cite sources in your answer. Context ends with PDF name, URL and updated date.

Context:
{context}

Question: {question}
Answer:"""

            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": "llama3-8b-8192",
                "stream": True,
                "messages": user_sessions[session_id] + [{"role": "user", "content": prompt}]
            }

            await websocket.send_text("THINKING...")

            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", "https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload) as response:
                    full_answer = ""
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                import json
                                line = line.replace("data: ", "")
                                delta = json.loads(line)["choices"][0]["delta"].get("content")
                                if delta:
                                    full_answer += delta
                                    await websocket.send_text(delta)
                            except Exception:
                                continue

            user_sessions[session_id].append({"role": "assistant", "content": full_answer})

    except WebSocketDisconnect:
        user_sessions.pop(session_id, None)

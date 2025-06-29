import os
import asyncio
import uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import requests
import cohere
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
COHERE_API_KEY = os.environ["COHERE_API_KEY"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
COLLECTION_NAME = "lhc_judgments"

co = cohere.Client(COHERE_API_KEY)
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def query_rewriter(conversation_history: str):
    prompt = f"""You are an AI assistant. Rewrite the following conversation into a concise and clear search query:

Conversation:
{conversation_history}

Rewritten Query:"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)

    try:
        return response.json()["choices"][0]["message"]["content"]
    except Exception:
        print("❌ Query rewriting failed:", response.status_code, response.text)
        return None

async def search_query(query, top_k=3):
    embed = co.embed(
        texts=[query],
        model="embed-english-light-v3.0",
        input_type="search_query"
    ).embeddings[0]

    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=embed,
        limit=top_k,
        with_payload=True
    )

    return [hit.payload.get("text", "") for hit in results]

async def rerank_context(query, context_list):
    if not context_list:
        return []

    results = co.rerank(
        query=query,
        documents=context_list,
        top_n=min(5, len(context_list)),
        model="rerank-english-v3.0"
    )

    return [r.document["text"] if isinstance(r.document, dict) else r.document for r in results]

async def generate_answer_groq(context, question):
    context_str = "\n\n".join(context)
    prompt = f"""Context:
{context_str}

Each context chunk ends with metadata (PDF name, URL, and updated date).
When answering, include relevant source(s) to justify your answer.

Question: {question}
Answer:"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)

    try:
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print("❌ LLM error:", e)
        return "Sorry, something went wrong while generating the answer."



@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ WebSocket connected.")

    conversation_history = []  # 

    try:
        while True:
            question = await websocket.receive_text()

            conversation_history.append(f"User: {question}")

            await websocket.send_text("THINKING...")

            # rewrite query based on full history
            full_convo = "\n".join(conversation_history)
            rewritten_query = await query_rewriter(full_convo)

            # run both vector searches in parallel
            orig_task = asyncio.create_task(search_query(question))
            rewrite_task = asyncio.create_task(search_query(rewritten_query)) if rewritten_query else None

            orig_results = await orig_task
            rewrite_results = await rewrite_task if rewrite_task else []

            combined_context = list(dict.fromkeys(orig_results + rewrite_results))

            # Rerank
            reranked_context = await rerank_context(question, combined_context)

            #  final answer
            answer = await generate_answer_groq(reranked_context, question)

            # Add assistant reply to history
            conversation_history.append(f"Assistant: {answer}")

            await websocket.send_text(answer)

    except WebSocketDisconnect:
        print("❌ WebSocket disconnected.")

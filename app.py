from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import cohere
from qdrant_client import QdrantClient
import requests
from fastapi.middleware.cors import CORSMiddleware
import httpx

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

class QueryRequest(BaseModel):
    question: str

def search_query(query, collection_name, co, top_k=3):
    try:
        query_embed = co.embed(
            texts=[query],
            model="embed-english-light-v3.0",
            input_type="search_query"
        ).embeddings[0]

        headers = {
            "Content-Type": "application/json",
            "api-key": QDRANT_API_KEY
        }

        payload = {
            "vector": query_embed,
            "top": top_k,
            "with_payload": True
        }

        url = f"{QDRANT_URL}/collections/{collection_name}/points/search"

        response = httpx.post(url, headers=headers, json=payload)
        response.raise_for_status()
        results = response.json()["result"]

        contexts = [hit["payload"]["text"] for hit in results if "text" in hit["payload"]]
        return "\n\n".join(contexts)
    except Exception as e:
        raise RuntimeError(f"Search failed: {e}")


def generate_answer_groq(context, question):
    try:
        prompt = f"""Use the following context to answer the question:
        
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

        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    except Exception as e:
        return f"❌ Error: {e}"
    

# API endpoint
@app.post("/query")
def query_api(request: QueryRequest):
    try:
        context = search_query(request.question, COLLECTION_NAME, co)
        answer = generate_answer_groq(context, request.question)
        return {"question": request.question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

from fastapi import WebSocket, WebSocketDisconnect
import asyncio

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

            context = search_query(question, COLLECTION_NAME, co)


            prompt = f"""Use the following context to answer the question:\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"""

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
        print("Client disconnected")
        user_sessions.pop(session_id, None)


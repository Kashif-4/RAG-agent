import os
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import AsyncQdrantClient
from dotenv import load_dotenv
import requests
import cohere
from datetime import datetime, timezone
from langsmith.client import Client  # Import Client

load_dotenv()

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COHERE_API_KEY = os.environ["COHERE_API_KEY"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY")  # Your LangSmith API key env var
COLLECTION_NAME = "lhc_judgments"

co = cohere.Client(COHERE_API_KEY)
qdrant = AsyncQdrantClient(url=QDRANT_URL)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate LangSmith client with API key (adjust endpoint if needed)
client = Client(api_key=LANGSMITH_API_KEY)


def now_iso():
    return datetime.now(timezone.utc).isoformat()


async def query_rewriter(conversation_history: str):
    prompt = f"""Given the conversation below, write a short and specific legal search query:

{conversation_history}

Search Query:"""

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
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("❌ Query rewriting failed:", e)
        return None


async def search_query(query, top_k=3):
    embed = co.embed(
        texts=[query],
        model="embed-english-light-v3.0",
        input_type="search_query"
    ).embeddings[0]

    results = await qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=embed,
        limit=top_k,
        with_payload=True
    )

    return [
        {
            "text": hit.payload.get("text", ""),
            "source": hit.payload.get("pdf_name"),
            "url": hit.payload.get("url"),
            "updated": hit.payload.get("updated_date"),
            "score": hit.score
        }
        for hit in results
    ]


async def rerank_context(query, context_list):
    if not context_list:
        return []

    documents = [str(item["text"]) for item in context_list]

    response = co.rerank(
        query=query,
        documents=documents,
        top_n=min(5, len(documents)),
        model="rerank-english-v3.0"
    )

    reranked = []
    for res in response.results:
        index = res.index
        original = context_list[index]
        reranked.append({
            "text": original["text"],
            "source": original.get("source"),
            "url": original.get("url"),
            "updated": original.get("updated"),
            "score": res.relevance_score
        })

    return reranked


async def generate_answer_groq(context, question):
    SYSTEM_PROMPT = """You are a legal assistant AI helping users understand legal concepts based on provided context.

Guidelines:
- Answer clearly and concisely
- Reference sources with name and updated date when available
- Avoid making up laws or interpreting beyond the given context
- If context is insufficient, state that clearly
- Use a formal and respectful tone
"""

    context_str = "\n\n".join([
        f"{chunk['text']}\n(Source: {chunk.get('source') or 'Unknown'}, Updated: {chunk.get('updated') or 'Unknown'})"
        for chunk in context
    ])

    full_prompt = f"""{SYSTEM_PROMPT}

Context:
{context_str}

Question: {question}

Answer:"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "user", "content": full_prompt}
        ]
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data
        )
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print("❌ LLM error:", e)
        return "Sorry, something went wrong while generating the answer."
    
from fastapi import WebSocket, WebSocketDisconnect
from langsmith.client import Client
from langsmith.schemas import RunTypeEnum
import uuid

client = Client()  

from typing import Optional, Dict, List
import re

class InteractionAnalyzer:
    @staticmethod
    def is_simple_interaction(text: str) -> bool:
        """Detect non-substantive interactions using multiple heuristics"""
        text = text.lower().strip()
        
        if len(text) < 4 and not any(c.isdigit() for c in text):
            return True
            
        patterns = [
            r'^(hi|hello|hey|greetings|good\s(morning|afternoon))\b',  # Greetings
            r'^(thanks|thank you|cheers)\b',  # Acknowledgments
            r'^(ok|okay|got it|understood)\b',  # Continuations
            r'^\W*$'  # Pure punctuation/empty
        ]
        
        return any(re.fullmatch(pattern, text) for pattern in patterns)
    
    @staticmethod
    def handle_simple_interaction(text: str) -> str:
        """Generate appropriate responses for simple interactions"""
        text = text.lower().strip()
        
        if re.match(r'^(hi|hello|hey)', text):
            return "Hello! I'm your legal assistant. How can I help you today?"
        elif 'thank' in text:
            return "You're welcome! Let me know if you have other legal questions."
        elif re.match(r'^(ok|got it)', text):
            return "Understood. Please continue with your legal inquiry."
        return "I'm here to help with legal matters. What would you like to know?"
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ WebSocket connected.")
    
    conversation_history = []
    analyzer = InteractionAnalyzer()

    try:
        while True:
            question = await websocket.receive_text()
            await websocket.send_text("THINKING...")
            conversation_history.append(f"User: {question}")

            try:
                # Handle simple interactions without tracing
                if analyzer.is_simple_interaction(question):
                    response = analyzer.handle_simple_interaction(question)
                    await websocket.send_text(response)
                    conversation_history.append(f"Assistant: {response}")
                    continue

                # Start tracing only for substantive queries
                trace_id = str(uuid.uuid4())
                root_run = client.create_run(
                    id=trace_id,
                    name="Legal Assistant Session",
                    run_type=RunTypeEnum.chain,
                    tags=["chat"],
                    inputs={"initial_question": question}
                )

                # 🔹 Step 1: Query Rewrite
                rewritten_query = await query_rewriter("\n".join(conversation_history))
                step1_id = str(uuid.uuid4())
                client.create_run(
                    id=step1_id,
                    name="Query Rewrite",
                    run_type=RunTypeEnum.llm,
                    parent_run_id=trace_id,
                    metadata={"conversation_history": "\n".join(conversation_history)},
                    inputs={"raw_input": question}
                )

                # 🔹 Step 2: Vector Search with confidence check
                search_results = await search_query(rewritten_query or question)
                step2_id = str(uuid.uuid4())
                client.create_run(
                    id=step2_id,
                    name="Vector Search",
                    run_type=RunTypeEnum.retriever,
                    parent_run_id=trace_id,
                    metadata={"query_used": rewritten_query or question},
                    inputs={"rewritten_query": rewritten_query}
                )

                # Check for low-confidence results
                if search_results and all(res['score'] < 0.3 for res in search_results):
                    low_conf_output = {
                        "warning": "low_confidence_results",
                        "top_score": max(res['score'] for res in search_results),
                        "count": len(search_results)
                    }
                    client.update_run(step2_id, outputs=low_conf_output)
                    await websocket.send_text(
                        "I couldn't find strongly relevant legal information.\n"
                        "Please try:\n- Using specific legal terms\n- Adding jurisdiction details\n- Providing more context"
                    )
                    continue

                client.update_run(step2_id, outputs={"top_k": len(search_results)})

                # 🔹 Step 3: Rerank Context
                reranked = await rerank_context(question, search_results)
                step3_id = str(uuid.uuid4())
                client.create_run(
                    id=step3_id,
                    name="Rerank Context",
                    run_type=RunTypeEnum.tool,
                    parent_run_id=trace_id,
                    inputs={"question": question, "results_count": len(search_results)}
                )
                client.update_run(step3_id, outputs={
                    "reranked_count": len(reranked),
                    "top_score": reranked[0]['score'] if reranked else 0
                })

                # 🔹 Step 4: Answer Generation with confidence awareness
                answer = await generate_answer_groq(reranked, question)
                step4_id = str(uuid.uuid4())
                client.create_run(
                    id=step4_id,
                    name="Answer Generation",
                    run_type=RunTypeEnum.llm,
                    parent_run_id=trace_id,
                    metadata={
                        "query": question,
                        "top_context_score": reranked[0]['score'] if reranked else 0
                    },
                    inputs={
                        "context": reranked,
                        "question": question,
                        "context_count": len(reranked)
                    }
                )
                
                # Add confidence disclaimer if needed
                if reranked and reranked[0]['score'] < 0.5:
                    answer = f"⚠️ Note: Confidence in this answer is limited.\n\n{answer}"

                client.update_run(step4_id, outputs={"answer": answer})
                conversation_history.append(f"Assistant: {answer}")
                await websocket.send_text(answer)

            except Exception as e:
                print("❌ Error during interaction:", e)
                error_id = str(uuid.uuid4())
                if 'trace_id' in locals():
                    client.create_run(
                        id=error_id,
                        name="Error",
                        run_type=RunTypeEnum.tool,
                        parent_run_id=trace_id,
                        error=str(e),
                        inputs={"question": question}
                    )
                await websocket.send_text("I encountered an error processing your legal request. Please try again.")

    except WebSocketDisconnect:
        print("❌ WebSocket disconnected.")

import os
import asyncio
import uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
#from qdrant_client import QdrantClient
from qdrant_client import AsyncQdrantClient
#from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams
import requests
import cohere
from dotenv import load_dotenv
from langsmith import traceable
from langsmith.run_trees import RunTree
from datetime import datetime



load_dotenv()

QDRANT_URL = os.environ["QDRANT_URL"]
#QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
COHERE_API_KEY = os.environ["COHERE_API_KEY"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
COLLECTION_NAME = "lhc_judgments"

co = cohere.Client(COHERE_API_KEY)
#qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
#qdrant = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
qdrant = AsyncQdrantClient(url="http://localhost:6333")  # No API for local


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@traceable(run_type="llm", name="Query Rewriter")
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

@traceable(run_type="retriever", name="Vector Search")
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

    return [hit.payload.get("text", "") for hit in results]

@traceable(run_type="tool", name="Rerank Context")
async def rerank_context(query, context_list):
    if not context_list:
        return []

    # Ensure all documents are strings
    documents = [str(doc) for doc in context_list]

    results = co.rerank(
        query=query,
        documents=documents,
        top_n=min(5, len(documents)),
        model="rerank-english-v3.0"
    )

    # Each result is a tuple like (document, relevance_score)
    return [doc for doc, _ in results]


@traceable(run_type="llm", name="Answer Generator")
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

    conversation_history = []

    try:
        while True:
            question = await websocket.receive_text()
            conversation_history.append(f"User: {question}")
            await websocket.send_text("THINKING...")

            root_trace = RunTree(
                name="LangGraph Interaction",
                run_type="chain",
                inputs={"question": question, "history": "\n".join(conversation_history)},
                metadata={"timestamp": str(datetime.utcnow())}
            )

            try:
                full_convo = "\n".join(conversation_history)

                step1 = root_trace.create_child(
                    name="Query Rewriter",
                    run_type="llm",
                    inputs={"conversation": full_convo}
                )
                try:
                    rewritten_query = await query_rewriter(full_convo)
                    step1.end(outputs={"rewritten_query": rewritten_query})
                except Exception as e:
                    step1.end(error=str(e))
                    raise

                step2a = root_trace.create_child(
                    name="Search (Original)",
                    run_type="retriever",
                    inputs={"query": question}
                )
                try:
                    orig_results = await search_query(question)
                    step2a.end(outputs={"results": orig_results})
                except Exception as e:
                    step2a.end(error=str(e))
                    raise

                if rewritten_query:
                    step2b = root_trace.create_child(
                        name="Search (Rewritten)",
                        run_type="retriever",
                        inputs={"query": rewritten_query}
                    )
                    try:
                        rewrite_results = await search_query(rewritten_query)
                        step2b.end(outputs={"results": rewrite_results})
                    except Exception as e:
                        step2b.end(error=str(e))
                        raise
                else:
                    rewrite_results = []

                combined_context = list(dict.fromkeys(orig_results + rewrite_results))

                step3 = root_trace.create_child(
                    name="Rerank Context",
                    run_type="reranker",
                    inputs={"query": question, "context": combined_context}
                )
                try:
                    reranked_context = await rerank_context(question, combined_context)
                    step3.end(outputs={"reranked": reranked_context})
                except Exception as e:
                    step3.end(error=str(e))
                    raise

                step4 = root_trace.create_child(
                    name="Answer Generator",
                    run_type="llm",
                    inputs={"question": question, "context": reranked_context}
                )
                try:
                    answer = await generate_answer_groq(reranked_context, question)
                    step4.end(outputs={"answer": answer})
                except Exception as e:
                    step4.end(error=str(e))
                    raise

                conversation_history.append(f"Assistant: {answer}")
                await websocket.send_text(answer)

                root_trace.end(outputs={"final_answer": answer})

            except Exception as e:
                print("❌ Error in chat pipeline:", str(e))
                root_trace.end(error=str(e))

    except WebSocketDisconnect:
        print("❌ WebSocket disconnected.")

import os
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import AsyncQdrantClient
from dotenv import load_dotenv
import requests
import cohere
from datetime import datetime
from langsmith.run_trees import RunTree

load_dotenv()

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COHERE_API_KEY = os.environ["COHERE_API_KEY"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
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

    # Cohere 5.x: returns a RerankResponse object with .results
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

            from langsmith.client import Client
            print("🔍 LangSmith API in loop:", Client().api_key[:8])

            root_trace = RunTree(
                name="Legal Assistant Interaction",
                run_type="chain",
                inputs={"question": question, "history": "\n".join(conversation_history)},
                metadata={"timestamp": str(datetime.utcnow())}
            )

            try:
                full_convo = "\n".join(conversation_history)

                step1 = root_trace.create_child(name="Query Rewriter", run_type="llm", inputs={"conversation": full_convo})
                rewritten_query = await query_rewriter(full_convo)
                step1.end(outputs={"rewritten_query": rewritten_query})

                search_input = rewritten_query or question
                step2 = root_trace.create_child(name="Vector Search", run_type="retriever", inputs={"query": search_input})
                search_results = await search_query(search_input)
                step2.end(outputs={"top_k": len(search_results), "documents": search_results})

                step3 = root_trace.create_child(name="Rerank Context", run_type="tool", inputs={"query": question})
                reranked_context = await rerank_context(question, search_results)
                step3.end(outputs={"reranked": reranked_context})

                step4 = root_trace.create_child(
                    name="Answer Generator",
                    run_type="llm",
                    inputs={"question": question, "context_count": len(reranked_context)}
                )
                answer = await generate_answer_groq(reranked_context, question)
                step4.end(outputs={"answer": answer})

                conversation_history.append(f"Assistant: {answer}")
                await websocket.send_text(answer)

                root_trace.end(outputs={"final_answer": answer})

            except Exception as e:
                print("❌ Error in chat pipeline:", str(e))
                root_trace.end(error=str(e))

    except WebSocketDisconnect:
        print("❌ WebSocket disconnected.")

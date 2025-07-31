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
    
from langsmith.run_helpers import trace  # ✅ New import for tracing

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ WebSocket connected.")

    conversation_history = []

    try:
        while True:
            question = await websocket.receive_text()
            await websocket.send_text("THINKING...")
            conversation_history.append(f"User: {question}")

            try:
                async with trace("Legal Assistant Session", run_type="chain", tags=["chat"]) as root_run:

                    async with trace("Query Rewrite", run_type="llm") as step1:
                        rewritten_query = await query_rewriter("\n".join(conversation_history))
                        step1.metadata["conversation_history"] = "\n".join(conversation_history)
                        step1.end(outputs={"rewritten_query": rewritten_query})

                    async with trace("Vector Search", run_type="retriever") as step2:
                        search_results = await search_query(rewritten_query or question)
                        step2.metadata["query_used"] = rewritten_query or question
                        step2.end(outputs={"top_k": len(search_results)})

                    async with trace("Rerank Context", run_type="tool") as step3:
                        reranked = await rerank_context(question, search_results)
                        step3.end(outputs={"reranked_count": len(reranked)})

                    async with trace("Answer Generation", run_type="llm") as step4:
                        answer = await generate_answer_groq(reranked, question)
                        step4.metadata["query"] = question
                        step4.end(outputs={"answer": answer})

                conversation_history.append(f"Assistant: {answer}")
                await websocket.send_text(answer)

            except Exception as e:
                print("❌ Error during interaction:", e)
                await websocket.send_text("Something went wrong. Please try again.")

    except WebSocketDisconnect:
        print("❌ WebSocket disconnected.")



# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     print("✅ WebSocket connected.")

#     conversation_history = []

#     try:
#         while True:
#             question = await websocket.receive_text()
#             conversation_history.append(f"User: {question}")
#             await websocket.send_text("THINKING...")

#             root_run = None
#             try:
#                 # Start parent trace (chain)
#                 root_run =  client.create_run(
#                     name="Legal Assistant Interaction",
#                     run_type="chain",
#                     tags=["chat"],
#                     inputs={"question": question, "conversation": conversation_history},
#                     start_time=now_iso()
#                 )
#                 print("root_run created:", root_run)
#                 if root_run is None:
#                     raise Exception("root_run is None! Check your LangSmith client initialization and API key.")

#                 # Step 1: Query Rewriter
#                 step1 =  client.create_run(
#                     name="Query Rewriter",
#                     run_type="llm",
#                     parent_run_id=root_run.id,
#                     inputs={"conversation_history": "\n".join(conversation_history)},
#                     start_time=now_iso()
#                 )

#                 rewritten_query = await query_rewriter("\n".join(conversation_history))
#                 step1.metadata["conversation"] = "\n".join(conversation_history)

#                 try:
#                     step1.end(outputs={"rewritten_query": rewritten_query}, end_time=now_iso())
#                 except Exception as e:
#                     print("❌ Failed to end step1 trace:", e)

#                 client.track_run(step1)

#                 # Step 2: Vector Search
#                 step2 =  client.create_run(
#                     name="Vector Search",
#                     run_type="retriever",
#                     parent_run_id=root_run.id,
#                     inputs={"search_input": rewritten_query or question},
#                     start_time=now_iso()
#                 )

#                 search_input = rewritten_query or question
#                 search_results = await search_query(search_input)

#                 try:
#                     step2.end(outputs={"top_k": len(search_results)}, end_time=now_iso())
#                 except Exception as e:
#                     print("❌ Failed to end step2 trace:", e)

#                 client.track_run(step2)

#                 # Step 3: Rerank Context
#                 step3 =  client.create_run(
#                     name="Rerank Context",
#                     run_type="tool",
#                     parent_run_id=root_run.id,
#                     inputs={"query": question, "num_contexts": len(search_results)},
#                     start_time=now_iso()
#                 )

#                 reranked_context = await rerank_context(question, search_results)

#                 try:
#                     step3.end(outputs={"reranked_count": len(reranked_context)}, end_time=now_iso())
#                 except Exception as e:
#                     print("❌ Failed to end step3 trace:", e)

#                 client.track_run(step3)

#                 # Step 4: Answer Generator
#                 step4 =  client.create_run(
#                     name="Answer Generator",
#                     run_type="llm",
#                     parent_run_id=root_run.id,
#                     inputs={"question": question},
#                     start_time=now_iso()
#                 )

#                 answer = await generate_answer_groq(reranked_context, question)

#                 try:
#                     step4.end(outputs={"answer": answer}, end_time=now_iso())
#                 except Exception as e:
#                     print("❌ Failed to end step4 trace:", e)

#                 client.track_run(step4)

#                 # End root run
#                 try:
#                     root_run.end(outputs={"final_answer": answer}, end_time=now_iso())
#                 except Exception as e:
#                     print("❌ Failed to end root trace:", e)

#                 client.track_run(root_run)

#                 conversation_history.append(f"Assistant: {answer}")
#                 await websocket.send_text(answer)

#             except Exception as e:
#                 print("❌ Error in chat pipeline:", e)
#                 if root_run:
#                     try:
#                         root_run.end(error=str(e), end_time=now_iso())
#                         client.track_run(root_run)
#                     except Exception as e2:
#                         print("❌ Failed to end root trace on error:", e2)

#     except WebSocketDisconnect:
#         print("❌ WebSocket disconnected.")



# # @app.get("/test-langsmith")
# # async def test_langsmith_trace():
# #     try:
# #         trace =  client.create_run(
# #             name="Test Trace",
# #             run_type="chain",
# #             start_time=now_iso(),
# #             inputs={"test_input": "ping"},
# #         )

# #         child =  client.create_run(
# #             name="Child Step",
# #             run_type="llm",
# #             start_time=now_iso(),
# #             inputs={"foo": "bar"},
# #             parent_run_id=trace.id,
# #         )
# #         try:
# #             child.end(outputs={"result": "baz"}, end_time=now_iso())
# #         except Exception as e:
# #             print("❌ Failed to end child trace:", e)
# #         client.track_run(child)

# #         try:
# #             trace.end(outputs={"done": True}, end_time=now_iso())
# #         except Exception as e:
# #             print("❌ Failed to end parent trace:", e)
# #         client.track_run(trace)

# #         return {"status": "trace sent"}
# #     except Exception as e:
# #         return {"status": "error", "message": str(e)}

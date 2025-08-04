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
    """Returns original input for greetings/simple text, otherwise rewrites legal queries"""
    last_message = conversation_history.split("\n")[-1].replace("User: ", "").strip()
    
    # Skip LLM rewrite for these cases
    if (len(last_message.split()) <= 2  # Short messages
        and not last_message.endswith("?")  # Not a question
        and not any(term in last_message.lower() for term in  # No legal terms
                   ["law", "act", "case", "clause", "contract", "section"])):
        return last_message
    
    # Improved prompt for legal queries only
    prompt = f"""Convert this into a professional legal search query. 
Return the original text ONLY if it's already a good legal query.

Input: {last_message}
Output:"""
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1  # More deterministic outputs
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data
        )
        rewritten = response.json()["choices"][0]["message"]["content"].strip()
        return rewritten if rewritten and len(rewritten) > 3 else last_message
    except Exception as e:
        print("Query rewrite error, using original:", e)
        return last_message
    

async def search_query(query, top_k=3):
    """Vector search with automatic legal context detection"""
    if not query.strip():
        return []

    embed = co.embed(
        texts=[query],
        model="embed-english-light-v3.0",
        input_type="search_document"  
    ).embeddings[0]

    results = await qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=embed,
        limit=top_k,
        with_payload=True,
        score_threshold=0.15  # Minimum relevance score
    )

    # Format results for both processing and LangSmith display
    formatted_results = [
        {
            "text": hit.payload.get("text", "")[:500] + "..." if len(hit.payload.get("text", "")) > 500 else hit.payload.get("text", ""),
            "source": hit.payload.get("pdf_name"),
            "url": hit.payload.get("url"),
            "updated": hit.payload.get("updated_date"),
            "score": float(hit.score)  # Convert to native float for LangSmith
        }
        for hit in results
    ]
    
    return formatted_results

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
    
import os
import uuid
import time
import re
from fastapi import WebSocket, WebSocketDisconnect
from langsmith.client import Client
from langsmith.schemas import RunTypeEnum

client = Client()

class InteractionAnalyzer:
    @staticmethod
    def is_simple_interaction(text: str) -> bool:
        """Detect non-substantive interactions"""
        text = text.lower().strip()
        if len(text) < 4 and not any(c.isdigit() for c in text):
            return True
            
        patterns = [
            r'^(hi|hello|hey|greetings|good\s(morning|afternoon))\b',
            r'^(thanks|thank you|cheers)\b',
            r'^(ok|okay|got it|understood)\b',
            r'^\W*$'
        ]
        return any(re.fullmatch(pattern, text) for pattern in patterns)
    
    @staticmethod
    def handle_simple_interaction(text: str) -> str:
        """Generate responses for simple interactions"""
        text = text.lower().strip()
        if re.match(r'^(hi|hello|hey)', text):
            return "Hello! I'm your legal assistant. How can I help you today?"
        elif 'thank' in text:
            return "You're welcome! Let me know if you have other legal questions."
        elif re.match(r'^(ok|got it)', text):
            return "Understood. Please continue with your legal inquiry."
        return "I'm here to help with legal matters. What would you like to know?"
    
    @staticmethod
    def get_interaction_type(text: str) -> str:
        """Classify interaction type"""
        text = text.lower().strip()
        if re.match(r'^(hi|hello|hey)', text):
            return "greeting"
        elif 'thank' in text:
            return "acknowledgment"
        elif re.match(r'^(ok|got it)', text):
            return "continuation"
        return "other"
    

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ WebSocket connected.")
    
    conversation_history = []
    analyzer = InteractionAnalyzer()

    def generate_id():
        return str(uuid.uuid4())

    try:
        while True:
            # Receive and validate input
            try:
                raw_question = await websocket.receive_text()
                question = raw_question.strip()
                if not question:
                    await websocket.send_text("Please enter a valid query")
                    continue
            except Exception as e:
                print(f"❌ Input error: {e}")
                await websocket.send_text("Error reading your input. Please try again.")
                continue

            await websocket.send_text("THINKING...")
            conversation_history.append(f"User: {question}")

            # Create root trace
            trace_id = generate_id()
            try:
                root_run = client.create_run(
                    id=trace_id,
                    name="Legal Assistant Session",
                    run_type=RunTypeEnum.chain,
                    tags=["chat"],
                    inputs={"initial_question": question}
                )
            except Exception as e:
                print(f"❌ Trace init failed: {e}")
                trace_id = None

            try:
                # 🔹 Step 1: Query Rewrite (always runs)
                rewritten_query = await query_rewriter("\n".join(conversation_history))
                if trace_id:
                    client.create_run(
                        id=generate_id(),
                        name="Query Rewrite",
                        run_type=RunTypeEnum.llm,
                        parent_run_id=trace_id,
                        inputs={
                            "conversation_history": "\n".join(conversation_history),
                            "raw_input": question
                        },
                        outputs={
                            "original": question,
                            "rewritten": rewritten_query,
                            "is_simple": analyzer.is_simple_interaction(question)
                        }
                    )

                # 🔹 Step 2: Vector Search (always runs)
                search_results = []
                if not analyzer.is_simple_interaction(question):
                    search_results = await search_query(rewritten_query or question)
                else:
                    # For simple interactions, return empty results
                    search_results = []
                
                if trace_id:
                    client.create_run(
                        id=generate_id(),
                        name="Vector Search",
                        run_type=RunTypeEnum.retriever,
                        parent_run_id=trace_id,
                        inputs={
                            "query": rewritten_query or question,
                            "is_simple": analyzer.is_simple_interaction(question)
                        },
                        outputs={
                            "results": [
                                {
                                    "source": r.get("source", "unknown"),
                                    "score": float(r["score"]),
                                    "preview": r["text"][:100] + "..." if len(r["text"]) > 100 else r["text"]
                                }
                                for r in search_results
                            ] if not analyzer.is_simple_interaction(question) else []
                        }
                    )

                # 🔹 Step 3: Rerank Context (always runs)
                reranked = []
                if not analyzer.is_simple_interaction(question) and search_results:
                    reranked = await rerank_context(question, search_results)
                else:
                    reranked = []
                
                if trace_id:
                    client.create_run(
                        id=generate_id(),
                        name="Rerank Context",
                        run_type=RunTypeEnum.tool,
                        parent_run_id=trace_id,
                        inputs={
                            "query": question,
                            "is_simple": analyzer.is_simple_interaction(question),
                            "original_count": len(search_results)
                        },
                        outputs={
                            "results": [
                                {
                                    "source": r.get("source", "unknown"),
                                    "score": float(r["score"]),
                                    "preview": r["text"][:100] + "..." if len(r["text"]) > 100 else r["text"]
                                }
                                for r in reranked[:3]
                            ] if not analyzer.is_simple_interaction(question) else []
                        }
                    )

                # 🔹 Step 4: Answer Generation (always runs)
                if analyzer.is_simple_interaction(question):
                    answer = analyzer.handle_simple_interaction(question)
                elif reranked:
                    answer = await generate_answer_groq(reranked, question)
                else:
                    answer = "I couldn't find relevant information. Please try rephrasing."
                
                if trace_id:
                    client.create_run(
                        id=generate_id(),
                        name="Answer Generation",
                        run_type=RunTypeEnum.llm,
                        parent_run_id=trace_id,
                        inputs={
                            "question": question,
                            "is_simple": analyzer.is_simple_interaction(question),
                            "context_count": len(reranked)
                        },
                        outputs={
                            "answer": answer,
                            "response_type": "simple" if analyzer.is_simple_interaction(question) else "full"
                        }
                    )

                # Final response
                conversation_history.append(f"Assistant: {answer}")
                await websocket.send_text(answer)

            except Exception as e:
                print(f"❌ Processing error: {e}")
                if trace_id:
                    client.create_run(
                        id=generate_id(),
                        name="Processing Error",
                        run_type=RunTypeEnum.tool,
                        parent_run_id=trace_id,
                        inputs={"question": question},
                        error=str(e)
                    )
                await websocket.send_text("Error processing your query. Please try again.")

    except WebSocketDisconnect:
        print("❌ Client disconnected")
    except Exception as e:
        print(f"❌ Fatal WS error: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass
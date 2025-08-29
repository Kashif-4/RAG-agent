import os
import uuid
import re
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import AsyncQdrantClient
from dotenv import load_dotenv
import requests
import cohere
from datetime import datetime, timezone
from langsmith.client import Client
from langsmith.schemas import RunTypeEnum

load_dotenv()

# Initialize clients
co = cohere.Client(os.environ["COHERE_API_KEY"])
qdrant = AsyncQdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))
client = Client(api_key=os.environ.get("LANGSMITH_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def now_iso():
    return datetime.now(timezone.utc).isoformat()

class InteractionAnalyzer:
    @staticmethod
    def is_simple_interaction(text: str) -> bool:
        text = text.lower().strip()
        patterns = [
            r'^(hi|hello|hey|greetings|good\s(morning|afternoon))\b',
            r'^(thanks|thank you|cheers)\b',
            r'^(ok|okay|got it|understood)\b',
            r'^\W*$'
        ]
        return any(re.fullmatch(pattern, text) for pattern in patterns)
    
    @staticmethod
    def handle_simple_interaction(text: str) -> str:
        text = text.lower().strip()
        if re.match(r'^(hi|hello|hey)', text):
            return "Hello! I'm your legal assistant. How can I help you today?"
        elif 'thank' in text:
            return "You're welcome! Let me know if you have other legal questions."
        return "I'm here to help with legal matters. What would you like to know?"

async def query_rewriter(conversation_history: list, current_query: str):
    """Rewrites query using only previous rewritten queries as context"""
    # Extract all previous rewritten queries for context and tracing
    previous_rewrites = [
        item["rewritten"] for item in conversation_history 
        if isinstance(item, dict) and "rewritten" in item
    ]
    
    context_used = "\n".join(f"- {q}" for q in previous_rewrites) if previous_rewrites else "None (first query)"
    
    # For first query or very simple queries
    if not previous_rewrites or len(current_query.split()) <= 3:
        return current_query
    
    prompt = f"""Given these previous legal queries:
{context_used}

Rewrite this query to be more precise for legal research:
"{current_query}"

Return ONLY the rewritten query text with no explanations:"""
    
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama3-8b-8192",
                "messages": [{
                    "role": "user",
                    "content": prompt
                }],
                "temperature": 0.1,
                "stop": ["\n\n"]
            }
        )
        
        raw_response = response.json()["choices"][0]["message"]["content"].strip()
        clean_query = re.sub(r'^"?([^"\n]+)"?$', r'\1', raw_response.split('\n')[0]).strip()
        return clean_query if clean_query else current_query
    
    except Exception as e:
        print(f"Query rewrite error: {e}")
        return current_query

async def search_query(query, top_k=3):
    if not query.strip():
        return []

    try:
        embed = co.embed(
            texts=[query],
            model="embed-english-light-v3.0",
            input_type="search_document"  
        ).embeddings[0]

        results = await qdrant.search(
            collection_name="lhc_judgments",
            query_vector=embed,
            limit=top_k,
            with_payload=True,
            score_threshold=0.15
        )

        return [
            {
                "text": hit.payload.get("text", "")[:500] + ("..." if len(hit.payload.get("text", "")) > 500 else ""),
                "source": hit.payload.get("pdf_name"),
                "url": hit.payload.get("url"),
                "updated": hit.payload.get("updated_date"),
                "score": float(hit.score)
            }
            for hit in results
        ]
    except Exception as e:
        print(f"Search error: {e}")
        return []

async def rerank_context(query, context_list):
    if not context_list:
        return []

    try:
        response = co.rerank(
            query=query,
            documents=[str(item["text"]) for item in context_list],
            top_n=min(5, len(context_list)),
            model="rerank-english-v3.0"
        )
        return [
            {
                **context_list[res.index],
                "score": res.relevance_score
            }
            for res in response.results
        ]
    except Exception as e:
        print(f"Rerank error: {e}")
        return context_list

async def generate_answer_groq(context, question):
    SYSTEM_PROMPT = """You are a legal assistant. Provide direct answers:
- No introductory phrases
- Include sources when relevant: [Source: name]
- Never say "based on the context"
- If unsure, say "The information doesn't specify\""""

    context_str = "\n".join(
        f"{chunk['text']}\n[Source: {chunk.get('source', 'Unknown')}]"
        for chunk in context
    )

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama3-8b-8192",
                "messages": [{
                    "role": "user", 
                    "content": f"{SYSTEM_PROMPT}\n\nContext:\n{context_str}\n\nQuestion: {question}\nAnswer:"
                }],
                "temperature": 0.3
            }
        )
        answer = response.json()["choices"][0]["message"]["content"]
        return re.sub(r'^(Based on .+?,\s*|According to .+?,\s*)', '', answer, flags=re.IGNORECASE).strip()
    except Exception as e:
        print(f"LLM error: {e}")
        return "Sorry, I couldn't generate a response. Please try again."

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ WebSocket connected")
    
    conversation_history = []
    analyzer = InteractionAnalyzer()

    def generate_id():
        return str(uuid.uuid4())

    try:
        while True:
            question = (await websocket.receive_text()).strip()
            if not question:
                await websocket.send_text("Please enter a valid query")
                continue

            # Start trace
            trace_id = generate_id()
            client.create_run(
                id=trace_id,
                name="Legal Assistant Session",
                run_type=RunTypeEnum.chain,
                inputs={"question": question, "conversation_history": conversation_history}
            )

            # Handle simple interactions
            if analyzer.is_simple_interaction(question):
                response = analyzer.handle_simple_interaction(question)
                await websocket.send_text(response)
                continue

            await websocket.send_text("THINKING...")
            current_interaction = {"original": question}
            conversation_history.append(current_interaction)

            try:
                # Step 1: Query Rewrite (now returns single value)
                rewritten_query = await query_rewriter(conversation_history, question)
                current_interaction["rewritten"] = rewritten_query
                
                # Prepare trace data
                previous_rewrites = [
                    h["rewritten"] for h in conversation_history[:-1] 
                    if "rewritten" in h
                ]
                
                client.create_run(
                    id=generate_id(),
                    name="Query Rewrite",
                    run_type=RunTypeEnum.llm,
                    parent_run_id=trace_id,
                    inputs={
                        "original": question,
                        "previous_rewrites": previous_rewrites,
                        "context_used": "\n".join(f"- {q}" for q in previous_rewrites) if previous_rewrites else "None"
                    },
                    outputs={"rewritten": rewritten_query}
                )

                # [Rest of your pipeline remains unchanged...]
                # Step 2: Vector Search
                search_results = await search_query(rewritten_query)
                
                client.create_run(
                    id=generate_id(),
                    name="Vector Search",
                    run_type=RunTypeEnum.retriever,
                    parent_run_id=trace_id,
                    inputs={"query": rewritten_query},
                    outputs={"results": [{"source": r["source"], "score": r["score"]} for r in search_results]}
                )

                # Step 3: Rerank
                reranked = await rerank_context(rewritten_query, search_results)
                
                client.create_run(
                    id=generate_id(),
                    name="Rerank Context",
                    run_type=RunTypeEnum.tool,
                    parent_run_id=trace_id,
                    inputs={"query": rewritten_query},
                    outputs={"top_results": [{"source": r["source"], "score": r["score"]} for r in reranked[:3]]}
                )

                # Step 4: Answer Generation
                answer = await generate_answer_groq(reranked, rewritten_query)
                
                client.create_run(
                    id=generate_id(),
                    name="Answer Generation",
                    run_type=RunTypeEnum.llm,
                    parent_run_id=trace_id,
                    inputs={"query": rewritten_query},
                    outputs={"answer": answer}
                )

                # Send response
                conversation_history.append({"response": answer})
                await websocket.send_text(answer)

            except Exception as e:
                print(f"❌ Error: {e}")
                await websocket.send_text("Error processing request. Please try again.")

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass
        print("Connection closed")
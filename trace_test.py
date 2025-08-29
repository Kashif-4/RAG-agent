from langsmith.run_trees import RunTree
import os

# Debug: print your environment setup
print("🔍 Tracing?", os.getenv("LANGCHAIN_TRACING_V2"))
print("🔑 API Key:", os.getenv("LANGCHAIN_API_KEY"))
print("📁 Project:", os.getenv("LANGCHAIN_PROJECT"))

# Create trace
rt = RunTree(
    name="Trace Test",
    run_type="chain",
    inputs={"test": "ping"},
    project_name=os.getenv("LANGCHAIN_PROJECT"),  # optional
    metadata={"source": "manual test"}
)

rt.create_child(
    name="Inner Step",
    run_type="llm",
    inputs={"step": "test"}
).end(outputs={"result": "ok"})

rt.end(outputs={"final": "complete"})

print("✅ Trace submitted.")

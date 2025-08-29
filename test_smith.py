from langsmith.client import LangSmithClient

client = LangSmithClient()

# Start a run manually
run = client.create_run(
    name="test_smith_run",
    run_type="chain",
    inputs={"query": "Hello LangSmith!"},
    metadata={"source": "manual_trace", "user": "Kashif"}
)

# Log outputs or intermediate steps
client.update_run(run_id=run.id, outputs={"response": "This is a test output"})

# End the run
client.end_run(run.id)

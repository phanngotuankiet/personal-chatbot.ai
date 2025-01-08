# Fast API
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# import smolagents services
from smolagents_examples.services.run_task_smolagents import run_task_smolagents
from types_api.types_api import RunTaskRequest

# environment
import os
from dotenv import load_dotenv
load_dotenv()

# init server
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/run-task-smolagents")
async def handle_run_task_smolagents(req: RunTaskRequest):
    """
    Run task with smolagents

    Args:
        req (RunTaskRequest): Task to run

    Returns:
        {result: str}: Result of the task
    """
    # curl:
    #  curl -X POST "http://localhost:8000/api/run-task-smolagents" \
    #      -H "Content-Type: application/json" \
    #      -d '{"task": "What is the 1st number in the Fibonacci sequence?"}'
    return await run_task_smolagents(req)

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # To run: uvicorn main:app --reload
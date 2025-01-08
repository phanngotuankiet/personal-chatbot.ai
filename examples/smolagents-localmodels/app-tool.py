# Fast API
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# smolagents
from smolagents import CodeAgent, HfApiModel

# types
from types_api.types_api import RunTaskRequest

import os
from dotenv import load_dotenv

load_dotenv()

# init FastAPI app
app: FastAPI = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# get hf token
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN is not set in the environment variables")

# model_id = "meta-llama/Llama-3.2-1B-Instruct"
model_id = "Qwen/Qwen2.5-Coder-32B-Instruct"

# init Hf model
model = HfApiModel(model_id=model_id, token=hf_token)
# init agent
agent = CodeAgent(tools=[], model=model, add_base_tools=True)

@app.post("/api/run-task-smolagents")
async def run_task_smolagents(req: RunTaskRequest):
    try:
        result = agent.run(req.task)
        
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # To run: uvicorn app-tool:app --reload
    
# curl:
#  curl -X POST "http://localhost:8000/api/run-task-smolagents" \
#      -H "Content-Type: application/json" \
#      -d '{"task": "What is the 1st number in the Fibonacci sequence?"}'
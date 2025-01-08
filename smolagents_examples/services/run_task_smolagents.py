from fastapi import HTTPException
from types_api.types_api import RunTaskRequest

# smolagents
from smolagents import CodeAgent, HfApiModel

import os
from dotenv import load_dotenv

load_dotenv()

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


async def run_task_smolagents(req: RunTaskRequest):
    try:
        result = agent.run(req.task)
        
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# export run_task_smolagents
__all__  = ["run_task_smolagents"]
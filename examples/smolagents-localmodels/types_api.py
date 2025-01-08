from pydantic import BaseModel

class RunTaskRequest(BaseModel):
    task: str

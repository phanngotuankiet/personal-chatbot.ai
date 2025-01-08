# đây là tính năng đọc nội dung từ website thông qua url và tóm tắt nội dung
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain
from langchain_ollama import OllamaLLM
from typing import Optional, List

import os
from dotenv import load_dotenv

from langchain_core.tools import Tool
from langchain_google_community import GoogleSearchAPIWrapper

load_dotenv()

search = GoogleSearchAPIWrapper()

tool = Tool(
    name="google_search",
    description="Search Google for recent results.",
    func=search.run,
)

# Thêm print để xem kết quả
result = tool.run("Obama's first name?")
print("Search Result:", result)

# Khởi tạo FastAPI app
app = FastAPI()

# Cấu hình CORS để cho phép frontend gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model cho request body
class SummarizeRequest(BaseModel):
    url: str
    chain_type: Optional[str] = "stuff"  # Mặc định là "stuff"
    model: Optional[str] = "codellama"   # Mặc định là "codellama"

    # chain_type="stuff": Đơn giản nhất - ghép tất cả văn bản thành một prompt và gọi LLM một lần để tóm tắt. 
    # Phù hợp với văn bản ngắn.

    # chain_type="refine": Tóm tắt từng phần văn bản và tinh chỉnh dần. Đầu tiên tóm tắt phần đầu, 
    # sau đó dùng kết quả đó để tóm tắt tiếp phần sau. Phù hợp với văn bản dài và cần độ chính xác cao.

    # chain_type="map_reduce": Chia văn bản thành nhiều phần nhỏ, tóm tắt từng phần (map), 
    # sau đó kết hợp các tóm tắt lại (reduce). Phù hợp với văn bản rất dài và có thể chạy song song.

    # chain_type="map_rerank": Tương tự map_reduce nhưng thêm bước xếp hạng các tóm tắt để chọn ra
    # những phần quan trọng nhất. Phù hợp khi cần tập trung vào những điểm chính của văn bản.

# Model cho search request
class SearchRequest(BaseModel):
    query: str
    k: Optional[int] = 3  # Số lượng kết quả trả về

@app.post("/summarize")
async def summarize_url(request: SummarizeRequest):
    try:
        # Khởi tạo loader với URL từ body của request - đoạn này đọc nội dung từ website
        # AI model vốn không thể truy cập được internet nên WebBaseLoader tải nội dung từ HTML và lọc ra web sạch
        loader = WebBaseLoader(request.url)
        # WebBaseLoader làm những điều sau:
        # 1. Sử dụng BeautifulSoup4 để tải nội dung HTML từ url
        # 2. parse HTML để lấy text content
        # 3. Loại bỏ các thẻ HTML, scripts JS, css
        # 4. Trả về nội dung text sạch
        docs = loader.load()

        llm = OllamaLLM(model=request.model)  # Khởi tạo model thông qua ollama server local
        chain = load_summarize_chain(llm, chain_type=request.chain_type) # tạo chain chuyên dụng để tóm tắt nội dung

        # Thực hiện tóm tắt
        result = chain.invoke(docs)  # thực hiện tóm tắt: 1. Lấy nội dung từ 'docs', 2. Tạo prompt lên model LLM yêu cầu tóm tắt, 3. Lưu vào 'result'

        return {
            "success": True,
            "summary": result,
            "metadata": {
                "url": request.url,
                "model": request.model,
                "chain_type": request.chain_type
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint kiểm tra health
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
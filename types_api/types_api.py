from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List, Dict, Literal

# Regular run task with empty tool array
class RunTaskRequest(BaseModel):
    task: str

class TranslateRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str

# Search request
class SearchRequest(BaseModel):
    query: str
    k: Optional[int] = 3  # Số lượng kết quả trả về

# /summarize
class SummarizeRequest(BaseModel):
    url: str
    chain_type: Optional[str] = "stuff"  # Mặc định là "stuff"
    model: Optional[str] = "llama2-uncensored"   # Mặc định là "codellama"

    # chain_type="stuff": Đơn giản nhất - ghép tất cả văn bản thành một prompt và gọi LLM một lần để tóm tắt. 
    # Phù hợp với văn bản ngắn.

    # chain_type="refine": Tóm tắt từng phần văn bản và tinh chỉnh dần. Đầu tiên tóm tắt phần đầu, 
    # sau đó dùng kết quả đó để tóm tắt tiếp phần sau. Phù hợp với văn bản dài và cần độ chính xác cao.

    # chain_type="map_reduce": Chia văn bản thành nhiều phần nhỏ, tóm tắt từng phần (map), 
    # sau đó kết hợp các tóm tắt lại (reduce). Phù hợp với văn bản rất dài và có thể chạy song song.

    # chain_type="map_rerank": Tương tự map_reduce nhưng thêm bước xếp hạng các tóm tắt để chọn ra
    # những phần quan trọng nhất. Phù hợp khi cần tập trung vào những điểm chính của văn bản.

# Summarize response
class SummarizeResponse(BaseModel):
    success: bool
    summary: str
    metadata: SummarizeRequest

class NewsRequest(BaseModel):
    query: str
    max_results: int = 5
    time_period: str = "day"

class NewsResponse(BaseModel):
    success: bool
    articles: List[Dict]
    summary: str
    key_insights: List[str]

class NewsParams(BaseModel):
    q: str # Query
    location: Optional[str] = None # Location Requested
    device: Optional[Literal["desktop", "mobile", "tablet"]] = "desktop"
    hl: Optional[str] = None # Google UI Language
    gl: Optional[str] = None # Google Country
    safe: Optional[Literal["active", "off"]] = "off" # Safe Search Flag
    num: Optional[int] = 10 # Number of Results
    start: Optional[int] = 0 # Pagination Offset
    api_key: str
    tbm: Optional[Literal["nws", "isch", "shop"]] = "nws" # To be match
    tbs: Optional[str] = None # custom to be search criteria
    async_: Optional[Literal["true", "false"]] = Field(default="false", alias="async")# allow async request
    output: Optional[Literal["json", "html"]] = "json" # output format
    
class WebContentResponse(BaseModel):
    title: str
    content: str
    metadata: Optional[Dict] = None

class ScrapeRequest(BaseModel):
    url: HttpUrl
    crawl_entire_site: bool = False
    max_pages: int = 10

class ProcessWebContentsResponse(BaseModel):
    title: str
    filter_prompt: str
    filtered_content: str
    num_chunks: int
    url: str
    persist_dir: str


from langchain_ollama import OllamaLLM
from tools.web_scraper_tool import WebScraperTool
from types_api.types_api import NewsParams, NewsRequest, NewsResponse, ProcessWebContentsResponse
from fastapi import HTTPException
from serpapi import GoogleSearch
import os

async def analyze_news(req: NewsRequest):
    # mục đích để test hàm gọi LLM extract ra content chính từ url để chunk và lưu vào Chroma
    try:
        # Get API key from environment
        serpapi_key: str | None = os.getenv("SERPAPI_API_KEY")
        if not serpapi_key:
            raise ValueError("SERPAPI_API_KEY not found in environment variables")

        # Calculate time range
        time_mapping = {
            "day": "d",
            "week": "w",
            "month": "m"
        }
    
        # Setup search parameters
        search_params = NewsParams(
            api_key=serpapi_key,
            # engine="google",
            q=req.query,
            tbm="nws",  # Specify khu vực tìm kiếm là chỉ có news
            num=req.max_results,
            tbs=f"qdr:{time_mapping.get(req.time_period, 'd')}",
        )

        # Tìm kiếm news trên google
        search = GoogleSearch(search_params.model_dump())
        results = search.get_dict()
        
        # lấy link của bài viết đầu tiên
        first_article = results["news_results"][0].get("link")
        # đưa link, lọc content từ link, lọc tiếp nội dung chính xác, chunk, lưu vào Chroma,
        webScraper = WebScraperTool()
        content: ProcessWebContentsResponse = await webScraper.check_content(
            url=first_article,
            query=req.query
        )
        
        return content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
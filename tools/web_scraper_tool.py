from bs4 import BeautifulSoup
import requests
from types_api.types_api import ProcessWebContentsResponse, WebContentResponse
from functions.web_content_processor import WebContentProcessor
class WebScraperTool:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
        }
    
    # nhận vào url của website, trả về content của website
    async def scrape_url(self, url: str) -> WebContentResponse:
        try:
            # Fetch webpage
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Get title
            title = soup.title.string if soup.title else ""
            
            # Get main content (customize selectors based on website structure)
            content = ""
            # Thường content sẽ nằm trong các thẻ article, main, div với class specific
            content_selectors = [
                'article', 'main',
                'div[class*="content"]',
                'div[class*="article"]',
                'div[class*="post"]'
            ]
            
            main_content = None
            for selector in content_selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            if main_content:
                # Loại bỏ script, style tags
                for tag in main_content(['script', 'style']):
                    tag.decompose()
                content = main_content.get_text(separator='\n', strip=True)
            
            return WebContentResponse(
                title=title or "",
                content=content,
                metadata={
                    "url": url,
                    "length": len(content)
                }
            )

        except Exception as e:
            raise Exception(f"Failed to scrape {url}: {str(e)}")
        
    # nhận vào url, trả về content, nhận content, trả về nội dung đã filtered (lưu trong Chroma)
    async def check_content(self, url: str, query: str) -> ProcessWebContentsResponse:
        # lấy content từ url
        extracted_content: WebContentResponse = await self.scrape_url(url)
        
        processor = WebContentProcessor()
        filtered_content = await processor.process_web_content(
            content=extracted_content.content,
            url=url,
            title=extracted_content.title,
            query=query
        )
        
        return filtered_content
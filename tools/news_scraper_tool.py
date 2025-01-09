from smolagents import Tool
from typing import List, Dict

# Duckduckgo search
from duckduckgo_search import DDGS

class NewsScraperTool(Tool):
    name = "news_scraper"
    description = "Scrapes and analyzes latest news articles based on keywords and topics"
    inputs = {
        "query": {
            "type": "string",
            "description": "Search query or topic"
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of news articles to fetch",
            "default": 5
        },
        "time_period": {
            "type": "string", 
            "description": "Time period for news (day, week, month)",
            "default": "day"
        }
    }
    output_type = "list"

    def __init__(self):
        super().__init__()
        self.ddgs = DDGS()

    def forward(self, query: str, max_results: int = 5, time_period: str = "day") -> List[Dict]:
        try:
            # Search news using DuckDuckGo
            news_results = self.ddgs.news(
                query,
                max_results=max_results,
                timelimit=time_period
            )

            formatted_results = []
            for article in news_results:
                formatted_results.append({
                    "title": article.get("title"),
                    "link": article.get("link"),
                    "date": article.get("date"),
                    "excerpt": article.get("excerpt"),
                    "source": article.get("source")
                })

            return formatted_results

        except Exception as e:
            error_msg = f"News scraping failed: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)
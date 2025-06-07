import requests
from bs4 import BeautifulSoup
import os
import sys
from typing import List, Dict

if __name__ == "__main__": 
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sources.tools.tools import Tools

class searxSearch(Tools):
    def __init__(self, base_url: str = None):
        """
        A tool for searching a SearxNG instance and extracting URLs and titles.
        """
        super().__init__()
        self.tag = "web_search"
        self.name = "searxSearch"
        self.description = "A tool for searching a SearxNG for web search"
        self.base_url = base_url or os.getenv("SEARXNG_BASE_URL", "http://127.0.0.1:8080")
        self.user_agent = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
        if not self.base_url:
            raise ValueError("SearxNG base URL must be provided either as an argument or via the SEARXNG_BASE_URL environment variable.")

    def execute(self, blocks: list, safety: bool = False) -> str:
        """Executes a search query against a SearxNG instance using POST and extracts URLs and titles."""
        if not blocks: return "Error: No search query provided."
        query = blocks[0].strip()
        if not query: return "Error: Empty search query provided."

        search_url = f"{self.base_url}/search"
        headers = {'User-Agent': self.user_agent}
        data = f"q={query}&categories=general&language=en".encode('utf-8')
        try:
            response = requests.post(search_url, headers=headers, data=data, verify=False, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            raise Exception(f"Searxng search failed. Is it running? Is SEARXNG_BASE_URL set? Error: {e}")

    def parse_results(self, html_content: str) -> List[Dict[str, str]]:
        """Parses the HTML content from SearXNG to extract search results."""
        soup = BeautifulSoup(html_content, 'html.parser')
        results = []
        for article in soup.find_all('article', class_='result'):
            url_header = article.find('a', class_='url_header')
            if url_header and url_header.has_attr('href'):
                results.append({
                    "url": url_header['href'],
                    "title": article.find('h3').text.strip() if article.find('h3') else "No Title",
                    "snippet": article.find('p', class_='content').text.strip() if article.find('p', class_='content') else "No Description"
                })
        return results
    
    def execution_failure_check(self, output: str) -> bool:
        """
        Checks if the execution failed based on the output.
        """
        return "Error" in output or "No search results" in output

    def interpreter_feedback(self, output: str) -> str:
        """
        Feedback of web search to agent.
        """
        if self.execution_failure_check(output):
            return f"Web search failed: {output}"
        return f"Web search result:\n{output}"

if __name__ == "__main__":
    search_tool = searxSearch(base_url="http://127.0.0.1:8080")
    result = search_tool.execute(["are dog better than cat?"])
    print(result)

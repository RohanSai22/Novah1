"""
Enhanced Search Agent - Pure web scraping implementation
Supports multiple search engines without APIs: DuckDuckGo, Brave, Bing, Yahoo Finance, Ask.com, Internet Archive
"""

import re
import time
import json
import asyncio
import aiohttp
import urllib.parse
from datetime import date, datetime
from typing import List, Tuple, Dict, Any, Optional, Union
from enum import Enum
from dataclasses import dataclass, asdict
import logging
from bs4 import BeautifulSoup
import requests
from fake_useragent import UserAgent

from sources.utility import pretty_print, animate_thinking
from sources.agents.agent import Agent, ExecutionManager
from sources.logger import Logger
from sources.memory import Memory
from sources.tools.searxSearch import searxSearch

class SearchEngine(Enum):
    DUCKDUCKGO = "duckduckgo"
    BRAVE = "brave"
    BING = "bing"
    YAHOO_FINANCE = "yahoo_finance"
    ASK = "ask"
    INTERNET_ARCHIVE = "internet_archive"
    STARTPAGE = "startpage"

class SearchType(Enum):
    WEB = "web"
    NEWS = "news"
    IMAGES = "images"
    VIDEOS = "videos"
    ACADEMIC = "academic"
    FINANCIAL = "financial"
    ARCHIVE = "archive"

@dataclass
class SearchResult:
    title: str
    snippet: str
    url: str
    source: str
    engine: str
    timestamp: datetime
    relevance_score: float = 0.0
    content_type: str = "text"
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class SearchQuery:
    query: str
    search_type: SearchType
    engines: List[SearchEngine]
    max_results_per_engine: int = 10
    timeout: int = 30
    filters: Dict[str, Any] = None

    def __post_init__(self):
        if self.filters is None:
            self.filters = {}

class EnhancedSearchAgent(Agent):
    def __init__(self, name, prompt_path, provider, verbose=False, browser=None):
        """
        Enhanced Search Agent with pure web scraping capabilities
        """
        super().__init__(name, prompt_path, provider, verbose, browser)
        self.tools = {
            "comprehensive_search": self.comprehensive_search,
            "duckduckgo_scrape": self.duckduckgo_scrape,
            "brave_scrape": self.brave_scrape,
            "bing_scrape": self.bing_scrape,
            "yahoo_finance_scrape": self.yahoo_finance_scrape,
            "ask_scrape": self.ask_scrape,
            "internet_archive_scrape": self.internet_archive_scrape,
            "startpage_scrape": self.startpage_scrape,
            "aggregate_results": self.aggregate_results,
            "filter_and_rank": self.filter_and_rank,
        }
        self.role = "enhanced_search"
        self.type = "enhanced_search_agent"
        self.current_query = ""
        self.search_history = []
        self.search_results = []
        self.date = self.get_today_date()
        self.logger = Logger("enhanced_search_agent.log")
        self.memory = Memory(
            self.load_prompt(prompt_path),
            recover_last_session=False,
            memory_compression=False,
            model_provider=provider.get_model_name()
        )
        
        # User agent rotation for avoiding detection
        self.ua = UserAgent()
        self.session_headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        # Rate limiting
        self.request_delays = {
            SearchEngine.DUCKDUCKGO: 1.0,
            SearchEngine.BRAVE: 1.5,
            SearchEngine.BING: 1.0,
            SearchEngine.YAHOO_FINANCE: 2.0,
            SearchEngine.ASK: 1.5,
            SearchEngine.INTERNET_ARCHIVE: 2.0,
            SearchEngine.STARTPAGE: 1.5,
        }
        
        # Search engine configurations
        self.search_configs = self._initialize_search_configs()
        
        self.search_tool = searxSearch()

    def get_today_date(self) -> str:
        """Get the current date"""
        return date.today().strftime("%B %d, %Y")
        
    def _initialize_search_configs(self) -> Dict[SearchEngine, Dict[str, Any]]:
        """Initialize search engine configurations"""
        return {
            SearchEngine.DUCKDUCKGO: {
                'base_url': 'https://duckduckgo.com/',
                'search_url': 'https://html.duckduckgo.com/html/',
                'selectors': {
                    'results': '.result',
                    'title': '.result__title a',
                    'snippet': '.result__snippet',
                    'url': '.result__title a'
                },
                'params': {
                    'q': '',
                    'b': '',  # pagination
                    'kl': 'us-en'
                }
            },
            SearchEngine.BRAVE: {
                'base_url': 'https://search.brave.com/',
                'search_url': 'https://search.brave.com/search',
                'selectors': {
                    'results': '.snippet',
                    'title': '.snippet-title',
                    'snippet': '.snippet-description',
                    'url': '.snippet-title'
                },
                'params': {
                    'q': '',
                    'source': 'web'
                }
            },
            SearchEngine.BING: {
                'base_url': 'https://www.bing.com/',
                'search_url': 'https://www.bing.com/search',
                'selectors': {
                    'results': '.b_algo',
                    'title': 'h2 a',
                    'snippet': '.b_caption p',
                    'url': 'h2 a'
                },
                'params': {
                    'q': '',
                    'form': 'QBLH',
                    'sp': '-1'
                }
            },
            SearchEngine.YAHOO_FINANCE: {
                'base_url': 'https://finance.yahoo.com/',
                'search_url': 'https://finance.yahoo.com/lookup/',
                'selectors': {
                    'results': '.Fz\\(s\\)',
                    'title': '.Fw\\(600\\)',
                    'snippet': '.C\\(\\$c-fuji-grey-k\\)',
                    'url': 'a'
                },
                'params': {
                    's': '',
                    'type': 'all'
                }
            },
            SearchEngine.ASK: {
                'base_url': 'https://www.ask.com/',
                'search_url': 'https://www.ask.com/web',
                'selectors': {
                    'results': '.PartialSearchResults-item',
                    'title': '.PartialSearchResults-item-title a',
                    'snippet': '.PartialSearchResults-item-abstract',
                    'url': '.PartialSearchResults-item-title a'
                },
                'params': {
                    'q': '',
                    'qsrc': '0',
                    'o': '0'
                }
            },
            SearchEngine.INTERNET_ARCHIVE: {
                'base_url': 'https://archive.org/',
                'search_url': 'https://archive.org/search.php',
                'selectors': {
                    'results': '.item-ia',
                    'title': '.ttl',
                    'snippet': '.snip',
                    'url': '.ttl'
                },
                'params': {
                    'query': '',
                    'and[]': 'mediatype:"texts"'
                }
            },
            SearchEngine.STARTPAGE: {
                'base_url': 'https://www.startpage.com/',
                'search_url': 'https://www.startpage.com/sp/search',
                'selectors': {
                    'results': '.w-gl__result',
                    'title': '.w-gl__result-title',
                    'snippet': '.w-gl__description',
                    'url': '.w-gl__result-title'
                },
                'params': {
                    'query': '',
                    'cat': 'web',
                    'pl': 'ext-ff',
                    'language': 'english'
                }
            }
        }

    async def comprehensive_search(self, query: str, search_type: SearchType = SearchType.WEB, 
                                 engines: List[SearchEngine] = None, max_results_per_engine: int = 10) -> Dict[str, Any]:
        """
        Perform comprehensive search across multiple engines
        """
        if engines is None:
            engines = [SearchEngine.DUCKDUCKGO, SearchEngine.BRAVE, SearchEngine.BING]
            
        search_query = SearchQuery(
            query=query,
            search_type=search_type,
            engines=engines,
            max_results_per_engine=max_results_per_engine
        )
        
        self.logger.log(f"Starting comprehensive search for: '{query}' across {len(engines)} engines")
        
        # Execute searches concurrently
        tasks = []
        for engine in engines:
            if engine == SearchEngine.DUCKDUCKGO:
                tasks.append(self.duckduckgo_scrape(query, max_results_per_engine))
            elif engine == SearchEngine.BRAVE:
                tasks.append(self.brave_scrape(query, max_results_per_engine))
            elif engine == SearchEngine.BING:
                tasks.append(self.bing_scrape(query, max_results_per_engine))
            elif engine == SearchEngine.YAHOO_FINANCE:
                tasks.append(self.yahoo_finance_scrape(query, max_results_per_engine))
            elif engine == SearchEngine.ASK:
                tasks.append(self.ask_scrape(query, max_results_per_engine))
            elif engine == SearchEngine.INTERNET_ARCHIVE:
                tasks.append(self.internet_archive_scrape(query, max_results_per_engine))
            elif engine == SearchEngine.STARTPAGE:
                tasks.append(self.startpage_scrape(query, max_results_per_engine))
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Aggregate all results
            all_results = []
            successful_engines = []
            failed_engines = []
            
            for i, result in enumerate(results):
                engine = engines[i]
                if isinstance(result, Exception):
                    self.logger.log(f"Search failed for {engine.value}: {str(result)}")
                    failed_engines.append(engine.value)
                else:
                    if result.get('success', False):
                        all_results.extend(result.get('results', []))
                        successful_engines.append(engine.value)
                    else:
                        failed_engines.append(engine.value)
            
            # Filter and rank results
            filtered_results = await self.filter_and_rank(all_results, query)
            
            self.search_results = filtered_results
            self.search_history.append({
                'query': query,
                'timestamp': datetime.now(),
                'engines': [e.value for e in engines],
                'results_count': len(filtered_results)
            })
            
            return {
                'success': True,
                'query': query,
                'results': filtered_results,
                'engines_used': successful_engines,
                'engines_failed': failed_engines,
                'total_results': len(filtered_results),
                'search_type': search_type.value,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.log(f"Comprehensive search failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'query': query,
                'results': [],
                'engines_used': [],
                'engines_failed': [e.value for e in engines]
            }

    async def duckduckgo_scrape(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """
        Scrape DuckDuckGo search results
        """
        try:
            await asyncio.sleep(self.request_delays[SearchEngine.DUCKDUCKGO])
            
            config = self.search_configs[SearchEngine.DUCKDUCKGO]
            params = config['params'].copy()
            params['q'] = query
            
            headers = self.session_headers.copy()
            headers['User-Agent'] = self.ua.random
            
            async with aiohttp.ClientSession(headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(config['search_url'], params=params) as response:
                    if response.status != 200:
                        raise Exception(f"HTTP {response.status}")
                    
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    results = []
                    result_elements = soup.select(config['selectors']['results'])[:max_results]
                    
                    for element in result_elements:
                        try:
                            title_elem = element.select_one(config['selectors']['title'])
                            snippet_elem = element.select_one(config['selectors']['snippet'])
                            url_elem = element.select_one(config['selectors']['url'])
                            
                            if title_elem and url_elem:
                                title = title_elem.get_text(strip=True)
                                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                                url = url_elem.get('href', '')
                                
                                # Clean up DuckDuckGo redirect URLs
                                if url.startswith('/l/?uddg='):
                                    url = urllib.parse.unquote(url.split('uddg=')[1])
                                
                                results.append(SearchResult(
                                    title=title,
                                    snippet=snippet,
                                    url=url,
                                    source='duckduckgo',
                                    engine=SearchEngine.DUCKDUCKGO.value,
                                    timestamp=datetime.now(),
                                    content_type='web'
                                ))
                        except Exception as e:
                            continue
                    
                    self.logger.log(f"DuckDuckGo scraping returned {len(results)} results")
                    return {
                        'success': True,
                        'results': [asdict(r) for r in results],
                        'engine': SearchEngine.DUCKDUCKGO.value,
                        'query': query
                    }
                    
        except Exception as e:
            self.logger.log(f"DuckDuckGo scraping failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'results': [],
                'engine': SearchEngine.DUCKDUCKGO.value,
                'query': query
            }

    async def brave_scrape(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """
        Scrape Brave search results
        """
        try:
            await asyncio.sleep(self.request_delays[SearchEngine.BRAVE])
            
            config = self.search_configs[SearchEngine.BRAVE]
            params = config['params'].copy()
            params['q'] = query
            
            headers = self.session_headers.copy()
            headers['User-Agent'] = self.ua.random
            
            async with aiohttp.ClientSession(headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(config['search_url'], params=params) as response:
                    if response.status != 200:
                        raise Exception(f"HTTP {response.status}")
                    
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    results = []
                    result_elements = soup.select(config['selectors']['results'])[:max_results]
                    
                    for element in result_elements:
                        try:
                            title_elem = element.select_one(config['selectors']['title'])
                            snippet_elem = element.select_one(config['selectors']['snippet'])
                            url_elem = element.select_one(config['selectors']['url'])
                            
                            if title_elem and url_elem:
                                title = title_elem.get_text(strip=True)
                                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                                url = url_elem.get('href', '')
                                
                                results.append(SearchResult(
                                    title=title,
                                    snippet=snippet,
                                    url=url,
                                    source='brave',
                                    engine=SearchEngine.BRAVE.value,
                                    timestamp=datetime.now(),
                                    content_type='web'
                                ))
                        except Exception as e:
                            continue
                    
                    self.logger.log(f"Brave scraping returned {len(results)} results")
                    return {
                        'success': True,
                        'results': [asdict(r) for r in results],
                        'engine': SearchEngine.BRAVE.value,
                        'query': query
                    }
                    
        except Exception as e:
            self.logger.log(f"Brave scraping failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'results': [],
                'engine': SearchEngine.BRAVE.value,
                'query': query
            }

    async def bing_scrape(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """
        Scrape Bing search results
        """
        try:
            await asyncio.sleep(self.request_delays[SearchEngine.BING])
            
            config = self.search_configs[SearchEngine.BING]
            params = config['params'].copy()
            params['q'] = query
            
            headers = self.session_headers.copy()
            headers['User-Agent'] = self.ua.random
            
            async with aiohttp.ClientSession(headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(config['search_url'], params=params) as response:
                    if response.status != 200:
                        raise Exception(f"HTTP {response.status}")
                    
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    results = []
                    result_elements = soup.select(config['selectors']['results'])[:max_results]
                    
                    for element in result_elements:
                        try:
                            title_elem = element.select_one(config['selectors']['title'])
                            snippet_elem = element.select_one(config['selectors']['snippet'])
                            url_elem = element.select_one(config['selectors']['url'])
                            
                            if title_elem and url_elem:
                                title = title_elem.get_text(strip=True)
                                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                                url = url_elem.get('href', '')
                                
                                results.append(SearchResult(
                                    title=title,
                                    snippet=snippet,
                                    url=url,
                                    source='bing',
                                    engine=SearchEngine.BING.value,
                                    timestamp=datetime.now(),
                                    content_type='web'
                                ))
                        except Exception as e:
                            continue
                    
                    self.logger.log(f"Bing scraping returned {len(results)} results")
                    return {
                        'success': True,
                        'results': [asdict(r) for r in results],
                        'engine': SearchEngine.BING.value,
                        'query': query
                    }
                    
        except Exception as e:
            self.logger.log(f"Bing scraping failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'results': [],
                'engine': SearchEngine.BING.value,
                'query': query
            }

    async def yahoo_finance_scrape(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """
        Scrape Yahoo Finance for financial data
        """
        try:
            await asyncio.sleep(self.request_delays[SearchEngine.YAHOO_FINANCE])
            
            config = self.search_configs[SearchEngine.YAHOO_FINANCE]
            params = config['params'].copy()
            params['s'] = query
            
            headers = self.session_headers.copy()
            headers['User-Agent'] = self.ua.random
            
            async with aiohttp.ClientSession(headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(config['search_url'], params=params) as response:
                    if response.status != 200:
                        raise Exception(f"HTTP {response.status}")
                    
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    results = []
                    # Yahoo Finance has a different structure, adapt accordingly
                    result_elements = soup.find_all('tr', {'data-row': True})[:max_results]
                    
                    for element in result_elements:
                        try:
                            symbol_elem = element.find('td', {'aria-label': 'Symbol'})
                            name_elem = element.find('td', {'aria-label': 'Name'})
                            
                            if symbol_elem and name_elem:
                                symbol = symbol_elem.get_text(strip=True)
                                name = name_elem.get_text(strip=True)
                                url = f"https://finance.yahoo.com/quote/{symbol}"
                                
                                results.append(SearchResult(
                                    title=f"{symbol} - {name}",
                                    snippet=f"Financial data for {name} ({symbol})",
                                    url=url,
                                    source='yahoo_finance',
                                    engine=SearchEngine.YAHOO_FINANCE.value,
                                    timestamp=datetime.now(),
                                    content_type='financial'
                                ))
                        except Exception as e:
                            continue
                    
                    self.logger.log(f"Yahoo Finance scraping returned {len(results)} results")
                    return {
                        'success': True,
                        'results': [asdict(r) for r in results],
                        'engine': SearchEngine.YAHOO_FINANCE.value,
                        'query': query
                    }
                    
        except Exception as e:
            self.logger.log(f"Yahoo Finance scraping failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'results': [],
                'engine': SearchEngine.YAHOO_FINANCE.value,
                'query': query
            }

    async def ask_scrape(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """
        Scrape Ask.com search results
        """
        try:
            await asyncio.sleep(self.request_delays[SearchEngine.ASK])
            
            config = self.search_configs[SearchEngine.ASK]
            params = config['params'].copy()
            params['q'] = query
            
            headers = self.session_headers.copy()
            headers['User-Agent'] = self.ua.random
            
            async with aiohttp.ClientSession(headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(config['search_url'], params=params) as response:
                    if response.status != 200:
                        raise Exception(f"HTTP {response.status}")
                    
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    results = []
                    result_elements = soup.select(config['selectors']['results'])[:max_results]
                    
                    for element in result_elements:
                        try:
                            title_elem = element.select_one(config['selectors']['title'])
                            snippet_elem = element.select_one(config['selectors']['snippet'])
                            url_elem = element.select_one(config['selectors']['url'])
                            
                            if title_elem and url_elem:
                                title = title_elem.get_text(strip=True)
                                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                                url = url_elem.get('href', '')
                                
                                results.append(SearchResult(
                                    title=title,
                                    snippet=snippet,
                                    url=url,
                                    source='ask',
                                    engine=SearchEngine.ASK.value,
                                    timestamp=datetime.now(),
                                    content_type='web'
                                ))
                        except Exception as e:
                            continue
                    
                    self.logger.log(f"Ask.com scraping returned {len(results)} results")
                    return {
                        'success': True,
                        'results': [asdict(r) for r in results],
                        'engine': SearchEngine.ASK.value,
                        'query': query
                    }
                    
        except Exception as e:
            self.logger.log(f"Ask.com scraping failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'results': [],
                'engine': SearchEngine.ASK.value,
                'query': query
            }

    async def internet_archive_scrape(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """
        Scrape Internet Archive for historical content
        """
        try:
            await asyncio.sleep(self.request_delays[SearchEngine.INTERNET_ARCHIVE])
            
            config = self.search_configs[SearchEngine.INTERNET_ARCHIVE]
            params = config['params'].copy()
            params['query'] = query
            
            headers = self.session_headers.copy()
            headers['User-Agent'] = self.ua.random
            
            async with aiohttp.ClientSession(headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(config['search_url'], params=params) as response:
                    if response.status != 200:
                        raise Exception(f"HTTP {response.status}")
                    
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    results = []
                    result_elements = soup.select(config['selectors']['results'])[:max_results]
                    
                    for element in result_elements:
                        try:
                            title_elem = element.select_one(config['selectors']['title'])
                            snippet_elem = element.select_one(config['selectors']['snippet'])
                            url_elem = element.select_one(config['selectors']['url'])
                            
                            if title_elem and url_elem:
                                title = title_elem.get_text(strip=True)
                                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                                url = url_elem.get('href', '')
                                if url.startswith('/'):
                                    url = f"https://archive.org{url}"
                                
                                results.append(SearchResult(
                                    title=title,
                                    snippet=snippet,
                                    url=url,
                                    source='internet_archive',
                                    engine=SearchEngine.INTERNET_ARCHIVE.value,
                                    timestamp=datetime.now(),
                                    content_type='archive'
                                ))
                        except Exception as e:
                            continue
                    
                    self.logger.log(f"Internet Archive scraping returned {len(results)} results")
                    return {
                        'success': True,
                        'results': [asdict(r) for r in results],
                        'engine': SearchEngine.INTERNET_ARCHIVE.value,
                        'query': query
                    }
                    
        except Exception as e:
            self.logger.log(f"Internet Archive scraping failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'results': [],
                'engine': SearchEngine.INTERNET_ARCHIVE.value,
                'query': query
            }

    async def startpage_scrape(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """
        Scrape Startpage search results
        """
        try:
            await asyncio.sleep(self.request_delays[SearchEngine.STARTPAGE])
            
            config = self.search_configs[SearchEngine.STARTPAGE]
            params = config['params'].copy()
            params['query'] = query
            
            headers = self.session_headers.copy()
            headers['User-Agent'] = self.ua.random
            
            async with aiohttp.ClientSession(headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(config['search_url'], params=params) as response:
                    if response.status != 200:
                        raise Exception(f"HTTP {response.status}")
                    
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    results = []
                    result_elements = soup.select(config['selectors']['results'])[:max_results]
                    
                    for element in result_elements:
                        try:
                            title_elem = element.select_one(config['selectors']['title'])
                            snippet_elem = element.select_one(config['selectors']['snippet'])
                            url_elem = element.select_one(config['selectors']['url'])
                            
                            if title_elem and url_elem:
                                title = title_elem.get_text(strip=True)
                                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                                url = url_elem.get('href', '')
                                
                                results.append(SearchResult(
                                    title=title,
                                    snippet=snippet,
                                    url=url,
                                    source='startpage',
                                    engine=SearchEngine.STARTPAGE.value,
                                    timestamp=datetime.now(),
                                    content_type='web'
                                ))
                        except Exception as e:
                            continue
                    
                    self.logger.log(f"Startpage scraping returned {len(results)} results")
                    return {
                        'success': True,
                        'results': [asdict(r) for r in results],
                        'engine': SearchEngine.STARTPAGE.value,
                        'query': query
                    }
                    
        except Exception as e:
            self.logger.log(f"Startpage scraping failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'results': [],
                'engine': SearchEngine.STARTPAGE.value,
                'query': query
            }

    async def filter_and_rank(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        Filter and rank search results based on relevance
        """
        try:
            # Convert dict results back to SearchResult objects if needed
            search_results = []
            for result in results:
                if isinstance(result, dict):
                    # Handle dict format
                    search_results.append(SearchResult(
                        title=result.get('title', ''),
                        snippet=result.get('snippet', ''),
                        url=result.get('url', ''),
                        source=result.get('source', ''),
                        engine=result.get('engine', ''),
                        timestamp=datetime.fromisoformat(result['timestamp']) if isinstance(result.get('timestamp'), str) else datetime.now(),
                        content_type=result.get('content_type', 'web'),
                        metadata=result.get('metadata', {})
                    ))
                else:
                    search_results.append(result)
            
            # Remove duplicates based on URL
            unique_results = {}
            for result in search_results:
                if result.url not in unique_results:
                    unique_results[result.url] = result
                else:
                    # Keep the one with better source or more recent timestamp
                    existing = unique_results[result.url]
                    if result.timestamp > existing.timestamp:
                        unique_results[result.url] = result
            
            # Calculate relevance scores
            query_terms = query.lower().split()
            for result in unique_results.values():
                score = 0.0
                title_lower = result.title.lower()
                snippet_lower = result.snippet.lower()
                
                # Title matching (higher weight)
                for term in query_terms:
                    if term in title_lower:
                        score += 3.0
                
                # Snippet matching
                for term in query_terms:
                    if term in snippet_lower:
                        score += 1.0
                
                # Source reliability bonus
                source_scores = {
                    'duckduckgo': 1.2,
                    'bing': 1.1,
                    'brave': 1.15,
                    'startpage': 1.1,
                    'yahoo_finance': 1.3,  # Higher for financial queries
                    'internet_archive': 1.0,
                    'ask': 0.9
                }
                score *= source_scores.get(result.source, 1.0)
                
                result.relevance_score = score
            
            # Sort by relevance score
            sorted_results = sorted(unique_results.values(), key=lambda x: x.relevance_score, reverse=True)
            
            # Convert back to dict format
            return [asdict(result) for result in sorted_results]
            
        except Exception as e:
            self.logger.log(f"Error filtering and ranking results: {str(e)}")
            return results  # Return original results if filtering fails

    async def aggregate_results(self, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate and analyze search results
        """
        try:
            if not search_results:
                return {
                    'success': False,
                    'message': 'No results to aggregate'
                }
            
            # Group by source
            by_source = {}
            for result in search_results:
                source = result.get('source', 'unknown')
                if source not in by_source:
                    by_source[source] = []
                by_source[source].append(result)
            
            # Calculate statistics
            total_results = len(search_results)
            avg_relevance = sum(r.get('relevance_score', 0) for r in search_results) / total_results if total_results > 0 else 0
            
            # Extract common themes
            all_titles = ' '.join([r.get('title', '') for r in search_results])
            all_snippets = ' '.join([r.get('snippet', '') for r in search_results])
            
            return {
                'success': True,
                'total_results': total_results,
                'average_relevance': avg_relevance,
                'results_by_source': {k: len(v) for k, v in by_source.items()},
                'top_results': search_results[:10],  # Top 10 results
                'content_analysis': {
                    'title_content': all_titles[:500],  # First 500 chars
                    'snippet_content': all_snippets[:1000]  # First 1000 chars
                }
            }
            
        except Exception as e:
            self.logger.log(f"Error aggregating results: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    async def process(self, prompt: str, execution_manager: Optional[ExecutionManager] = None, speech_module=None):
        return await self.execute(prompt, execution_manager)

    async def execute(self, prompt: str, execution_manager: Optional[ExecutionManager] = None) -> dict:
        self.logger.info(f"Executing search task: {prompt}")
        
        try:
            # The search tool expects a list of queries
            raw_results = self.search_tool.execute([prompt])
            
            # Parse the raw string into a list of dictionaries
            parsed_results = self.parse_raw_results(raw_results)

            if execution_manager:
                execution_manager.update_state({
                    "search": {"results": parsed_results, "search_queries": [prompt]}
                })
            
            summary = f"Found {len(parsed_results)} results for '{prompt}'."
            self.logger.info(summary)
            return {"success": True, "summary": summary, "results": parsed_results}
        except Exception as e:
            self.logger.error(f"Search execution failed: {e}")
            return {"success": False, "error": str(e)}

    def parse_raw_results(self, raw_data: str) -> List[Dict[str, str]]:
        """Parses the 'Title:...\nSnippet:...\nLink:...' string into a list of dicts."""
        if "No search results" in raw_data:
            return []
        
        results = []
        entries = raw_data.strip().split("\n\n")
        for entry in entries:
            lines = entry.strip().split("\n")
            result = {}
            for line in lines:
                if line.startswith("Title:"):
                    result["title"] = line[6:].strip()
                elif line.startswith("Snippet:"):
                    result["snippet"] = line[8:].strip()
                elif line.startswith("Link:"):
                    result["url"] = line[5:].strip() # Changed from 'link' to 'url'
            if result:
                result["source"] = "SearxNG" # Add source
                results.append(result)
        return results

    async def comprehensive_search_and_extract(self, query: str, max_results_per_source: int = 5) -> dict:
        await asyncio.sleep(2)
        mock_results = [
            {'title': f'Result 1 for {query}', 'snippet': 'This is a great result about AI investment.', 'url': 'https://example.com/ai-invest-1', 'source': 'google'},
            {'title': f'Result 2 for {query}', 'snippet': 'Another useful finding on AI startups.', 'url': 'https://example.com/ai-invest-2', 'source': 'bing'}
        ]
        return {
            "success": True, "query": query, "total_results": 2, 
            "search_results": mock_results, "extracted_data": [], "links_processed": []
        }

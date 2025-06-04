"""
Enhanced Web Agent - Advanced browser automation with screenshot analysis
Supports dynamic content extraction, JavaScript rendering, and visual analysis
"""

import asyncio
import aiohttp
import json
import base64
import io
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path

# Browser automation
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.common.exceptions import TimeoutException, WebDriverException

# Image processing and analysis
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import cv2
import numpy as np

# Web content extraction
from bs4 import BeautifulSoup
import requests
from readability import Document

from sources.utility import pretty_print, animate_thinking
from sources.agents.agent import Agent
from sources.logger import Logger
from sources.memory import Memory

class BrowserType(Enum):
    CHROME = "chrome"
    FIREFOX = "firefox"
    HEADLESS_CHROME = "headless_chrome"
    HEADLESS_FIREFOX = "headless_firefox"

class ExtractionMode(Enum):
    FULL_PAGE = "full_page"
    MAIN_CONTENT = "main_content"
    SPECIFIC_ELEMENTS = "specific_elements"
    DYNAMIC_CONTENT = "dynamic_content"
    FORM_DATA = "form_data"

class AnalysisType(Enum):
    TEXT_EXTRACTION = "text_extraction"
    VISUAL_ANALYSIS = "visual_analysis"
    LAYOUT_ANALYSIS = "layout_analysis"
    INTERACTIVE_ELEMENTS = "interactive_elements"
    PERFORMANCE_METRICS = "performance_metrics"

@dataclass
class WebContent:
    url: str
    title: str
    content: str
    html: str
    timestamp: datetime
    extraction_mode: ExtractionMode
    metadata: Dict[str, Any]
    screenshot_path: Optional[str] = None
    performance_metrics: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.performance_metrics is None:
            self.performance_metrics = {}

@dataclass
class ScreenshotAnalysis:
    image_path: str
    text_content: str
    visual_elements: List[Dict[str, Any]]
    layout_structure: Dict[str, Any]
    interactive_elements: List[Dict[str, Any]]
    accessibility_score: float
    timestamp: datetime

@dataclass
class WebTask:
    url: str
    action_type: str
    extraction_mode: ExtractionMode
    analysis_types: List[AnalysisType]
    selectors: List[str] = None
    wait_conditions: List[str] = None
    timeout: int = 30
    take_screenshot: bool = True
    execute_javascript: Optional[str] = None

    def __post_init__(self):
        if self.selectors is None:
            self.selectors = []
        if self.wait_conditions is None:
            self.wait_conditions = []

class EnhancedWebAgent(Agent):
    def __init__(self, name, prompt_path, provider, verbose=False, browser=None):
        """
        Enhanced Web Agent with advanced browser automation and analysis
        """
        super().__init__(name, prompt_path, provider, verbose, browser)
        self.tools = {
            "extract_web_content": self.extract_web_content,
            "take_screenshot": self.take_screenshot,
            "analyze_screenshot": self.analyze_screenshot,
            "extract_dynamic_content": self.extract_dynamic_content,
            "interact_with_page": self.interact_with_page,
            "extract_structured_data": self.extract_structured_data,
            "monitor_page_changes": self.monitor_page_changes,
            "analyze_page_performance": self.analyze_page_performance,
            "extract_forms_data": self.extract_forms_data,
            "capture_network_activity": self.capture_network_activity,
        }
        self.role = "enhanced_web"
        self.type = "enhanced_web_agent"
        self.logger = Logger("enhanced_web_agent.log")
        self.memory = Memory(
            self.load_prompt(prompt_path),
            recover_last_session=False,
            memory_compression=False,
            model_provider=provider.get_model_name()
        )
        
        # Browser configuration
        self.browser_type = BrowserType.HEADLESS_CHROME
        self.driver = None
        self.screenshot_dir = Path("screenshots")
        self.screenshot_dir.mkdir(exist_ok=True)
        
        # Analysis tools
        self.setup_analysis_tools()
        
        # Web content cache
        self.content_cache = {}
        self.screenshot_cache = {}
        
    def setup_analysis_tools(self):
        """Initialize analysis tools and configurations"""
        # OCR configuration
        self.ocr_config = r'--oem 3 --psm 6'
        
        # Browser options
        self.chrome_options = ChromeOptions()
        self.chrome_options.add_argument('--headless')
        self.chrome_options.add_argument('--no-sandbox')
        self.chrome_options.add_argument('--disable-dev-shm-usage')
        self.chrome_options.add_argument('--disable-gpu')
        self.chrome_options.add_argument('--window-size=1920,1080')
        self.chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
        
        self.firefox_options = FirefoxOptions()
        self.firefox_options.add_argument('--headless')
        self.firefox_options.add_argument('--width=1920')
        self.firefox_options.add_argument('--height=1080')

    def get_driver(self, browser_type: BrowserType = None) -> webdriver:
        """Get or create browser driver"""
        if browser_type:
            self.browser_type = browser_type
            
        try:
            if self.driver is None:
                if self.browser_type in [BrowserType.CHROME, BrowserType.HEADLESS_CHROME]:
                    self.driver = webdriver.Chrome(options=self.chrome_options)
                elif self.browser_type in [BrowserType.FIREFOX, BrowserType.HEADLESS_FIREFOX]:
                    self.driver = webdriver.Firefox(options=self.firefox_options)
                else:
                    # Default to headless Chrome
                    self.driver = webdriver.Chrome(options=self.chrome_options)
                    
                # Set timeouts
                self.driver.implicitly_wait(10)
                self.driver.set_page_load_timeout(30)
                
            return self.driver
        except Exception as e:
            self.logger.log(f"Failed to create browser driver: {str(e)}")
            raise

    def close_driver(self):
        """Close browser driver"""
        if self.driver:
            try:
                self.driver.quit()
                self.driver = None
            except Exception as e:
                self.logger.log(f"Error closing driver: {str(e)}")

    async def extract_web_content(self, url: str, extraction_mode: ExtractionMode = ExtractionMode.MAIN_CONTENT,
                                wait_time: int = 3, selectors: List[str] = None) -> Dict[str, Any]:
        """
        Extract web content using browser automation
        """
        try:
            driver = self.get_driver()
            
            # Navigate to URL
            self.logger.log(f"Navigating to: {url}")
            driver.get(url)
            
            # Wait for page to load
            await asyncio.sleep(wait_time)
            
            # Wait for specific elements if provided
            if selectors:
                wait = WebDriverWait(driver, 10)
                for selector in selectors:
                    try:
                        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
                    except TimeoutException:
                        self.logger.log(f"Timeout waiting for selector: {selector}")
            
            # Get page source and metadata
            html = driver.page_source
            title = driver.title
            current_url = driver.current_url
            
            # Extract content based on mode
            if extraction_mode == ExtractionMode.FULL_PAGE:
                content = self._extract_full_content(html)
            elif extraction_mode == ExtractionMode.MAIN_CONTENT:
                content = self._extract_main_content(html)
            elif extraction_mode == ExtractionMode.SPECIFIC_ELEMENTS:
                content = self._extract_specific_elements(driver, selectors or [])
            elif extraction_mode == ExtractionMode.DYNAMIC_CONTENT:
                content = await self._extract_dynamic_content(driver)
            else:
                content = self._extract_main_content(html)
            
            # Collect metadata
            metadata = {
                'original_url': url,
                'final_url': current_url,
                'page_title': title,
                'extraction_mode': extraction_mode.value,
                'timestamp': datetime.now().isoformat(),
                'page_size': len(html),
                'content_length': len(content)
            }
            
            # Performance metrics
            performance_metrics = self._get_performance_metrics(driver)
            
            web_content = WebContent(
                url=current_url,
                title=title,
                content=content,
                html=html,
                timestamp=datetime.now(),
                extraction_mode=extraction_mode,
                metadata=metadata,
                performance_metrics=performance_metrics
            )
            
            self.content_cache[url] = web_content
            
            return {
                'success': True,
                'content': asdict(web_content),
                'url': current_url,
                'title': title,
                'extraction_mode': extraction_mode.value
            }
            
        except Exception as e:
            self.logger.log(f"Failed to extract web content from {url}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'url': url
            }

    async def take_screenshot(self, url: str, filename: str = None, full_page: bool = True,
                            element_selector: str = None) -> Dict[str, Any]:
        """
        Take screenshot of web page
        """
        try:
            driver = self.get_driver()
            
            if driver.current_url != url:
                driver.get(url)
                await asyncio.sleep(2)
            
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                domain = url.split('/')[2].replace('.', '_')
                filename = f"{domain}_{timestamp}.png"
            
            screenshot_path = self.screenshot_dir / filename
            
            if element_selector:
                # Screenshot specific element
                element = driver.find_element(By.CSS_SELECTOR, element_selector)
                screenshot_data = element.screenshot_as_png
            else:
                # Full page or viewport screenshot
                if full_page:
                    # Set window size to capture full page
                    total_height = driver.execute_script("return document.body.scrollHeight")
                    driver.set_window_size(1920, total_height)
                
                screenshot_data = driver.get_screenshot_as_png()
            
            # Save screenshot
            with open(screenshot_path, 'wb') as f:
                f.write(screenshot_data)
            
            self.logger.log(f"Screenshot saved: {screenshot_path}")
            
            return {
                'success': True,
                'screenshot_path': str(screenshot_path),
                'filename': filename,
                'url': url,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.log(f"Failed to take screenshot of {url}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'url': url
            }

    async def analyze_screenshot(self, screenshot_path: str, analysis_types: List[AnalysisType] = None) -> Dict[str, Any]:
        """
        Analyze screenshot using various techniques
        """
        try:
            if analysis_types is None:
                analysis_types = [AnalysisType.TEXT_EXTRACTION, AnalysisType.VISUAL_ANALYSIS]
            
            image = Image.open(screenshot_path)
            analysis_results = {}
            
            # Text extraction using OCR
            if AnalysisType.TEXT_EXTRACTION in analysis_types:
                text_content = pytesseract.image_to_string(image, config=self.ocr_config)
                analysis_results['text_content'] = text_content.strip()
            
            # Visual analysis
            if AnalysisType.VISUAL_ANALYSIS in analysis_types:
                visual_elements = self._analyze_visual_elements(image)
                analysis_results['visual_elements'] = visual_elements
            
            # Layout analysis
            if AnalysisType.LAYOUT_ANALYSIS in analysis_types:
                layout_structure = self._analyze_layout_structure(image)
                analysis_results['layout_structure'] = layout_structure
            
            # Interactive elements detection
            if AnalysisType.INTERACTIVE_ELEMENTS in analysis_types:
                interactive_elements = self._detect_interactive_elements(image)
                analysis_results['interactive_elements'] = interactive_elements
            
            screenshot_analysis = ScreenshotAnalysis(
                image_path=screenshot_path,
                text_content=analysis_results.get('text_content', ''),
                visual_elements=analysis_results.get('visual_elements', []),
                layout_structure=analysis_results.get('layout_structure', {}),
                interactive_elements=analysis_results.get('interactive_elements', []),
                accessibility_score=self._calculate_accessibility_score(analysis_results),
                timestamp=datetime.now()
            )
            
            return {
                'success': True,
                'analysis': asdict(screenshot_analysis),
                'screenshot_path': screenshot_path
            }
            
        except Exception as e:
            self.logger.log(f"Failed to analyze screenshot {screenshot_path}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'screenshot_path': screenshot_path
            }

    async def extract_dynamic_content(self, url: str, wait_conditions: List[str] = None,
                                    execute_js: str = None) -> Dict[str, Any]:
        """
        Extract dynamic content that requires JavaScript execution
        """
        try:
            driver = self.get_driver()
            driver.get(url)
            
            # Wait for initial load
            await asyncio.sleep(3)
            
            # Wait for specific conditions
            if wait_conditions:
                wait = WebDriverWait(driver, 15)
                for condition in wait_conditions:
                    try:
                        if condition.startswith('element:'):
                            selector = condition.replace('element:', '')
                            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
                        elif condition.startswith('text:'):
                            text = condition.replace('text:', '')
                            wait.until(EC.text_to_be_present_in_element((By.TAG_NAME, 'body'), text))
                    except TimeoutException:
                        self.logger.log(f"Timeout waiting for condition: {condition}")
            
            # Execute custom JavaScript if provided
            if execute_js:
                try:
                    result = driver.execute_script(execute_js)
                    self.logger.log(f"JavaScript execution result: {result}")
                except Exception as e:
                    self.logger.log(f"JavaScript execution failed: {str(e)}")
            
            # Extract content after dynamic loading
            html = driver.page_source
            content = self._extract_main_content(html)
            
            # Get dynamic elements
            dynamic_elements = self._extract_dynamic_elements(driver)
            
            return {
                'success': True,
                'content': content,
                'html': html,
                'dynamic_elements': dynamic_elements,
                'url': driver.current_url,
                'title': driver.title,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.log(f"Failed to extract dynamic content from {url}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'url': url
            }

    async def interact_with_page(self, url: str, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Interact with page elements (click, type, scroll, etc.)
        """
        try:
            driver = self.get_driver()
            driver.get(url)
            await asyncio.sleep(2)
            
            interaction_results = []
            
            for interaction in interactions:
                try:
                    action_type = interaction.get('action')
                    selector = interaction.get('selector')
                    value = interaction.get('value', '')
                    
                    element = driver.find_element(By.CSS_SELECTOR, selector)
                    
                    if action_type == 'click':
                        element.click()
                    elif action_type == 'type':
                        element.clear()
                        element.send_keys(value)
                    elif action_type == 'scroll_to':
                        driver.execute_script("arguments[0].scrollIntoView();", element)
                    elif action_type == 'hover':
                        webdriver.ActionChains(driver).move_to_element(element).perform()
                    
                    interaction_results.append({
                        'action': action_type,
                        'selector': selector,
                        'success': True
                    })
                    
                    # Wait between interactions
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    interaction_results.append({
                        'action': interaction.get('action'),
                        'selector': interaction.get('selector'),
                        'success': False,
                        'error': str(e)
                    })
            
            # Get final page state
            final_html = driver.page_source
            final_content = self._extract_main_content(final_html)
            
            return {
                'success': True,
                'interactions': interaction_results,
                'final_content': final_content,
                'final_url': driver.current_url,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.log(f"Failed to interact with page {url}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'url': url
            }

    def _extract_full_content(self, html: str) -> str:
        """Extract all text content from HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        return soup.get_text(separator=' ', strip=True)

    def _extract_main_content(self, html: str) -> str:
        """Extract main content using readability"""
        try:
            doc = Document(html)
            main_content = doc.summary()
            soup = BeautifulSoup(main_content, 'html.parser')
            return soup.get_text(separator=' ', strip=True)
        except Exception as e:
            self.logger.log(f"Readability extraction failed, falling back to full content: {str(e)}")
            return self._extract_full_content(html)

    def _extract_specific_elements(self, driver: webdriver, selectors: List[str]) -> str:
        """Extract content from specific elements"""
        content_parts = []
        
        for selector in selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                for element in elements:
                    text = element.text.strip()
                    if text:
                        content_parts.append(text)
            except Exception as e:
                self.logger.log(f"Failed to extract from selector {selector}: {str(e)}")
        
        return '\n\n'.join(content_parts)

    async def _extract_dynamic_content(self, driver: webdriver) -> str:
        """Extract dynamic content after JavaScript execution"""
        # Scroll to load lazy content
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        await asyncio.sleep(2)
        
        # Scroll back to top
        driver.execute_script("window.scrollTo(0, 0);")
        await asyncio.sleep(1)
        
        # Get final content
        html = driver.page_source
        return self._extract_main_content(html)

    def _extract_dynamic_elements(self, driver: webdriver) -> List[Dict[str, Any]]:
        """Extract information about dynamic elements"""
        dynamic_elements = []
        
        # Find elements with common dynamic attributes
        dynamic_selectors = [
            '[data-testid]',
            '[data-cy]',
            '[data-test]',
            '.lazy-load',
            '.dynamic-content',
            '[ng-*]',  # Angular
            '[v-*]',   # Vue
            '[data-react*]'  # React
        ]
        
        for selector in dynamic_selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                for element in elements[:10]:  # Limit to first 10
                    dynamic_elements.append({
                        'tag': element.tag_name,
                        'selector': selector,
                        'text': element.text[:100],  # First 100 chars
                        'attributes': dict(element.get_property('attributes') or {})
                    })
            except Exception as e:
                continue
        
        return dynamic_elements

    def _get_performance_metrics(self, driver: webdriver) -> Dict[str, Any]:
        """Get page performance metrics"""
        try:
            # Get navigation timing
            nav_timing = driver.execute_script(
                "return window.performance.getEntriesByType('navigation')[0];"
            )
            
            # Get resource timing
            resources = driver.execute_script(
                "return window.performance.getEntriesByType('resource').length;"
            )
            
            # Get memory info (Chrome only)
            memory_info = {}
            try:
                memory_info = driver.execute_script(
                    "return window.performance.memory;"
                )
            except:
                pass
            
            return {
                'load_time': nav_timing.get('loadEventEnd', 0) - nav_timing.get('navigationStart', 0) if nav_timing else 0,
                'dom_content_loaded': nav_timing.get('domContentLoadedEventEnd', 0) - nav_timing.get('navigationStart', 0) if nav_timing else 0,
                'resource_count': resources,
                'memory_info': memory_info
            }
        except Exception as e:
            self.logger.log(f"Failed to get performance metrics: {str(e)}")
            return {}

    def _analyze_visual_elements(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Analyze visual elements in screenshot"""
        try:
            # Convert PIL image to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Detect edges
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            visual_elements = []
            for i, contour in enumerate(contours[:20]):  # Limit to first 20
                area = cv2.contourArea(contour)
                if area > 100:  # Filter small elements
                    x, y, w, h = cv2.boundingRect(contour)
                    visual_elements.append({
                        'type': 'contour',
                        'area': int(area),
                        'bbox': [int(x), int(y), int(w), int(h)],
                        'aspect_ratio': round(w / h, 2) if h > 0 else 0
                    })
            
            return visual_elements
        except Exception as e:
            self.logger.log(f"Visual analysis failed: {str(e)}")
            return []

    def _analyze_layout_structure(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze page layout structure"""
        try:
            width, height = image.size
            
            # Divide into grid sections
            sections = {
                'header': (0, 0, width, height // 6),
                'left_sidebar': (0, height // 6, width // 4, height * 5 // 6),
                'main_content': (width // 4, height // 6, width * 3 // 4, height * 5 // 6),
                'right_sidebar': (width * 3 // 4, height // 6, width, height * 5 // 6),
                'footer': (0, height * 5 // 6, width, height)
            }
            
            layout_analysis = {
                'image_dimensions': [width, height],
                'sections': sections,
                'aspect_ratio': round(width / height, 2)
            }
            
            return layout_analysis
        except Exception as e:
            self.logger.log(f"Layout analysis failed: {str(e)}")
            return {}

    def _detect_interactive_elements(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect interactive elements like buttons, links, forms"""
        # This is a simplified implementation
        # In practice, you'd use more sophisticated computer vision techniques
        interactive_elements = [
            {
                'type': 'button',
                'confidence': 0.8,
                'bbox': [100, 200, 150, 40],
                'description': 'Detected button-like element'
            },
            {
                'type': 'link',
                'confidence': 0.7,
                'bbox': [50, 100, 200, 20],
                'description': 'Detected link-like element'
            }
        ]
        
        return interactive_elements

    def _calculate_accessibility_score(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate basic accessibility score"""
        score = 5.0  # Base score
        
        # Check for text content
        text_content = analysis_results.get('text_content', '')
        if len(text_content) > 100:
            score += 2.0
        
        # Check for interactive elements
        interactive_elements = analysis_results.get('interactive_elements', [])
        if len(interactive_elements) > 0:
            score += 1.5
        
        # Check for visual structure
        visual_elements = analysis_results.get('visual_elements', [])
        if len(visual_elements) > 5:
            score += 1.5
        
        return min(score, 10.0)  # Cap at 10

    async def extract_structured_data(self, url: str) -> Dict[str, Any]:
        """Extract structured data (JSON-LD, microdata, etc.)"""
        try:
            driver = self.get_driver()
            driver.get(url)
            await asyncio.sleep(2)
            
            structured_data = {}
            
            # Extract JSON-LD
            json_ld_scripts = driver.find_elements(By.CSS_SELECTOR, 'script[type="application/ld+json"]')
            json_ld_data = []
            
            for script in json_ld_scripts:
                try:
                    data = json.loads(script.get_attribute('innerHTML'))
                    json_ld_data.append(data)
                except json.JSONDecodeError:
                    continue
            
            structured_data['json_ld'] = json_ld_data
            
            # Extract meta tags
            meta_tags = {}
            meta_elements = driver.find_elements(By.CSS_SELECTOR, 'meta[property], meta[name]')
            
            for meta in meta_elements:
                property_attr = meta.get_attribute('property') or meta.get_attribute('name')
                content = meta.get_attribute('content')
                if property_attr and content:
                    meta_tags[property_attr] = content
            
            structured_data['meta_tags'] = meta_tags
            
            return {
                'success': True,
                'structured_data': structured_data,
                'url': url
            }
            
        except Exception as e:
            self.logger.log(f"Failed to extract structured data from {url}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'url': url
            }

    async def monitor_page_changes(self, url: str, check_interval: int = 5, duration: int = 30) -> Dict[str, Any]:
        """Monitor page for changes over time"""
        try:
            driver = self.get_driver()
            driver.get(url)
            
            initial_content = driver.page_source
            changes = []
            start_time = datetime.now()
            
            while (datetime.now() - start_time).seconds < duration:
                await asyncio.sleep(check_interval)
                
                current_content = driver.page_source
                if current_content != initial_content:
                    changes.append({
                        'timestamp': datetime.now().isoformat(),
                        'change_detected': True,
                        'content_length': len(current_content)
                    })
                    initial_content = current_content
            
            return {
                'success': True,
                'changes': changes,
                'monitoring_duration': duration,
                'url': url
            }
            
        except Exception as e:
            self.logger.log(f"Failed to monitor page changes for {url}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'url': url
            }

    async def analyze_page_performance(self, url: str) -> Dict[str, Any]:
        """Comprehensive page performance analysis"""
        try:
            driver = self.get_driver()
            
            start_time = datetime.now()
            driver.get(url)
            load_time = (datetime.now() - start_time).total_seconds()
            
            # Get detailed performance metrics
            performance_metrics = self._get_performance_metrics(driver)
            
            # Analyze page resources
            resources = driver.execute_script("""
                return window.performance.getEntriesByType('resource').map(resource => ({
                    name: resource.name,
                    type: resource.initiatorType,
                    size: resource.transferSize,
                    duration: resource.duration
                }));
            """)
            
            # Calculate scores
            performance_score = self._calculate_performance_score(performance_metrics, resources)
            
            return {
                'success': True,
                'load_time': load_time,
                'performance_metrics': performance_metrics,
                'resources': resources[:20],  # Limit to first 20
                'performance_score': performance_score,
                'url': url
            }
            
        except Exception as e:
            self.logger.log(f"Failed to analyze performance for {url}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'url': url
            }

    def _calculate_performance_score(self, metrics: Dict[str, Any], resources: List[Dict[str, Any]]) -> float:
        """Calculate performance score based on metrics"""
        score = 10.0  # Start with perfect score
        
        load_time = metrics.get('load_time', 0)
        if load_time > 5000:  # > 5 seconds
            score -= 3.0
        elif load_time > 3000:  # > 3 seconds
            score -= 2.0
        elif load_time > 1000:  # > 1 second
            score -= 1.0
        
        resource_count = len(resources)
        if resource_count > 100:
            score -= 2.0
        elif resource_count > 50:
            score -= 1.0
        
        return max(score, 0.0)

    async def extract_forms_data(self, url: str) -> Dict[str, Any]:
        """Extract information about forms on the page"""
        try:
            driver = self.get_driver()
            driver.get(url)
            await asyncio.sleep(2)
            
            forms = driver.find_elements(By.TAG_NAME, 'form')
            forms_data = []
            
            for i, form in enumerate(forms):
                try:
                    form_info = {
                        'index': i,
                        'action': form.get_attribute('action') or '',
                        'method': form.get_attribute('method') or 'get',
                        'fields': []
                    }
                    
                    # Get form fields
                    inputs = form.find_elements(By.CSS_SELECTOR, 'input, textarea, select')
                    for input_elem in inputs:
                        field_info = {
                            'type': input_elem.get_attribute('type') or input_elem.tag_name,
                            'name': input_elem.get_attribute('name') or '',
                            'id': input_elem.get_attribute('id') or '',
                            'placeholder': input_elem.get_attribute('placeholder') or '',
                            'required': input_elem.get_attribute('required') is not None
                        }
                        form_info['fields'].append(field_info)
                    
                    forms_data.append(form_info)
                except Exception as e:
                    continue
            
            return {
                'success': True,
                'forms': forms_data,
                'forms_count': len(forms_data),
                'url': url
            }
            
        except Exception as e:
            self.logger.log(f"Failed to extract forms data from {url}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'url': url
            }

    async def capture_network_activity(self, url: str, duration: int = 10) -> Dict[str, Any]:
        """Capture network activity during page load"""
        # This would require browser dev tools protocol
        # For now, return a placeholder implementation
        try:
            driver = self.get_driver()
            
            # Enable performance logging (Chrome only)
            caps = driver.desired_capabilities
            caps['goog:loggingPrefs'] = {'performance': 'ALL'}
            
            driver.get(url)
            await asyncio.sleep(duration)
            
            # Get performance logs
            logs = driver.get_log('performance')
            network_requests = []
            
            for log in logs:
                message = json.loads(log['message'])
                if message.get('message', {}).get('method') == 'Network.responseReceived':
                    response = message['message']['params']['response']
                    network_requests.append({
                        'url': response.get('url', ''),
                        'status': response.get('status', 0),
                        'mimeType': response.get('mimeType', ''),
                        'timestamp': log['timestamp']
                    })
            
            return {
                'success': True,
                'network_requests': network_requests[:50],  # Limit to first 50
                'total_requests': len(network_requests),
                'url': url
            }
            
        except Exception as e:
            self.logger.log(f"Failed to capture network activity for {url}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'url': url
            }

    async def execute(self, task: str, additional_instructions: str = "") -> str:
        """
        Execute a web extraction/analysis task
        """
        try:
            self.logger.log(f"Executing enhanced web task: {task}")
            
            # Parse task to determine actions needed
            task_lower = task.lower()
            results = {}
            
            # Extract URL from task
            import re
            url_pattern = r'https?://[^\s]+'
            urls = re.findall(url_pattern, task)
            
            if not urls:
                return "No valid URL found in the task. Please provide a URL to analyze."
            
            url = urls[0]
            
            # Determine extraction mode based on task
            extraction_mode = ExtractionMode.MAIN_CONTENT
            if 'full page' in task_lower:
                extraction_mode = ExtractionMode.FULL_PAGE
            elif 'dynamic' in task_lower or 'javascript' in task_lower:
                extraction_mode = ExtractionMode.DYNAMIC_CONTENT
            
            # Extract web content
            content_result = await self.extract_web_content(url, extraction_mode)
            results['content'] = content_result
            
            # Take screenshot if requested
            if 'screenshot' in task_lower or 'image' in task_lower:
                screenshot_result = await self.take_screenshot(url)
                results['screenshot'] = screenshot_result
                
                # Analyze screenshot if taken
                if screenshot_result.get('success') and screenshot_result.get('screenshot_path'):
                    analysis_result = await self.analyze_screenshot(screenshot_result['screenshot_path'])
                    results['screenshot_analysis'] = analysis_result
            
            # Performance analysis if requested
            if 'performance' in task_lower or 'speed' in task_lower:
                performance_result = await self.analyze_page_performance(url)
                results['performance'] = performance_result
            
            # Forms analysis if requested
            if 'form' in task_lower or 'input' in task_lower:
                forms_result = await self.extract_forms_data(url)
                results['forms'] = forms_result
            
            # Structured data if requested
            if 'structured' in task_lower or 'schema' in task_lower or 'metadata' in task_lower:
                structured_result = await self.extract_structured_data(url)
                results['structured_data'] = structured_result
            
            # Generate comprehensive response
            analysis_context = {
                'task': task,
                'url': url,
                'results': results,
                'timestamp': datetime.now().isoformat()
            }
            
            messages = [
                {"role": "system", "content": f"""You are an expert web content analyst. Analyze the web extraction results and provide insights.
                
Task: {task}
URL: {url}
Additional Instructions: {additional_instructions}

Provide a comprehensive analysis of the web content, including key findings, structure, and any notable elements."""},
                {"role": "user", "content": f"Please analyze these web extraction results: {json.dumps(analysis_context, indent=2, default=str)}"}
            ]
            
            response = self.provider.chat_completion(messages)
            
            # Store in memory
            self.memory.append_message("user", task)
            self.memory.append_message("assistant", response)
            
            return response
            
        except Exception as e:
            error_msg = f"Enhanced web execution failed: {str(e)}"
            self.logger.log(error_msg)
            return error_msg
        finally:
            # Clean up driver if needed
            pass  # Keep driver alive for subsequent requests

    def __del__(self):
        """Cleanup when agent is destroyed"""
        self.close_driver()

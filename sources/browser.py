import os
import sys
import time
import random
import shutil
import tempfile
import re
from typing import List
from urllib.parse import urlparse

# Add project root to path to ensure correct imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException, ElementClickInterceptedException
import undetected_chromedriver as uc
import markdownify
from bs4 import BeautifulSoup

from sources.utility import pretty_print
from sources.logger import Logger

def get_chrome_path() -> str:
    """Finds the path to the Chrome executable."""
    if sys.platform.startswith("win"):
        # Common paths for Windows
        possible_paths = [
            os.path.join(os.environ["ProgramFiles"], "Google", "Chrome", "Application", "chrome.exe"),
            os.path.join(os.environ["ProgramFiles(x86)"], "Google", "Chrome", "Application", "chrome.exe"),
        ]
    elif sys.platform == "darwin": # macOS
        possible_paths = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]
    else: # Linux
        possible_paths = ["/usr/bin/google-chrome", "/usr/bin/chromium-browser"]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # Fallback to asking user if not found
    user_path = shutil.which("google-chrome") or shutil.which("chrome") or shutil.which("chromium")
    if user_path:
        return user_path
        
    raise FileNotFoundError(
        "Google Chrome executable not found. Please install it or ensure it's in your system's PATH."
    )

def create_driver(headless=False, stealth_mode=True):
    """Creates a robust Chrome WebDriver instance."""
    options = uc.ChromeOptions()
    
    if headless:
        options.add_argument('--headless=new') # Use the new headless mode

    # Common options to make the browser appear more "human"
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument(f'--accept-lang=en-US,en;q=0.9')

    try:
        pretty_print("Initializing WebDriver...", "status")
        driver = uc.Chrome(options=options, use_subprocess=True)
        pretty_print("✅ WebDriver initialized successfully.", "success")
        return driver
    except Exception as e:
        pretty_print(f"CRITICAL: Failed to create WebDriver. Error: {e}", "failure")
        pretty_print("This might be due to a version mismatch between your Chrome browser and the chromedriver.", "warning")
        pretty_print("Try running 'chromedriver-autoinstaller --install' or manually updating chromedriver.", "warning")
        raise

class Browser:
    def __init__(self, driver: uc.Chrome):
        self.driver = driver
        self.wait = WebDriverWait(self.driver, 15)
        self.logger = Logger("browser.log")
        self.screenshot_folder = os.path.join(os.getcwd(), ".screenshots")
        os.makedirs(self.screenshot_folder, exist_ok=True)
        self.js_scripts_folder = os.path.join(os.path.dirname(__file__), 'web_scripts')

    def load_js(self, file_name: str) -> str:
        path = os.path.join(self.js_scripts_folder, file_name)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Could not load JS script {file_name}: {e}")
            return ""

    def apply_stealth_and_safety(self):
        """Injects JS to reduce detection and improve safety."""
        self.driver.execute_script(self.load_js("spoofing.js"))
        self.driver.execute_script(self.load_js("inject_safety_script.js"))

    def go_to(self, url: str) -> bool:
        try:
            self.driver.get(url)
            time.sleep(random.uniform(1.5, 3.0)) # Wait for page to settle
            self.apply_stealth_and_safety()
            self.logger.info(f"Navigated to: {url}")
            return True
        except (TimeoutException, WebDriverException) as e:
            self.logger.error(f"Error navigating to {url}: {e}")
            return False

    def get_text(self) -> str:
        try:
            # Use Readability.js logic for main content extraction
            doc_js = self.load_js("readability.js")
            self.driver.execute_script(doc_js)
            article = self.driver.execute_script("return new Readability(document).parse();")
            
            if article and article.get('textContent'):
                html_content = markdownify.markdownify(article['content'], heading_style="ATX")
                # Final cleaning
                text = re.sub(r'\n{3,}', '\n\n', html_content).strip()
                return text[:20000] # Limit context size
            
            # Fallback to BeautifulSoup if Readability fails
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            for element in soup(['script', 'style', 'header', 'footer', 'nav']):
                element.decompose()
            return soup.get_text(separator='\n', strip=True)[:20000]
            
        except Exception as e:
            self.logger.error(f"Error getting text content: {e}")
            return "Could not extract page content."

    def screenshot(self, filename: str = 'screenshot.png') -> str:
        try:
            path = os.path.join(self.screenshot_folder, filename)
            # Full-page screenshot logic
            height = self.driver.execute_script("return document.body.scrollHeight")
            self.driver.set_window_size(1920, height)
            time.sleep(0.5) # Allow repaint
            self.driver.save_screenshot(path)
            self.driver.set_window_size(1920, 1080) # Reset window size
            self.logger.info(f"Screenshot saved to {path}")
            return os.path.join("/", path).replace("\\", "/") # Return web-accessible path
        except Exception as e:
            self.logger.error(f"Error taking screenshot: {e}")
            return ""
            
    def get_page_title(self) -> str:
        return self.driver.title
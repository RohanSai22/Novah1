import asyncio
from typing import Any, Dict, List

try:
    from fastapi import WebSocket
except Exception:  # pragma: no cover - optional dependency for tests
    class WebSocket:  # minimal stub for tests if FastAPI isn't installed
        async def send_json(self, data: Any) -> None:
            pass

# Dummy planner implementation
async def planner(query: str) -> Dict[str, Any]:
    await asyncio.sleep(0.1)
    return {"steps": [f"plan for {query}"]}

# Dummy browser using Playwright placeholder
async def browser(plan: Dict[str, Any]) -> List[str]:
    await asyncio.sleep(0.1)
    return ["/static/screenshots/mock.png"]

# Dummy search
async def search(plan: Dict[str, Any]) -> List[Dict[str, str]]:
    await asyncio.sleep(0.1)
    return [{"title": "Python vs Rust", "url": "https://example.com"}]

# Dummy coder
async def coder(plan: Dict[str, Any], links: List[Dict[str, str]]) -> Dict[str, Any]:
    await asyncio.sleep(0.1)
    return {"stdout": "print('done')", "images": []}

# Dummy reporter
async def reporter(plan: Dict[str, Any], links: List[Dict[str, str]], code_out: Dict[str, Any], screenshots: List[str]) -> str:
    await asyncio.sleep(0.1)
    return "/static/reports/report.pdf"

async def summarize(plan: Dict[str, Any], links: List[Dict[str, str]], code_out: Dict[str, Any]) -> str:
    await asyncio.sleep(0.1)
    return "summary"

async def run(query: str, deep: bool, ws: WebSocket) -> None:
    plan = await planner(query)
    await ws.send_json({"phase": "plan", "data": plan})

    browser_task = asyncio.create_task(browser(plan))
    search_task = asyncio.create_task(search(plan))

    screenshots = await browser_task
    await ws.send_json({"phase": "browser", "data": screenshots})

    links = await search_task
    await ws.send_json({"phase": "search", "data": links})

    code_out = await coder(plan, links)
    await ws.send_json({"phase": "code", "data": code_out})

    if deep:
        pdf = await reporter(plan, links, code_out, screenshots)
        await ws.send_json({"phase": "report", "data": pdf})
    else:
        summary = await summarize(plan, links, code_out)
        await ws.send_json({"phase": "complete", "data": summary})

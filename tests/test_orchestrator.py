import asyncio
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from orchestrator import run

class DummyWS:
    def __init__(self):
        self.sent = []
    async def send_json(self, data):
        self.sent.append(data)

def test_run_pipeline(monkeypatch):
    async def planner(q):
        return {'steps': ['p']}
    async def browser(p):
        return ['b']
    async def search(p):
        return ['s']
    async def coder(p,l):
        return {'stdout': 'out', 'images': []}
    async def reporter(p,l,c,s):
        return 'r'
    monkeypatch.setattr('orchestrator.planner', planner)
    monkeypatch.setattr('orchestrator.browser', browser)
    monkeypatch.setattr('orchestrator.search', search)
    monkeypatch.setattr('orchestrator.coder', coder)
    monkeypatch.setattr('orchestrator.reporter', reporter)
    ws = DummyWS()
    asyncio.run(run('Python vs Rust', True, ws))
    phases = [m['phase'] for m in ws.sent]
    assert phases == ['plan', 'browser', 'search', 'code', 'report']

def test_run_pipeline_normal(monkeypatch):
    async def planner(q):
        return {'steps': ['p']}
    async def browser(p):
        return ['b']
    async def search(p):
        return ['s']
    async def coder(p,l):
        return {'stdout': 'out', 'images': []}
    async def summarize(p,l,c):
        return 'summary'
    monkeypatch.setattr('orchestrator.planner', planner)
    monkeypatch.setattr('orchestrator.browser', browser)
    monkeypatch.setattr('orchestrator.search', search)
    monkeypatch.setattr('orchestrator.coder', coder)
    monkeypatch.setattr('orchestrator.summarize', summarize)
    ws = DummyWS()
    asyncio.run(run('Python vs Rust', False, ws))
    phases = [m['phase'] for m in ws.sent]
    assert phases == ['plan', 'browser', 'search', 'code', 'complete']

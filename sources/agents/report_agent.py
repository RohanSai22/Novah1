import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from sources.agents.agent import Agent
from sources.memory import Memory

class ReportAgent(Agent):
    """Agent responsible for compiling a final markdown report and saving to PDF."""

    def __init__(self, name: str, prompt_path: str, provider, verbose: bool = False):
        super().__init__(name, prompt_path, provider, verbose, None)
        self.role = "report"
        self.type = "report_agent"
        self.memory = Memory(self.load_prompt(prompt_path), recover_last_session=False, memory_compression=False, model_provider=provider.get_model_name())

    async def process(self, prompt: str, speech_module=None):
        self.memory.push('user', prompt)
        answer, reasoning = await self.llm_request()
        self.last_answer = answer
        pdf_path = os.path.join('reports', 'final_report.pdf')
        os.makedirs('reports', exist_ok=True)
        c = canvas.Canvas(pdf_path, pagesize=letter)
        text_object = c.beginText(50, 750)
        for line in answer.split('\n'):
            text_object.textLine(line)
        c.drawText(text_object)
        c.save()
        self.status_message = "Ready"
        return answer, reasoning

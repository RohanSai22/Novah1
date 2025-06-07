"""
Enhanced Report Agent - Advanced PDF Generation and Infographics
This agent creates comprehensive reports with advanced visualizations, and professional formatting
"""
import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from typing import Dict, Any

from sources.agents.agent import Agent, ExecutionManager

class EnhancedReportAgent(Agent):
    """Generates comprehensive PDF reports."""

    def __init__(self, name="Enhanced Report Agent", prompt_path="prompts/base/report_agent.txt", provider=None, verbose=False):
        # *** CRITICAL FIX: Call the parent class constructor ***
        super().__init__(name, prompt_path, provider, verbose)
        self.role = "report"
        self.type = "enhanced_report_agent"

    async def execute(self, prompt: str, execution_manager: 'ExecutionManager' = None) -> Dict[str, str]:
        if not execution_manager or not execution_manager.execution_state:
            return {"error": "Execution context not provided for report generation."}

        report_path = self.generate_comprehensive_report(execution_manager.execution_state)
        
        # Update central state
        web_path = os.path.join('/', report_path).replace('\\', '/')
        execution_manager.update_state({"report": {"final_report_url": web_path}})
        
        summary = f"Generated comprehensive report at {web_path}"
        execution_manager.update_state({"execution": {"agent_progress": {self.agent_name: {"status": "completed", "output": summary}}}})

        return {"report_path": web_path, "status": "completed", "summary": summary}

    def generate_comprehensive_report(self, execution_data: dict) -> str:
        reports_dir = 'reports'
        os.makedirs(reports_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"execution_report_{timestamp}.pdf"
        pdf_path = os.path.join(reports_dir, pdf_filename)
        
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("Comprehensive Research Report", styles['h1']))
        story.append(Spacer(1, 12))

        intent = execution_data.get('execution', {}).get('intent', 'N/A')
        summary_text = f"<b>Research Query:</b> {intent}"
        story.append(Paragraph(summary_text, styles['Normal']))
        story.append(Spacer(1, 12))
        
        story.append(Paragraph("Execution Plan", styles['h2']))
        plan_steps = execution_data.get('plan', {}).get('steps', [])
        if plan_steps:
            for i, step in enumerate(plan_steps, 1):
                story.append(Paragraph(f"{i}. <b>{step['tool']}</b>: {step['task']}", styles['Normal']))
        else:
            story.append(Paragraph("No execution plan was available.", styles['Normal']))
        story.append(Spacer(1, 12))

        story.append(Paragraph("Agent Findings", styles['h2']))
        agent_progress = execution_data.get('execution', {}).get('agent_progress', {})
        if agent_progress:
            for agent, progress in agent_progress.items():
                story.append(Paragraph(f"<b>Findings from {agent}:</b>", styles['h3']))
                output = progress.get('output', 'No output recorded.').replace('\n', '<br/>')
                story.append(Paragraph(output, styles['p']))
                story.append(Spacer(1, 6))
        else:
            story.append(Paragraph("No agent outputs were recorded.", styles['Normal']))

        doc.build(story)
        print(f"📄 Comprehensive report generated: {pdf_path}")
        return pdf_path

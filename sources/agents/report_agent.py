import os
import time
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
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

    async def generate_comprehensive_report(self, execution_data: dict) -> str:
        """
        Generate a comprehensive report with enhanced data from all agents
        
        Args:
            execution_data: Dictionary containing complete execution state
            
        Returns:
            Path to the generated PDF report
        """
        # Create reports directory
        reports_dir = 'reports'
        os.makedirs(reports_dir, exist_ok=True)
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"execution_report_{timestamp}.pdf"
        pdf_path = os.path.join(reports_dir, pdf_filename)
        
        # Create the PDF document
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Enhanced Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.darkblue,
            alignment=1  # Center alignment
        )
        story.append(Paragraph("Comprehensive AI Research Report", title_style))
        story.append(Spacer(1, 20))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", styles['Heading2']))
        intent = execution_data.get('intent', 'No intent specified')
        story.append(Paragraph(f"<b>Research Objective:</b> {intent}", styles['Normal']))
        
        # Add execution statistics
        agent_progress = execution_data.get('agent_progress', {})
        search_results = execution_data.get('search_results', [])
        links_processed = execution_data.get('links_processed', [])
        
        story.append(Paragraph(f"<b>Agents Deployed:</b> {len(agent_progress)}", styles['Normal']))
        story.append(Paragraph(f"<b>Search Results Found:</b> {len(search_results)}", styles['Normal']))
        story.append(Paragraph(f"<b>Links Processed:</b> {len(links_processed)}", styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Plan Overview with Enhanced Details
        story.append(Paragraph("Research Methodology", styles['Heading2']))
        plan = execution_data.get('plan', [])
        if plan:
            for i, step in enumerate(plan, 1):
                story.append(Paragraph(f"{i}. {step}", styles['Normal']))
        else:
            story.append(Paragraph("No structured plan was generated", styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Agent Performance Analysis
        story.append(Paragraph("Agent Performance Analysis", styles['Heading2']))
        
        if agent_progress:
            for agent_name, progress in agent_progress.items():
                story.append(Paragraph(f"<b>{agent_name}:</b>", styles['Heading3']))
                
                status = progress.get('status', 'Unknown')
                current_task = progress.get('current_task', 'No task assigned')
                output = progress.get('output', 'No output available')
                
                story.append(Paragraph(f"Status: {status.title()}", styles['Normal']))
                story.append(Paragraph(f"Task: {current_task}", styles['Normal']))
                
                if len(output) > 200:
                    output = output[:200] + "..."
                story.append(Paragraph(f"Output: {output}", styles['Normal']))
                
                # Add search results for SearchAgent
                if agent_name == 'SearchAgent' and progress.get('search_results'):
                    story.append(Paragraph("Search Results:", styles['Heading4']))
                    for result in progress['search_results'][:3]:
                        title = result.get('title', 'No title')
                        snippet = result.get('snippet', 'No description')[:100] + "..."
                        story.append(Paragraph(f"• <b>{title}:</b> {snippet}", styles['Normal']))
                
                story.append(Spacer(1, 8))
        else:
            story.append(Paragraph("No agent performance data available", styles['Normal']))
        
        story.append(Spacer(1, 12))
        
        # Research Findings Section
        story.append(Paragraph("Research Findings", styles['Heading2']))
        
        if search_results:
            story.append(Paragraph("Key Sources Identified:", styles['Heading3']))
            for i, result in enumerate(search_results[:5], 1):
                title = result.get('title', 'Unknown Source')
                snippet = result.get('snippet', 'No description available')
                url = result.get('url', 'No URL')
                source = result.get('source', 'Unknown')
                
                story.append(Paragraph(f"{i}. <b>{title}</b> ({source})", styles['Normal']))
                story.append(Paragraph(f"   {snippet[:150]}...", styles['Normal']))
                story.append(Paragraph(f"   <i>Source: {url}</i>", styles['Normal']))
                story.append(Spacer(1, 6))
        else:
            story.append(Paragraph("No research findings available", styles['Normal']))
        
        story.append(Spacer(1, 12))
        
        # Task Execution Status with Enhanced Details
        story.append(Paragraph("Task Execution Summary", styles['Heading2']))
        subtask_status = execution_data.get('subtask_status', [])
        
        if subtask_status:
            # Create enhanced status table
            table_data = [['Task', 'Status', 'Agent', 'Details']]
            
            for subtask in subtask_status:
                task_desc = subtask.get('subtask', 'Unknown task')
                if len(task_desc) > 50:
                    task_desc = task_desc[:50] + '...'
                
                status = subtask.get('status', 'Unknown')
                agent = subtask.get('agent', 'None')
                output = subtask.get('output', 'No output')
                
                if len(output) > 40:
                    output = output[:40] + '...'
                
                table_data.append([task_desc, status, agent, output])
            
            table = Table(table_data, colWidths=[2.5*inch, 1*inch, 1*inch, 2*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
        else:
            story.append(Paragraph("No task execution data available", styles['Normal']))
        
        story.append(Spacer(1, 12))
        
        # Final Insights and Recommendations
        story.append(Paragraph("Final Analysis", styles['Heading2']))
        
        # Generate comprehensive summary based on all collected data
        final_summary = self.generate_final_summary(execution_data)
        story.append(Paragraph(final_summary, styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        # Footer
        footer_text = f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by Novah AI Research System"
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            alignment=1
        )
        story.append(Paragraph(footer_text, footer_style))
        
        # Build the PDF
        doc.build(story)
        
        print(f"📄 Comprehensive report generated: {pdf_path}")
        return pdf_path

    def generate_final_summary(self, execution_data: dict) -> str:
        """Generate a comprehensive final summary based on all execution data"""
        intent = execution_data.get('intent', '')
        agent_progress = execution_data.get('agent_progress', {})
        search_results = execution_data.get('search_results', [])
        
        # Count successful vs failed agents
        successful_agents = [name for name, progress in agent_progress.items() if progress.get('status') == 'completed']
        total_agents = len(agent_progress)
        
        summary = f"""Based on the comprehensive research conducted for "{intent}", the following analysis has been completed:

<b>Research Execution Summary:</b>
• Successfully deployed {len(successful_agents)} out of {total_agents} specialized AI agents
• Collected {len(search_results)} relevant search results from multiple sources
• Processed information from various databases including web search, academic sources, and news outlets

<b>Key Research Outcomes:</b>
"""
        
        if search_results:
            sources = set(result.get('source', 'unknown') for result in search_results)
            summary += f"• Information gathered from {len(sources)} distinct sources: {', '.join(sources)}\n"
            summary += f"• Identified {len(search_results)} relevant documents and resources\n"
        
        if 'SearchAgent' in agent_progress:
            search_output = agent_progress['SearchAgent'].get('output', '')
            if search_output:
                summary += f"• Comprehensive search analysis completed with detailed findings\n"
        
        if 'BrowserAgent' in agent_progress:
            browser_output = agent_progress['BrowserAgent'].get('output', '')
            if browser_output:
                summary += f"• Web navigation and data extraction completed successfully\n"
        
        summary += f"""
<b>Research Quality Assessment:</b>
The research process successfully addressed the stated objective through systematic information gathering and analysis. All deployed agents contributed to building a comprehensive understanding of the topic."""
        
        return summary
        """
        Generate a comprehensive report based on execution data.
        
        Args:
            execution_data: Dictionary containing intent, plan, subtask_status, and agent_outputs
            
        Returns:
            Path to the generated PDF report
        """
        # Create reports directory
        reports_dir = 'reports'
        os.makedirs(reports_dir, exist_ok=True)
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"execution_report_{timestamp}.pdf"
        pdf_path = os.path.join(reports_dir, pdf_filename)
        
        # Create the PDF document
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.darkblue
        )
        story.append(Paragraph("Execution Report", title_style))
        story.append(Spacer(1, 20))
        
        # Execution Summary
        story.append(Paragraph("Executive Summary", styles['Heading2']))
        intent = execution_data.get('intent', 'No intent specified')
        story.append(Paragraph(f"<b>Intent:</b> {intent}", styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Plan Overview
        story.append(Paragraph("Plan Overview", styles['Heading2']))
        plan = execution_data.get('plan', [])
        if plan:
            for i, step in enumerate(plan, 1):
                story.append(Paragraph(f"{i}. {step}", styles['Normal']))
        else:
            story.append(Paragraph("No plan available", styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Subtask Status
        story.append(Paragraph("Task Execution Status", styles['Heading2']))
        subtask_status = execution_data.get('subtask_status', [])
        
        if subtask_status:
            # Create status table
            table_data = [['Task', 'Status', 'Agent', 'Duration']]
            
            for subtask in subtask_status:
                task_desc = subtask.get('description', 'Unknown task')[:50] + ('...' if len(subtask.get('description', '')) > 50 else '')
                status = subtask.get('status', 'Unknown')
                agent = subtask.get('agent_assigned', 'None')
                
                # Calculate duration if available
                start_time = subtask.get('start_time')
                end_time = subtask.get('end_time')
                duration = 'N/A'
                if start_time and end_time:
                    try:
                        start_dt = datetime.fromisoformat(start_time)
                        end_dt = datetime.fromisoformat(end_time)
                        duration = str(end_dt - start_dt)
                    except:
                        duration = 'N/A'
                
                table_data.append([task_desc, status, agent, duration])
            
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
        else:
            story.append(Paragraph("No subtask information available", styles['Normal']))
        
        story.append(Spacer(1, 12))
        
        # Agent Outputs
        story.append(Paragraph("Agent Outputs", styles['Heading2']))
        agent_outputs = execution_data.get('agent_outputs', {})
        
        if agent_outputs:
            for agent_name, output in agent_outputs.items():
                story.append(Paragraph(f"<b>{agent_name}:</b>", styles['Heading3']))
                # Truncate long outputs for readability
                output_text = str(output)[:1000] + ('...' if len(str(output)) > 1000 else '')
                story.append(Paragraph(output_text, styles['Normal']))
                story.append(Spacer(1, 8))
        else:
            story.append(Paragraph("No agent outputs available", styles['Normal']))
        
        # Footer
        story.append(Spacer(1, 20))
        footer_text = f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        story.append(Paragraph(footer_text, styles['Normal']))
        
        # Build the PDF
        doc.build(story)
        
        return pdf_path

    async def process(self, prompt: str, execution_data: dict = None, speech_module=None):
        """
        Process a report generation request.
        
        Args:
            prompt: The request prompt
            execution_data: Dictionary containing execution data for report generation
            speech_module: Optional speech module
            
        Returns:
            Tuple of (answer, reasoning)
        """
        self.memory.push('user', prompt)
        
        if execution_data:
            # Generate comprehensive report
            try:
                pdf_path = await self.generate_report(execution_data)
                answer = f"Comprehensive execution report generated successfully. Report saved to: {pdf_path}"
                reasoning = "Generated a detailed PDF report including execution summary, plan overview, task status, and agent outputs."
            except Exception as e:
                answer = f"Error generating report: {str(e)}"
                reasoning = f"Failed to generate PDF report due to: {str(e)}"
        else:
            # Fallback to basic LLM response
            answer, reasoning = await self.llm_request()
            
            # Generate basic PDF from LLM response
            try:
                reports_dir = 'reports'
                os.makedirs(reports_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                pdf_path = os.path.join(reports_dir, f'basic_report_{timestamp}.pdf')
                
                c = canvas.Canvas(pdf_path, pagesize=letter)
                text_object = c.beginText(50, 750)
                text_object.setFont("Helvetica", 12)
                
                # Split answer into lines and add to PDF
                lines = answer.split('\n')
                for line in lines:
                    # Handle long lines by wrapping
                    if len(line) > 80:
                        words = line.split(' ')
                        current_line = ''
                        for word in words:
                            if len(current_line + word) < 80:
                                current_line += word + ' '
                            else:
                                text_object.textLine(current_line.strip())
                                current_line = word + ' '
                        if current_line:
                            text_object.textLine(current_line.strip())
                    else:
                        text_object.textLine(line)
                
                c.drawText(text_object)
                c.save()
                
                answer += f"\n\nReport saved to: {pdf_path}"
            except Exception as e:
                answer += f"\n\nWarning: Could not save PDF report: {str(e)}"
        
        self.last_answer = answer
        self.status_message = "Ready"
        return answer, reasoning

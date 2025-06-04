"""
Enhanced Report Agent - Advanced PDF Generation and Infographics
This agent creates comprehensive reports with advanced visualizations, infographics, and professional formatting
"""
import asyncio
import json
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass, asdict
import logging
import io
import tempfile
import os

# PDF and document generation
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4, legal
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.platypus.flowables import HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.graphics.shapes import Drawing, String, Circle, Rect, Line
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.widgets.markers import makeMarker
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

# Image and visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from PIL import Image as PILImage, ImageDraw, ImageFont
import numpy as np
import pandas as pd

# Word cloud and advanced graphics
from wordcloud import WordCloud
import cv2

from sources.utility import pretty_print, animate_thinking
from sources.agents.agent import Agent


class ReportType(Enum):
    """Types of reports that can be generated"""
    EXECUTIVE_SUMMARY = "executive_summary"
    DETAILED_ANALYSIS = "detailed_analysis"
    TECHNICAL_REPORT = "technical_report"
    INFOGRAPHIC_REPORT = "infographic_report"
    DASHBOARD_REPORT = "dashboard_report"
    COMPARATIVE_REPORT = "comparative_report"
    TREND_REPORT = "trend_report"
    RESEARCH_REPORT = "research_report"


class OutputFormat(Enum):
    """Output formats for reports"""
    PDF = "pdf"
    HTML = "html"
    DOCX = "docx"
    PPTX = "pptx"
    JSON = "json"
    INTERACTIVE_HTML = "interactive_html"


class VisualizationType(Enum):
    """Types of visualizations available"""
    BAR_CHART = "bar_chart"
    LINE_CHART = "line_chart"
    PIE_CHART = "pie_chart"
    SCATTER_PLOT = "scatter_plot"
    HEATMAP = "heatmap"
    HISTOGRAM = "histogram"
    BOX_PLOT = "box_plot"
    VIOLIN_PLOT = "violin_plot"
    TREEMAP = "treemap"
    SANKEY_DIAGRAM = "sankey_diagram"
    WORD_CLOUD = "word_cloud"
    NETWORK_GRAPH = "network_graph"
    GEOGRAPHIC_MAP = "geographic_map"
    GANTT_CHART = "gantt_chart"
    INFOGRAPHIC = "infographic"


@dataclass
class ReportSection:
    """Definition of a report section"""
    title: str
    content: str
    visualizations: List[Dict[str, Any]]
    subsections: List['ReportSection']
    metadata: Dict[str, Any]
    style_config: Dict[str, Any]


@dataclass
class ReportTemplate:
    """Template configuration for reports"""
    name: str
    report_type: str
    sections: List[str]
    style_config: Dict[str, Any]
    layout_config: Dict[str, Any]
    branding: Dict[str, Any]


@dataclass
class InfographicElement:
    """Definition of an infographic element"""
    element_type: str
    position: Tuple[int, int]
    size: Tuple[int, int]
    data: Dict[str, Any]
    style: Dict[str, Any]
    animation: Optional[Dict[str, Any]] = None


@dataclass
class ReportMetadata:
    """Metadata for generated reports"""
    title: str
    author: str
    creation_date: str
    report_type: str
    data_sources: List[str]
    total_pages: int
    file_size: Optional[int] = None
    version: str = "1.0"


class EnhancedReportAgent(Agent):
    """Enhanced Report Agent for advanced report generation"""
    
    def __init__(self, provider):
        super().__init__(provider=provider, name="Enhanced Report Agent")
        self.report_templates = self._load_default_templates()
        self.style_manager = StyleManager()
        self.visualization_generator = AdvancedVisualizationGenerator()
        self.infographic_creator = InfographicCreator()
        self.pdf_generator = AdvancedPDFGenerator()
        
    async def generate_comprehensive_report(self,
                                          data: Dict[str, Any],
                                          report_type: ReportType = ReportType.DETAILED_ANALYSIS,
                                          output_format: OutputFormat = OutputFormat.PDF,
                                          template_name: Optional[str] = None,
                                          custom_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive report with advanced formatting and visualizations
        
        Args:
            data: Data to include in the report
            report_type: Type of report to generate
            output_format: Output format for the report
            template_name: Name of template to use
            custom_config: Custom configuration options
            
        Returns:
            Dictionary containing report information and file paths
        """
        if custom_config is None:
            custom_config = {}
            
        # Select appropriate template
        template = self._select_template(report_type, template_name)
        
        # Process and structure data
        structured_data = await self._structure_report_data(data, template)
        
        # Generate sections
        sections = await self._generate_report_sections(structured_data, template, custom_config)
        
        # Create visualizations
        visualizations = await self._create_visualizations(structured_data, template)
        
        # Generate report based on output format
        if output_format == OutputFormat.PDF:
            report_file = await self._generate_pdf_report(sections, visualizations, template, custom_config)
        elif output_format == OutputFormat.HTML:
            report_file = await self._generate_html_report(sections, visualizations, template, custom_config)
        elif output_format == OutputFormat.INTERACTIVE_HTML:
            report_file = await self._generate_interactive_html_report(sections, visualizations, template, custom_config)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        # Generate metadata
        metadata = ReportMetadata(
            title=custom_config.get('title', 'Analysis Report'),
            author=custom_config.get('author', 'Enhanced Report Agent'),
            creation_date=datetime.now().isoformat(),
            report_type=report_type.value,
            data_sources=custom_config.get('data_sources', ['Various']),
            total_pages=len(sections)
        )
        
        return {
            'report_file': report_file,
            'metadata': asdict(metadata),
            'sections': [asdict(section) for section in sections],
            'template_used': template.name,
            'generation_stats': {
                'sections_created': len(sections),
                'visualizations_created': len(visualizations),
                'generation_time': datetime.now().isoformat()
            }
        }
    
    async def create_infographic_report(self,
                                      data: Dict[str, Any],
                                      theme: str = "modern",
                                      color_scheme: str = "blue",
                                      custom_elements: Optional[List[InfographicElement]] = None) -> Dict[str, Any]:
        """
        Create an infographic-style report with visual elements
        
        Args:
            data: Data to visualize
            theme: Visual theme for the infographic
            color_scheme: Color scheme to use
            custom_elements: Custom infographic elements
            
        Returns:
            Dictionary containing infographic information and file paths
        """
        if custom_elements is None:
            custom_elements = []
            
        # Create infographic canvas
        canvas_config = {
            'width': 1920,
            'height': 1080,
            'theme': theme,
            'color_scheme': color_scheme
        }
        
        # Generate infographic elements
        elements = await self._generate_infographic_elements(data, canvas_config)
        elements.extend(custom_elements)
        
        # Create infographic
        infographic_file = await self.infographic_creator.create_infographic(elements, canvas_config)
        
        # Generate interactive version
        interactive_file = await self._create_interactive_infographic(elements, canvas_config)
        
        return {
            'infographic_file': infographic_file,
            'interactive_file': interactive_file,
            'elements': [asdict(element) for element in elements],
            'config': canvas_config,
            'metadata': {
                'creation_date': datetime.now().isoformat(),
                'theme': theme,
                'color_scheme': color_scheme
            }
        }
    
    async def generate_dashboard_report(self,
                                      data: Dict[str, Any],
                                      layout: str = "grid",
                                      interactive: bool = True,
                                      auto_refresh: bool = False) -> Dict[str, Any]:
        """
        Generate an interactive dashboard report
        
        Args:
            data: Data for the dashboard
            layout: Layout style for the dashboard
            interactive: Whether to include interactive elements
            auto_refresh: Whether dashboard should auto-refresh
            
        Returns:
            Dictionary containing dashboard information and files
        """
        # Create dashboard configuration
        dashboard_config = {
            'layout': layout,
            'interactive': interactive,
            'auto_refresh': auto_refresh,
            'grid_columns': 3,
            'responsive': True
        }
        
        # Generate dashboard widgets
        widgets = await self._generate_dashboard_widgets(data, dashboard_config)
        
        # Create dashboard layout
        dashboard_layout = await self._create_dashboard_layout(widgets, dashboard_config)
        
        # Generate dashboard files
        if interactive:
            dashboard_file = await self._generate_interactive_dashboard(dashboard_layout, dashboard_config)
        else:
            dashboard_file = await self._generate_static_dashboard(dashboard_layout, dashboard_config)
        
        return {
            'dashboard_file': dashboard_file,
            'widgets': widgets,
            'layout': dashboard_layout,
            'config': dashboard_config,
            'metadata': {
                'creation_date': datetime.now().isoformat(),
                'widget_count': len(widgets),
                'interactive': interactive
            }
        }
    
    def _load_default_templates(self) -> Dict[str, ReportTemplate]:
        """Load default report templates"""
        templates = {}
        
        # Executive Summary Template
        templates['executive_summary'] = ReportTemplate(
            name="Executive Summary",
            report_type=ReportType.EXECUTIVE_SUMMARY.value,
            sections=['cover', 'executive_summary', 'key_findings', 'recommendations', 'appendix'],
            style_config={
                'font_family': 'Helvetica',
                'font_size': 12,
                'line_spacing': 1.2,
                'color_scheme': 'corporate'
            },
            layout_config={
                'margins': {'top': 1, 'bottom': 1, 'left': 1, 'right': 1},
                'page_size': 'letter',
                'orientation': 'portrait'
            },
            branding={
                'logo_position': 'header_right',
                'company_colors': ['#1f4e79', '#4472c4'],
                'footer_text': 'Confidential Analysis Report'
            }
        )
        
        # Detailed Analysis Template
        templates['detailed_analysis'] = ReportTemplate(
            name="Detailed Analysis",
            report_type=ReportType.DETAILED_ANALYSIS.value,
            sections=['cover', 'table_of_contents', 'methodology', 'analysis', 'findings', 'conclusions', 'appendix'],
            style_config={
                'font_family': 'Times-Roman',
                'font_size': 11,
                'line_spacing': 1.5,
                'color_scheme': 'academic'
            },
            layout_config={
                'margins': {'top': 1.25, 'bottom': 1, 'left': 1.25, 'right': 1},
                'page_size': 'letter',
                'orientation': 'portrait'
            },
            branding={
                'logo_position': 'cover_center',
                'company_colors': ['#2e4057', '#3e5266'],
                'footer_text': 'Detailed Analysis Report'
            }
        )
        
        # Infographic Template
        templates['infographic'] = ReportTemplate(
            name="Infographic Report",
            report_type=ReportType.INFOGRAPHIC_REPORT.value,
            sections=['header', 'key_stats', 'main_visual', 'insights', 'footer'],
            style_config={
                'font_family': 'Helvetica-Bold',
                'font_size': 14,
                'line_spacing': 1.0,
                'color_scheme': 'vibrant'
            },
            layout_config={
                'margins': {'top': 0.5, 'bottom': 0.5, 'left': 0.5, 'right': 0.5},
                'page_size': 'A4',
                'orientation': 'portrait'
            },
            branding={
                'logo_position': 'header_left',
                'company_colors': ['#ff6b6b', '#4ecdc4', '#45b7d1'],
                'footer_text': 'Visual Analysis Report'
            }
        )
        
        return templates
    
    def _select_template(self, report_type: ReportType, template_name: Optional[str]) -> ReportTemplate:
        """Select appropriate template for report generation"""
        if template_name and template_name in self.report_templates:
            return self.report_templates[template_name]
        
        # Default template mapping
        template_mapping = {
            ReportType.EXECUTIVE_SUMMARY: 'executive_summary',
            ReportType.DETAILED_ANALYSIS: 'detailed_analysis',
            ReportType.INFOGRAPHIC_REPORT: 'infographic'
        }
        
        template_key = template_mapping.get(report_type, 'detailed_analysis')
        return self.report_templates[template_key]
    
    async def _structure_report_data(self, data: Dict[str, Any], template: ReportTemplate) -> Dict[str, Any]:
        """Structure data according to template requirements"""
        structured_data = {
            'raw_data': data,
            'processed_sections': {},
            'visualizations_data': {},
            'metadata': {
                'processing_date': datetime.now().isoformat(),
                'template_used': template.name
            }
        }
        
        # Process data for each section
        for section in template.sections:
            if section == 'executive_summary':
                structured_data['processed_sections'][section] = await self._create_executive_summary(data)
            elif section == 'key_findings':
                structured_data['processed_sections'][section] = await self._extract_key_findings(data)
            elif section == 'analysis':
                structured_data['processed_sections'][section] = await self._create_analysis_section(data)
            elif section == 'recommendations':
                structured_data['processed_sections'][section] = await self._generate_recommendations(data)
            else:
                structured_data['processed_sections'][section] = await self._create_generic_section(data, section)
        
        return structured_data
    
    async def _generate_report_sections(self,
                                      structured_data: Dict[str, Any],
                                      template: ReportTemplate,
                                      config: Dict[str, Any]) -> List[ReportSection]:
        """Generate report sections based on structured data and template"""
        sections = []
        
        for section_name in template.sections:
            section_data = structured_data['processed_sections'].get(section_name, {})
            
            section = ReportSection(
                title=section_name.replace('_', ' ').title(),
                content=section_data.get('content', ''),
                visualizations=section_data.get('visualizations', []),
                subsections=section_data.get('subsections', []),
                metadata={
                    'section_type': section_name,
                    'word_count': len(section_data.get('content', '').split()),
                    'creation_time': datetime.now().isoformat()
                },
                style_config=template.style_config
            )
            
            sections.append(section)
        
        return sections
    
    async def _create_visualizations(self,
                                   structured_data: Dict[str, Any],
                                   template: ReportTemplate) -> List[Dict[str, Any]]:
        """Create visualizations for the report"""
        visualizations = []
        
        data = structured_data['raw_data']
        
        # Generate different types of visualizations based on data
        if 'numerical_data' in data:
            visualizations.extend(await self._create_numerical_visualizations(data['numerical_data']))
        
        if 'categorical_data' in data:
            visualizations.extend(await self._create_categorical_visualizations(data['categorical_data']))
        
        if 'time_series_data' in data:
            visualizations.extend(await self._create_time_series_visualizations(data['time_series_data']))
        
        if 'text_data' in data:
            visualizations.extend(await self._create_text_visualizations(data['text_data']))
        
        # Apply template styling
        for viz in visualizations:
            viz['style'] = template.style_config
        
        return visualizations
    
    async def _generate_pdf_report(self,
                                 sections: List[ReportSection],
                                 visualizations: List[Dict[str, Any]],
                                 template: ReportTemplate,
                                 config: Dict[str, Any]) -> str:
        """Generate PDF report using advanced PDF generation"""
        return await self.pdf_generator.create_advanced_pdf(sections, visualizations, template, config)
    
    async def _generate_html_report(self,
                                  sections: List[ReportSection],
                                  visualizations: List[Dict[str, Any]],
                                  template: ReportTemplate,
                                  config: Dict[str, Any]) -> str:
        """Generate HTML report"""
        html_content = await self._create_html_content(sections, visualizations, template, config)
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"report_{timestamp}.html"
        filepath = os.path.join(tempfile.gettempdir(), filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return filepath
    
    async def _generate_interactive_html_report(self,
                                              sections: List[ReportSection],
                                              visualizations: List[Dict[str, Any]],
                                              template: ReportTemplate,
                                              config: Dict[str, Any]) -> str:
        """Generate interactive HTML report with JavaScript components"""
        # Create interactive visualizations using Plotly
        interactive_viz = await self._create_interactive_visualizations(visualizations)
        
        html_content = await self._create_interactive_html_content(sections, interactive_viz, template, config)
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"interactive_report_{timestamp}.html"
        filepath = os.path.join(tempfile.gettempdir(), filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return filepath
    
    # Section creation methods
    async def _create_executive_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create executive summary section"""
        summary_points = []
        
        # Extract key metrics
        if 'metrics' in data:
            for metric, value in data['metrics'].items():
                summary_points.append(f"• {metric}: {value}")
        
        # Extract key insights
        if 'insights' in data:
            summary_points.extend([f"• {insight}" for insight in data['insights'][:5]])
        
        content = "\n".join(summary_points) if summary_points else "Executive summary content based on analysis results."
        
        return {
            'content': content,
            'visualizations': [],
            'subsections': []
        }
    
    async def _extract_key_findings(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key findings from data"""
        findings = []
        
        # Process different types of findings
        if 'analysis_results' in data:
            for result in data['analysis_results']:
                if 'insights' in result:
                    findings.extend(result['insights'])
        
        content = "\n\n".join([f"{i+1}. {finding}" for i, finding in enumerate(findings)])
        
        return {
            'content': content or "Key findings will be displayed here based on the analysis.",
            'visualizations': [],
            'subsections': []
        }
    
    async def _create_analysis_section(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed analysis section"""
        analysis_content = []
        
        if 'analysis_results' in data:
            for result in data['analysis_results']:
                analysis_type = result.get('analysis_type', 'Analysis')
                analysis_content.append(f"## {analysis_type.replace('_', ' ').title()}")
                
                if 'results' in result:
                    analysis_content.append("Results:")
                    # Format results based on type
                    results = result['results']
                    if isinstance(results, dict):
                        for key, value in results.items():
                            analysis_content.append(f"- {key}: {value}")
                
                if 'insights' in result:
                    analysis_content.append("\nKey Insights:")
                    for insight in result['insights']:
                        analysis_content.append(f"• {insight}")
                
                analysis_content.append("")  # Add spacing
        
        content = "\n".join(analysis_content) if analysis_content else "Detailed analysis content will be presented here."
        
        return {
            'content': content,
            'visualizations': [],
            'subsections': []
        }
    
    async def _generate_recommendations(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        if 'analysis_results' in data:
            for result in data['analysis_results']:
                if result.get('analysis_type') == 'trend_analysis':
                    recommendations.append("Monitor identified trends for strategic planning")
                elif result.get('analysis_type') == 'sentiment_analysis':
                    recommendations.append("Address sentiment patterns to improve engagement")
                elif result.get('analysis_type') == 'statistical_analysis':
                    recommendations.append("Leverage statistical insights for data-driven decisions")
        
        # Add generic recommendations if none found
        if not recommendations:
            recommendations = [
                "Implement continuous monitoring systems",
                "Establish regular review cycles",
                "Create actionable metrics dashboard",
                "Develop response protocols for key findings"
            ]
        
        content = "\n\n".join([f"{i+1}. {rec}" for i, rec in enumerate(recommendations)])
        
        return {
            'content': content,
            'visualizations': [],
            'subsections': []
        }
    
    async def _create_generic_section(self, data: Dict[str, Any], section_name: str) -> Dict[str, Any]:
        """Create a generic section based on section name"""
        content_map = {
            'cover': "This is the cover page of the report.",
            'table_of_contents': "Table of contents will be generated automatically.",
            'methodology': "Methodology section describing the analysis approach.",
            'conclusions': "Conclusions drawn from the analysis results.",
            'appendix': "Additional supporting information and detailed data."
        }
        
        content = content_map.get(section_name, f"Content for {section_name.replace('_', ' ')} section.")
        
        return {
            'content': content,
            'visualizations': [],
            'subsections': []
        }
    
    # Visualization creation methods
    async def _create_numerical_visualizations(self, data: Any) -> List[Dict[str, Any]]:
        """Create visualizations for numerical data"""
        visualizations = []
        
        if isinstance(data, dict):
            # Bar chart for key-value pairs
            visualizations.append({
                'type': VisualizationType.BAR_CHART.value,
                'data': data,
                'config': {
                    'title': 'Numerical Data Overview',
                    'x_label': 'Categories',
                    'y_label': 'Values'
                }
            })
            
            # Pie chart if appropriate
            if len(data) <= 10:
                visualizations.append({
                    'type': VisualizationType.PIE_CHART.value,
                    'data': data,
                    'config': {
                        'title': 'Data Distribution'
                    }
                })
        
        return visualizations
    
    async def _create_categorical_visualizations(self, data: Any) -> List[Dict[str, Any]]:
        """Create visualizations for categorical data"""
        visualizations = []
        
        if isinstance(data, dict):
            visualizations.append({
                'type': VisualizationType.BAR_CHART.value,
                'data': data,
                'config': {
                    'title': 'Categorical Data Distribution',
                    'orientation': 'horizontal'
                }
            })
        
        return visualizations
    
    async def _create_time_series_visualizations(self, data: Any) -> List[Dict[str, Any]]:
        """Create visualizations for time series data"""
        visualizations = []
        
        visualizations.append({
            'type': VisualizationType.LINE_CHART.value,
            'data': data,
            'config': {
                'title': 'Time Series Analysis',
                'x_label': 'Time',
                'y_label': 'Value'
            }
        })
        
        return visualizations
    
    async def _create_text_visualizations(self, data: Any) -> List[Dict[str, Any]]:
        """Create visualizations for text data"""
        visualizations = []
        
        # Word cloud
        visualizations.append({
            'type': VisualizationType.WORD_CLOUD.value,
            'data': data,
            'config': {
                'title': 'Text Analysis Word Cloud',
                'max_words': 100
            }
        })
        
        return visualizations
    
    # HTML generation methods
    async def _create_html_content(self,
                                 sections: List[ReportSection],
                                 visualizations: List[Dict[str, Any]],
                                 template: ReportTemplate,
                                 config: Dict[str, Any]) -> str:
        """Create HTML content for the report"""
        html_parts = []
        
        # HTML header
        html_parts.append("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
                .section { margin-bottom: 30px; page-break-after: auto; }
                .section h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
                .section h2 { color: #34495e; margin-top: 25px; }
                .visualization { margin: 20px 0; text-align: center; }
                .key-finding { background: #f8f9fa; padding: 15px; border-left: 4px solid #3498db; margin: 10px 0; }
                .recommendation { background: #e8f5e8; padding: 15px; border-left: 4px solid #27ae60; margin: 10px 0; }
                .footer { margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; text-align: center; color: #666; }
            </style>
        </head>
        <body>
        """)
        
        # Title
        title = config.get('title', 'Analysis Report')
        html_parts.append(f"<h1 style='text-align: center; color: #2c3e50;'>{title}</h1>")
        html_parts.append(f"<p style='text-align: center; color: #666;'>Generated on {datetime.now().strftime('%B %d, %Y')}</p>")
        
        # Sections
        for section in sections:
            html_parts.append(f'<div class="section">')
            html_parts.append(f'<h1>{section.title}</h1>')
            
            # Convert content to HTML paragraphs
            paragraphs = section.content.split('\n\n')
            for paragraph in paragraphs:
                if paragraph.strip():
                    # Handle markdown-style formatting
                    if paragraph.startswith('##'):
                        html_parts.append(f'<h2>{paragraph[2:].strip()}</h2>')
                    elif paragraph.startswith('•') or paragraph.startswith('-'):
                        # Handle bullet points
                        html_parts.append('<ul>')
                        for line in paragraph.split('\n'):
                            if line.strip().startswith(('•', '-')):
                                html_parts.append(f'<li>{line.strip()[1:].strip()}</li>')
                        html_parts.append('</ul>')
                    else:
                        html_parts.append(f'<p>{paragraph}</p>')
            
            # Add visualizations for this section
            for viz in section.visualizations:
                html_parts.append(f'<div class="visualization">')
                html_parts.append(f'<h3>{viz.get("title", "Visualization")}</h3>')
                html_parts.append('[Visualization placeholder - would be rendered in actual implementation]')
                html_parts.append('</div>')
            
            html_parts.append('</div>')
        
        # Footer
        html_parts.append(f'''
        <div class="footer">
            <p>Report generated by Enhanced Report Agent</p>
            <p>© {datetime.now().year} - Analysis Report</p>
        </div>
        </body>
        </html>
        ''')
        
        return '\n'.join(html_parts)
    
    async def _create_interactive_visualizations(self, visualizations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create interactive versions of visualizations using Plotly"""
        interactive_viz = []
        
        for viz in visualizations:
            viz_type = viz.get('type')
            data = viz.get('data', {})
            config = viz.get('config', {})
            
            if viz_type == VisualizationType.BAR_CHART.value:
                fig = go.Figure(data=[
                    go.Bar(x=list(data.keys()), y=list(data.values()))
                ])
                fig.update_layout(title=config.get('title', 'Bar Chart'))
                
            elif viz_type == VisualizationType.PIE_CHART.value:
                fig = go.Figure(data=[
                    go.Pie(labels=list(data.keys()), values=list(data.values()))
                ])
                fig.update_layout(title=config.get('title', 'Pie Chart'))
                
            elif viz_type == VisualizationType.LINE_CHART.value:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=list(range(len(data))), y=list(data.values()),
                                       mode='lines+markers'))
                fig.update_layout(title=config.get('title', 'Line Chart'))
                
            else:
                # Default to bar chart
                fig = go.Figure(data=[
                    go.Bar(x=['Data'], y=[1])
                ])
                fig.update_layout(title='Placeholder Chart')
            
            # Convert to HTML
            html_str = pio.to_html(fig, include_plotlyjs='cdn', div_id=f"viz_{len(interactive_viz)}")
            
            interactive_viz.append({
                'type': viz_type,
                'html': html_str,
                'config': config
            })
        
        return interactive_viz
    
    async def _create_interactive_html_content(self,
                                             sections: List[ReportSection],
                                             interactive_viz: List[Dict[str, Any]],
                                             template: ReportTemplate,
                                             config: Dict[str, Any]) -> str:
        """Create interactive HTML content with JavaScript components"""
        html_parts = []
        
        # HTML header with enhanced styling and JavaScript
        html_parts.append("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Interactive Analysis Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    margin: 0; 
                    padding: 0; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: #333;
                }
                .container { max-width: 1200px; margin: 0 auto; background: white; box-shadow: 0 0 20px rgba(0,0,0,0.1); }
                .header { background: #2c3e50; color: white; padding: 30px; text-align: center; }
                .nav { background: #34495e; padding: 15px; }
                .nav a { color: white; text-decoration: none; margin: 0 15px; padding: 10px; border-radius: 5px; transition: background 0.3s; }
                .nav a:hover { background: #4a5f7a; }
                .section { margin: 30px; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                .section h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 15px; }
                .visualization { margin: 30px 0; padding: 20px; background: #f8f9fa; border-radius: 10px; }
                .interactive-element { cursor: pointer; transition: transform 0.2s; }
                .interactive-element:hover { transform: scale(1.02); }
                .footer { background: #2c3e50; color: white; padding: 20px; text-align: center; }
                .toggle-section { background: #3498db; color: white; padding: 10px; border: none; border-radius: 5px; cursor: pointer; margin: 10px 0; }
                .collapsible-content { display: none; }
                .collapsible-content.active { display: block; }
            </style>
        </head>
        <body>
        <div class="container">
        """)
        
        # Header
        title = config.get('title', 'Interactive Analysis Report')
        html_parts.append(f'''
        <div class="header">
            <h1>{title}</h1>
            <p>Interactive Report Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        </div>
        ''')
        
        # Navigation
        html_parts.append('<div class="nav">')
        for section in sections:
            section_id = section.title.lower().replace(' ', '_')
            html_parts.append(f'<a href="#{section_id}">{section.title}</a>')
        html_parts.append('</div>')
        
        # Sections with interactive elements
        viz_counter = 0
        for section in sections:
            section_id = section.title.lower().replace(' ', '_')
            html_parts.append(f'<div class="section" id="{section_id}">')
            html_parts.append(f'<h1>{section.title}</h1>')
            
            # Add collapsible functionality
            html_parts.append(f'<button class="toggle-section" onclick="toggleSection(\'{section_id}_content\')">Toggle Content</button>')
            html_parts.append(f'<div id="{section_id}_content" class="collapsible-content active">')
            
            # Content
            paragraphs = section.content.split('\n\n')
            for paragraph in paragraphs:
                if paragraph.strip():
                    if paragraph.startswith('##'):
                        html_parts.append(f'<h2>{paragraph[2:].strip()}</h2>')
                    elif paragraph.startswith('•') or paragraph.startswith('-'):
                        html_parts.append('<ul>')
                        for line in paragraph.split('\n'):
                            if line.strip().startswith(('•', '-')):
                                html_parts.append(f'<li>{line.strip()[1:].strip()}</li>')
                        html_parts.append('</ul>')
                    else:
                        html_parts.append(f'<p>{paragraph}</p>')
            
            # Add interactive visualizations
            if viz_counter < len(interactive_viz):
                viz = interactive_viz[viz_counter]
                html_parts.append(f'<div class="visualization interactive-element">')
                html_parts.append(viz['html'])
                html_parts.append('</div>')
                viz_counter += 1
            
            html_parts.append('</div>')  # Close collapsible content
            html_parts.append('</div>')  # Close section
        
        # JavaScript for interactivity
        html_parts.append('''
        <script>
            function toggleSection(sectionId) {
                const content = document.getElementById(sectionId);
                content.classList.toggle('active');
            }
            
            // Add smooth scrolling
            document.querySelectorAll('a[href^="#"]').forEach(anchor => {
                anchor.addEventListener('click', function (e) {
                    e.preventDefault();
                    document.querySelector(this.getAttribute('href')).scrollIntoView({
                        behavior: 'smooth'
                    });
                });
            });
            
            // Add animation on scroll
            const observer = new IntersectionObserver((entries) => {
                entries.forEach((entry) => {
                    if (entry.isIntersecting) {
                        entry.target.style.animation = 'fadeInUp 0.6s ease-out';
                    }
                });
            });
            
            document.querySelectorAll('.section').forEach((section) => {
                observer.observe(section);
            });
        </script>
        
        <style>
            @keyframes fadeInUp {
                from {
                    opacity: 0;
                    transform: translateY(30px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
        </style>
        ''')
        
        # Footer
        html_parts.append(f'''
        <div class="footer">
            <p>Interactive Report generated by Enhanced Report Agent</p>
            <p>© {datetime.now().year} - Advanced Analysis Platform</p>
        </div>
        </div>
        </body>
        </html>
        ''')
        
        return '\n'.join(html_parts)
    
    # Dashboard generation methods
    async def _generate_dashboard_widgets(self, data: Dict[str, Any], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate widgets for dashboard"""
        widgets = []
        
        # KPI widgets
        if 'metrics' in data:
            for metric, value in data['metrics'].items():
                widgets.append({
                    'type': 'kpi',
                    'title': metric.replace('_', ' ').title(),
                    'value': value,
                    'size': 'small'
                })
        
        # Chart widgets
        if 'analysis_results' in data:
            for result in data['analysis_results']:
                widgets.append({
                    'type': 'chart',
                    'title': result.get('analysis_type', 'Analysis').replace('_', ' ').title(),
                    'data': result.get('results', {}),
                    'size': 'medium'
                })
        
        return widgets
    
    async def _create_dashboard_layout(self, widgets: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        """Create layout configuration for dashboard"""
        layout = {
            'type': config.get('layout', 'grid'),
            'columns': config.get('grid_columns', 3),
            'rows': [],
            'responsive': config.get('responsive', True)
        }
        
        # Arrange widgets in grid
        current_row = []
        for widget in widgets:
            current_row.append(widget)
            if len(current_row) >= layout['columns']:
                layout['rows'].append(current_row)
                current_row = []
        
        # Add remaining widgets
        if current_row:
            layout['rows'].append(current_row)
        
        return layout
    
    async def _generate_interactive_dashboard(self, layout: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Generate interactive dashboard HTML"""
        # This would create a full interactive dashboard
        # For now, returning a placeholder path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dashboard_{timestamp}.html"
        filepath = os.path.join(tempfile.gettempdir(), filename)
        
        # Create basic dashboard HTML
        dashboard_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Interactive Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .dashboard { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }
                .widget { background: white; border: 1px solid #ddd; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .kpi { text-align: center; }
                .kpi .value { font-size: 2em; font-weight: bold; color: #3498db; }
            </style>
        </head>
        <body>
            <h1>Interactive Dashboard</h1>
            <div class="dashboard">
                <div class="widget kpi">
                    <h3>Sample KPI</h3>
                    <div class="value">1,234</div>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(dashboard_html)
        
        return filepath
    
    async def _generate_static_dashboard(self, layout: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Generate static dashboard image/PDF"""
        # This would create a static dashboard
        # For now, returning a placeholder path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"static_dashboard_{timestamp}.pdf"
        filepath = os.path.join(tempfile.gettempdir(), filename)
        
        # Create placeholder file
        with open(filepath, 'w') as f:
            f.write("Static dashboard placeholder")
        
        return filepath
    
    # Infographic methods
    async def _generate_infographic_elements(self, data: Dict[str, Any], canvas_config: Dict[str, Any]) -> List[InfographicElement]:
        """Generate infographic elements based on data"""
        elements = []
        
        # Title element
        elements.append(InfographicElement(
            element_type='title',
            position=(100, 50),
            size=(1720, 100),
            data={'text': 'Data Analysis Infographic'},
            style={'font_size': 48, 'color': '#2c3e50', 'align': 'center'}
        ))
        
        # Key metrics
        if 'metrics' in data:
            y_pos = 200
            for metric, value in list(data['metrics'].items())[:4]:
                elements.append(InfographicElement(
                    element_type='metric_box',
                    position=(200 + len(elements) * 300, y_pos),
                    size=(250, 150),
                    data={'label': metric, 'value': value},
                    style={'background': '#3498db', 'text_color': 'white'}
                ))
        
        # Chart element
        elements.append(InfographicElement(
            element_type='chart',
            position=(200, 400),
            size=(600, 400),
            data={'type': 'bar', 'data': data.get('chart_data', {})},
            style={'theme': canvas_config.get('theme', 'modern')}
        ))
        
        return elements
    
    async def _create_interactive_infographic(self, elements: List[InfographicElement], config: Dict[str, Any]) -> str:
        """Create interactive infographic"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"interactive_infographic_{timestamp}.html"
        filepath = os.path.join(tempfile.gettempdir(), filename)
        
        # Create basic interactive infographic HTML
        infographic_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Interactive Infographic</title>
            <style>
                body { margin: 0; font-family: Arial, sans-serif; }
                .infographic { width: 100vw; height: 100vh; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
            </style>
        </head>
        <body>
            <div class="infographic">
                <h1 style="text-align: center; color: white; padding-top: 50px;">Interactive Infographic</h1>
            </div>
        </body>
        </html>
        """
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(infographic_html)
        
        return filepath


class StyleManager:
    """Manages styling and theming for reports"""
    
    def __init__(self):
        self.color_schemes = self._load_color_schemes()
        self.font_combinations = self._load_font_combinations()
    
    def _load_color_schemes(self) -> Dict[str, Dict[str, str]]:
        """Load predefined color schemes"""
        return {
            'corporate': {
                'primary': '#1f4e79',
                'secondary': '#4472c4',
                'accent': '#70ad47',
                'text': '#333333',
                'background': '#ffffff'
            },
            'modern': {
                'primary': '#2c3e50',
                'secondary': '#3498db',
                'accent': '#e74c3c',
                'text': '#2c3e50',
                'background': '#ffffff'
            },
            'vibrant': {
                'primary': '#ff6b6b',
                'secondary': '#4ecdc4',
                'accent': '#45b7d1',
                'text': '#2c3e50',
                'background': '#ffffff'
            }
        }
    
    def _load_font_combinations(self) -> Dict[str, Dict[str, str]]:
        """Load font combinations"""
        return {
            'professional': {
                'heading': 'Helvetica-Bold',
                'body': 'Helvetica',
                'caption': 'Helvetica-Oblique'
            },
            'academic': {
                'heading': 'Times-Bold',
                'body': 'Times-Roman',
                'caption': 'Times-Italic'
            },
            'modern': {
                'heading': 'Futura-Medium',
                'body': 'Avenir-Roman',
                'caption': 'Avenir-Oblique'
            }
        }


class AdvancedVisualizationGenerator:
    """Generates advanced visualizations for reports"""
    
    async def create_advanced_chart(self, chart_type: VisualizationType, data: Any, config: Dict[str, Any]) -> str:
        """Create advanced chart and return file path"""
        # This would create actual charts using matplotlib/plotly
        # For now, returning placeholder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chart_{chart_type.value}_{timestamp}.png"
        filepath = os.path.join(tempfile.gettempdir(), filename)
        
        # Create placeholder chart
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'{chart_type.value.replace("_", " ").title()}\nChart Placeholder', 
                ha='center', va='center', fontsize=16)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath


class InfographicCreator:
    """Creates infographic-style visualizations"""
    
    async def create_infographic(self, elements: List[InfographicElement], config: Dict[str, Any]) -> str:
        """Create infographic from elements"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"infographic_{timestamp}.png"
        filepath = os.path.join(tempfile.gettempdir(), filename)
        
        # Create infographic using PIL
        width = config.get('width', 1920)
        height = config.get('height', 1080)
        
        img = PILImage.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw placeholder infographic
        draw.text((width//2, 100), 'Infographic Placeholder', fill='black', anchor='mm')
        
        img.save(filepath)
        return filepath


class AdvancedPDFGenerator:
    """Generates advanced PDF reports with professional formatting"""
    
    async def create_advanced_pdf(self,
                                sections: List[ReportSection],
                                visualizations: List[Dict[str, Any]],
                                template: ReportTemplate,
                                config: Dict[str, Any]) -> str:
        """Create advanced PDF report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"report_{timestamp}.pdf"
        filepath = os.path.join(tempfile.gettempdir(), filename)
        
        # Create PDF document
        doc = SimpleDocTemplate(
            filepath,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Styles
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        title = config.get('title', 'Analysis Report')
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 12))
        
        # Date
        date_style = ParagraphStyle(
            'DateStyle',
            parent=styles['Normal'],
            fontSize=10,
            alignment=TA_CENTER,
            textColor=colors.grey
        )
        story.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y')}", date_style))
        story.append(Spacer(1, 20))
        
        # Sections
        for section in sections:
            # Section title
            story.append(Paragraph(section.title, styles['Heading1']))
            story.append(Spacer(1, 12))
            
            # Section content
            paragraphs = section.content.split('\n\n')
            for paragraph in paragraphs:
                if paragraph.strip():
                    if paragraph.startswith('##'):
                        story.append(Paragraph(paragraph[2:].strip(), styles['Heading2']))
                    else:
                        story.append(Paragraph(paragraph, styles['Normal']))
                    story.append(Spacer(1, 12))
            
            story.append(Spacer(1, 20))
        
        # Build PDF
        doc.build(story)
        return filepath

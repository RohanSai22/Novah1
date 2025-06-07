"""
Enhanced Coding Agent - E2B integration for code generation and data visualization
Supports Python, JavaScript, data analysis, and interactive visualizations
"""

import asyncio
import json
import tempfile
import subprocess
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path
import base64
import io

# E2B imports (placeholder - would need actual E2B SDK)
# from e2b import Session, Sandbox

# Data analysis and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Code execution and analysis
import ast
import inspect
from jinja2 import Template

from sources.utility import pretty_print, animate_thinking
from sources.agents.agent import Agent, ExecutionManager
from sources.logger import Logger
from sources.memory import Memory

class CodeLanguage(Enum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    R = "r"
    SQL = "sql"
    BASH = "bash"
    HTML = "html"
    CSS = "css"

class ExecutionEnvironment(Enum):
    LOCAL = "local"
    E2B_SANDBOX = "e2b_sandbox"
    DOCKER = "docker"
    JUPYTER = "jupyter"

class VisualizationType(Enum):
    BAR_CHART = "bar_chart"
    LINE_CHART = "line_chart"
    SCATTER_PLOT = "scatter_plot"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    PIE_CHART = "pie_chart"
    BOX_PLOT = "box_plot"
    VIOLIN_PLOT = "violin_plot"
    TREEMAP = "treemap"
    SUNBURST = "sunburst"
    INTERACTIVE_DASHBOARD = "interactive_dashboard"

@dataclass
class CodeTask:
    task_id: str
    language: CodeLanguage
    code: str
    environment: ExecutionEnvironment
    dependencies: List[str]
    expected_output: str = ""
    timeout: int = 30
    visualization_type: Optional[VisualizationType] = None
    data_source: Optional[str] = None

@dataclass
class ExecutionResult:
    task_id: str
    success: bool
    output: str
    error: str
    execution_time: float
    generated_files: List[str]
    visualization_path: Optional[str] = None
    metadata: Dict[str, Any] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class DataAnalysisRequest:
    data: Union[str, pd.DataFrame, Dict[str, Any]]
    analysis_type: str
    visualization_types: List[VisualizationType]
    filters: Dict[str, Any] = None
    groupby_columns: List[str] = None
    metrics: List[str] = None

    def __post_init__(self):
        if self.filters is None:
            self.filters = {}
        if self.groupby_columns is None:
            self.groupby_columns = []
        if self.metrics is None:
            self.metrics = []

class EnhancedCodingAgent(Agent):
    def __init__(self, name, prompt_path, provider, verbose=False, browser=None):
        """
        Enhanced Coding Agent with E2B integration and advanced data visualization
        """
        super().__init__(name, prompt_path, provider, verbose, browser)
        self.tools = {
            "execute_code": self.execute_code,
            "generate_code": self.generate_code,
            "create_visualization": self.create_visualization,
            "analyze_data": self.analyze_data,
            "create_dashboard": self.create_dashboard,
            "optimize_code": self.optimize_code,
            "debug_code": self.debug_code,
            "install_dependencies": self.install_dependencies,
            "create_jupyter_notebook": self.create_jupyter_notebook,
            "execute_in_sandbox": self.execute_in_sandbox,
            "generate_data_report": self.generate_data_report,
        }
        self.role = "enhanced_coding"
        self.type = "enhanced_coding_agent"
        self.logger = Logger("enhanced_coding_agent.log")
        self.memory = Memory(
            self.load_prompt(prompt_path),
            recover_last_session=False,
            memory_compression=False,
            model_provider=provider.get_model_name()
        )
        
        # File management
        self.workspace_dir = Path("coding_workspace")
        self.workspace_dir.mkdir(exist_ok=True)
        self.visualizations_dir = self.workspace_dir / "visualizations"
        self.visualizations_dir.mkdir(exist_ok=True)
        self.notebooks_dir = self.workspace_dir / "notebooks"
        self.notebooks_dir.mkdir(exist_ok=True)
        
        # Execution tracking
        self.execution_history = []
        self.generated_files = []
        
        # E2B configuration (placeholder)
        self.e2b_session = None
        self.sandbox_initialized = False
        
        # Code templates
        self.code_templates = self._initialize_code_templates()
        
        # Visualization settings
        plt.style.use('default')
        sns.set_theme()

    def _initialize_code_templates(self) -> Dict[str, str]:
        """Initialize code templates for different tasks"""
        return {
            "data_analysis": """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load and analyze data
def analyze_data(data_path):
    df = pd.read_csv(data_path) if isinstance(data_path, str) else data_path
    
    # Basic statistics
    print("Dataset Shape:", df.shape)
    print("\\nColumn Info:")
    print(df.info())
    print("\\nBasic Statistics:")
    print(df.describe())
    
    # Missing values
    missing = df.isnull().sum()
    if missing.any():
        print("\\nMissing Values:")
        print(missing[missing > 0])
    
    return df

# Visualization function
def create_visualizations(df):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Add your visualization code here
    
    plt.tight_layout()
    plt.savefig('analysis_results.png', dpi=300, bbox_inches='tight')
    return fig
""",
            "web_scraping": """
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def scrape_website(url, headers=None):
    if headers is None:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup

def extract_data(soup):
    # Extract data based on requirements
    data = []
    # Add extraction logic here
    return pd.DataFrame(data)
""",
            "api_integration": """
import requests
import json
import pandas as pd
from datetime import datetime

class APIClient:
    def __init__(self, base_url, api_key=None):
        self.base_url = base_url
        self.api_key = api_key
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({'Authorization': f'Bearer {api_key}'})
    
    def get(self, endpoint, params=None):
        url = f"{self.base_url}/{endpoint}"
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()
    
    def post(self, endpoint, data=None):
        url = f"{self.base_url}/{endpoint}"
        response = self.session.post(url, json=data)
        response.raise_for_status()
        return response.json()
""",
            "machine_learning": """
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, regression_metrics
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_data(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Handle categorical variables
    categorical_columns = X.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    return X, y

def train_model(X, y, model_type='classification'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Add model training logic here
    
    return model, X_test_scaled, y_test
"""
        }

    async def execute_code(self, code: str, language: CodeLanguage = CodeLanguage.PYTHON,
                          environment: ExecutionEnvironment = ExecutionEnvironment.LOCAL,
                          timeout: int = 30) -> Dict[str, Any]:
        """
        Execute code in specified environment
        """
        try:
            task_id = f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            if environment == ExecutionEnvironment.E2B_SANDBOX:
                result = await self._execute_in_e2b(code, language, task_id, timeout)
            elif environment == ExecutionEnvironment.LOCAL:
                result = await self._execute_locally(code, language, task_id, timeout)
            else:
                result = ExecutionResult(
                    task_id=task_id,
                    success=False,
                    output="",
                    error=f"Unsupported environment: {environment.value}",
                    execution_time=0.0,
                    generated_files=[]
                )
            
            self.execution_history.append(result)
            return asdict(result)
            
        except Exception as e:
            self.logger.log(f"Code execution failed: {str(e)}")
            return {
                'task_id': task_id,
                'success': False,
                'output': '',
                'error': str(e),
                'execution_time': 0.0,
                'generated_files': []
            }

    async def _execute_locally(self, code: str, language: CodeLanguage, task_id: str, timeout: int) -> ExecutionResult:
        """Execute code locally"""
        start_time = datetime.now()
        
        try:
            if language == CodeLanguage.PYTHON:
                # Create temporary file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(code)
                    temp_file = f.name
                
                # Execute Python code
                result = subprocess.run(
                    [sys.executable, temp_file],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=str(self.workspace_dir)
                )
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                # Clean up
                Path(temp_file).unlink(missing_ok=True)
                
                # Check for generated files
                generated_files = self._find_generated_files()
                
                return ExecutionResult(
                    task_id=task_id,
                    success=result.returncode == 0,
                    output=result.stdout,
                    error=result.stderr,
                    execution_time=execution_time,
                    generated_files=generated_files
                )
                
            elif language == CodeLanguage.JAVASCRIPT:
                # Execute JavaScript using Node.js
                with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                    f.write(code)
                    temp_file = f.name
                
                result = subprocess.run(
                    ['node', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=str(self.workspace_dir)
                )
                
                execution_time = (datetime.now() - start_time).total_seconds()
                Path(temp_file).unlink(missing_ok=True)
                
                return ExecutionResult(
                    task_id=task_id,
                    success=result.returncode == 0,
                    output=result.stdout,
                    error=result.stderr,
                    execution_time=execution_time,
                    generated_files=self._find_generated_files()
                )
            
            else:
                return ExecutionResult(
                    task_id=task_id,
                    success=False,
                    output="",
                    error=f"Language {language.value} not supported for local execution",
                    execution_time=0.0,
                    generated_files=[]
                )
                
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                task_id=task_id,
                success=False,
                output="",
                error=f"Execution timed out after {timeout} seconds",
                execution_time=timeout,
                generated_files=[]
            )
        except Exception as e:
            return ExecutionResult(
                task_id=task_id,
                success=False,
                output="",
                error=str(e),
                execution_time=(datetime.now() - start_time).total_seconds(),
                generated_files=[]
            )

    async def _execute_in_e2b(self, code: str, language: CodeLanguage, task_id: str, timeout: int) -> ExecutionResult:
        """Execute code in E2B sandbox (placeholder implementation)"""
        start_time = datetime.now()
        
        try:
            # Placeholder for E2B integration
            # In practice, you would use the E2B SDK:
            # session = await Session.create()
            # result = await session.execute(code, language=language.value, timeout=timeout)
            
            # For now, simulate E2B execution
            await asyncio.sleep(1)  # Simulate processing time
            
            # Mock successful execution
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ExecutionResult(
                task_id=task_id,
                success=True,
                output=f"E2B execution completed for {language.value} code",
                error="",
                execution_time=execution_time,
                generated_files=[],
                metadata={'environment': 'e2b_sandbox'}
            )
            
        except Exception as e:
            return ExecutionResult(
                task_id=task_id,
                success=False,
                output="",
                error=f"E2B execution failed: {str(e)}",
                execution_time=(datetime.now() - start_time).total_seconds(),
                generated_files=[]
            )

    async def generate_code(self, task_description: str, language: CodeLanguage = CodeLanguage.PYTHON,
                           template_type: str = None, additional_requirements: str = "") -> Dict[str, Any]:
        """
        Generate code based on task description
        """
        try:
            # Get template if specified
            template_code = ""
            if template_type and template_type in self.code_templates:
                template_code = self.code_templates[template_type]
            
            # Generate code using LLM
            messages = [
                {"role": "system", "content": f"""You are an expert {language.value} programmer. Generate clean, efficient, and well-documented code.

Language: {language.value}
Task: {task_description}
Additional Requirements: {additional_requirements}

Guidelines:
1. Write production-ready code with proper error handling
2. Include comprehensive comments and docstrings
3. Follow best practices and coding standards
4. Add type hints where applicable
5. Include example usage if relevant
6. Handle edge cases appropriately

Template (if applicable):
{template_code}"""},
                {"role": "user", "content": f"Generate {language.value} code for: {task_description}"}
            ]
            
            generated_code = self.provider.chat_completion(messages)
            
            # Extract code from response
            code = self._extract_code_from_response(generated_code, language)
            
            # Validate generated code
            validation_result = await self._validate_code(code, language)
            
            return {
                'success': True,
                'generated_code': code,
                'language': language.value,
                'validation': validation_result,
                'task_description': task_description,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.log(f"Code generation failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'task_description': task_description
            }

    async def create_visualization(self, data: Union[pd.DataFrame, str, Dict[str, Any]],
                                 visualization_type: VisualizationType,
                                 title: str = "", config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create data visualization
        """
        try:
            if config is None:
                config = {}
            
            # Load data if it's a file path
            if isinstance(data, str):
                if data.endswith('.csv'):
                    df = pd.read_csv(data)
                elif data.endswith('.json'):
                    df = pd.read_json(data)
                else:
                    return {'success': False, 'error': 'Unsupported data format'}
            elif isinstance(data, dict):
                df = pd.DataFrame(data)
            else:
                df = data
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{visualization_type.value}_{timestamp}"
            
            # Create visualization based on type
            if visualization_type == VisualizationType.BAR_CHART:
                fig = self._create_bar_chart(df, config)
            elif visualization_type == VisualizationType.LINE_CHART:
                fig = self._create_line_chart(df, config)
            elif visualization_type == VisualizationType.SCATTER_PLOT:
                fig = self._create_scatter_plot(df, config)
            elif visualization_type == VisualizationType.HISTOGRAM:
                fig = self._create_histogram(df, config)
            elif visualization_type == VisualizationType.HEATMAP:
                fig = self._create_heatmap(df, config)
            elif visualization_type == VisualizationType.PIE_CHART:
                fig = self._create_pie_chart(df, config)
            elif visualization_type == VisualizationType.BOX_PLOT:
                fig = self._create_box_plot(df, config)
            elif visualization_type == VisualizationType.TREEMAP:
                fig = self._create_treemap(df, config)
            else:
                fig = self._create_default_chart(df, config)
            
            # Save visualization
            if hasattr(fig, 'write_html'):  # Plotly figure
                output_path = self.visualizations_dir / f"{filename}.html"
                fig.write_html(str(output_path))
                
                # Also save as static image if possible
                try:
                    static_path = self.visualizations_dir / f"{filename}.png"
                    fig.write_image(str(static_path))
                except:
                    static_path = None
            else:  # Matplotlib figure
                output_path = self.visualizations_dir / f"{filename}.png"
                fig.savefig(str(output_path), dpi=300, bbox_inches='tight')
                plt.close(fig)
                static_path = output_path
            
            return {
                'success': True,
                'visualization_path': str(output_path),
                'static_path': str(static_path) if static_path else None,
                'visualization_type': visualization_type.value,
                'title': title,
                'data_shape': df.shape,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.log(f"Visualization creation failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'visualization_type': visualization_type.value
            }

    def _create_bar_chart(self, df: pd.DataFrame, config: Dict[str, Any]):
        """Create bar chart"""
        x_col = config.get('x_column', df.columns[0])
        y_col = config.get('y_column', df.columns[1] if len(df.columns) > 1 else df.columns[0])
        
        fig = px.bar(df, x=x_col, y=y_col, title=config.get('title', 'Bar Chart'))
        return fig

    def _create_line_chart(self, df: pd.DataFrame, config: Dict[str, Any]):
        """Create line chart"""
        x_col = config.get('x_column', df.columns[0])
        y_col = config.get('y_column', df.columns[1] if len(df.columns) > 1 else df.columns[0])
        
        fig = px.line(df, x=x_col, y=y_col, title=config.get('title', 'Line Chart'))
        return fig

    def _create_scatter_plot(self, df: pd.DataFrame, config: Dict[str, Any]):
        """Create scatter plot"""
        x_col = config.get('x_column', df.columns[0])
        y_col = config.get('y_column', df.columns[1] if len(df.columns) > 1 else df.columns[0])
        
        fig = px.scatter(df, x=x_col, y=y_col, title=config.get('title', 'Scatter Plot'))
        return fig

    def _create_histogram(self, df: pd.DataFrame, config: Dict[str, Any]):
        """Create histogram"""
        column = config.get('column', df.select_dtypes(include=[np.number]).columns[0])
        
        fig = px.histogram(df, x=column, title=config.get('title', 'Histogram'))
        return fig

    def _create_heatmap(self, df: pd.DataFrame, config: Dict[str, Any]):
        """Create heatmap"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if config.get('correlation', True):
            corr_matrix = numeric_df.corr()
            fig = px.imshow(corr_matrix, title=config.get('title', 'Correlation Heatmap'))
        else:
            fig = px.imshow(numeric_df, title=config.get('title', 'Heatmap'))
        
        return fig

    def _create_pie_chart(self, df: pd.DataFrame, config: Dict[str, Any]):
        """Create pie chart"""
        values_col = config.get('values_column', df.columns[0])
        names_col = config.get('names_column', df.columns[1] if len(df.columns) > 1 else None)
        
        if names_col:
            fig = px.pie(df, values=values_col, names=names_col, title=config.get('title', 'Pie Chart'))
        else:
            value_counts = df[values_col].value_counts()
            fig = px.pie(values=value_counts.values, names=value_counts.index, title=config.get('title', 'Pie Chart'))
        
        return fig

    def _create_box_plot(self, df: pd.DataFrame, config: Dict[str, Any]):
        """Create box plot"""
        y_col = config.get('y_column', df.select_dtypes(include=[np.number]).columns[0])
        x_col = config.get('x_column', None)
        
        if x_col:
            fig = px.box(df, x=x_col, y=y_col, title=config.get('title', 'Box Plot'))
        else:
            fig = px.box(df, y=y_col, title=config.get('title', 'Box Plot'))
        
        return fig

    def _create_treemap(self, df: pd.DataFrame, config: Dict[str, Any]):
        """Create treemap"""
        values_col = config.get('values_column', df.select_dtypes(include=[np.number]).columns[0])
        names_col = config.get('names_column', df.columns[0])
        
        fig = px.treemap(df, values=values_col, names=names_col, title=config.get('title', 'Treemap'))
        return fig

    def _create_default_chart(self, df: pd.DataFrame, config: Dict[str, Any]):
        """Create default chart based on data"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) >= 2:
            fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], title='Data Visualization')
        elif len(numeric_cols) == 1:
            fig = px.histogram(df, x=numeric_cols[0], title='Data Distribution')
        else:
            # Categorical data
            cat_col = df.columns[0]
            value_counts = df[cat_col].value_counts()
            fig = px.bar(x=value_counts.index, y=value_counts.values, title='Value Counts')
        
        return fig

    async def analyze_data(self, analysis_request: DataAnalysisRequest) -> Dict[str, Any]:
        """
        Perform comprehensive data analysis
        """
        try:
            # Load data
            if isinstance(analysis_request.data, str):
                if analysis_request.data.endswith('.csv'):
                    df = pd.read_csv(analysis_request.data)
                elif analysis_request.data.endswith('.json'):
                    df = pd.read_json(analysis_request.data)
                else:
                    return {'success': False, 'error': 'Unsupported data format'}
            elif isinstance(analysis_request.data, dict):
                df = pd.DataFrame(analysis_request.data)
            else:
                df = analysis_request.data
            
            # Apply filters
            filtered_df = df.copy()
            for column, value in analysis_request.filters.items():
                if column in filtered_df.columns:
                    if isinstance(value, list):
                        filtered_df = filtered_df[filtered_df[column].isin(value)]
                    else:
                        filtered_df = filtered_df[filtered_df[column] == value]
            
            # Perform analysis
            analysis_results = {
                'basic_stats': {
                    'shape': filtered_df.shape,
                    'columns': list(filtered_df.columns),
                    'dtypes': filtered_df.dtypes.to_dict(),
                    'missing_values': filtered_df.isnull().sum().to_dict(),
                    'memory_usage': filtered_df.memory_usage().sum()
                },
                'numeric_summary': {},
                'categorical_summary': {},
                'correlations': {},
                'visualizations': []
            }
            
            # Numeric analysis
            numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                analysis_results['numeric_summary'] = filtered_df[numeric_cols].describe().to_dict()
                
                # Correlation matrix
                if len(numeric_cols) > 1:
                    analysis_results['correlations'] = filtered_df[numeric_cols].corr().to_dict()
            
            # Categorical analysis
            categorical_cols = filtered_df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                cat_summary = {}
                for col in categorical_cols:
                    cat_summary[col] = {
                        'unique_count': filtered_df[col].nunique(),
                        'top_values': filtered_df[col].value_counts().head(10).to_dict()
                    }
                analysis_results['categorical_summary'] = cat_summary
            
            # Create visualizations
            for viz_type in analysis_request.visualization_types:
                viz_result = await self.create_visualization(
                    filtered_df, 
                    viz_type, 
                    f"{analysis_request.analysis_type} - {viz_type.value}"
                )
                if viz_result.get('success'):
                    analysis_results['visualizations'].append(viz_result)
            
            # Group by analysis if requested
            if analysis_request.groupby_columns:
                groupby_results = {}
                for groupby_col in analysis_request.groupby_columns:
                    if groupby_col in filtered_df.columns:
                        if analysis_request.metrics:
                            for metric in analysis_request.metrics:
                                if metric in filtered_df.columns:
                                    group_result = filtered_df.groupby(groupby_col)[metric].agg(['mean', 'sum', 'count']).to_dict()
                                    groupby_results[f"{groupby_col}_{metric}"] = group_result
                        else:
                            group_result = filtered_df.groupby(groupby_col).size().to_dict()
                            groupby_results[groupby_col] = group_result
                
                analysis_results['groupby_analysis'] = groupby_results
            
            return {
                'success': True,
                'analysis_results': analysis_results,
                'analysis_type': analysis_request.analysis_type,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.log(f"Data analysis failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'analysis_type': analysis_request.analysis_type
            }

    async def create_dashboard(self, data: Union[pd.DataFrame, str], 
                             dashboard_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create interactive dashboard
        """
        try:
            if dashboard_config is None:
                dashboard_config = {}
            
            # Load data
            if isinstance(data, str):
                if data.endswith('.csv'):
                    df = pd.read_csv(data)
                elif data.endswith('.json'):
                    df = pd.read_json(data)
                else:
                    return {'success': False, 'error': 'Unsupported data format'}
            else:
                df = data
            
            # Create subplots
            charts = dashboard_config.get('charts', [])
            if not charts:
                # Auto-generate charts based on data
                charts = self._auto_generate_charts(df)
            
            # Create dashboard with subplots
            rows = dashboard_config.get('rows', 2)
            cols = dashboard_config.get('cols', 2)
            
            fig = make_subplots(
                rows=rows, 
                cols=cols,
                subplot_titles=[chart.get('title', f'Chart {i+1}') for i, chart in enumerate(charts)],
                specs=[[{"secondary_y": False} for _ in range(cols)] for _ in range(rows)]
            )
            
            # Add charts to subplots
            for i, chart_config in enumerate(charts[:rows*cols]):
                row = (i // cols) + 1
                col = (i % cols) + 1
                
                chart_fig = await self._create_chart_for_dashboard(df, chart_config)
                
                # Add traces from chart_fig to main fig
                for trace in chart_fig.data:
                    fig.add_trace(trace, row=row, col=col)
            
            # Update layout
            fig.update_layout(
                title=dashboard_config.get('title', 'Interactive Dashboard'),
                height=dashboard_config.get('height', 800),
                showlegend=True
            )
            
            # Save dashboard
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dashboard_path = self.visualizations_dir / f"dashboard_{timestamp}.html"
            fig.write_html(str(dashboard_path))
            
            return {
                'success': True,
                'dashboard_path': str(dashboard_path),
                'charts_count': len(charts),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.log(f"Dashboard creation failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    def _auto_generate_charts(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Auto-generate chart configurations based on data"""
        charts = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Correlation heatmap for numeric data
        if len(numeric_cols) > 1:
            charts.append({
                'type': 'heatmap',
                'title': 'Correlation Matrix',
                'config': {'correlation': True}
            })
        
        # Distribution of first numeric column
        if len(numeric_cols) > 0:
            charts.append({
                'type': 'histogram',
                'title': f'Distribution of {numeric_cols[0]}',
                'config': {'column': numeric_cols[0]}
            })
        
        # Scatter plot if multiple numeric columns
        if len(numeric_cols) >= 2:
            charts.append({
                'type': 'scatter',
                'title': f'{numeric_cols[0]} vs {numeric_cols[1]}',
                'config': {'x_column': numeric_cols[0], 'y_column': numeric_cols[1]}
            })
        
        # Value counts for categorical data
        if len(categorical_cols) > 0:
            charts.append({
                'type': 'bar',
                'title': f'Counts by {categorical_cols[0]}',
                'config': {'x_column': categorical_cols[0]}
            })
        
        return charts

    async def _create_chart_for_dashboard(self, df: pd.DataFrame, chart_config: Dict[str, Any]):
        """Create individual chart for dashboard"""
        chart_type = chart_config.get('type', 'bar')
        config = chart_config.get('config', {})
        
        if chart_type == 'bar':
            return self._create_bar_chart(df, config)
        elif chart_type == 'line':
            return self._create_line_chart(df, config)
        elif chart_type == 'scatter':
            return self._create_scatter_plot(df, config)
        elif chart_type == 'histogram':
            return self._create_histogram(df, config)
        elif chart_type == 'heatmap':
            return self._create_heatmap(df, config)
        elif chart_type == 'pie':
            return self._create_pie_chart(df, config)
        elif chart_type == 'box':
            return self._create_box_plot(df, config)
        else:
            return self._create_default_chart(df, config)

    def _extract_code_from_response(self, response: str, language: CodeLanguage) -> str:
        """Extract code from LLM response"""
        # Look for code blocks
        code_patterns = [
            f"```{language.value}\n(.*?)\n```",
            f"```\n(.*?)\n```",
            f"<code>\n(.*?)\n</code>"
        ]
        
        import re
        for pattern in code_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                return matches[0].strip()
        
        # If no code blocks found, return the entire response
        return response.strip()

    async def _validate_code(self, code: str, language: CodeLanguage) -> Dict[str, Any]:
        """Validate generated code"""
        try:
            if language == CodeLanguage.PYTHON:
                # Parse Python code
                ast.parse(code)
                return {
                    'valid': True,
                    'syntax_errors': [],
                    'warnings': []
                }
            else:
                # For other languages, basic validation
                return {
                    'valid': True,
                    'syntax_errors': [],
                    'warnings': ['Validation not implemented for this language']
                }
        except SyntaxError as e:
            return {
                'valid': False,
                'syntax_errors': [str(e)],
                'warnings': []
            }
        except Exception as e:
            return {
                'valid': False,
                'syntax_errors': [f"Validation error: {str(e)}"],
                'warnings': []
            }

    def _find_generated_files(self) -> List[str]:
        """Find files generated during code execution"""
        # Look for common output files
        generated_files = []
        common_extensions = ['.png', '.jpg', '.svg', '.html', '.csv', '.json', '.txt']
        
        for file_path in self.workspace_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix in common_extensions:
                # Check if file was created recently (within last minute)
                if (datetime.now() - datetime.fromtimestamp(file_path.stat().st_mtime)).seconds < 60:
                    generated_files.append(str(file_path))
        
        return generated_files

    async def optimize_code(self, code: str, language: CodeLanguage = CodeLanguage.PYTHON) -> Dict[str, Any]:
        """Optimize existing code"""
        try:
            messages = [
                {"role": "system", "content": f"""You are an expert {language.value} code optimizer. Analyze the provided code and suggest optimizations for:

1. Performance improvements
2. Memory efficiency  
3. Code readability
4. Best practices
5. Error handling
6. Security considerations

Provide the optimized code along with explanations of the changes made."""},
                {"role": "user", "content": f"Optimize this {language.value} code:\n\n{code}"}
            ]
            
            optimization_response = self.provider.chat_completion(messages)
            optimized_code = self._extract_code_from_response(optimization_response, language)
            
            return {
                'success': True,
                'original_code': code,
                'optimized_code': optimized_code,
                'optimization_notes': optimization_response,
                'language': language.value,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.log(f"Code optimization failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'original_code': code
            }

    async def debug_code(self, code: str, error_message: str, language: CodeLanguage = CodeLanguage.PYTHON) -> Dict[str, Any]:
        """Debug code with error message"""
        try:
            messages = [
                {"role": "system", "content": f"""You are an expert {language.value} debugger. Analyze the provided code and error message to:

1. Identify the root cause of the error
2. Explain why the error occurred
3. Provide a fixed version of the code
4. Suggest preventive measures
5. Add appropriate error handling

Be thorough in your analysis and provide clear explanations."""},
                {"role": "user", "content": f"Debug this {language.value} code:\n\nCode:\n{code}\n\nError Message:\n{error_message}"}
            ]
            
            debug_response = self.provider.chat_completion(messages)
            fixed_code = self._extract_code_from_response(debug_response, language)
            
            return {
                'success': True,
                'original_code': code,
                'error_message': error_message,
                'fixed_code': fixed_code,
                'debug_analysis': debug_response,
                'language': language.value,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.log(f"Code debugging failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'original_code': code,
                'error_message': error_message
            }

    async def install_dependencies(self, packages: List[str], package_manager: str = "pip") -> Dict[str, Any]:
        """Install code dependencies"""
        try:
            results = {}
            
            for package in packages:
                try:
                    if package_manager == "pip":
                        result = subprocess.run(
                            [sys.executable, "-m", "pip", "install", package],
                            capture_output=True,
                            text=True,
                            timeout=300  # 5 minutes timeout
                        )
                    elif package_manager == "npm":
                        result = subprocess.run(
                            ["npm", "install", package],
                            capture_output=True,
                            text=True,
                            timeout=300,
                            cwd=str(self.workspace_dir)
                        )
                    else:
                        results[package] = {
                            'success': False,
                            'error': f'Unsupported package manager: {package_manager}'
                        }
                        continue
                    
                    results[package] = {
                        'success': result.returncode == 0,
                        'output': result.stdout,
                        'error': result.stderr if result.returncode != 0 else None
                    }
                    
                except subprocess.TimeoutExpired:
                    results[package] = {
                        'success': False,
                        'error': 'Installation timed out'
                    }
                except Exception as e:
                    results[package] = {
                        'success': False,
                        'error': str(e)
                    }
            
            return {
                'success': True,
                'results': results,
                'package_manager': package_manager,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.log(f"Dependency installation failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'packages': packages
            }

    async def create_jupyter_notebook(self, cells: List[Dict[str, Any]], notebook_name: str = None) -> Dict[str, Any]:
        """Create Jupyter notebook with specified cells"""
        try:
            if not notebook_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                notebook_name = f"generated_notebook_{timestamp}.ipynb"
            
            # Create notebook structure
            notebook = {
                "cells": [],
                "metadata": {
                    "kernelspec": {
                        "display_name": "Python 3",
                        "language": "python",
                        "name": "python3"
                    },
                    "language_info": {
                        "name": "python",
                        "version": "3.8.0"
                    }
                },
                "nbformat": 4,
                "nbformat_minor": 4
            }
            
            # Add cells
            for cell_config in cells:
                cell = {
                    "cell_type": cell_config.get("type", "code"),
                    "source": cell_config.get("content", ""),
                    "metadata": {}
                }
                
                if cell["cell_type"] == "code":
                    cell["execution_count"] = None
                    cell["outputs"] = []
                
                notebook["cells"].append(cell)
            
            # Save notebook
            notebook_path = self.notebooks_dir / notebook_name
            with open(notebook_path, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, indent=2)
            
            return {
                'success': True,
                'notebook_path': str(notebook_path),
                'notebook_name': notebook_name,
                'cells_count': len(cells),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.log(f"Jupyter notebook creation failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'notebook_name': notebook_name
            }

    async def execute_in_sandbox(self, code: str, language: CodeLanguage = CodeLanguage.PYTHON,
                                timeout: int = 30) -> Dict[str, Any]:
        """Execute code in E2B sandbox"""
        try:
            # This is a placeholder for E2B integration
            # In practice, you would use the actual E2B SDK
            
            execution_result = await self._execute_in_e2b(code, language, f"sandbox_{datetime.now().strftime('%Y%m%d_%H%M%S')}", timeout)
            
            return {
                'success': execution_result.success,
                'output': execution_result.output,
                'error': execution_result.error,
                'execution_time': execution_result.execution_time,
                'environment': 'e2b_sandbox',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.log(f"Sandbox execution failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'code': code
            }

    async def generate_data_report(self, data: Union[pd.DataFrame, str], 
                                 report_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate comprehensive data analysis report"""
        try:
            if report_config is None:
                report_config = {}
            
            # Perform comprehensive analysis
            analysis_request = DataAnalysisRequest(
                data=data,
                analysis_type="comprehensive",
                visualization_types=[
                    VisualizationType.HISTOGRAM,
                    VisualizationType.HEATMAP,
                    VisualizationType.BAR_CHART,
                    VisualizationType.SCATTER_PLOT
                ]
            )
            
            analysis_result = await self.analyze_data(analysis_request)
            
            if not analysis_result.get('success'):
                return analysis_result
            
            # Generate HTML report
            report_template = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Data Analysis Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                    .section { margin: 20px 0; }
                    .stats-table { border-collapse: collapse; width: 100%; }
                    .stats-table th, .stats-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    .visualization { margin: 20px 0; }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Data Analysis Report</h1>
                    <p>Generated on: {{ timestamp }}</p>
                    <p>Dataset Shape: {{ shape[0] }} rows × {{ shape[1] }} columns</p>
                </div>
                
                <div class="section">
                    <h2>Basic Statistics</h2>
                    <table class="stats-table">
                        <tr><th>Metric</th><th>Value</th></tr>
                        <tr><td>Total Records</td><td>{{ shape[0] }}</td></tr>
                        <tr><td>Total Columns</td><td>{{ shape[1] }}</td></tr>
                        <tr><td>Missing Values</td><td>{{ missing_count }}</td></tr>
                        <tr><td>Memory Usage</td><td>{{ memory_usage }} bytes</td></tr>
                    </table>
                </div>
                
                <div class="section">
                    <h2>Visualizations</h2>
                    {% for viz in visualizations %}
                    <div class="visualization">
                        <h3>{{ viz.visualization_type }}</h3>
                        <p>Path: {{ viz.visualization_path }}</p>
                    </div>
                    {% endfor %}
                </div>
            </body>
            </html>
            """
            
            template = Template(report_template)
            analysis_data = analysis_result['analysis_results']
            
            report_html = template.render(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                shape=analysis_data['basic_stats']['shape'],
                missing_count=sum(analysis_data['basic_stats']['missing_values'].values()),
                memory_usage=analysis_data['basic_stats']['memory_usage'],
                visualizations=analysis_data['visualizations']
            )
            
            # Save report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.workspace_dir / f"data_report_{timestamp}.html"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_html)
            
            return {
                'success': True,
                'report_path': str(report_path),
                'analysis_results': analysis_result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.log(f"Data report generation failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    async def execute(self, prompt: str, execution_manager: Optional[ExecutionManager] = None) -> dict:
        self.logger.info(f"Executing coding task: {prompt}")

        try:
            # Use LLM to generate the code first
            code_generation_result = await self.generate_code(prompt, CodeLanguage.PYTHON)
            if not code_generation_result['success']:
                return code_generation_result

            generated_code = code_generation_result['generated_code']
            
            # Now, execute the generated code
            execution_result = await self.execute_code(generated_code)

            if execution_manager:
                current_executions = execution_manager.execution_state['coding'].get('executions', [])
                current_executions.append(execution_result)
                execution_manager.update_state({"coding": {"executions": current_executions}})

            return {
                "success": execution_result['success'],
                "summary": f"Generated and executed code for task: {prompt}",
                "code": generated_code,
                "output": execution_result.get('output', ''),
                "error": execution_result.get('error', '')
            }
        except Exception as e:
            self.logger.error(f"Coding task failed: {e}")
            return {"success": False, "error": str(e)}

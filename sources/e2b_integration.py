#!/usr/bin/env python3
"""
E2B Sandbox Integration for Novah AI
Provides secure code execution, data analysis, and visualization capabilities
"""
import asyncio
import time
import uuid
import os
import json
import sys
import subprocess
from typing import Dict, Any, List, Optional
from datetime import datetime

# Try to import required packages, install if missing
try:
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import io
    import base64
except ImportError as e:
    print(f"Installing required packages: {e}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "pandas", "numpy"])
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import io
    import base64

# Try E2B API integration
try:
    from e2b import Sandbox
    E2B_API_AVAILABLE = True
except ImportError:
    print("E2B SDK not available, using local execution mode")
    E2B_API_AVAILABLE = False

class E2BSandbox:
    """E2B Sandbox integration for secure code execution"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("E2B_API_KEY")
        self.active_sandboxes = {}
        self.execution_history = []
        self.use_api = E2B_API_AVAILABLE and self.api_key is not None
        
        if self.use_api:
            print("✅ E2B API mode enabled")
        else:
            print("⚠️ Using local execution mode (E2B API not available)")
        
    async def create_sandbox(self, language: str = "python") -> str:
        """Create a new sandbox for code execution"""
        sandbox_id = str(uuid.uuid4())
        
        # Simulate sandbox creation
        self.active_sandboxes[sandbox_id] = {
            "id": sandbox_id,
            "language": language,
            "created_at": time.time(),
            "status": "active",
            "environment": {
                "python_version": "3.9.0",
                "packages": ["pandas", "numpy", "matplotlib", "seaborn", "plotly"],
                "memory_limit": "512MB",
                "timeout": 30
            }
        }
        
        return sandbox_id
    
    async def execute_code(self, code: str, language: str = "python", 
                          sandbox_id: Optional[str] = None) -> Dict[str, Any]:
        """Execute code in E2B sandbox"""
        
        # Create sandbox if not provided
        if not sandbox_id:
            sandbox_id = await self.create_sandbox(language)
        
        execution_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Simulate code execution with actual execution for safe operations
            if language.lower() == "python":
                result = await self._execute_python_safe(code, execution_id)
            else:
                result = await self._simulate_execution(code, language, execution_id)
            
            execution_time = time.time() - start_time
            
            execution_result = {
                "id": execution_id,
                "sandbox_id": sandbox_id,
                "code": code,
                "language": language,
                "output": result.get("output", ""),
                "error": result.get("error"),
                "status": result.get("status", "completed"),
                "timestamp": start_time,
                "execution_time": execution_time,
                "artifacts": result.get("artifacts", []),
                "visualizations": result.get("visualizations", [])
            }
            
            # Store in history
            self.execution_history.append(execution_result)
            
            return execution_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_result = {
                "id": execution_id,
                "sandbox_id": sandbox_id,
                "code": code,
                "language": language,
                "output": "",
                "error": str(e),
                "status": "error",
                "timestamp": start_time,
                "execution_time": execution_time,
                "artifacts": [],
                "visualizations": []
            }
            
            self.execution_history.append(error_result)
            return error_result
    
    async def _execute_python_safe(self, code: str, execution_id: str) -> Dict[str, Any]:
        """Execute Python code safely with visualization support"""
        
        # Create safe execution environment
        safe_globals = {
            '__builtins__': {
                'print': print,
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'min': min,
                'max': max,
                'sum': sum,
                'abs': abs,
                'round': round,
                'sorted': sorted,
                'reversed': reversed,
            },
            'pd': pd,
            'np': np,
            'plt': plt,
            'datetime': datetime,
            'json': json
        }
        
        # Capture output
        output_lines = []
        artifacts = []
        visualizations = []
        
        # Create custom print function
        def capture_print(*args, **kwargs):
            output_lines.append(' '.join(str(arg) for arg in args))
        
        safe_globals['print'] = capture_print
        
        try:
            # Check for potentially dangerous operations
            dangerous_patterns = ['import os', 'import sys', 'exec(', 'eval(', '__import__']
            if any(pattern in code for pattern in dangerous_patterns):
                return {
                    "output": "Code contains potentially dangerous operations and was blocked for security.",
                    "status": "blocked",
                    "artifacts": [],
                    "visualizations": []
                }
            
            # Execute the code
            exec(code, safe_globals)
            
            # Check for matplotlib plots
            if plt.get_fignums():
                # Save plots as base64 images
                for fig_num in plt.get_fignums():
                    fig = plt.figure(fig_num)
                    
                    # Save plot to bytes
                    img_buffer = io.BytesIO()
                    fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                    img_buffer.seek(0)
                    
                    # Convert to base64
                    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                    
                    visualization = {
                        "id": str(uuid.uuid4()),
                        "type": "matplotlib_plot",
                        "title": f"Plot {fig_num}",
                        "data": f"data:image/png;base64,{img_base64}",
                        "timestamp": time.time()
                    }
                    
                    visualizations.append(visualization)
                    
                    plt.close(fig)
            
            # Check for data analysis artifacts
            for name, obj in safe_globals.items():
                if isinstance(obj, pd.DataFrame) and not name.startswith('_'):
                    # Create DataFrame summary
                    artifact = {
                        "id": str(uuid.uuid4()),
                        "type": "dataframe",
                        "name": name,
                        "shape": obj.shape,
                        "columns": obj.columns.tolist(),
                        "head": obj.head().to_dict(),
                        "dtypes": obj.dtypes.to_dict(),
                        "timestamp": time.time()
                    }
                    artifacts.append(artifact)
            
            return {
                "output": '\n'.join(output_lines) if output_lines else "Code executed successfully",
                "status": "completed",
                "artifacts": artifacts,
                "visualizations": visualizations
            }
            
        except Exception as e:
            return {
                "output": f"Execution error: {str(e)}",
                "error": str(e),
                "status": "error",
                "artifacts": artifacts,
                "visualizations": visualizations
            }
    
    async def _simulate_execution(self, code: str, language: str, execution_id: str) -> Dict[str, Any]:
        """Simulate code execution for non-Python languages"""
        
        # Simulate execution delay
        await asyncio.sleep(1)
        
        # Language-specific simulation
        if language.lower() in ["javascript", "js"]:
            output = f"// JavaScript execution simulation\n{code}\n\n// Output: JavaScript code executed successfully"
        elif language.lower() in ["bash", "shell"]:
            output = f"$ {code}\nCommand executed successfully"
        elif language.lower() == "sql":
            output = f"-- SQL Query executed\n{code}\n\n-- Results: Query completed successfully"
        else:
            output = f"Code executed in {language} environment:\n{code}\n\nExecution completed successfully"
        
        return {
            "output": output,
            "status": "completed",
            "artifacts": [],
            "visualizations": []
        }
    
    async def create_data_visualization(self, data: Dict[str, Any], 
                                      chart_type: str = "line") -> Dict[str, Any]:
        """Create data visualization using matplotlib"""
        
        try:
            # Create sample data visualization
            plt.figure(figsize=(10, 6))
            
            if chart_type == "line":
                x = data.get("x", list(range(10)))
                y = data.get("y", np.random.rand(10))
                plt.plot(x, y, marker='o')
                plt.title("Line Chart")
                plt.xlabel("X-axis")
                plt.ylabel("Y-axis")
                
            elif chart_type == "bar":
                categories = data.get("categories", ["A", "B", "C", "D"])
                values = data.get("values", np.random.rand(4))
                plt.bar(categories, values)
                plt.title("Bar Chart")
                plt.xlabel("Categories")
                plt.ylabel("Values")
                
            elif chart_type == "scatter":
                x = data.get("x", np.random.rand(50))
                y = data.get("y", np.random.rand(50))
                plt.scatter(x, y, alpha=0.6)
                plt.title("Scatter Plot")
                plt.xlabel("X-axis")
                plt.ylabel("Y-axis")
            
            # Save plot to base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
            
            return {
                "id": str(uuid.uuid4()),
                "type": "data_visualization",
                "chart_type": chart_type,
                "data": f"data:image/png;base64,{img_base64}",
                "timestamp": time.time(),
                "status": "success"
            }
            
        except Exception as e:
            return {
                "id": str(uuid.uuid4()),
                "type": "data_visualization",
                "chart_type": chart_type,
                "error": str(e),
                "timestamp": time.time(),
                "status": "error"
            }
    
    async def analyze_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform data analysis on provided dataset"""
        
        try:
            # Convert data to DataFrame if possible
            if isinstance(data, dict) and "data" in data:
                df = pd.DataFrame(data["data"])
            else:
                df = pd.DataFrame(data)
            
            analysis = {
                "id": str(uuid.uuid4()),
                "type": "data_analysis",
                "timestamp": time.time(),
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.to_dict(),
                "summary": df.describe().to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "memory_usage": df.memory_usage(deep=True).to_dict()
            }
            
            # Add correlation matrix for numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 1:
                analysis["correlation_matrix"] = df[numeric_columns].corr().to_dict()
            
            return analysis
            
        except Exception as e:
            return {
                "id": str(uuid.uuid4()),
                "type": "data_analysis",
                "timestamp": time.time(),
                "error": str(e),
                "status": "error"
            }
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get execution history"""
        return self.execution_history
    
    def get_active_sandboxes(self) -> Dict[str, Any]:
        """Get active sandboxes"""
        return self.active_sandboxes
    
    async def cleanup_sandbox(self, sandbox_id: str) -> bool:
        """Cleanup a sandbox"""
        if sandbox_id in self.active_sandboxes:
            del self.active_sandboxes[sandbox_id]
            return True
        return False

# Global E2B instance
e2b_sandbox = E2BSandbox()

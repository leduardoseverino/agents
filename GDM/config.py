"""
Configuration and Compatibility Module for AG2 Enhanced Documentation Flow
===========================================================================

Handles system configuration, compatibility fixes, and data models.
"""

import os
import sys
import logging
import subprocess
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json

# =============================================================================
# TORCH + STREAMLIT COMPATIBILITY FIX
# =============================================================================

def comprehensive_torch_fix():
    """Complete and robust fix for torch/streamlit issues"""
    try:
        # 1. Set environment variables BEFORE any imports
        os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
        os.environ["STREAMLIT_SERVER_HEADLESS"] = "true" 
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "0"
        
        # 2. Fix torch if already imported
        if 'torch' in sys.modules:
            torch = sys.modules['torch']
            
            # Fix classes.__path__
            if hasattr(torch, 'classes'):
                if not hasattr(torch.classes, '__path__'):
                    torch.classes.__path__ = []
                elif hasattr(torch.classes.__path__, '_path'):
                    # Replace problematic implementation
                    class SafePath:
                        def __init__(self):
                            self._path = []
                        def __iter__(self):
                            return iter(self._path)
                        def __getattr__(self, name):
                            return []
                        def __getitem__(self, key):
                            return []
                        def __len__(self):
                            return 0
                    
                    torch.classes.__path__ = SafePath()
            
            # Fix other problematic attributes
            problematic_attrs = ['_path', '__path__', '_modules']
            for attr in problematic_attrs:
                if hasattr(torch.classes, attr):
                    try:
                        setattr(torch.classes, attr, [])
                    except:
                        pass
        
        # 3. Mock future torch imports
        if 'torch' not in sys.modules:
            class MockTorchClasses:
                def __init__(self):
                    self.__path__ = []
                def __getattr__(self, name):
                    return []
            
            class MockTorch:
                def __init__(self):
                    self.classes = MockTorchClasses()
                def __getattr__(self, name):
                    return lambda *args, **kwargs: None
            
            sys.modules['torch'] = MockTorch()
            sys.modules['torch.classes'] = MockTorchClasses()
        
        print("🔧 Torch/Streamlit fix applied")
        
    except Exception as e:
        print(f"⚠️ Warning torch fix: {e}")

# Apply fix BEFORE any other imports
comprehensive_torch_fix()

# Now safe imports
import streamlit as st

# =============================================================================
# PYDANTIC COMPATIBILITY
# =============================================================================

try:
    from pydantic import BaseModel, Field, ConfigDict
    PYDANTIC_V2 = True
    print("✅ Pydantic V2 detected")
except ImportError:
    from pydantic import BaseModel, Field
    PYDANTIC_V2 = False
    print("⚠️ Pydantic V1 in use")

# =============================================================================
# AG2 IMPORTS
# =============================================================================

try:
    import autogen
    from autogen import ConversableAgent, UserProxyAgent, GroupChat, GroupChatManager
    AG2_AVAILABLE = True
    print("✅ AG2 available")
except ImportError as e:
    AG2_AVAILABLE = False
    print(f"❌ AG2 not available: {e}")

# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# DATA MODELS - PYDANTIC V1/V2 COMPATIBLE
# =============================================================================

class DocItem(BaseModel):
    """Documentation item - Compatible with Pydantic V1/V2"""
    title: str = Field(description="Documentation section title")
    description: str = Field(description="Detailed content description")
    prerequisites: str = Field(description="Required prerequisites")
    examples: List[str] = Field(description="List of practical examples", default_factory=list)
    goal: str = Field(description="Specific documentation goal")
    
    # V2 configuration (ignored if V1)
    if PYDANTIC_V2:
        model_config = ConfigDict(
            validate_assignment=True,
            extra='forbid'
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """V1/V2 compatible serialization method"""
        if PYDANTIC_V2:
            return self.model_dump()
        else:
            return self.dict()

class DocPlan(BaseModel):
    """Documentation plan - Compatible with Pydantic V1/V2"""
    overview: str = Field(description="Project overview")
    docs: List[DocItem] = Field(description="List of documentation items", default_factory=list)
    
    # V2 configuration (ignored if V1)
    if PYDANTIC_V2:
        model_config = ConfigDict(
            validate_assignment=True,
            extra='forbid'
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """V1/V2 compatible serialization method"""
        if PYDANTIC_V2:
            return self.model_dump()
        else:
            return self.dict()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocPlan':
        """V1/V2 compatible deserialization method"""
        if PYDANTIC_V2:
            return cls.model_validate(data)
        else:
            return cls.parse_obj(data)

class DocumentationState(BaseModel):
    """Workflow state - Compatible with Pydantic V1/V2"""
    project_url: str
    repo_path: Optional[str] = None
    current_phase: str = "init"
    plan: Optional[DocPlan] = None
    generated_docs: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # V2 configuration (ignored if V1)
    if PYDANTIC_V2:
        model_config = ConfigDict(
            validate_assignment=True,
            extra='allow',
            arbitrary_types_allowed=True
        )
    else:
        class Config:
            arbitrary_types_allowed = True
    
    def to_dict(self) -> Dict[str, Any]:
        """V1/V2 compatible method"""
        if PYDANTIC_V2:
            return self.model_dump()
        else:
            return self.dict()

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

@dataclass
class ModelConfig:
    """LLM model configuration"""
    llm_model: str = "devstral:latest"
    context_window: int = 128000
    max_tokens: int = 96000
    timeout: int = 120
    temperature: float = 0.1
    fast_review_mode: bool = True  # Enable fast reviewer for better performance

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def check_system_requirements() -> Dict[str, bool]:
    """Check if all system requirements are met"""
    requirements = {
        "ag2_available": AG2_AVAILABLE,
        "pydantic_available": True,  # We have it if we got here
        "ollama_available": False,
        "devstral_available": False,
        "git_available": False
    }
    
    # Check Ollama
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            requirements["ollama_available"] = True
            
            # Check for devstral model
            if "devstral:latest" in result.stdout:
                requirements["devstral_available"] = True
                
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    
    # Check Git
    try:
        result = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            requirements["git_available"] = True
            
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    
    return requirements

def get_system_info() -> Dict[str, str]:
    """Get system information"""
    return {
        "pydantic_version": "V2" if PYDANTIC_V2 else "V1",
        "ag2_available": "Yes" if AG2_AVAILABLE else "No",
        "python_version": sys.version.split()[0],
        "platform": sys.platform
    }

def create_default_advanced_config() -> Dict[str, Any]:
    """Create default advanced configuration"""
    return {
        "analysis_mode": "all",  # all, smart, or limited
        "focus_languages": [],
        "include_config": True,
        "deep_functions": True,
        "fast_review_mode": True  # Enable fast reviewer by default
    }

# =============================================================================
# CONSTANTS
# =============================================================================

SUPPORTED_LANGUAGES = {
    '.py': 'Python',
    '.js': 'JavaScript', 
    '.ts': 'TypeScript',
    '.java': 'Java',
    '.cpp': 'C++',
    '.c': 'C',
    '.h': 'C/C++ Header',
    '.go': 'Go',
    '.rs': 'Rust',
    '.php': 'PHP',
    '.rb': 'Ruby',
    '.scala': 'Scala',
    '.kt': 'Kotlin',
    '.swift': 'Swift',
    '.json': 'JSON',
    '.yaml': 'YAML',
    '.yml': 'YAML',
    '.toml': 'TOML',
    '.xml': 'XML',
    '.html': 'HTML',
    '.css': 'CSS',
    '.md': 'Markdown',
    '.txt': 'Text',
    '.sql': 'SQL',
    '.sh': 'Shell Script',
    '.bat': 'Batch',
    '.ps1': 'PowerShell'
}

KEY_FILE_PATTERNS = {
    "🚀 Entry Points": [
        "main.py", "index.js", "app.py", "server.py", "main.go", 
        "index.html", "App.js", "__init__.py", "main.java", "index.php"
    ],
    "📋 Project Configuration": [
        "package.json", "requirements.txt", "pom.xml", "Cargo.toml", 
        "go.mod", "setup.py", "pyproject.toml", "composer.json", "build.gradle"
    ],
    "📖 Documentation": [
        "README.md", "README.rst", "README.txt", "CHANGELOG.md", 
        "LICENSE", "CONTRIBUTING.md", "docs/", "INSTALL.md"
    ],
    "🔧 Build and Deploy": [
        "Makefile", "Dockerfile", "docker-compose.yml", 
        ".github/workflows/", "Jenkinsfile", "build.gradle", "webpack.config.js"
    ],
    "⚙️ Environment Configuration": [
        "config.py", "settings.py", ".env", "config.json",
        "webpack.config.js", "tsconfig.json", ".eslintrc", "pytest.ini"
    ],
    "🧪 Tests": [
        "test_", "_test.py", ".test.js", "spec.js", "tests/", 
        "test/", "pytest.ini", "jest.config.js"
    ],
    "🎨 Interface/Frontend": [
        "style.css", "main.css", "app.css", "index.html", 
        "template", "static/", "public/", "assets/"
    ]
}

SYSTEM_VERSION = "Enhanced AG2 Flow v2.0"
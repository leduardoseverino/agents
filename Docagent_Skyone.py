#!/usr/bin/env python3
"""
DocAgent LangGraph - Sistema de Documentação Automática de Repositórios
=======================================================================

Sistema avançado baseado em LangGraph para análise e documentação completa
de repositórios GitHub, com suporte para OpenAI e Ollama.

Características:
- Arquitetura baseada em LangGraph com fluxo de estados
- Agentes especializados para análise e documentação
- Suporte para OpenAI GPT-4 e modelos Ollama locais
- Interface web moderna com autenticação
- Relatórios anônimos e documentação técnica completa
- Sistema de tools avançadas para análise de código
- Arquitetura C4 Model para documentação arquitetural

- DocAgent Skyone v3.0 LangGraph 
"""

import os
import sys
import json
import subprocess
import tempfile
import shutil
import time
import logging
import traceback
import hashlib
import re
import fnmatch
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, TypedDict, Annotated
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import asyncio
from contextlib import asynccontextmanager

# Carregar variáveis de ambiente primeiro
try:
    from dotenv import load_dotenv
    load_dotenv()  # Carrega variáveis do arquivo .env
    print("✅ Variáveis de ambiente carregadas do .env")
except ImportError:
    print("⚠️ python-dotenv não disponível - usando apenas variáveis do sistema")
    print("   Para instalar: pip install python-dotenv")
except Exception as e:
    print(f"⚠️ Erro ao carregar .env: {e}")

# GitHub Enterprise SSO imports
try:
    from github_enterprise_sso import (
        integrate_github_enterprise_sso,
        GitHubEnterpriseConfig,
        GitHubEnterpriseAuth,
        GitHubEnterpriseRepositoryFetcher,
        AuthenticationMiddleware,
        setup_environment_variables
    )
    GITHUB_ENTERPRISE_AVAILABLE = True
    print("✅ GitHub Enterprise SSO disponível")
except ImportError as e:
    GITHUB_ENTERPRISE_AVAILABLE = False
    print(f"⚠️ GitHub Enterprise SSO não disponível: {e}")
    print("   Para habilitar: coloque github_enterprise_sso.py no mesmo diretório")


# Importar configurações
try:
    from config import config
    CONFIG_AVAILABLE = True
    print("✅ Configurações carregadas")
except ImportError:
    CONFIG_AVAILABLE = False
    print("⚠️ Arquivo de configuração não encontrado - usando padrões")

# LangGraph e LangChain imports
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
    from langchain_core.tools import tool
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
    from langchain_community.llms import Ollama
    from langchain_core.runnables import RunnableConfig
    
    # Tentar importar ToolNode da versão mais recente
    try:
        from langgraph.prebuilt import ToolNode
    except ImportError:
        ToolNode = None
        print("⚠️ ToolNode não disponível")
    
    # Importações para Mermaid nativo do LangGraph
    try:
        from IPython.display import Image, display
        MERMAID_NATIVE_AVAILABLE = True
        print("✅ Mermaid nativo disponível")
    except ImportError:
        MERMAID_NATIVE_AVAILABLE = False
        print("⚠️ Mermaid nativo não disponível")
    
    # Importações para geração PNG com Playwright
    try:
        from playwright.sync_api import sync_playwright
        PLAYWRIGHT_AVAILABLE = True
        print("✅ Playwright disponível para PNG")
    except ImportError:
        PLAYWRIGHT_AVAILABLE = False
        print("⚠️ Playwright não disponível")
    
    LANGGRAPH_AVAILABLE = True
    print("✅ LangGraph e LangChain disponíveis")
except ImportError as e:
    LANGGRAPH_AVAILABLE = False
    MERMAID_NATIVE_AVAILABLE = False
    print(f"❌ LangGraph/LangChain não disponível: {e}")
    print("Execute: pip install langgraph langchain langchain-openai langchain-community")

# FastAPI e dependências web
try:
    from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
    from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    WEB_AVAILABLE = True
    print("✅ FastAPI disponível")
except ImportError as e:
    WEB_AVAILABLE = False
    print(f"❌ FastAPI não disponível: {e}")

# Pydantic
try:
    from pydantic import BaseModel, Field, ConfigDict
    PYDANTIC_V2 = True
    print("✅ Pydantic V2 detectado")
except ImportError:
    try:
        from pydantic import BaseModel, Field
        PYDANTIC_V2 = False
        print("⚠️ Pydantic V1 em uso")
    except ImportError as e:
        print(f"❌ Pydantic não disponível: {e}")
        exit(1)

# Configurar logging
if CONFIG_AVAILABLE:
    # Criar diretório de logs
    config.LOGS_DIR.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )
else:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

logger = logging.getLogger(__name__)

# =============================================================================
# ESTADO DO LANGGRAPH - DEFINIÇÕES PRINCIPAIS
# =============================================================================

class DocumentationState(TypedDict):
    """Estado principal do fluxo LangGraph para documentação"""
    # Entrada
    repo_url: str
    model_provider: str  # "openai" ou "ollama"
    model_name: str
    anonymous: bool
    max_files: int
    deep_analysis: bool
    
    # Estado do processamento
    repo_path: Optional[str]
    current_phase: str
    progress: int
    logs: List[str]
    
    # Análise
    file_structure: Optional[Dict[str, Any]]
    code_analysis: Optional[Dict[str, Any]]
    architecture_analysis: Optional[Dict[str, Any]]
    
    # Documentação
    documentation_plan: Optional[Dict[str, Any]]
    generated_docs: List[str]
    
    # Metadata
    metadata: Dict[str, Any]
    error_count: int

# =============================================================================
# MODELOS DE DADOS
# =============================================================================

@dataclass
class RepositoryInfo:
    """Informações de um repositório GitHub"""
    nome: str
    nome_completo: str
    descricao: str
    url: str
    linguagem_principal: str
    estrelas: int
    forks: int
    tamanho_kb: int
    atualizado_em: str
    topicos: List[str]
    privado: bool

@dataclass
class FileAnalysis:
    """Análise detalhada de um arquivo"""
    name: str
    path: str
    language: str
    size: int
    lines: int
    functions: List[str]
    classes: List[str]
    imports: List[str]
    purpose: str
    summary: str
    complexity: str

class AnalysisRequest(BaseModel):
    """Requisição de análise"""
    model_config = {"protected_namespaces": ()}
    
    repo_url: str
    model_provider: str = "ollama"  # "openai" ou "ollama"
    model_name: str = "qwen2.5:7b"
    max_files: int = 50
    deep_analysis: bool = True
    anonymous: bool = True
    local_directory: Optional[str] = None  # Para análise de diretório local

class AnalysisStatus(BaseModel):
    """Status da análise"""
    status: str
    phase: str
    progress: int
    message: str
    logs: List[str] = []
    current_step: str = ""

class SearchRequest(BaseModel):
    """Requisição de busca de repositórios"""
    usuario: str
    incluir_forks: bool = False

class ModelConfig(BaseModel):
    """Configuração de modelos"""
    model_config = {"protected_namespaces": ()}
    
    provider: str  # "openai" ou "ollama"
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 8192

# =============================================================================
# SISTEMA DE ANONIMIZAÇÃO
# =============================================================================

class AnonymizationSystem:
    """Sistema para anonimizar informações pessoais"""
    
    def __init__(self):
        self.user_mapping = {}
        self.repo_mapping = {}
        self.counter = 1
        print("🔒 Sistema de anonimização inicializado")
    
    def anonymize_repo_url(self, url: str) -> str:
        """Anonimiza URL do repositório ou diretório local"""
        try:
            # Anonimizar diretório local
            if url.startswith("file://"):
                return "file:///projeto_local_anonimo"
            
            # Anonimizar URL GitHub
            match = re.search(r'github\.com/([^/]+)/([^/]+)', url)
            if match:
                user, repo = match.groups()
                
                if user not in self.user_mapping:
                    self.user_mapping[user] = f"usuario_anonimo_{self.counter}"
                    self.counter += 1
                
                if repo not in self.repo_mapping:
                    self.repo_mapping[repo] = f"projeto_anonimo_{len(self.repo_mapping) + 1}"
                
                anon_user = self.user_mapping[user]
                anon_repo = self.repo_mapping[repo]
                
                return f"https://github.com/{anon_user}/{anon_repo}"
            
            return "https://github.com/usuario_anonimo/projeto_anonimo"
            
        except Exception as e:
            logger.warning(f"Erro na anonimização: {e}")
            return "https://github.com/usuario_anonimo/projeto_anonimo"
    
    def anonymize_content(self, content: str, anonymous: bool = True) -> str:
        """Anonimiza conteúdo textual se necessário"""
        if not anonymous:
            return content
        
        try:
            # Anonimizar caminhos de arquivo
            content = re.sub(r'/home/[^/]+/', '/home/usuario_anonimo/', content)
            content = re.sub(r'/Users/[^/]+/', '/Users/usuario_anonimo/', content)
            content = re.sub(r'C:\\Users\\[^\\]+\\', 'C:\\Users\\usuario_anonimo\\', content)
            
            # Anonimizar nomes de usuário em caminhos
            content = re.sub(r'[/\\]([a-zA-Z][a-zA-Z0-9_-]{2,})[/\\]', r'/usuario_anonimo/', content)
            
            return content
        except Exception as e:
            logger.warning(f"Erro na anonimização de conteúdo: {e}")
            return content

# =============================================================================
# SISTEMA DE BUSCA GITHUB
# =============================================================================

class GitHubRepositoryFetcher:
    """Sistema para buscar repositórios do GitHub"""
    
    def __init__(self):
        self.session_cache = {}
        self.rate_limit_info = {}
        self.last_request_time = 0
        self.min_request_interval = 1.0
        print("🔍 Sistema de busca GitHub inicializado")
    
    def _rate_limit_wait(self):
        """Implementa rate limiting básico"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            wait_time = self.min_request_interval - time_since_last
            time.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    def search_repositories(self, user_or_org: str, include_forks: bool = False) -> List[RepositoryInfo]:
        """Busca repositórios de um usuário ou organização"""
        try:
            clean_user = self._extract_user_from_input(user_or_org)
            logger.info(f"Buscando repositórios de: {clean_user}")
            
            if not self._verify_user_exists(clean_user):
                logger.error(f"Usuário/organização não encontrado: {clean_user}")
                return []
            
            repositories = []
            page = 1
            
            while len(repositories) < 50 and page <= 5:
                repos_api_url = f"https://api.github.com/users/{clean_user}/repos?per_page=30&sort=updated&page={page}"
                
                try:
                    self._rate_limit_wait()
                    repos_data = self._make_github_request(repos_api_url)
                    
                    if not repos_data:
                        break
                    
                    for repo_data in repos_data:
                        if not include_forks and repo_data.get('fork', False):
                            continue
                        
                        if repo_data.get('private', False) and not os.environ.get('GITHUB_TOKEN'):
                            continue
                        
                        repo_info = self._process_repository_data(repo_data)
                        if repo_info:
                            repositories.append(repo_info)
                    
                    if len(repos_data) < 30:
                        break
                    
                    page += 1
                    
                except Exception as e:
                    logger.warning(f"Erro na página {page}: {e}")
                    break
            
            logger.info(f"Encontrados {len(repositories)} repositórios")
            return sorted(repositories, key=lambda x: x.estrelas, reverse=True)
            
        except Exception as e:
            logger.error(f"Erro ao buscar repositórios: {e}")
            return []
    
    def _extract_user_from_input(self, input_str: str) -> str:
        """Extrai nome de usuário de diferentes formatos"""
        input_str = input_str.strip()
        
        if 'github.com' in input_str:
            match = re.search(r'github\.com/([^/]+)', input_str)
            if match:
                return match.group(1)
        
        return re.sub(r'[^a-zA-Z0-9\-_]', '', input_str)
    
    def _verify_user_exists(self, user: str) -> bool:
        """Verifica se usuário existe"""
        try:
            import urllib.request
            url = f"https://api.github.com/users/{user}"
            self._rate_limit_wait()
            response = self._make_github_request(url)
            return response is not None
        except Exception as e:
            logger.warning(f"Erro ao verificar usuário: {e}")
            return False
    
    def _make_github_request(self, url: str) -> Optional[Dict]:
        """Faz requisição para API do GitHub"""
        try:
            import urllib.request
            import urllib.error
            
            request = urllib.request.Request(url)
            request.add_header('User-Agent', 'Mozilla/5.0 (compatible; DocAgent-LangGraph/3.0)')
            request.add_header('Accept', 'application/vnd.github.v3+json')
            
            github_token = os.environ.get('GITHUB_TOKEN')
            if github_token:
                request.add_header('Authorization', f'token {github_token}')
            
            with urllib.request.urlopen(request, timeout=30) as response:
                if response.getcode() == 200:
                    return json.loads(response.read().decode('utf-8'))
                else:
                    logger.warning(f"Resposta HTTP {response.getcode()}")
                    return None
                    
        except urllib.error.HTTPError as e:
            if e.code == 404:
                logger.error(f"Recurso não encontrado (404): {url}")
            elif e.code == 403:
                logger.error("403: Rate limit ou permissão insuficiente")
            elif e.code == 401:
                logger.error("401: Token inválido ou expirado")
            else:
                logger.error(f"Erro HTTP {e.code}: {e.reason}")
            return None
        except Exception as e:
            logger.error(f"Erro na requisição: {e}")
            return None
    
    def _process_repository_data(self, repo_data: Dict) -> Optional[RepositoryInfo]:
        """Processa dados do repositório"""
        try:
            return RepositoryInfo(
                nome=repo_data.get('name', ''),
                nome_completo=repo_data.get('full_name', ''),
                descricao=(repo_data.get('description') or 'Sem descrição')[:200],
                url=repo_data.get('html_url', ''),
                linguagem_principal=repo_data.get('language') or 'Desconhecida',
                estrelas=repo_data.get('stargazers_count', 0),
                forks=repo_data.get('forks_count', 0),
                tamanho_kb=repo_data.get('size', 0),
                atualizado_em=repo_data.get('updated_at', ''),
                topicos=repo_data.get('topics', []),
                privado=repo_data.get('private', False)
            )
        except Exception as e:
            logger.warning(f"Erro ao processar repositório: {e}")
            return None

# =============================================================================
# SISTEMA DE GERAÇÃO MERMAID NATIVO
# =============================================================================

class MermaidGenerator:
    """Gerador de diagramas Mermaid usando LangGraph nativo + Playwright"""
    
    def __init__(self, workflow_graph=None):
        self.workflow_graph = workflow_graph
        self.docs_dir = Path("docs")
        self.docs_dir.mkdir(exist_ok=True)
        logger.info("Gerador Mermaid inicializado")
    
    def generate_workflow_diagram(self) -> str:
        """Gera diagrama Mermaid do workflow LangGraph"""
        try:
            if not self.workflow_graph:
                return "Workflow não disponível"
            
            # Usar função nativa do LangGraph
            if MERMAID_NATIVE_AVAILABLE:
                try:
                    mermaid_code = self.workflow_graph.get_graph().draw_mermaid()
                    return mermaid_code
                except Exception as e:
                    logger.warning(f"Erro ao usar Mermaid nativo: {e}")
                    return self._generate_fallback_workflow()
            else:
                return self._generate_fallback_workflow()
                
        except Exception as e:
            logger.error(f"Erro na geração do diagrama workflow: {e}")
            return self._generate_fallback_workflow()
    
    def _generate_fallback_workflow(self) -> str:
        """Gera diagrama workflow de fallback"""
        return """
```mermaid
flowchart TD
    A[Início] --> B[Clone/Local Directory]
    B --> C[Analyze Structure]
    C --> D[Generate Plan]
    D --> E[Generate C4 Context]
    E --> F[Generate C4 Container]
    F --> G[Generate C4 Component]
    G --> H[Generate C4 Code]
    H --> I[Generate Detailed Analysis]
    I --> J[Generate Structure Report]
    J --> K[Generate Implementation Guide]
    K --> L[Generate Flowcharts]
    L --> M[Concluído]
    
    style A fill:#e1f5fe
    style M fill:#c8e6c9
    style E fill:#f3e5f5
    style F fill:#f3e5f5
    style G fill:#f3e5f5
    style H fill:#f3e5f5
    style I fill:#e8f5e8
    style J fill:#e8f5e8
    style K fill:#e8f5e8
    style L fill:#e8f5e8
```
"""
    
    def save_workflow_diagram(self, filename: str = "skyone_workflow_diagram.md") -> str:
        """Salva diagrama do workflow em arquivo"""
        try:
            mermaid_content = f"""# Skyone DocAgent - Workflow Diagram

## 🔄 Fluxo de Processamento LangGraph

Este diagrama mostra o fluxo completo de processamento do Skyone DocAgent.

{self.generate_workflow_diagram()}

## 📋 Descrição das Etapas

### Fase 1: Preparação
- **Clone/Local Directory**: Preparação do repositório ou diretório local
- **Analyze Structure**: Análise da estrutura de arquivos e código

### Fase 2: Planejamento
- **Generate Plan**: Criação do plano de documentação

### Fase 3: Documentação C4
- **Generate C4 Context**: Diagrama de contexto
- **Generate C4 Container**: Diagrama de contêineres
- **Generate C4 Component**: Diagrama de componentes
- **Generate C4 Code**: Análise de código C4

### Fase 4: Análise Detalhada
- **Generate Detailed Analysis**: Análise técnica profunda
- **Generate Structure Report**: Relatório estrutural
- **Generate Implementation Guide**: Guia de implementação
- **Generate Flowcharts**: Fluxogramas detalhados

---
*Gerado pelo Skyone DocAgent v3.0 • LangGraph Native*
"""
            
            file_path = self.docs_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(mermaid_content)
            
            logger.info(f"Diagrama workflow salvo: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Erro ao salvar diagrama workflow: {e}")
            return ""
    
    def generate_mermaid_png(self, mermaid_code: str, filename: str = None) -> str:
        """Gera PNG do diagrama Mermaid usando Playwright"""
        try:
            if not PLAYWRIGHT_AVAILABLE:
                logger.warning("Playwright não disponível para geração PNG")
                return ""
            
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"skyone_mermaid_{timestamp}.png"
            
            png_path = self.docs_dir / filename
            
            # HTML template para renderizar Mermaid
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <style>
        body {{ margin: 0; padding: 20px; background: white; }}
        .mermaid {{ text-align: center; }}
    </style>
</head>
<body>
    <div class="mermaid">
{mermaid_code}
    </div>
    <script>
        mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
    </script>
</body>
</html>
"""
            
            # Gerar PNG com Playwright
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.set_content(html_content)
                page.wait_for_timeout(2000)  # Aguardar renderização
                page.screenshot(path=str(png_path), full_page=True)
                browser.close()
            
            logger.info(f"PNG Mermaid gerado: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Erro ao gerar PNG Mermaid: {e}")
            return ""

# =============================================================================
# SISTEMA DE TOOLS PARA ANÁLISE DE REPOSITÓRIO
# =============================================================================

class RepositoryAnalysisTools:
    """Tools especializadas para análise de repositório"""
    
    def __init__(self, repo_path: Union[str, Path]):
        self.repo_path = Path(repo_path)
        self.file_cache = {}
        self.analysis_cache = {}
        logger.info(f"Inicializando tools de análise para: {self.repo_path}")
    
    def get_file_structure(self) -> Dict[str, Any]:
        """Analisa estrutura de arquivos do repositório"""
        try:
            structure = {
                "total_files": 0,
                "code_files": 0,
                "languages": {},
                "directories": [],
                "important_files": [],
                "config_files": []
            }
            
            code_extensions = {
                '.py': 'Python', '.js': 'JavaScript', '.ts': 'TypeScript',
                '.java': 'Java', '.cpp': 'C++', '.c': 'C', '.go': 'Go',
                '.rs': 'Rust', '.php': 'PHP', '.rb': 'Ruby', '.swift': 'Swift'
            }
            
            for root, dirs, files in os.walk(self.repo_path):
                # Filtrar diretórios irrelevantes
                dirs[:] = [d for d in dirs if not d.startswith('.') 
                          and d not in ['node_modules', '__pycache__', 'target', 'build', 'dist']]
                
                # Adicionar diretórios importantes
                for dir_name in dirs:
                    if dir_name in ['src', 'lib', 'app', 'components', 'services', 'models']:
                        structure["directories"].append(str(Path(root) / dir_name))
                
                for file in files:
                    if file.startswith('.'):
                        continue
                    
                    structure["total_files"] += 1
                    file_path = Path(root) / file
                    ext = file_path.suffix.lower()
                    
                    # Arquivos de código
                    if ext in code_extensions:
                        structure["code_files"] += 1
                        lang = code_extensions[ext]
                        structure["languages"][lang] = structure["languages"].get(lang, 0) + 1
                    
                    # Arquivos importantes
                    if (file.lower() in ['readme.md', 'package.json', 'requirements.txt', 'dockerfile'] or
                        'main' in file.lower() or 'index' in file.lower() or 'app' in file.lower()):
                        structure["important_files"].append(str(file_path.relative_to(self.repo_path)))
                    
                    # Arquivos de configuração
                    if ext in ['.json', '.yml', '.yaml', '.toml', '.ini', '.cfg']:
                        structure["config_files"].append(str(file_path.relative_to(self.repo_path)))
            
            return structure
            
        except Exception as e:
            logger.error(f"Erro na análise de estrutura: {e}")
            return {"error": str(e)}
    
    def analyze_code_files(self, max_files: int = 50) -> List[FileAnalysis]:
        """Analisa arquivos de código em detalhes"""
        try:
            analyses = []
            
            # Padrões de arquivos importantes (expandido)
            important_patterns = [
                'main.py', 'app.py', 'index.js', 'server.py', 'api.py',
                'models.py', 'views.py', 'controllers.py', 'routes.py',
                'utils.py', 'helpers.js', 'config.py', 'settings.py',
                'service.py', 'handler.py', 'manager.py', 'client.py',
                'database.py', 'db.py', 'auth.py', 'middleware.py',
                'component.js', 'module.py', 'processor.py', 'worker.py',
                'scheduler.py', 'task.py', 'job.py', 'queue.py'
            ]
            
            for root, dirs, files in os.walk(self.repo_path):
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                
                for file in files:
                    if len(analyses) >= max_files:
                        break
                    
                    file_path = Path(root) / file
                    
                    # Priorizar arquivos importantes
                    is_important = any(pattern in file.lower() for pattern in important_patterns)
                    is_code = file_path.suffix.lower() in ['.py', '.js', '.ts', '.java', '.go', '.rs']
                    
                    if not (is_important or is_code):
                        continue
                    
                    try:
                        if file_path.stat().st_size > 500 * 1024:  # Skip files > 500KB
                            continue
                        
                        content = file_path.read_text(encoding='utf-8', errors='ignore')
                        language = self._get_language(file_path.suffix.lower())
                        
                        analysis = self._analyze_file_content(file_path, content, language)
                        analyses.append(analysis)
                        
                    except Exception as e:
                        logger.warning(f"Erro ao analisar {file}: {e}")
                        continue
                
                if len(analyses) >= max_files:
                    break
            
            logger.info(f"Analisados {len(analyses)} arquivos de código")
            return analyses
            
        except Exception as e:
            logger.error(f"Erro na análise de arquivos: {e}")
            return []
    
    def _analyze_file_content(self, file_path: Path, content: str, language: str) -> FileAnalysis:
        """Analisa conteúdo de um arquivo"""
        try:
            lines = [line for line in content.split('\n') if line.strip()]
            
            # Extrair funções, classes e imports
            functions = self._extract_functions(content, language)
            classes = self._extract_classes(content, language)
            imports = self._extract_imports(content, language)
            
            # Determinar propósito
            purpose = self._determine_file_purpose(file_path, content, language)
            
            # Gerar resumo
            summary = self._generate_file_summary(file_path, content, functions, classes)
            
            # Calcular complexidade
            complexity = self._calculate_complexity(content, functions, classes)
            
            return FileAnalysis(
                name=file_path.name,
                path=str(file_path.relative_to(self.repo_path)),
                language=language,
                size=len(content.encode('utf-8')),
                lines=len(lines),
                functions=functions[:10],
                classes=classes[:10],
                imports=imports[:15],
                purpose=purpose,
                summary=summary,
                complexity=complexity
            )
            
        except Exception as e:
            logger.warning(f"Erro ao analisar arquivo {file_path}: {e}")
            return FileAnalysis(
                name=file_path.name,
                path=str(file_path),
                language=language,
                size=0,
                lines=0,
                functions=[],
                classes=[],
                imports=[],
                purpose="Arquivo de código",
                summary=f"Arquivo {language}",
                complexity="Baixa"
            )
    
    def _get_language(self, ext: str) -> str:
        """Identifica linguagem pela extensão"""
        language_map = {
            '.py': 'Python', '.js': 'JavaScript', '.ts': 'TypeScript',
            '.java': 'Java', '.cpp': 'C++', '.c': 'C', '.go': 'Go',
            '.rs': 'Rust', '.php': 'PHP', '.rb': 'Ruby', '.swift': 'Swift',
            '.html': 'HTML', '.css': 'CSS', '.sql': 'SQL', '.sh': 'Shell'
        }
        return language_map.get(ext, 'Unknown')
    
    def _extract_functions(self, content: str, language: str) -> List[str]:
        """Extrai funções do código com mais detalhes"""
        functions = []
        try:
            if language == 'Python':
                # Capturar funções normais e async
                pattern = r'(?:async\s+)?def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
                # Também capturar métodos de classe
                class_method_pattern = r'^\s+(?:async\s+)?def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
                
                matches = re.findall(pattern, content, re.MULTILINE)
                class_matches = re.findall(class_method_pattern, content, re.MULTILINE)
                functions.extend(matches)
                functions.extend([f"method_{m}" for m in class_matches])
                
            elif language in ['JavaScript', 'TypeScript']:
                # Funções normais, arrow functions, métodos
                patterns = [
                    r'function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
                    r'const\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*\(',
                    r'([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*function\s*\(',
                    r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*=>'
                ]
                for pattern in patterns:
                    matches = re.findall(pattern, content, re.MULTILINE)
                    functions.extend(matches)
                    
            elif language == 'Java':
                pattern = r'(?:public|private|protected)?\s*(?:static)?\s*\w+\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
                matches = re.findall(pattern, content, re.MULTILINE)
                functions.extend(matches)
            else:
                return []
            
            functions = list(set(functions)) if functions else []
        except Exception:
            pass
        
        return functions[:20]  # Limitar a 20 funções mais importantes
    
    def _extract_classes(self, content: str, language: str) -> List[str]:
        """Extrai classes do código"""
        classes = []
        try:
            if language in ['Python', 'Java', 'JavaScript', 'TypeScript']:
                pattern = r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)'
                matches = re.findall(pattern, content, re.MULTILINE)
                classes = list(set(matches)) if matches else []
        except Exception:
            pass
        
        return classes
    
    def _extract_imports(self, content: str, language: str) -> List[str]:
        """Extrai imports do código"""
        imports = []
        try:
            if language == 'Python':
                pattern = r'(?:from\s+[\w.]+\s+)?import\s+([\w\s,.*]+)'
            elif language in ['JavaScript', 'TypeScript']:
                pattern = r'import\s+.*?from\s+[\'"]([^\'"]+)[\'"]'
            elif language == 'Java':
                pattern = r'import\s+([\w.]+)'
            else:
                return []
            
            matches = re.findall(pattern, content, re.MULTILINE)
            for match in matches:
                if isinstance(match, str):
                    import_name = match.strip().split(',')[0].strip()
                    if import_name and import_name not in imports:
                        imports.append(import_name)
        except Exception:
            pass
        
        return imports
    
    def _determine_file_purpose(self, file_path: Path, content: str, language: str) -> str:
        """Determina o propósito do arquivo"""
        filename = file_path.name.lower()
        
        if 'test' in filename:
            return "Arquivo de testes"
        elif filename in ['main.py', 'app.py', 'index.js']:
            return "Ponto de entrada da aplicação"
        elif filename in ['config.py', 'settings.py']:
            return "Arquivo de configuração"
        elif 'api' in filename or 'controller' in filename:
            return "Controlador de API"
        elif 'model' in filename:
            return "Modelo de dados"
        elif 'view' in filename or 'component' in filename:
            return "Componente de interface"
        elif filename.endswith('.md'):
            return "Documentação"
        else:
            return f"Arquivo {language} do projeto"
    
    def _generate_file_summary(self, file_path: Path, content: str, functions: List[str], classes: List[str]) -> str:
        """Gera resumo do arquivo"""
        lines_count = len([l for l in content.split('\n') if l.strip()])
        summary_parts = [f"Arquivo com {lines_count} linhas"]
        
        if classes:
            summary_parts.append(f"{len(classes)} classe(s)")
        if functions:
            summary_parts.append(f"{len(functions)} função(ões)")
        
        return ", ".join(summary_parts)
    
    def _calculate_complexity(self, content: str, functions: List[str], classes: List[str]) -> str:
        """Calcula complexidade do arquivo"""
        lines = len([l for l in content.split('\n') if l.strip()])
        complexity_score = 0
        
        if lines > 200:
            complexity_score += 2
        elif lines > 50:
            complexity_score += 1
        
        complexity_score += len(functions) + len(classes)
        
        control_structures = len(re.findall(r'\b(if|for|while|try|catch)\b', content))
        if control_structures > 20:
            complexity_score += 2
        elif control_structures > 10:
            complexity_score += 1
        
        if complexity_score >= 5:
            return "Alta"
        elif complexity_score >= 3:
            return "Média"
        else:
            return "Baixa"

# =============================================================================
# TOOLS LANGGRAPH PARA ANÁLISE
# =============================================================================

# Definir tools apenas se LangGraph estiver disponível
if LANGGRAPH_AVAILABLE:
    @tool
    def analyze_repository_structure(repo_path: str) -> str:
        """Analisa a estrutura completa do repositório"""
        try:
            tools = RepositoryAnalysisTools(repo_path)
            structure = tools.get_file_structure()
            
            if "error" in structure:
                return f"Erro na análise: {structure['error']}"
            
            result = f"""## 📁 Estrutura do Repositório

### 📊 Estatísticas Gerais:
- **Total de arquivos:** {structure['total_files']:,}
- **Arquivos de código:** {structure['code_files']:,}
- **Linguagens detectadas:** {len(structure['languages'])}

### 💻 Distribuição por Linguagem:
"""
            for lang, count in sorted(structure['languages'].items(), key=lambda x: x[1], reverse=True):
                percentage = (count / structure['code_files']) * 100 if structure['code_files'] > 0 else 0
                result += f"- **{lang}:** {count} arquivos ({percentage:.1f}%)\n"
            
            result += "\n### 📁 Diretórios Importantes:\n"
            for directory in structure['directories'][:10]:
                result += f"- {directory}\n"
            
            result += "\n### 🎯 Arquivos Importantes:\n"
            for file in structure['important_files'][:15]:
                result += f"- {file}\n"
            
            return result
            
        except Exception as e:
            return f"Erro na análise de estrutura: {str(e)}"

    @tool
    def analyze_code_files(repo_path: str, max_files: int = 20) -> str:
        """Analisa arquivos de código em detalhes"""
        try:
            tools = RepositoryAnalysisTools(repo_path)
            analyses = tools.analyze_code_files(max_files)
            
            if not analyses:
                return "Nenhum arquivo de código encontrado para análise"
            
            result = f"## 🔬 Análise Detalhada de Arquivos ({len(analyses)} arquivos)\n\n"
            
            for i, analysis in enumerate(analyses[:15], 1):
                result += f"### {i}. {analysis.name} ({analysis.language})\n"
                result += f"**Localização:** `{analysis.path}`\n"
                result += f"**Tamanho:** {analysis.size:,} bytes | **Linhas:** {analysis.lines:,} | **Complexidade:** {analysis.complexity}\n"
                result += f"**Propósito:** {analysis.purpose}\n"
                result += f"**Resumo:** {analysis.summary}\n"
                
                if analysis.functions:
                    result += f"**Funções:** {', '.join(analysis.functions[:5])}"
                    if len(analysis.functions) > 5:
                        result += f" e mais {len(analysis.functions) - 5}"
                    result += "\n"
                
                if analysis.classes:
                    result += f"**Classes:** {', '.join(analysis.classes[:3])}"
                    if len(analysis.classes) > 3:
                        result += f" e mais {len(analysis.classes) - 3}"
                    result += "\n"
                
                if analysis.imports:
                    result += f"**Principais imports:** {', '.join(analysis.imports[:5])}\n"
                
                result += "\n---\n\n"
            
            return result
            
        except Exception as e:
            return f"Erro na análise de arquivos: {str(e)}"

    @tool
    def read_file_content(repo_path: str, file_path: str) -> str:
        """Lê conteúdo de um arquivo específico"""
        try:
            full_path = Path(repo_path) / file_path
            
            if not full_path.exists():
                return f"Arquivo não encontrado: {file_path}"
            
            if full_path.stat().st_size > 100 * 1024:  # 100KB limit
                return f"Arquivo muito grande: {file_path}"
            
            content = full_path.read_text(encoding='utf-8', errors='ignore')
            
            return f"""## 📄 Arquivo: {file_path}

**Tamanho:** {full_path.stat().st_size:,} bytes
**Linhas:** {len(content.split(chr(10)))}

### Conteúdo:
```{full_path.suffix[1:] if full_path.suffix else 'text'}
{content[:2000]}{'...' if len(content) > 2000 else ''}
```
"""
            
        except Exception as e:
            return f"Erro ao ler arquivo {file_path}: {str(e)}"

    @tool
    def find_dependencies(repo_path: str) -> str:
        """Encontra e analisa arquivos de dependências"""
        try:
            dependency_files = {
                'package.json': 'Node.js',
                'requirements.txt': 'Python',
                'Pipfile': 'Python (Pipenv)',
                'pyproject.toml': 'Python (Poetry)',
                'pom.xml': 'Java (Maven)',
                'build.gradle': 'Java (Gradle)',
                'Cargo.toml': 'Rust',
                'go.mod': 'Go',
                'composer.json': 'PHP'
            }
            
            found_deps = []
            repo_path = Path(repo_path)
            
            for dep_file, tech in dependency_files.items():
                file_path = repo_path / dep_file
                if file_path.exists():
                    try:
                        content = file_path.read_text(encoding='utf-8', errors='ignore')
                        size = file_path.stat().st_size
                        
                        found_deps.append({
                            'file': dep_file,
                            'technology': tech,
                            'size': size,
                            'content_preview': content[:500]
                        })
                    except Exception as e:
                        logger.warning(f"Erro ao ler {dep_file}: {e}")
            
            if not found_deps:
                return "Nenhum arquivo de dependências encontrado"
            
            result = "## 📦 Arquivos de Dependências Encontrados\n\n"
            
            for dep in found_deps:
                result += f"### {dep['file']} ({dep['technology']})\n"
                result += f"**Tamanho:** {dep['size']:,} bytes\n\n"
                
                if dep['file'] == 'package.json':
                    try:
                        pkg_data = json.loads(dep['content_preview'])
                        if 'dependencies' in pkg_data:
                            result += "**Dependências principais:**\n"
                            for pkg, version in list(pkg_data['dependencies'].items())[:10]:
                                result += f"- {pkg}: {version}\n"
                    except:
                        pass
                elif dep['file'] == 'requirements.txt':
                    lines = dep['content_preview'].split('\n')[:15]
                    result += "**Dependências:**\n"
                    for line in lines:
                        if line.strip() and not line.startswith('#'):
                            result += f"- {line.strip()}\n"
                
                result += "\n---\n\n"
            
            return result
            
        except Exception as e:
            return f"Erro na análise de dependências: {str(e)}"

else:
    # Definir tools vazias se LangGraph não estiver disponível
    def analyze_repository_structure(repo_path: str) -> str:
        return "LangGraph não disponível"
    
    def analyze_code_files(repo_path: str, max_files: int = 20) -> str:
        return "LangGraph não disponível"
    
    def read_file_content(repo_path: str, file_path: str) -> str:
        return "LangGraph não disponível"
    
    def find_dependencies(repo_path: str) -> str:
        return "LangGraph não disponível"

# =============================================================================
# SISTEMA DE MODELOS LLM
# =============================================================================

class LLMManager:
    """Gerenciador de modelos LLM (OpenAI e Ollama)"""
    
    def __init__(self):
        self.current_config = None
        self.available_models = {
            "openai": [
                "gpt-4o", 
                "gpt-4o-mini", 
                "gpt-4-turbo",
                "gpt-4",
                "gpt-3.5-turbo",
                "gpt-3.5-turbo-16k",
                "text-davinci-003",
                "text-davinci-002"
            ],
            "ollama": []
        }
        self._check_ollama_models()
        logger.info("LLM Manager inicializado")
    
    def _check_ollama_models(self):
        """Verifica modelos Ollama disponíveis"""
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                models = []
                for line in result.stdout.strip().split('\n')[1:]:
                    if line.strip():
                        parts = line.split()
                        if parts:
                            model_name = parts[0]
                            # Adicionar informações extras se disponíveis
                            size_info = parts[1] if len(parts) > 1 else ""
                            models.append({
                                "name": model_name,
                                "size": size_info,
                                "display_name": f"{model_name} ({size_info})" if size_info else model_name
                            })
                self.available_models["ollama"] = models
                logger.info(f"Modelos Ollama encontrados: {[m['name'] for m in models]}")
            else:
                logger.warning("Ollama não está rodando")
                self.available_models["ollama"] = []
        except Exception as e:
            logger.warning(f"Erro ao verificar Ollama: {e}")
            self.available_models["ollama"] = []
    
    def get_ollama_models_detailed(self):
        """Retorna modelos Ollama com detalhes"""
        self._check_ollama_models()
        return self.available_models["ollama"]
    
    def configure_model(self, config: ModelConfig):
        """Configura modelo LLM"""
        try:
            self.current_config = config
            
            if config.provider == "openai":
                if not config.api_key:
                    config.api_key = os.environ.get("OPENAI_API_KEY")
                
                if not config.api_key:
                    raise ValueError("OPENAI_API_KEY não configurada")
                
                self.llm = ChatOpenAI(
                    model=config.model_name,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    api_key=config.api_key
                )
                
            elif config.provider == "ollama":
                self.llm = Ollama(
                    model=config.model_name,
                    temperature=config.temperature,
                    base_url=config.base_url or "http://localhost:11434"
                )
            
            else:
                raise ValueError(f"Provider não suportado: {config.provider}")
            
            logger.info(f"Modelo configurado: {config.provider}/{config.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao configurar modelo: {e}")
            return False
    
    def get_llm(self):
        """Retorna instância do LLM configurado"""
        if not hasattr(self, 'llm'):
            # Configuração padrão
            default_config = ModelConfig(
                provider="ollama",
                model_name="qwen2.5:7b"
            )
            self.configure_model(default_config)
        
        return self.llm
    
    def list_available_models(self) -> Dict[str, List[str]]:
        """Lista modelos disponíveis"""
        self._check_ollama_models()  # Atualizar lista Ollama
        return self.available_models.copy()

# =============================================================================
# NÓS DO LANGGRAPH - AGENTES ESPECIALIZADOS
# =============================================================================

class DocumentationAgents:
    """Agentes especializados para documentação usando LangGraph"""
    
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        self.anonymizer = AnonymizationSystem()
        logger.info("Agentes de documentação inicializados")
    
    async def clone_repository_node(self, state: DocumentationState) -> DocumentationState:
        """Nó para clonar repositório ou usar diretório local"""
        try:
            repo_url = state["repo_url"]
            logger.info(f"Processando: {repo_url}")
            
            # Verificar se é um diretório local
            if repo_url.startswith("file://"):
                local_path = repo_url.replace("file://", "")
                logger.info(f"Usando diretório local: {local_path}")
                
                if not Path(local_path).exists():
                    state["logs"].append(f"❌ Diretório não encontrado: {local_path}")
                    state["error_count"] += 1
                    return state
                
                if not Path(local_path).is_dir():
                    state["logs"].append(f"❌ Caminho não é um diretório: {local_path}")
                    state["error_count"] += 1
                    return state
                
                state["repo_path"] = local_path
                state["current_phase"] = "local_ready"
                state["progress"] = 20
                state["logs"].append("✅ Diretório local configurado")
                logger.info(f"Diretório local pronto: {local_path}")
                return state
            
            # Validar URL GitHub
            if not self._validate_github_url(repo_url):
                state["logs"].append("❌ URL inválida")
                state["error_count"] += 1
                return state
            
            # Preparar diretório para clone
            repo_name = repo_url.split("/")[-1].replace(".git", "")
            workdir = Path("workdir").resolve()
            workdir.mkdir(exist_ok=True)
            repo_path = workdir / repo_name
            
            # Remover se existir
            if repo_path.exists():
                shutil.rmtree(repo_path, ignore_errors=True)
            
            # Clone
            cmd = ["git", "clone", "--depth", "1", repo_url, str(repo_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0 and repo_path.exists():
                state["repo_path"] = str(repo_path)
                state["current_phase"] = "cloned"
                state["progress"] = 20
                state["logs"].append("✅ Repositório clonado com sucesso")
                logger.info(f"Clone concluído: {repo_path}")
            else:
                state["logs"].append(f"❌ Falha no clone: {result.stderr}")
                state["error_count"] += 1
            
            return state
            
        except Exception as e:
            logger.error(f"Erro no processamento: {e}")
            state["logs"].append(f"❌ Erro no processamento: {str(e)}")
            state["error_count"] += 1
            return state
    
    async def analyze_structure_node(self, state: DocumentationState) -> DocumentationState:
        """Nó para análise de estrutura"""
        try:
            repo_path = state.get("repo_path")
            if not repo_path:
                state["logs"].append("❌ Caminho do repositório não encontrado")
                state["error_count"] += 1
                return state
            
            logger.info("Analisando estrutura do repositório")
            
            # Usar tool de análise
            structure_result = analyze_repository_structure.invoke({"repo_path": repo_path})
            
            # Analisar arquivos de código
            code_result = analyze_code_files.invoke({
                "repo_path": repo_path, 
                "max_files": state.get("max_files", 20)
            })
            
            # Encontrar dependências
            deps_result = find_dependencies.invoke({"repo_path": repo_path})
            
            state["file_structure"] = {
                "structure_analysis": structure_result,
                "code_analysis": code_result,
                "dependencies": deps_result
            }
            
            state["current_phase"] = "analyzed"
            state["progress"] = 40
            state["logs"].append("✅ Estrutura analisada")
            
            return state
            
        except Exception as e:
            logger.error(f"Erro na análise de estrutura: {e}")
            state["logs"].append(f"❌ Erro na análise: {str(e)}")
            state["error_count"] += 1
            return state
    
    async def generate_documentation_plan_node(self, state: DocumentationState) -> DocumentationState:
        """Nó para gerar plano de documentação"""
        try:
            logger.info("Gerando plano de documentação")
            
            llm = self.llm_manager.get_llm()
            
            # Preparar contexto da análise
            structure_info = state.get("file_structure", {})
            
            # Prompt para planejamento com C4 e análise detalhada
            planning_prompt = ChatPromptTemplate.from_template("""
Você é um especialista em documentação técnica e arquitetura C4. Baseado na análise do repositório abaixo, 
crie um plano detalhado para documentação completa com arquitetura C4 e análise detalhada em formato JSON.

ANÁLISE DO REPOSITÓRIO:
{structure_analysis}

{code_analysis}

{dependencies}

Crie um plano JSON com 8 seções seguindo o modelo C4 + Análise Detalhada:
1. "C4 Context Diagram" - visão geral do sistema e interações externas
2. "C4 Container Diagram" - contêineres e tecnologias principais
3. "C4 Component Diagram" - componentes internos e suas responsabilidades
4. "C4 Code Analysis" - análise detalhada do código e estrutura
5. "Detailed Code Analysis" - análise técnica profunda linha por linha
6. "Code Structure Report" - relatório estrutural detalhado
7. "Technical Implementation Guide" - guia técnico de implementação
8. "Mermaid Flowcharts" - fluxogramas detalhados dos componentes

Formato JSON obrigatório:
{{
  "overview": "Documentação completa C4 + Análise Detalhada do projeto Skyone",
  "sections": [
    {{
      "title": "C4 Context Diagram",
      "description": "Visão contextual do sistema e suas interações externas",
      "content_type": "c4_context"
    }},
    {{
      "title": "C4 Container Diagram", 
      "description": "Contêineres, tecnologias e comunicação entre componentes",
      "content_type": "c4_container"
    }},
    {{
      "title": "C4 Component Diagram",
      "description": "Componentes internos, interfaces e responsabilidades",
      "content_type": "c4_component"
    }},
    {{
      "title": "C4 Code Analysis",
      "description": "Análise detalhada do código, classes e implementação",
      "content_type": "c4_code"
    }},
    {{
      "title": "Detailed Code Analysis",
      "description": "Análise técnica profunda linha por linha do código",
      "content_type": "detailed_code_analysis"
    }},
    {{
      "title": "Code Structure Report",
      "description": "Relatório estrutural detalhado dos arquivos e módulos",
      "content_type": "code_structure_report"
    }},
    {{
      "title": "Technical Implementation Guide",
      "description": "Guia técnico de implementação e padrões utilizados",
      "content_type": "technical_implementation"
    }},
    {{
      "title": "Mermaid Flowcharts",
      "description": "Fluxogramas detalhados dos componentes e processos",
      "content_type": "mermaid_flowcharts"
    }}
  ]
}}

Responda APENAS com o JSON válido.
""")
            
            # Executar planejamento
            chain = planning_prompt | llm
            
            planning_result = await chain.ainvoke({
                "structure_analysis": structure_info.get("structure_analysis", ""),
                "code_analysis": structure_info.get("code_analysis", ""),
                "dependencies": structure_info.get("dependencies", "")
            })
            
            # Extrair JSON do resultado
            try:
                if hasattr(planning_result, 'content'):
                    plan_text = planning_result.content
                else:
                    plan_text = str(planning_result)
                
                # Extrair JSON
                json_match = re.search(r'\{.*\}', plan_text, re.DOTALL)
                if json_match:
                    plan_data = json.loads(json_match.group())
                    state["documentation_plan"] = plan_data
                    state["logs"].append("✅ Plano de documentação criado")
                else:
                    raise ValueError("JSON não encontrado na resposta")
                    
            except Exception as e:
                logger.warning(f"Erro ao extrair plano JSON: {e}")
                # Plano padrão C4 com análise detalhada de código
                state["documentation_plan"] = {
                    "overview": "Documentação completa C4 + Análise Detalhada do projeto Skyone",
                    "sections": [
                        {"title": "C4 Context Diagram", "description": "Visão contextual do sistema", "content_type": "c4_context"},
                        {"title": "C4 Container Diagram", "description": "Contêineres e tecnologias", "content_type": "c4_container"},
                        {"title": "C4 Component Diagram", "description": "Componentes internos", "content_type": "c4_component"},
                        {"title": "C4 Code Analysis", "description": "Análise detalhada do código", "content_type": "c4_code"},
                        {"title": "Detailed Code Analysis", "description": "Análise técnica profunda do código", "content_type": "detailed_code_analysis"},
                        {"title": "Code Structure Report", "description": "Relatório estrutural detalhado", "content_type": "code_structure_report"},
                        {"title": "Technical Implementation Guide", "description": "Guia técnico de implementação", "content_type": "technical_implementation"},
                        {"title": "Mermaid Flowcharts", "description": "Fluxogramas detalhados dos componentes", "content_type": "mermaid_flowcharts"}
                    ]
                }
                state["logs"].append("⚠️ Usando plano padrão")
            
            state["current_phase"] = "planned"
            state["progress"] = 60
            
            return state
            
        except Exception as e:
            logger.error(f"Erro no planejamento: {e}")
            state["logs"].append(f"❌ Erro no planejamento: {str(e)}")
            state["error_count"] += 1
            return state
    
    async def generate_documentation_node(self, state: DocumentationState) -> DocumentationState:
        """Nó para gerar documentação"""
        try:
            logger.info("Gerando documentação")
            
            plan = state.get("documentation_plan", {})
            sections = plan.get("sections", [])
            
            if not sections:
                state["logs"].append("❌ Plano de documentação não encontrado")
                state["error_count"] += 1
                return state
            
            llm = self.llm_manager.get_llm()
            docs_dir = Path("docs")
            docs_dir.mkdir(exist_ok=True)
            
            generated_files = []
            
            for i, section in enumerate(sections):
                try:
                    logger.info(f"Gerando seção: {section['title']}")
                    
                    # Prompt específico por tipo de seção
                    doc_content = await self._generate_section_content(
                        llm, section, state, i + 1
                    )
                    
                    # Salvar arquivo
                    filename = self._get_section_filename(section['title'], i, state.get("anonymous", True))
                    file_path = docs_dir / filename
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(doc_content)
                    
                    generated_files.append(filename)
                    state["logs"].append(f"✅ Seção criada: {section['title']}")
                    
                except Exception as e:
                    logger.error(f"Erro na seção {section['title']}: {e}")
                    state["logs"].append(f"❌ Erro na seção {section['title']}: {str(e)}")
            
            # Gerar diagrama de workflow usando LangGraph nativo
            try:
                # Criar gerador Mermaid para esta instância
                temp_generator = MermaidGenerator()
                workflow_file = temp_generator.save_workflow_diagram()
                if workflow_file:
                    generated_files.append(workflow_file)
                    state["logs"].append("✅ Diagrama de workflow LangGraph criado")
            except Exception as e:
                logger.warning(f"Erro ao gerar diagrama de workflow: {e}")
            
            state["generated_docs"] = generated_files
            state["current_phase"] = "completed"
            state["progress"] = 100
            state["logs"].append(f"🎉 Documentação completa: {len(generated_files)} arquivos")
            
            return state
            
        except Exception as e:
            logger.error(f"Erro na geração de documentação: {e}")
            state["logs"].append(f"❌ Erro na documentação: {str(e)}")
            state["error_count"] += 1
            return state
    
    def _validate_github_url(self, url: str) -> bool:
        """Valida URL do GitHub"""
        pattern = r"^https://github\.com/[\w\-\.]+/[\w\-\.]+/?$"
        return bool(re.match(pattern, url.strip()))
    
    def _limit_context_size(self, structure_info: Dict[str, Any], max_chars: int = 8000) -> Dict[str, Any]:
        """Limita o tamanho do contexto para evitar 'Chunk too big'"""
        limited_info = {}
        
        for key, value in structure_info.items():
            if isinstance(value, str):
                if len(value) > max_chars:
                    # Truncar mantendo informações importantes
                    limited_info[key] = value[:max_chars] + "\n\n[... Análise truncada para evitar limite de tokens ...]"
                else:
                    limited_info[key] = value
            else:
                limited_info[key] = value
        
        return limited_info
    
    async def _generate_section_content(self, llm, section: Dict, state: DocumentationState, section_num: int) -> str:
        """Gera conteúdo de uma seção específica"""
        try:
            content_type = section.get("content_type", "general")
            repo_url = state["repo_url"]
            anonymous = state.get("anonymous", True)
            
            # Anonimizar URL se necessário
            final_url = self.anonymizer.anonymize_repo_url(repo_url) if anonymous else repo_url
            
            # Limitar tamanho do contexto para evitar "Chunk too big"
            structure_info = state.get("file_structure", {})
            limited_structure_info = self._limit_context_size(structure_info)
            
            if content_type == "c4_context":
                prompt = self._create_c4_context_prompt(section, limited_structure_info, final_url)
            elif content_type == "c4_container":
                prompt = self._create_c4_container_prompt(section, limited_structure_info, final_url)
            elif content_type == "c4_component":
                prompt = self._create_c4_component_prompt(section, limited_structure_info, final_url)
            elif content_type == "c4_code":
                prompt = self._create_c4_code_prompt(section, limited_structure_info, final_url)
            elif content_type == "mermaid_flowcharts":
                prompt = self._create_mermaid_flowcharts_prompt(section, limited_structure_info, final_url)
            elif content_type == "detailed_code_analysis":
                prompt = self._create_detailed_code_analysis_prompt(section, limited_structure_info, final_url)
            elif content_type == "code_structure_report":
                prompt = self._create_code_structure_report_prompt(section, limited_structure_info, final_url)
            elif content_type == "technical_implementation":
                prompt = self._create_technical_implementation_prompt(section, limited_structure_info, final_url)
            elif content_type == "overview":
                prompt = self._create_overview_prompt(section, state, final_url)
            elif content_type == "installation":
                prompt = self._create_installation_prompt(section, state, final_url)
            elif content_type == "technical":
                prompt = self._create_technical_prompt(section, state, final_url)
            else:
                prompt = self._create_general_prompt(section, state, final_url)
            
            # Gerar conteúdo
            result = await llm.ainvoke([HumanMessage(content=prompt)])
            
            if hasattr(result, 'content'):
                content = result.content
            else:
                content = str(result)
            
            # Aplicar anonimização no conteúdo se necessário
            anonymous = state.get("anonymous", True)
            content = self.anonymizer.anonymize_content(content, anonymous)
            
            return content
                
        except Exception as e:
            logger.error(f"Erro ao gerar seção {section.get('title', 'unknown')}: {e}")
            return self._create_fallback_content(section, state)
    
    def _create_overview_prompt(self, section: Dict, state: DocumentationState, final_url: str) -> str:
        """Cria prompt para visão geral"""
        structure_info = state.get("file_structure", {})
        
        return f"""
Crie uma documentação completa de VISÃO GERAL DO PROJETO baseada na análise real:

TÍTULO: {section['title']}
REPOSITÓRIO: {final_url}

ANÁLISE ESTRUTURAL:
{structure_info.get('structure_analysis', 'Análise não disponível')}

ANÁLISE DE CÓDIGO:
{structure_info.get('code_analysis', 'Análise de código não disponível')}

DEPENDÊNCIAS:
{structure_info.get('dependencies', 'Dependências não identificadas')}

Crie documentação em Markdown com:
# {section['title']}

## 🎯 Propósito do Projeto
[Baseado na análise real dos arquivos]

## 🛠️ Stack Tecnológico
[Tecnologias identificadas na análise]

## 🏗️ Arquitetura
[Estrutura e padrões identificados]

## 📊 Estatísticas
[Dados quantitativos da análise]

Use APENAS informações da análise real fornecida.
"""
    
    def _create_installation_prompt(self, section: Dict, state: DocumentationState, final_url: str) -> str:
        """Cria prompt para instalação"""
        structure_info = state.get("file_structure", {})
        
        return f"""
Crie um GUIA DE INSTALAÇÃO baseado na análise das dependências:

TÍTULO: {section['title']}

DEPENDÊNCIAS ENCONTRADAS:
{structure_info.get('dependencies', 'Nenhuma dependência identificada')}

ESTRUTURA DO PROJETO:
{structure_info.get('structure_analysis', 'Estrutura não analisada')}

Crie documentação em Markdown com:
# {section['title']}

## 📋 Pré-requisitos
[Baseado nas tecnologias identificadas]

## 🚀 Instalação
[Passos baseados nos arquivos de dependência encontrados]

## ⚙️ Configuração
[Configurações necessárias]

## ▶️ Execução
[Como executar o projeto]

Use APENAS informações das dependências e estrutura analisadas.
"""
    
    def _create_technical_prompt(self, section: Dict, state: DocumentationState, final_url: str) -> str:
        """Cria prompt para relatório técnico"""
        structure_info = state.get("file_structure", {})
        
        return f"""
Crie um RELATÓRIO TÉCNICO DETALHADO baseado na análise de código:

TÍTULO: {section['title']}

ANÁLISE DE ARQUIVOS:
{structure_info.get('code_analysis', 'Análise de código não disponível')}

ESTRUTURA:
{structure_info.get('structure_analysis', 'Estrutura não analisada')}

Crie documentação técnica em Markdown com:
# {section['title']}

## 📁 Estrutura do Projeto
[Organização identificada]

## 🔧 Arquivos Principais
[Para cada arquivo analisado:]
### [Nome do Arquivo]
- **Propósito:** [identificado na análise]
- **Linguagem:** [detectada]
- **Funções:** [listadas na análise]
- **Classes:** [encontradas]
- **Complexidade:** [calculada]

## 🏗️ Arquitetura
[Padrões identificados]

Use APENAS dados da análise real dos arquivos.
"""
    
    def _create_c4_context_prompt(self, section: Dict, structure_info: Dict[str, Any], final_url: str) -> str:
        """Cria prompt para C4 Context Diagram"""
        
        return f"""
Crie uma documentação C4 CONTEXT DIAGRAM baseada na análise real:

TÍTULO: {section['title']}
PROJETO: {final_url}

ANÁLISE ESTRUTURAL:
{structure_info.get('structure_analysis', 'Análise não disponível')}

DEPENDÊNCIAS:
{structure_info.get('dependencies', 'Dependências não identificadas')}

Crie documentação em Markdown seguindo o modelo C4 Context:

# {section['title']}

## 🎯 Visão Contextual do Sistema

### Sistema Principal
[Nome e propósito do sistema baseado na análise]

### Usuários e Atores
[Identifique os tipos de usuários que interagem com o sistema]

### Sistemas Externos
[Sistemas, APIs e serviços externos identificados nas dependências]

### Interações Principais
[Como o sistema se comunica com o mundo externo]

## 📊 Diagrama de Contexto C4

```mermaid
C4Context
    title Diagrama de Contexto - [Nome do Sistema]
    
    Person(user, "Usuário", "Descrição do usuário principal")
    System(system, "[Nome do Sistema]", "Descrição do sistema")
    System_Ext(external, "Sistema Externo", "Descrição")
    
    Rel(user, system, "Usa")
    Rel(system, external, "Consome API")
```

## 🔗 Integrações Identificadas
[Liste as integrações encontradas na análise]

Use APENAS informações da análise real fornecida.
"""

    def _create_c4_container_prompt(self, section: Dict, structure_info: Dict[str, Any], final_url: str) -> str:
        """Cria prompt para C4 Container Diagram"""
        
        return f"""
Você é um especialista em arquitetura C4. Crie uma documentação C4 CONTAINER DIAGRAM baseada EXCLUSIVAMENTE na análise real:

TÍTULO: {section['title']}

ANÁLISE DE CÓDIGO:
{structure_info.get('code_analysis', 'Análise de código não disponível')}

DEPENDÊNCIAS E TECNOLOGIAS REAIS:
{structure_info.get('dependencies', 'Dependências não identificadas')}

ESTRUTURA REAL:
{structure_info.get('structure_analysis', 'Estrutura não analisada')}

INSTRUÇÕES CRÍTICAS:
1. Identifique APENAS tecnologias REAIS encontradas nas dependências
2. Use APENAS linguagens de programação REAIS detectadas na análise
3. Identifique contêineres REAIS baseados na estrutura de diretórios
4. NÃO invente bancos de dados ou APIs se não foram identificados
5. Base-se APENAS nos arquivos e tecnologias encontrados

# {section['title']}

## 🏗️ Arquitetura de Contêineres Reais

### Contêineres Identificados na Análise
[Baseado na estrutura de diretórios REAL: frontend/, backend/, api/, etc.]

### Stack Tecnológico Real
[APENAS as tecnologias REAIS das dependências:]
- **Linguagens:** [Linguagens REAIS detectadas]
- **Frameworks:** [Frameworks REAIS dos arquivos de dependência]
- **Bibliotecas:** [Bibliotecas REAIS identificadas]

### Comunicação e Protocolos
[Baseado nos imports e configurações REAIS encontrados]

## 📦 Diagrama de Contêineres C4 (Dados Reais)

```mermaid
C4Container
    title Diagrama de Contêineres - [Nome Real do Projeto]
    
    Person(user, "Usuário", "Usuário do sistema")
    
    Container_Boundary(system, "[Nome Real do Sistema]") {{
        [Para cada contêiner REAL identificado na estrutura:]
        [Exemplo: Container(app_real, "NomeAplicacaoReal", "TecnologiaReal", "Função real")]
        [APENAS se houver banco de dados identificado: ContainerDb(db_real, "BancoDadosReal", "TipoReal", "Função real")]
    }}
    
    [APENAS se houver APIs externas REAIS nas dependências:]
    [System_Ext(api_real, "NomeAPIReal", "Função real")]
    
    [Relacionamentos REAIS baseados na análise:]
    [Rel(user, app_real, "Interage", "ProtocoloReal")]
    [Rel(app_real, db_real, "Acessa", "ProtocoloReal") - APENAS se DB foi identificado]
```

## 🔧 Detalhes Técnicos dos Contêineres Reais
[Para cada contêiner REAL identificado:]

### [Nome Real do Contêiner]
- **Tecnologia:** [Tecnologia REAL detectada]
- **Localização:** [Diretório REAL na estrutura]
- **Responsabilidades:** [Baseadas nos arquivos REAIS encontrados]
- **Dependências:** [Dependências REAIS identificadas]
- **Configuração:** [Arquivos de config REAIS encontrados]

## 🌐 Integrações Externas Reais
[APENAS se identificadas nas dependências ou imports:]

IMPORTANTE: Use SOMENTE dados REAIS da análise. NÃO invente contêineres, bancos de dados ou APIs.
"""

    def _create_c4_component_prompt(self, section: Dict, structure_info: Dict[str, Any], final_url: str) -> str:
        """Cria prompt para C4 Component Diagram"""
        
        return f"""
Você é um especialista em arquitetura de software e documentação C4. Crie uma documentação C4 COMPONENT DIAGRAM baseada EXCLUSIVAMENTE na análise detalhada real fornecida.

TÍTULO: {section['title']}

ANÁLISE DETALHADA DE ARQUIVOS:
{structure_info.get('code_analysis', 'Análise de código não disponível')}

ESTRUTURA DO PROJETO:
{structure_info.get('structure_analysis', 'Estrutura não analisada')}

DEPENDÊNCIAS IDENTIFICADAS:
{structure_info.get('dependencies', 'Dependências não identificadas')}

INSTRUÇÕES CRÍTICAS:
1. Use APENAS os arquivos, classes, funções e componentes REAIS identificados na análise
2. NÃO invente ou use componentes genéricos como "Controller", "Service", "Repository"
3. Use os NOMES REAIS dos arquivos e classes encontrados na análise
4. Base as responsabilidades nas FUNÇÕES REAIS identificadas
5. Use as TECNOLOGIAS REAIS encontradas nas dependências

Crie documentação em Markdown seguindo o modelo C4 Component:

# {section['title']}

## 🧩 Componentes Reais Identificados

### Arquivos e Módulos Principais
[Liste APENAS os arquivos reais da análise com suas funções específicas]

### Classes e Interfaces Reais
[Liste APENAS as classes reais encontradas na análise de código]

### Funções e Métodos Principais
[Liste APENAS as funções reais identificadas na análise]

## 🔗 Diagrama de Componentes C4 (Baseado na Análise Real)

```mermaid
C4Component
    title Diagrama de Componentes - [Nome Real do Sistema]
    
    Container_Boundary(main_container, "[Nome Real da Aplicação]") {{
        [Para cada arquivo/classe REAL da análise, crie um Component com nome, tecnologia e propósito REAIS]
        [Exemplo: Component(arquivo_real, "NomeArquivoReal.py", "Python", "Função real identificada")]
    }}
    
    [Adicione sistemas externos REAIS encontrados nas dependências]
    [Adicione bancos de dados REAIS se identificados]
    
    [Crie relacionamentos REAIS baseados nos imports e dependências da análise]
```

## 📋 Detalhes dos Componentes Reais
[Para cada arquivo/classe REAL da análise:]

### [Nome Real do Arquivo/Classe]
- **Localização:** [Caminho real do arquivo]
- **Linguagem:** [Linguagem real detectada]
- **Propósito:** [Propósito real identificado na análise]
- **Funções Principais:** [Funções reais listadas]
- **Dependências:** [Imports reais identificados]
- **Complexidade:** [Complexidade real calculada]

## 🔄 Fluxo de Dados Real
[Baseado nos imports e dependências REAIS, descreva como os dados fluem entre os componentes REAIS]

## 🏗️ Padrões Arquiteturais Identificados
[Identifique padrões REAIS baseados na estrutura e organização dos arquivos analisados]

IMPORTANTE: Use SOMENTE informações REAIS da análise fornecida. NÃO invente componentes genéricos.
"""

    def _create_c4_code_prompt(self, section: Dict, structure_info: Dict[str, Any], final_url: str) -> str:
        """Cria prompt para C4 Code Analysis"""
        
        return f"""
Crie uma documentação C4 CODE ANALYSIS baseada na análise detalhada do código:

TÍTULO: {section['title']}

ANÁLISE COMPLETA DOS ARQUIVOS:
{structure_info.get('code_analysis', 'Análise de código não disponível')}

ESTRUTURA DETALHADA:
{structure_info.get('structure_analysis', 'Estrutura não analisada')}

Crie documentação em Markdown seguindo o modelo C4 Code:

# {section['title']}

## 💻 Análise Detalhada do Código

### Estrutura de Classes e Funções
[Baseado na análise real dos arquivos]

### Padrões de Código Identificados
[Padrões arquiteturais encontrados na análise]

### Dependências Internas
[Como as classes e módulos se relacionam]

## 🏗️ Estrutura de Código

### Arquivos Principais Analisados
[Para cada arquivo analisado:]

#### [Nome do Arquivo]
- **Linguagem:** [Detectada na análise]
- **Propósito:** [Identificado na análise]
- **Classes:** [Listadas na análise]
- **Funções:** [Listadas na análise]
- **Complexidade:** [Calculada na análise]
- **Imports:** [Dependências identificadas]

## 🔍 Métricas de Código
[Estatísticas extraídas da análise]

## 🏛️ Arquitetura do Código
[Padrões arquiteturais identificados]

## 📈 Qualidade e Complexidade
[Avaliação baseada na análise realizada]

## 🔗 Diagrama de Classes (se aplicável)

```mermaid
classDiagram
    [Baseado nas classes identificadas na análise]
```

Use APENAS dados reais da análise dos arquivos fornecida.
"""

    def _create_mermaid_flowcharts_prompt(self, section: Dict, structure_info: Dict[str, Any], final_url: str) -> str:
        """Cria prompt para fluxogramas Mermaid detalhados"""
        
        return f"""
Você é um especialista em fluxogramas e diagramas Mermaid. Crie fluxogramas detalhados baseados EXCLUSIVAMENTE na análise real do código.

TÍTULO: {section['title']}

ANÁLISE COMPLETA DOS ARQUIVOS:
{structure_info.get('code_analysis', 'Análise de código não disponível')}

ESTRUTURA DETALHADA:
{structure_info.get('structure_analysis', 'Estrutura não analisada')}

DEPENDÊNCIAS IDENTIFICADAS:
{structure_info.get('dependencies', 'Dependências não identificadas')}

INSTRUÇÕES CRÍTICAS:
1. Use APENAS arquivos, funções e fluxos REAIS identificados na análise
2. Crie fluxogramas para os processos REAIS encontrados no código
3. Use nomes REAIS das funções e classes
4. Base os fluxos nas chamadas de função REAIS identificadas
5. NÃO invente processos genéricos

# {section['title']}

## 🔄 Fluxogramas dos Componentes Reais

### Fluxograma Principal do Sistema
[Baseado no arquivo principal identificado na análise]

```mermaid
flowchart TD
    [Para cada função/processo REAL identificado, crie um nó]
    [Exemplo: A[FuncaoRealPrincipal] --> B[FuncaoRealSecundaria]]
    [Use nomes REAIS das funções da análise]
    
    [Conecte baseado nas chamadas de função REAIS identificadas]
    [Adicione decisões baseadas em condicionais REAIS do código]
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
```

### Fluxograma de Processamento de Dados
[APENAS se identificado processamento de dados na análise]

```mermaid
flowchart LR
    [Baseado nos fluxos REAIS de dados identificados]
    [Use funções REAIS que manipulam dados]
    
    subgraph "Módulo Real"
        [FuncaoRealEntrada] --> [FuncaoRealProcessamento]
        [FuncaoRealProcessamento] --> [FuncaoRealSaida]
    end
```

### Fluxograma de Interação entre Módulos
[Baseado nos imports REAIS identificados]

```mermaid
flowchart TB
    [Para cada arquivo/módulo REAL:]
    
    subgraph "[NomeModuloReal1]"
        [FuncaoReal1]
        [FuncaoReal2]
    end
    
    subgraph "[NomeModuloReal2]"
        [FuncaoReal3]
        [FuncaoReal4]
    end
    
    [Conecte baseado nos imports REAIS]
    [FuncaoReal1] --> [FuncaoReal3]
    [FuncaoReal2] --> [FuncaoReal4]
```

### Fluxograma de Tratamento de Erros
[APENAS se identificado tratamento de erro na análise]

```mermaid
flowchart TD
    [Baseado em try/catch ou tratamento de erro REAL encontrado]
```

### Fluxograma de Configuração e Inicialização
[APENAS se identificados arquivos de config na análise]

```mermaid
flowchart TD
    [Baseado nos arquivos de configuração REAIS encontrados]
    [Use funções REAIS de inicialização identificadas]
```

## 📋 Descrição dos Fluxogramas

### [Nome do Fluxograma Real]
- **Baseado em:** [Arquivo/função REAL da análise]
- **Entrada:** [Parâmetros REAIS identificados]
- **Processamento:** [Lógica REAL encontrada no código]
- **Saída:** [Retorno REAL da função]
- **Dependências:** [Chamadas REAIS para outras funções]

## 🔧 Detalhes de Implementação
[Para cada fluxo REAL identificado:]

### Processo: [Nome Real do Processo]
- **Arquivo:** [Localização REAL]
- **Função Principal:** [Nome REAL da função]
- **Complexidade:** [Complexidade REAL calculada]
- **Chamadas:** [Funções REAIS que chama]

IMPORTANTE: Crie APENAS fluxogramas baseados em código REAL analisado. NÃO invente processos genéricos.
"""

    def _create_detailed_code_analysis_prompt(self, section: Dict, structure_info: Dict[str, Any], final_url: str) -> str:
        """Cria prompt para análise detalhada linha por linha do código"""
        
        return f"""
Você é um especialista em análise de código e engenharia de software. Crie uma ANÁLISE TÉCNICA PROFUNDA linha por linha baseada EXCLUSIVAMENTE na análise real do código.

TÍTULO: {section['title']}

ANÁLISE COMPLETA DOS ARQUIVOS:
{structure_info.get('code_analysis', 'Análise de código não disponível')}

ESTRUTURA DETALHADA:
{structure_info.get('structure_analysis', 'Estrutura não analisada')}

DEPENDÊNCIAS IDENTIFICADAS:
{structure_info.get('dependencies', 'Dependências não identificadas')}

INSTRUÇÕES CRÍTICAS:
1. Analise CADA arquivo identificado na análise com detalhes técnicos profundos
2. Examine imports, dependências, padrões de código e estruturas de dados REAIS
3. Identifique vulnerabilidades, otimizações e melhorias possíveis
4. Use APENAS dados da análise fornecida - NÃO invente código ou estruturas
5. Foque em aspectos técnicos: performance, segurança, manutenibilidade

# {section['title']}

## 🔬 Análise Técnica Profunda por Arquivo

### Metodologia de Análise
- **Escopo:** Análise linha por linha dos arquivos principais
- **Critérios:** Performance, Segurança, Manutenibilidade, Padrões
- **Ferramentas:** Análise estática baseada na estrutura identificada

### Arquivos Analisados em Detalhes

[Para cada arquivo REAL da análise:]

#### [Nome Real do Arquivo] - Análise Técnica

**📍 Localização:** `[Caminho REAL do arquivo]`
**🔤 Linguagem:** [Linguagem REAL detectada]
**📏 Métricas:** [Linhas, tamanho, complexidade REAIS]

##### Estrutura do Código
- **Imports/Dependências:**
  [Liste os imports REAIS identificados na análise]
  - Análise de cada import: propósito, versão, segurança
  
- **Classes Identificadas:**
  [Para cada classe REAL encontrada:]
  - `[NomeClasseReal]`: [Propósito baseado na análise]
    - Métodos: [Métodos REAIS identificados]
    - Atributos: [Baseados na análise do código]
    - Padrões aplicados: [Padrões REAIS identificados]

- **Funções Principais:**
  [Para cada função REAL encontrada:]
  - `[NomeFuncaoReal]()`: [Propósito baseado na análise]
    - Parâmetros: [Baseados na análise]
    - Lógica: [Resumo da lógica identificada]
    - Complexidade: [Complexidade REAL calculada]

##### Análise de Qualidade
- **Performance:**
  - Pontos de otimização identificados
  - Estruturas de dados utilizadas
  - Algoritmos e complexidade
  
- **Segurança:**
  - Validação de inputs
  - Tratamento de erros
  - Exposição de dados sensíveis
  
- **Manutenibilidade:**
  - Legibilidade do código
  - Documentação interna
  - Padrões de nomenclatura

##### Padrões e Arquitetura
- **Padrões de Design:** [Padrões REAIS identificados]
- **Arquitetura:** [Estrutura arquitetural identificada]
- **Acoplamento:** [Análise de dependências REAIS]
- **Coesão:** [Análise da organização do código]

##### Recomendações Técnicas
- **Melhorias de Performance:** [Baseadas na análise real]
- **Refatorações Sugeridas:** [Baseadas no código analisado]
- **Correções de Segurança:** [Se identificadas vulnerabilidades]
- **Otimizações:** [Específicas para o código analisado]

---

## 📊 Resumo da Análise Técnica

### Métricas Gerais
[Baseadas na análise real:]
- **Total de Arquivos Analisados:** [Número REAL]
- **Linhas de Código:** [Total REAL]
- **Complexidade Média:** [Calculada na análise]
- **Linguagens Principais:** [Identificadas na análise]

### Pontos Críticos Identificados
[APENAS se identificados na análise real:]
1. **Performance:** [Problemas específicos encontrados]
2. **Segurança:** [Vulnerabilidades específicas]
3. **Manutenibilidade:** [Problemas de código específicos]

### Padrões Arquiteturais Detectados
[Baseados na estrutura REAL do código:]
- [Padrão1]: [Onde foi identificado]
- [Padrão2]: [Como está implementado]

### Tecnologias e Frameworks
[APENAS os identificados nas dependências:]
- [Framework1]: [Versão e uso identificado]
- [Biblioteca1]: [Propósito no projeto]

IMPORTANTE: Use SOMENTE informações REAIS da análise fornecida. Seja específico e técnico.
"""

    def _create_code_structure_report_prompt(self, section: Dict, structure_info: Dict[str, Any], final_url: str) -> str:
        """Cria prompt para relatório estrutural detalhado"""
        
        return f"""
Você é um arquiteto de software especialista. Crie um RELATÓRIO ESTRUTURAL DETALHADO baseado EXCLUSIVAMENTE na análise real da estrutura do projeto.

TÍTULO: {section['title']}

ANÁLISE ESTRUTURAL COMPLETA:
{structure_info.get('structure_analysis', 'Análise estrutural não disponível')}

ANÁLISE DE CÓDIGO:
{structure_info.get('code_analysis', 'Análise de código não disponível')}

DEPENDÊNCIAS:
{structure_info.get('dependencies', 'Dependências não identificadas')}

INSTRUÇÕES CRÍTICAS:
1. Use APENAS a estrutura de diretórios e arquivos REAIS identificados
2. Analise a organização REAL do projeto baseada na análise
3. Identifique padrões organizacionais e convenções REAIS
4. NÃO invente estruturas ou organize informações não presentes na análise

# {section['title']}

## 🏗️ Estrutura Organizacional do Projeto

### Hierarquia de Diretórios
[Baseado na estrutura REAL identificada:]

```
[Reproduza a estrutura REAL de diretórios identificada na análise]
```

### Organização por Responsabilidade

#### Diretórios Principais
[Para cada diretório REAL identificado:]

**📁 [Nome do Diretório Real]**
- **Propósito:** [Baseado nos arquivos encontrados]
- **Arquivos:** [Arquivos REAIS encontrados]
- **Linguagens:** [Linguagens REAIS detectadas]
- **Responsabilidade:** [Baseada na análise dos arquivos]

#### Padrões de Organização
[Baseados na estrutura REAL:]
- **Convenções de nomenclatura:** [Padrões REAIS identificados]
- **Separação de responsabilidades:** [Como está organizado]
- **Modularização:** [Estrutura modular identificada]

### Análise de Arquivos por Categoria

#### Arquivos de Configuração
[APENAS os identificados na análise:]
- `[arquivo.config]`: [Propósito baseado na análise]
- `[requirements.txt]`: [Dependências identificadas]

#### Arquivos Principais
[APENAS os identificados como importantes na análise:]
- `[main.py]`: [Função no projeto baseada na análise]
- `[app.js]`: [Responsabilidade identificada]

#### Arquivos de Teste
[APENAS se identificados na análise:]
- Localização: [Onde estão os testes]
- Cobertura: [Baseada nos arquivos encontrados]

### Dependências e Integrações

#### Dependências Externas
[APENAS as identificadas nos arquivos de dependência:]
- **[Framework1]**: [Versão e uso identificado]
- **[Biblioteca1]**: [Propósito no projeto]

#### Dependências Internas
[Baseadas nos imports REAIS identificados:]
- **Módulos internos:** [Módulos REAIS que se importam]
- **Acoplamento:** [Nível de dependência entre módulos]

### Métricas Estruturais

#### Distribuição de Código
[Baseada na análise REAL:]
- **Total de arquivos:** [Número REAL]
- **Arquivos por linguagem:** [Distribuição REAL]
- **Tamanho médio dos arquivos:** [Calculado da análise]

#### Complexidade Estrutural
[Baseada na análise REAL:]
- **Profundidade de diretórios:** [Níveis REAIS]
- **Arquivos por diretório:** [Média REAL]
- **Interdependências:** [Baseadas nos imports]

### Padrões Arquiteturais Identificados

#### Padrões de Organização
[APENAS os identificados na estrutura REAL:]
- **[Padrão1]**: [Onde é aplicado no projeto]
- **[Padrão2]**: [Como está implementado]

#### Convenções do Projeto
[Baseadas na análise REAL:]
- **Nomenclatura:** [Padrões REAIS de nomes]
- **Estrutura:** [Convenções REAIS de organização]
- **Separação:** [Como responsabilidades estão divididas]

### Recomendações Estruturais

#### Pontos Fortes
[Baseados na estrutura analisada:]
- [Aspecto positivo específico identificado]
- [Boa prática estrutural encontrada]

#### Oportunidades de Melhoria
[APENAS se identificadas na análise:]
- [Problema estrutural específico]
- [Sugestão de reorganização específica]

## 📋 Resumo Executivo da Estrutura

### Características Principais
- **Tipo de projeto:** [Identificado pela estrutura]
- **Padrão arquitetural:** [Principal padrão identificado]
- **Nível de organização:** [Baseado na análise]

### Pontos de Atenção
[APENAS os identificados na análise real]

IMPORTANTE: Use SOMENTE dados REAIS da análise estrutural fornecida.
"""

    def _create_technical_implementation_prompt(self, section: Dict, structure_info: Dict[str, Any], final_url: str) -> str:
        """Cria prompt para guia técnico de implementação"""
        
        return f"""
Você é um especialista em implementação de software. Crie um GUIA TÉCNICO DE IMPLEMENTAÇÃO baseado EXCLUSIVAMENTE na análise real do projeto.

TÍTULO: {section['title']}

ANÁLISE COMPLETA:
{structure_info.get('code_analysis', 'Análise de código não disponível')}

ESTRUTURA:
{structure_info.get('structure_analysis', 'Estrutura não analisada')}

DEPENDÊNCIAS:
{structure_info.get('dependencies', 'Dependências não identificadas')}

INSTRUÇÕES CRÍTICAS:
1. Base-se APENAS nas tecnologias, padrões e implementações REAIS identificadas
2. Documente APENAS os padrões e práticas REAIS encontradas no código
3. Use exemplos REAIS das funções e classes identificadas na análise
4. NÃO invente implementações ou padrões não presentes no código analisado

# {section['title']}

## 🛠️ Guia de Implementação Técnica

### Stack Tecnológico Identificado
[Baseado nas dependências e arquivos REAIS:]

#### Tecnologias Principais
- **Linguagem Principal:** [Linguagem REAL mais usada]
- **Framework:** [Framework REAL identificado nas dependências]
- **Bibliotecas:** [Bibliotecas REAIS encontradas]

#### Ferramentas de Desenvolvimento
[APENAS se identificadas nos arquivos de config:]
- **Gerenciamento de dependências:** [Tool REAL identificado]
- **Testes:** [Framework de teste se encontrado]
- **Build:** [Sistema de build se identificado]

### Padrões de Implementação Identificados

#### Padrões de Código
[Baseados no código REAL analisado:]

**Exemplo de Implementação Real:**
```[linguagem]
// Baseado em função/classe REAL identificada na análise
[Trecho de código real ou estrutura identificada]
```

#### Convenções de Nomenclatura
[Baseadas nos nomes REAIS identificados:]
- **Variáveis:** [Padrão REAL identificado]
- **Funções:** [Convenção REAL encontrada]
- **Classes:** [Padrão REAL de nomenclatura]

### Arquitetura de Implementação

#### Estrutura de Módulos
[Baseada na organização REAL:]
- **[Módulo Real 1]:** [Responsabilidade identificada]
  - Implementação: [Como está implementado]
  - Dependências: [Dependências REAIS]
  
- **[Módulo Real 2]:** [Função no sistema]
  - Padrões aplicados: [Padrões REAIS identificados]
  - Interfaces: [Interfaces REAIS identificadas]

#### Fluxo de Dados
[Baseado nas funções e imports REAIS:]
1. **Entrada:** [Como dados entram no sistema - baseado na análise]
2. **Processamento:** [Funções REAIS de processamento identificadas]
3. **Saída:** [Como dados saem - baseado no código analisado]

### Implementações Específicas

#### Gerenciamento de Estado
[APENAS se identificado no código:]
- **Padrão utilizado:** [Padrão REAL identificado]
- **Implementação:** [Como está implementado no código]

#### Tratamento de Erros
[Baseado no código REAL analisado:]
- **Estratégia:** [Como erros são tratados no código]
- **Implementação:** [Exemplos REAIS encontrados]

#### Configuração
[Baseada nos arquivos de config REAIS:]
- **Método:** [Como configuração é gerenciada]
- **Arquivos:** [Arquivos de config REAIS identificados]

### Padrões de Qualidade Implementados

#### Práticas de Código
[Identificadas na análise REAL:]
- **Documentação:** [Nível de documentação encontrado]
- **Testes:** [Cobertura de testes se identificada]
- **Validação:** [Validações implementadas no código]

#### Performance
[Baseada na análise do código:]
- **Otimizações:** [Otimizações REAIS identificadas]
- **Estruturas de dados:** [Estruturas REAIS utilizadas]

### Guia de Extensão

#### Como Adicionar Funcionalidades
[Baseado nos padrões REAIS identificados:]
1. **Seguir padrão:** [Padrão REAL identificado no projeto]
2. **Estrutura:** [Como novas funcionalidades devem se integrar]
3. **Dependências:** [Como gerenciar novas dependências]

#### Pontos de Extensão Identificados
[Baseados na arquitetura REAL:]
- **[Ponto de extensão 1]:** [Onde e como estender]
- **[Ponto de extensão 2]:** [Padrão para seguir]

### Configuração de Ambiente

#### Dependências
[Baseadas nos arquivos REAIS de dependência:]
```bash
# Comandos baseados nos arquivos de dependência identificados
[Comandos REAIS de instalação baseados nos arquivos encontrados]
```

#### Variáveis de Ambiente
[APENAS se identificadas no código:]
- `[VAR_REAL]`: [Propósito baseado no código]
- `[CONFIG_REAL]`: [Uso identificado na análise]

### Debugging e Troubleshooting

#### Logs
[Baseados no sistema de log identificado:]
- **Sistema:** [Sistema de log REAL identificado]
- **Níveis:** [Níveis de log encontrados no código]

#### Monitoramento
[APENAS se implementado no código:]
- **Métricas:** [Métricas REAIS implementadas]
- **Health checks:** [Se implementados no código]

## 📋 Checklist de Implementação

### Pré-requisitos
[Baseados nas dependências REAIS:]
- [ ] [Requisito 1 baseado na análise]
- [ ] [Requisito 2 identificado]

### Implementação
[Baseada nos padrões REAIS:]
- [ ] [Passo 1 baseado na estrutura real]
- [ ] [Passo 2 seguindo padrões identificados]

### Validação
[Baseada nas práticas REAIS encontradas:]
- [ ] [Validação 1 baseada no código]
- [ ] [Teste 1 baseado na estrutura]

IMPORTANTE: Use SOMENTE padrões e implementações REAIS identificadas na análise.
"""

    def _create_general_prompt(self, section: Dict, state: DocumentationState, final_url: str) -> str:
        """Cria prompt genérico"""
        return f"""
Crie documentação para: {section['title']}

Descrição: {section.get('description', 'Documentação geral')}

Repositório: {final_url}

Crie documentação útil e informativa em formato Markdown.
"""
    
    def _create_fallback_content(self, section: Dict, state: DocumentationState) -> str:
        """Cria conteúdo de fallback C4"""
        title = section.get('title', 'Documentação')
        repo_url = state.get("repo_url", "")
        
        return f"""# {title}

## 🏗️ Documentação Arquitetural C4

Esta seção documenta {title.lower()} seguindo o modelo de arquitetura C4.

## 🚀 Informações do Projeto

- **Projeto:** {repo_url}
- **Gerado em:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
- **Sistema:** Skyone DocAgent v3.0
- **Modelo:** Arquitetura C4

## 📝 Sobre o Modelo C4

O modelo C4 (Context, Container, Component, Code) fornece uma abordagem estruturada 
para visualizar a arquitetura de software em quatro níveis hierárquicos.

## 🔍 Análise Necessária

Para uma documentação completa desta seção, é necessária uma análise mais 
detalhada do código-fonte do projeto.

---
*Gerado pelo Skyone DocAgent v3.0 • Arquitetura C4*
"""
    
    def _get_section_filename(self, title: str, index: int, anonymous: bool) -> str:
        """Gera nome do arquivo para seção C4"""
        suffix = "_anonimo" if anonymous else ""
        
        title_lower = title.lower()
        if "context" in title_lower:
            return f"01_C4_Context_Diagram{suffix}.md"
        elif "container" in title_lower:
            return f"02_C4_Container_Diagram{suffix}.md"
        elif "component" in title_lower:
            return f"03_C4_Component_Diagram{suffix}.md"
        elif "c4" in title_lower and "code" in title_lower:
            return f"04_C4_Code_Analysis{suffix}.md"
        elif "detailed" in title_lower and "code" in title_lower:
            return f"05_Detailed_Code_Analysis{suffix}.md"
        elif "structure" in title_lower and "report" in title_lower:
            return f"06_Code_Structure_Report{suffix}.md"
        elif "technical" in title_lower and "implementation" in title_lower:
            return f"07_Technical_Implementation_Guide{suffix}.md"
        elif "mermaid" in title_lower or "flowchart" in title_lower:
            return f"08_Mermaid_Flowcharts{suffix}.md"
        elif "visão" in title_lower or "geral" in title_lower:
            return f"01_visao_geral{suffix}.md"
        elif "instalação" in title_lower or "guia" in title_lower:
            return f"02_guia_instalacao{suffix}.md"
        elif "técnico" in title_lower or "relatório" in title_lower:
            return f"03_relatorio_tecnico{suffix}.md"
        else:
            safe_title = re.sub(r'[^\w\s-]', '', title)
            safe_title = re.sub(r'[-\s]+', '_', safe_title)
            return f"{index:02d}_{safe_title.lower()}{suffix}.md"

# =============================================================================
# CONSTRUTOR DO GRAFO LANGGRAPH
# =============================================================================

class DocAgentLangGraph:
    """Sistema principal de documentação baseado em LangGraph"""
    
    def __init__(self):
        self.llm_manager = LLMManager()
        self.agents = DocumentationAgents(self.llm_manager)
        self.graph = None
        self.app = None
        self.mermaid_generator = None
        self._build_graph()
        logger.info("DocAgent LangGraph inicializado")
    
    def _build_graph(self):
        """Constrói o grafo LangGraph"""
        try:
            if not LANGGRAPH_AVAILABLE:
                logger.error("LangGraph não disponível")
                return
            
            # Criar grafo
            workflow = StateGraph(DocumentationState)
            
            # Adicionar nós
            workflow.add_node("clone_repository", self.agents.clone_repository_node)
            workflow.add_node("analyze_structure", self.agents.analyze_structure_node)
            workflow.add_node("generate_plan", self.agents.generate_documentation_plan_node)
            workflow.add_node("generate_docs", self.agents.generate_documentation_node)
            
            # Definir fluxo
            workflow.set_entry_point("clone_repository")
            workflow.add_edge("clone_repository", "analyze_structure")
            workflow.add_edge("analyze_structure", "generate_plan")
            workflow.add_edge("generate_plan", "generate_docs")
            workflow.add_edge("generate_docs", END)
            
            # Compilar com memória
            memory = MemorySaver()
            self.app = workflow.compile(checkpointer=memory)
            
            # Inicializar gerador Mermaid com o workflow
            self.mermaid_generator = MermaidGenerator(self.app)
            
            logger.info("Grafo LangGraph construído com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao construir grafo: {e}")
    
    async def execute_documentation_flow(self, request: AnalysisRequest) -> Dict[str, Any]:
        """Executa o fluxo completo de documentação"""
        try:
            if not self.app:
                raise Exception("Grafo LangGraph não inicializado")
            
            logger.info(f"Iniciando fluxo de documentação para: {request.repo_url}")
            
            # Configurar modelo LLM
            model_config = ModelConfig(
                provider=request.model_provider,
                model_name=request.model_name,
                api_key=os.environ.get("OPENAI_API_KEY") if request.model_provider == "openai" else None
            )
            
            if not self.llm_manager.configure_model(model_config):
                raise Exception("Falha na configuração do modelo LLM")
            
            # Estado inicial
            initial_state = DocumentationState(
                repo_url=request.repo_url,
                model_provider=request.model_provider,
                model_name=request.model_name,
                anonymous=request.anonymous,
                max_files=request.max_files,
                deep_analysis=request.deep_analysis,
                repo_path=None,
                current_phase="init",
                progress=0,
                logs=["🚀 Iniciando análise LangGraph"],
                file_structure=None,
                code_analysis=None,
                architecture_analysis=None,
                documentation_plan=None,
                generated_docs=[],
                metadata={
                    "start_time": datetime.now().isoformat(),
                    "system": "DocAgent LangGraph v3.0"
                },
                error_count=0
            )
            
            # Configuração do thread
            config = RunnableConfig(
                thread_id=f"doc_session_{int(time.time())}",
                recursion_limit=50
            )
            
            # Executar fluxo
            final_state = await self.app.ainvoke(initial_state, config=config)
            
            # Preparar resultado
            result = {
                "status": "success" if final_state["error_count"] == 0 else "partial_success",
                "message": f"Documentação gerada: {len(final_state['generated_docs'])} arquivos",
                "generated_docs": final_state["generated_docs"],
                "metadata": {
                    **final_state["metadata"],
                    "end_time": datetime.now().isoformat(),
                    "total_errors": final_state["error_count"],
                    "final_phase": final_state["current_phase"],
                    "model_used": f"{request.model_provider}/{request.model_name}"
                },
                "logs": final_state["logs"]
            }
            
            logger.info(f"Fluxo concluído: {result['status']}")
            return result
            
        except Exception as e:
            logger.error(f"Erro no fluxo de documentação: {e}")
            return {
                "status": "error",
                "message": f"Erro crítico: {str(e)}",
                "generated_docs": [],
                "metadata": {
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                },
                "logs": [f"❌ Erro crítico: {str(e)}"]
            }

# =============================================================================
# GERENCIAMENTO DO CICLO DE VIDA DA APLICAÇÃO
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerenciamento do ciclo de vida da aplicação"""
    # Startup
    try:
        print("🚀 DocAgent LangGraph iniciando...")
        
        # Setup GitHub Enterprise SSO se disponível
        if GITHUB_ENTERPRISE_AVAILABLE:
            try:
                github_config = GitHubEnterpriseConfig()
                if not github_config.is_configured():
                    print("⚠️ GitHub Enterprise SSO não configurado")
                    setup_environment_variables()
                else:
                    print("✅ GitHub Enterprise SSO configurado e pronto")
            except Exception as e:
                print(f"❌ Erro no setup SSO: {e}")
        
        # Outras inicializações...
        print("✅ DocAgent LangGraph pronto!")
        
    except Exception as e:
        print(f"❌ Erro na inicialização: {e}")
    
    yield
    
    # Shutdown
    print("👋 Encerrando DocAgent LangGraph...")

# =============================================================================
# API WEB COM FASTAPI
# =============================================================================

if not WEB_AVAILABLE:
    logger.error("FastAPI não disponível")
    exit(1)

# Configurar aplicação FastAPI
if CONFIG_AVAILABLE:
    app = FastAPI(
        title=config.SYSTEM_NAME,
        version=config.VERSION,
        description=config.DESCRIPTION,
        debug=config.DEBUG,
        lifespan=lifespan
    )
    
    # Configurar CORS baseado nas configurações
    if config.ENABLE_CORS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.ALLOWED_ORIGINS,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
else:
    app = FastAPI(
        title="DocAgent LangGraph",
        version="3.0",
        description="Sistema de Documentação Automática com LangGraph, OpenAI e Ollama",
        lifespan=lifespan
    )
    
    # CORS padrão
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Configurar diretórios
if CONFIG_AVAILABLE:
    config.create_directories()
    static_dir = config.STATIC_DIR
    templates_dir = config.TEMPLATES_DIR
    docs_dir = config.DOCS_DIR
    workdir = config.WORKDIR
else:
    static_dir = Path("static")
    templates_dir = Path("templates")
    docs_dir = Path("docs")
    workdir = Path("workdir")
    
    for directory in [static_dir, templates_dir, docs_dir, workdir]:
        directory.mkdir(exist_ok=True)

try:
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    templates = Jinja2Templates(directory=str(templates_dir))
except Exception as e:
    logger.warning(f"Erro ao configurar arquivos estáticos: {e}")

# Estado global da aplicação
app_state = {
    "doc_agent": DocAgentLangGraph(),
    "github_fetcher": GitHubRepositoryFetcher(),
    "current_analysis": None,
    "analysis_status": AnalysisStatus(
        status="idle",
        phase="Aguardando",
        progress=0,
        message="Sistema LangGraph pronto"
    ),
    "user_sessions": {}
}

# =============================================================================
# ROTAS DA API
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Página principal com suporte a autenticação"""
    try:
        # Verificar se usuário está autenticado (se SSO estiver habilitado)
        user_authenticated = False
        current_user = None
        
        if GITHUB_ENTERPRISE_AVAILABLE and "github_enterprise" in app_state:
            try:
                auth_middleware = app_state["github_enterprise"]["middleware"]
                current_user = await auth_middleware.get_current_user(request)
                user_authenticated = current_user is not None
            except Exception as e:
                logger.debug(f"Erro ao verificar autenticação: {e}")
        
        # Preparar contexto para template com configuração dinâmica
        config_status = {
            "github_enterprise_enabled": False,
            "github_enterprise_url": "",
            "openai_configured": bool(os.environ.get("OPENAI_API_KEY")),
            "github_token_configured": bool(os.environ.get("GITHUB_TOKEN")),
            "ollama_available": False
        }
        
        # Verificar GitHub Enterprise
        if GITHUB_ENTERPRISE_AVAILABLE:
            try:
                github_config = GitHubEnterpriseConfig()
                config_status["github_enterprise_enabled"] = github_config.is_configured()
                config_status["github_enterprise_url"] = github_config.GITHUB_ENTERPRISE_URL
            except Exception:
                pass
        
        # Verificar Ollama
        try:
            if app_state.get("llm_manager"):
                ollama_models = app_state["llm_manager"].list_ollama_models()
                config_status["ollama_available"] = len(ollama_models) > 0
        except Exception:
            pass
        
        template_context = {
            "request": request,
            "github_enterprise_enabled": GITHUB_ENTERPRISE_AVAILABLE,
            "user_authenticated": user_authenticated,
            "current_user": current_user,
            "config": config_status
        }
        
        return templates.TemplateResponse("index.html", template_context)
    except Exception as e:
        logger.error(f"Erro ao carregar template: {e}")
        # Fallback HTML com SSO
        return HTMLResponse(content=f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>DocAgent LangGraph v3.0</title>
            <meta charset="utf-8">
        </head>
        <body>
            <h1>🤖 DocAgent LangGraph v3.0</h1>
            <p>Sistema de Documentação Automática com IA</p>
            {'<p>✅ GitHub Enterprise SSO Habilitado</p>' if GITHUB_ENTERPRISE_AVAILABLE else ''}
            <p>⚠️ Erro ao carregar interface completa</p>
            <p>Verifique se o arquivo templates/index.html existe</p>
        </body>
        </html>
        """)

@app.post("/api/search")
async def search_repositories(search_request: SearchRequest):
    """Busca repositórios"""
    try:
        logger.info(f"Buscando repositórios para: {search_request.usuario}")
        
        repositories = app_state["github_fetcher"].search_repositories(
            search_request.usuario,
            search_request.incluir_forks
        )
        
        return {
            "success": True,
            "repositories": [asdict(repo) for repo in repositories],
            "count": len(repositories),
            "user": search_request.usuario
        }
    except Exception as e:
        logger.error(f"Erro na busca: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze")
async def start_analysis(analysis_request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Inicia análise com LangGraph"""
    try:
        logger.info(f"Iniciando análise LangGraph: {analysis_request.repo_url}")
        
        # Reset status
        app_state["analysis_status"] = AnalysisStatus(
            status="starting",
            phase="Iniciando LangGraph",
            progress=0,
            message="Preparando sistema LangGraph...",
            logs=["🚀 Sistema LangGraph iniciado"]
        )
        
        # Iniciar em background
        background_tasks.add_task(run_langgraph_analysis, analysis_request)
        
        return {
            "success": True,
            "message": "Análise LangGraph iniciada",
            "analysis_id": f"langgraph_{int(time.time())}",
            "model": f"{analysis_request.model_provider}/{analysis_request.model_name}"
        }
    except Exception as e:
        logger.error(f"Erro ao iniciar análise: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
async def get_analysis_status():
    """Obtém status da análise"""
    try:
        return app_state["analysis_status"]
    except Exception as e:
        logger.error(f"Erro ao obter status: {e}")
        return AnalysisStatus(
            status="error",
            phase="Erro",
            progress=0,
            message=f"Erro no sistema: {str(e)}"
        )

@app.get("/api/results")
async def get_analysis_results():
    """Obtém resultados da análise"""
    try:
        if app_state["current_analysis"]:
            return app_state["current_analysis"]
        else:
            raise HTTPException(status_code=404, detail="Nenhuma análise disponível")
    except Exception as e:
        logger.error(f"Erro ao obter resultados: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models")
async def list_models():
    """Lista modelos disponíveis"""
    try:
        available = app_state["doc_agent"].llm_manager.list_available_models()
        return {
            "success": True,
            "available": available,
            "providers": ["openai", "ollama"]
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    """Download de arquivo gerado"""
    try:
        if ".." in filename or "/" in filename:
            raise HTTPException(status_code=400, detail="Nome de arquivo inválido")
        
        file_path = Path("docs") / filename
        
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path, filename=filename)
        else:
            raise HTTPException(status_code=404, detail="Arquivo não encontrado")
            
    except Exception as e:
        logger.error(f"Erro no download: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download-all-zip")
async def download_all_docs():
    """Download de todos os documentos em ZIP"""
    try:
        docs_dir = Path("docs")
        if not docs_dir.exists():
            raise HTTPException(status_code=404, detail="Nenhum documento encontrado")
        
        doc_files = list(docs_dir.glob("*.md")) + list(docs_dir.glob("*.json")) + list(docs_dir.glob("*.png"))
        
        if not doc_files:
            raise HTTPException(status_code=404, detail="Nenhum documento disponível")
        
        # Criar ZIP
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"skyone_docagent_{timestamp}.zip"
        zip_path = docs_dir / zip_filename
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for doc_file in doc_files:
                zipf.write(doc_file, doc_file.name)
        
        return FileResponse(
            zip_path,
            filename=zip_filename,
            media_type="application/zip"
        )
        
    except Exception as e:
        logger.error(f"Erro ao criar ZIP: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/generate-mermaid-chart")
async def generate_mermaid_chart():
    """Gera diagrama Mermaid do workflow como imagem SVG"""
    try:
        # Gerar diagrama do workflow Skyone DocAgent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_filename = f"skyone_workflow_chart_{timestamp}.html"
        chart_path = Path("docs") / chart_filename
        
        # HTML com Mermaid incorporado para visualização como imagem
        chart_content = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Skyone DocAgent - Workflow Chart</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10.6.1/dist/mermaid.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #1a365d 0%, #4a90e2 100%);
            font-family: 'Arial', sans-serif;
            color: white;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            color: #1a365d;
        }}
        .mermaid {{
            text-align: center;
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border: 2px solid #1a365d;
        }}
        .footer {{
            text-align: center;
            margin-top: 20px;
            color: #666;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Skyone DocAgent - Workflow Chart</h1>
            <p>Fluxo completo de documentação C4 + Análise Detalhada</p>
            <p><strong>Gerado em:</strong> {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')}</p>
        </div>
        
        <div class="mermaid">
flowchart TD
    A[🚀 Início] --> B[🧠 Configurar Modelo IA]
    B --> C[📁 Selecionar Diretório Local]
    C --> D[🔍 Analisar Estrutura]
    D --> E[📋 Gerar Plano C4]
    
    E --> F[🌐 C4 Context Diagram]
    F --> G[📦 C4 Container Diagram]
    G --> H[🧩 C4 Component Diagram]
    H --> I[💻 C4 Code Analysis]
    
    I --> J[🔬 Detailed Code Analysis]
    J --> K[📊 Structure Report]
    K --> L[🛠️ Implementation Guide]
    L --> M[🔄 Mermaid Flowcharts]
    
    M --> N[✅ Documentação Completa]
    N --> O[📥 Download ZIP]
    
    subgraph "Fase 1: Preparação"
        A
        B
        C
        D
        E
    end
    
    subgraph "Fase 2: Arquitetura C4"
        F
        G
        H
        I
    end
    
    subgraph "Fase 3: Análise Detalhada"
        J
        K
        L
        M
    end
    
    subgraph "Fase 4: Finalização"
        N
        O
    end
    
    style A fill:#1a365d,stroke:#fff,stroke-width:3px,color:#fff
    style O fill:#4a90e2,stroke:#fff,stroke-width:3px,color:#fff
    style F fill:#e1f5fe,stroke:#1a365d,stroke-width:2px
    style G fill:#e1f5fe,stroke:#1a365d,stroke-width:2px
    style H fill:#e1f5fe,stroke:#1a365d,stroke-width:2px
    style I fill:#e1f5fe,stroke:#1a365d,stroke-width:2px
    style J fill:#e8f5e8,stroke:#4a90e2,stroke-width:2px
    style K fill:#e8f5e8,stroke:#4a90e2,stroke-width:2px
    style L fill:#e8f5e8,stroke:#4a90e2,stroke-width:2px
    style M fill:#e8f5e8,stroke:#4a90e2,stroke-width:2px
        </div>
        
        <div class="footer">
            <p><strong>Skyone DocAgent v3.0</strong> • Documentação Automática com IA</p>
            <p>C4 Architecture + Análise Detalhada • 8 Documentos Técnicos</p>
            <p><em>Para salvar como imagem: Clique com botão direito → "Salvar imagem como..."</em></p>
        </div>
    </div>
    
    <script>
        mermaid.initialize({{
            startOnLoad: true,
            theme: 'default',
            themeVariables: {{
                primaryColor: '#1a365d',
                primaryTextColor: '#fff',
                primaryBorderColor: '#4a90e2',
                lineColor: '#2d5a87'
            }}
        }});
    </script>
</body>
</html>"""
        
        # Salvar arquivo HTML
        with open(chart_path, 'w', encoding='utf-8') as f:
            f.write(chart_content)
        
        logger.info(f"Chart HTML gerado: {chart_filename}")
        
        return FileResponse(
            chart_path,
            filename=chart_filename,
            media_type="text/html"
        )
        
    except Exception as e:
        logger.error(f"Erro ao gerar chart: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/auth/user")
async def get_current_user(request: Request):
    """Obtém informações do usuário autenticado"""
    try:
        if GITHUB_ENTERPRISE_AVAILABLE and "github_enterprise" in app_state:
            auth_middleware = app_state["github_enterprise"]["middleware"]
            current_user = await auth_middleware.get_current_user(request)
            
            if current_user:
                return {
                    "success": True,
                    "user": {
                        "id": current_user.get("id"),
                        "login": current_user.get("login"),
                        "name": current_user.get("name"),
                        "email": current_user.get("email"),
                        "avatar_url": current_user.get("avatar_url")
                    }
                }
        
        raise HTTPException(status_code=401, detail="Não autenticado")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao obter usuário: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/auth/config")
async def get_auth_config():
    """Obtém configuração de autenticação"""
    try:
        github_enterprise_enabled = GITHUB_ENTERPRISE_AVAILABLE
        github_enterprise_configured = False
        
        if GITHUB_ENTERPRISE_AVAILABLE:
            from github_enterprise_sso import GitHubEnterpriseConfig
            config = GitHubEnterpriseConfig()
            github_enterprise_configured = config.is_configured()
        
        return {
            "success": True,
            "github_enterprise_enabled": github_enterprise_enabled,
            "github_enterprise_configured": github_enterprise_configured
        }
    except Exception as e:
        logger.error(f"Erro ao obter config de auth: {e}")
        return {
            "success": False,
            "github_enterprise_enabled": False,
            "github_enterprise_configured": False
        }

@app.get("/api/repositories/user")
async def get_user_repositories(request: Request, include_private: bool = True, include_forks: bool = False):
    """Obtém repositórios do usuário autenticado"""
    try:
        if not GITHUB_ENTERPRISE_AVAILABLE or "github_enterprise" not in app_state:
            raise HTTPException(status_code=400, detail="GitHub Enterprise SSO não disponível")
        
        auth_middleware = app_state["github_enterprise"]["middleware"]
        current_user = await auth_middleware.get_current_user(request)
        
        if not current_user:
            raise HTTPException(status_code=401, detail="Não autenticado")
        
        # Obter access token do usuário
        access_token = current_user.get("access_token")
        if not access_token:
            raise HTTPException(status_code=401, detail="Token de acesso não disponível")
        
        # Usar o fetcher do GitHub Enterprise
        repo_fetcher = app_state["github_enterprise"]["repo_fetcher"]
        repositories = await repo_fetcher.search_user_repositories(
            access_token=access_token,
            include_forks=include_forks,
            include_private=include_private
        )
        
        return {
            "success": True,
            "repositories": repositories,
            "count": len(repositories)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao obter repositórios do usuário: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/repositories/org/{org_name}")
async def get_org_repositories(org_name: str, request: Request, include_forks: bool = False):
    """Obtém repositórios de uma organização"""
    try:
        if not GITHUB_ENTERPRISE_AVAILABLE or "github_enterprise" not in app_state:
            raise HTTPException(status_code=400, detail="GitHub Enterprise SSO não disponível")
        
        auth_middleware = app_state["github_enterprise"]["middleware"]
        current_user = await auth_middleware.get_current_user(request)
        
        if not current_user:
            raise HTTPException(status_code=401, detail="Não autenticado")
        
        # Obter access token do usuário
        access_token = current_user.get("access_token")
        if not access_token:
            raise HTTPException(status_code=401, detail="Token de acesso não disponível")
        
        # Usar o fetcher do GitHub Enterprise
        repo_fetcher = app_state["github_enterprise"]["repo_fetcher"]
        repositories = await repo_fetcher.search_organization_repositories(
            access_token=access_token,
            org_name=org_name,
            include_forks=include_forks
        )
        
        return {
            "success": True,
            "repositories": repositories,
            "count": len(repositories),
            "organization": org_name
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao obter repositórios da organização {org_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class GitHubConfigRequest(BaseModel):
    """Requisição de configuração GitHub Enterprise"""
    github_enterprise_url: str
    client_id: str
    client_secret: str

@app.post("/api/github/configure")
async def configure_github_enterprise(config_request: GitHubConfigRequest):
    """Configura GitHub Enterprise dinamicamente"""
    try:
        if not GITHUB_ENTERPRISE_AVAILABLE:
            raise HTTPException(status_code=400, detail="GitHub Enterprise SSO não disponível")
        
        # Validar URL
        if not config_request.github_enterprise_url.startswith(('http://', 'https://')):
            raise HTTPException(status_code=400, detail="URL inválida")
        
        # Atualizar configuração no sistema SSO
        from github_enterprise_sso import GitHubEnterpriseConfig
        
        # Criar nova instância de configuração
        github_config = GitHubEnterpriseConfig()
        github_config.GITHUB_ENTERPRISE_URL = config_request.github_enterprise_url
        github_config.CLIENT_ID = config_request.client_id
        github_config.CLIENT_SECRET = config_request.client_secret
        github_config.REDIRECT_URI = "http://localhost:8001/auth/github/callback"
        
        # Atualizar no app_state se já existe integração
        if "github_enterprise" in app_state:
            # Recriar auth system com nova configuração
            from github_enterprise_sso import GitHubEnterpriseAuth, GitHubEnterpriseRepositoryFetcher, AuthenticationMiddleware
            
            auth_system = GitHubEnterpriseAuth(github_config)
            repo_fetcher = GitHubEnterpriseRepositoryFetcher(github_config)
            auth_middleware = AuthenticationMiddleware(auth_system)
            
            app_state["github_enterprise"] = {
                "config": github_config,
                "auth": auth_system,
                "repo_fetcher": repo_fetcher,
                "middleware": auth_middleware
            }
        
        logger.info(f"Configuração GitHub Enterprise atualizada: {config_request.github_enterprise_url}")
        
        return {
            "success": True,
            "message": "Configuração GitHub Enterprise salva com sucesso",
            "config": {
                "github_enterprise_url": config_request.github_enterprise_url,
                "client_id": config_request.client_id[:10] + "...",  # Mostrar apenas parte do ID
                "configured": True
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao configurar GitHub Enterprise: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/github/test-config")
async def test_github_config(config_request: GitHubConfigRequest):
    """Testa configuração GitHub Enterprise"""
    try:
        import httpx
        
        # Testar se a URL é acessível
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get(f"{config_request.github_enterprise_url}/api/v3")
                if response.status_code == 200:
                    api_data = response.json()
                    return {
                        "success": True,
                        "message": "Configuração válida",
                        "server_info": {
                            "version": api_data.get("version", "unknown"),
                            "url": config_request.github_enterprise_url
                        }
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Servidor retornou status {response.status_code}"
                    }
            except httpx.TimeoutException:
                return {
                    "success": False,
                    "error": "Timeout - Servidor não responde"
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Erro de conexão: {str(e)}"
                }
                
    except Exception as e:
        logger.error(f"Erro ao testar configuração: {e}")
        return {
            "success": False,
            "error": f"Erro interno: {str(e)}"
        }

class ModelSelectionRequest(BaseModel):
    """Requisição para configurar modelo"""
    model_config = {"protected_namespaces": ()}
    
    provider: str  # "openai" ou "ollama"
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None

@app.post("/api/configure-model")
async def configure_model(model_request: ModelSelectionRequest):
    """Configura modelo LLM (OpenAI ou Ollama)"""
    try:
        logger.info(f"Configurando modelo: {model_request.provider}/{model_request.model_name}")
        
        # Validar provider
        if model_request.provider not in ["openai", "ollama"]:
            raise HTTPException(status_code=400, detail="Provider deve ser 'openai' ou 'ollama'")
        
        # Configurar modelo no DocAgent
        doc_agent = app_state["doc_agent"]
        config = ModelConfig(
            provider=model_request.provider,
            model_name=model_request.model_name,
            api_key=model_request.api_key,
            base_url=model_request.base_url
        )
        
        # Se for OpenAI e não tiver API key na requisição, tentar pegar do ambiente
        if model_request.provider == "openai" and not model_request.api_key:
            config.api_key = os.environ.get("OPENAI_API_KEY")
            if not config.api_key:
                raise HTTPException(
                    status_code=400, 
                    detail="API Key OpenAI necessária. Configure OPENAI_API_KEY ou forneça na requisição."
                )
        
        # Configurar no LLM manager
        success = doc_agent.llm_manager.configure_model(config)
        
        if success:
            return {
                "success": True,
                "message": f"Modelo configurado: {model_request.provider}/{model_request.model_name}",
                "provider": model_request.provider,
                "model": model_request.model_name
            }
        else:
            raise HTTPException(status_code=500, detail="Falha ao configurar modelo")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao configurar modelo: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sso/setup")
async def sso_setup_guide():
    """Guia de configuração do GitHub Enterprise SSO"""
    if not GITHUB_ENTERPRISE_AVAILABLE:
        return {
            "available": False,
            "message": "GitHub Enterprise SSO não disponível",
            "requirements": [
                "pip install httpx pyjwt",
                "Adicionar github_enterprise_sso.py ao projeto"
            ]
        }
    
    github_config = GitHubEnterpriseConfig()
    
    return {
        "available": True,
        "configured": github_config.is_configured(),
        "github_enterprise_url": github_config.GITHUB_ENTERPRISE_URL,
        "client_id_configured": bool(github_config.CLIENT_ID),
        "client_secret_configured": bool(github_config.CLIENT_SECRET),
        "redirect_uri": github_config.REDIRECT_URI,
        "required_env_vars": [
            "GITHUB_ENTERPRISE_URL",
            "GITHUB_ENTERPRISE_CLIENT_ID", 
            "GITHUB_ENTERPRISE_CLIENT_SECRET",
            "GITHUB_ENTERPRISE_REDIRECT_URI"
        ],
        "setup_instructions": [
            "1. Acesse seu GitHub Enterprise em Settings > Developer settings > OAuth Apps",
            "2. Crie uma nova OAuth App",
            "3. Configure Homepage URL como http://localhost:8001",
            "4. Configure Authorization callback URL como http://localhost:8001/auth/github/callback",
            "5. Configure as variáveis de ambiente com Client ID e Client Secret",
            "6. Reinicie o DocAgent"
        ]
    }

@app.get("/api/models/available")
async def get_available_models():
    """Lista todos os modelos disponíveis"""
    try:
        doc_agent = app_state["doc_agent"]
        available = doc_agent.llm_manager.list_available_models()
        
        # Obter modelos Ollama detalhados
        ollama_models_detailed = doc_agent.llm_manager.get_ollama_models_detailed()
        
        # Se ollama_detailed estiver vazio, tentar converter da lista simples
        if not ollama_models_detailed and available.get("ollama"):
            ollama_models_detailed = [
                {
                    "name": model,
                    "size": "Tamanho desconhecido",
                    "display_name": model
                }
                for model in available["ollama"]
            ]
        
        # Verificar se OpenAI está configurada
        openai_configured = bool(os.environ.get("OPENAI_API_KEY"))
        
        # Verificar status do Ollama
        ollama_status = "unknown"
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
            ollama_status = "running" if result.returncode == 0 else "stopped"
        except:
            ollama_status = "not_installed"
        
        return {
            "success": True,
            "models": available,
            "ollama_detailed": ollama_models_detailed,
            "status": {
                "openai_configured": openai_configured,
                "ollama_status": ollama_status
            },
            "recommended": {
                "openai": "gpt-4o",
                "ollama": ollama_models_detailed[0]["name"] if ollama_models_detailed else "qwen2.5:7b"
            }
        }
    except Exception as e:
        logger.error(f"Erro ao listar modelos: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/test-model")
async def test_model_connection(model_request: ModelSelectionRequest):
    """Testa conexão com modelo LLM"""
    try:
        logger.info(f"Testando modelo: {model_request.provider}/{model_request.model_name}")
        
        # Criar configuração temporária
        config = ModelConfig(
            provider=model_request.provider,
            model_name=model_request.model_name,
            api_key=model_request.api_key,
            base_url=model_request.base_url
        )
        
        # Testar configuração
        temp_manager = LLMManager()
        success = temp_manager.configure_model(config)
        
        if not success:
            raise Exception("Falha na configuração do modelo")
        
        # Teste simples de geração
        test_llm = temp_manager.get_llm()
        test_message = HumanMessage(content="Responda apenas 'OK' se você está funcionando.")
        
        # Timeout para o teste
        import asyncio
        try:
            result = await asyncio.wait_for(
                test_llm.ainvoke([test_message]),
                timeout=30.0
            )
            
            response_text = result.content if hasattr(result, 'content') else str(result)
            
            return {
                "success": True,
                "message": f"Modelo {model_request.provider}/{model_request.model_name} testado com sucesso",
                "test_response": response_text[:100],
                "provider": model_request.provider,
                "model": model_request.model_name
            }
            
        except asyncio.TimeoutError:
            raise Exception("Timeout na resposta do modelo (30s)")
            
    except Exception as e:
        logger.error(f"Erro no teste do modelo: {e}")
        return {
            "success": False,
            "error": str(e),
            "provider": model_request.provider,
            "model": model_request.model_name
        }

@app.get("/health")
async def health_check():
    """Verificação de saúde com suporte a SSO"""
    try:
        checks = {
            "langgraph_available": LANGGRAPH_AVAILABLE,
            "doc_agent_ready": app_state["doc_agent"] is not None,
            "github_fetcher_ready": app_state["github_fetcher"] is not None,
            "docs_directory": Path("docs").exists(),
            "workdir_exists": Path("workdir").exists(),
            "github_enterprise_sso_available": GITHUB_ENTERPRISE_AVAILABLE
        }
        
        # Verificar configuração SSO se disponível
        sso_status = {}
        if GITHUB_ENTERPRISE_AVAILABLE:
            try:
                github_config = GitHubEnterpriseConfig()
                sso_status = {
                    "github_enterprise_configured": github_config.is_configured(),
                    "github_enterprise_url": github_config.GITHUB_ENTERPRISE_URL,
                    "oauth_ready": bool(github_config.CLIENT_ID and github_config.CLIENT_SECRET)
                }
                checks["github_enterprise_configured"] = github_config.is_configured()
            except Exception as e:
                sso_status = {"error": str(e)}
                checks["github_enterprise_configured"] = False
        
        # Verificar status do Ollama
        ollama_status = "unknown"
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
            ollama_status = "running" if result.returncode == 0 else "stopped"
        except:
            ollama_status = "not_installed"
        
        # Verificar OpenAI
        openai_configured = bool(os.environ.get("OPENAI_API_KEY"))
        
        all_healthy = all(checks.values())
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "checks": checks,
            "langgraph_enabled": LANGGRAPH_AVAILABLE,
            "github_enterprise_sso": sso_status,
            "model_status": {
                "ollama_status": ollama_status,
                "openai_configured": openai_configured
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# =============================================================================
# ENDPOINTS DE CONFIGURAÇÃO DINÂMICA
# =============================================================================

@app.get("/api/config/status")
async def get_config_status():
    """Retorna status de configuração para interface dinâmica"""
    try:
        # Verificar configurações
        github_enterprise_enabled = False
        github_enterprise_url = ""
        
        if GITHUB_ENTERPRISE_AVAILABLE:
            try:
                github_config = GitHubEnterpriseConfig()
                github_enterprise_enabled = github_config.is_configured()
                github_enterprise_url = github_config.GITHUB_ENTERPRISE_URL
            except Exception:
                pass
        
        # Verificar outras configurações
        openai_configured = bool(os.environ.get("OPENAI_API_KEY"))
        github_token_configured = bool(os.environ.get("GITHUB_TOKEN"))
        
        # Verificar Ollama
        ollama_available = False
        try:
            if app_state["llm_manager"]:
                ollama_models = app_state["llm_manager"].list_ollama_models()
                ollama_available = len(ollama_models) > 0
        except Exception:
            pass
        
        return {
            "github_enterprise_enabled": github_enterprise_enabled,
            "github_enterprise_url": github_enterprise_url,
            "openai_configured": openai_configured,
            "github_token_configured": github_token_configured,
            "ollama_available": ollama_available,
            "system_ready": all([
                app_state.get("doc_agent") is not None,
                app_state.get("github_fetcher") is not None
            ])
        }
    except Exception as e:
        logger.error(f"Erro ao obter status de configuração: {e}")
        return {
            "error": str(e),
            "github_enterprise_enabled": False,
            "openai_configured": False,
            "github_token_configured": False,
            "ollama_available": False,
            "system_ready": False
        }

@app.get("/api/test/openai")
async def test_openai():
    """Testa configuração OpenAI"""
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return {"success": False, "message": "OPENAI_API_KEY não configurada"}
        
        # Teste simples com OpenAI
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        response = llm.invoke("Teste de conexão. Responda apenas: OK")
        
        return {
            "success": True,
            "message": "OpenAI configurado e funcionando",
            "response": response.content
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Erro ao testar OpenAI: {str(e)}"
        }

@app.get("/api/test/github")
async def test_github():
    """Testa configuração GitHub"""
    try:
        token = os.environ.get("GITHUB_TOKEN")
        if not token:
            return {"success": False, "message": "GITHUB_TOKEN não configurado"}
        
        # Teste simples com GitHub API
        import httpx
        headers = {"Authorization": f"token {token}"}
        
        async with httpx.AsyncClient() as client:
            response = await client.get("https://api.github.com/user", headers=headers)
            
        if response.status_code == 200:
            user_data = response.json()
            return {
                "success": True,
                "message": f"GitHub Token válido para usuário: {user_data.get('login', 'Unknown')}",
                "user": user_data.get('login')
            }
        else:
            return {
                "success": False,
                "message": f"Token inválido - Status: {response.status_code}"
            }
    except Exception as e:
        return {
            "success": False,
            "message": f"Erro ao testar GitHub: {str(e)}"
        }

@app.get("/api/logs/download")
async def download_logs():
    """Download dos logs do sistema"""
    try:
        log_file = Path("logs/docagent.log")
        if not log_file.exists():
            raise HTTPException(status_code=404, detail="Arquivo de log não encontrado")
        
        return FileResponse(
            log_file,
            filename="docagent.log",
            media_type="text/plain"
        )
    except Exception as e:
        logger.error(f"Erro ao baixar logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# FUNÇÃO DE ANÁLISE EM BACKGROUND
# =============================================================================

async def run_langgraph_analysis(analysis_request: AnalysisRequest):
    """Executa análise LangGraph em background"""
    
    def update_status(phase: str, progress: int, message: str, step: str = ""):
        """Atualiza status da análise"""
        current_logs = app_state["analysis_status"].logs.copy()
        if step:
            current_logs.append(f"🔄 {step}: {message}")
        
        app_state["analysis_status"] = AnalysisStatus(
            status="running",
            phase=phase,
            progress=progress,
            message=message,
            logs=current_logs,
            current_step=step
        )
        logger.info(f"Status: {phase} ({progress}%) - {message}")
    
    try:
        logger.info(f"Background: Iniciando análise LangGraph de {analysis_request.repo_url}")
        
        # Fase 1: Inicialização
        update_status("Inicialização LangGraph", 5, "Preparando sistema...", "Setup")
        
        doc_agent = app_state["doc_agent"]
        if not doc_agent or not doc_agent.app:
            raise Exception("DocAgent LangGraph não inicializado")
        
        # Verificar se modelo está configurado
        if not doc_agent.llm_manager.current_config:
            # Configurar modelo padrão se não estiver configurado
            default_config = ModelConfig(
                provider=analysis_request.model_provider,
                model_name=analysis_request.model_name
            )
            
            if analysis_request.model_provider == "openai":
                default_config.api_key = os.environ.get("OPENAI_API_KEY")
                if not default_config.api_key:
                    raise Exception("API Key OpenAI não configurada")
            
            success = doc_agent.llm_manager.configure_model(default_config)
            if not success:
                raise Exception("Falha ao configurar modelo LLM")
        
        update_status("Configuração", 8, f"Modelo {analysis_request.model_provider}/{analysis_request.model_name} configurado", "LLM")
        
        # Fase 2: Execução do fluxo
        update_status("Execução LangGraph", 10, "Executando fluxo de documentação...", "Flow")
        
        result = await doc_agent.execute_documentation_flow(analysis_request)
        
        if result["status"] in ["success", "partial_success"]:
            # Sucesso
            app_state["current_analysis"] = {
                "status": result["status"],
                "message": result["message"],
                "repository_url": analysis_request.repo_url,
                "analysis_data": result,
                "generated_docs": result["generated_docs"],
                "timestamp": datetime.now().isoformat(),
                "langgraph_enabled": True,
                "analysis_type": "LangGraph_enhanced",
                "model_used": f"{analysis_request.model_provider}/{analysis_request.model_name}"
            }
            
            app_state["analysis_status"] = AnalysisStatus(
                status="completed",
                phase="Concluído",
                progress=100,
                message=f"Análise LangGraph concluída! {len(result['generated_docs'])} documentos gerados.",
                logs=result["logs"] + ["🎉 Análise LangGraph finalizada"],
                current_step="Pronto para download"
            )
            
            logger.info(f"Background: Análise concluída - {len(result['generated_docs'])} arquivos")
        else:
            # Erro
            raise Exception(result.get("message", "Erro desconhecido"))
        
    except Exception as e:
        error_msg = f"Erro na análise LangGraph: {str(e)}"
        logger.error(f"Background: {error_msg}")
        traceback.print_exc()
        
        app_state["analysis_status"] = AnalysisStatus(
            status="error",
            phase="Erro",
            progress=0,
            message=error_msg,
            logs=app_state["analysis_status"].logs + [f"❌ {error_msg}"],
            current_step="Falha"
        )


# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================

def main():
    """Função principal com suporte a GitHub Enterprise SSO"""
    try:
        print("🚀 Iniciando DocAgent LangGraph v3.0 com GitHub Enterprise SSO")
        print("=" * 80)
        
        # Verificar dependências
        if not LANGGRAPH_AVAILABLE:
            print("❌ LangGraph não disponível")
            print("Execute: pip install langgraph langchain langchain-openai langchain-community")
            return 1
        
        if not WEB_AVAILABLE:
            print("❌ FastAPI não disponível")
            print("Execute: pip install fastapi uvicorn jinja2")
            return 1
        
        # Verificar se GitHub Enterprise SSO está disponível
        if GITHUB_ENTERPRISE_AVAILABLE:
            print("🔐 GitHub Enterprise SSO detectado")
            # Verificar configuração
            try:
                config_test = GitHubEnterpriseConfig()
                if config_test.is_configured():
                    print("✅ GitHub Enterprise OAuth configurado")
                else:
                    print("⚠️ GitHub Enterprise OAuth não configurado")
                    print("   Configure as variáveis de ambiente para habilitar SSO")
            except Exception as e:
                print(f"⚠️ Erro na configuração GitHub Enterprise: {e}")
        
        # Validar ambiente se configuração disponível
        if CONFIG_AVAILABLE:
            print("🔍 Validando ambiente...")
            checks = config.validate_environment()
            
            for check_name, status in checks.items():
                icon = "✅" if status else "⚠️"
                print(f"   {icon} {check_name}: {'OK' if status else 'Não disponível'}")
            
            # Verificar se há problemas críticos
            critical_checks = ["git_available", "directories_writable"]
            if not all(checks[check] for check in critical_checks if check in checks):
                print("❌ Verificações críticas falharam")
                return 1
        else:
            # Verificações básicas
            try:
                subprocess.run(["git", "--version"], capture_output=True, check=True)
                print("✅ Git disponível")
            except:
                print("❌ Git não encontrado")
                return 1
            
            # Criar diretórios básicos
            for dir_name in ["docs", "workdir", "static", "templates", "logs"]:
                Path(dir_name).mkdir(exist_ok=True)
        
        print("\n" + "="*80)
        print("🤖 DocAgent LangGraph v3.0 - Sistema de Documentação Avançada")
        print("="*80)
        print("🚀 Funcionalidades Ativas:")
        print("   ✅ LangGraph para fluxo de agentes especializados")
        print("   ✅ Suporte OpenAI GPT-4 e Ollama local")
        print("   ✅ Análise completa de repositórios GitHub")
        print("   ✅ Documentação técnica automática")
        print("   ✅ Relatórios anônimos profissionais")
        print("   ✅ Interface web moderna e responsiva")
        print("   ✅ API REST completa com validação")
        print("   ✅ Download individual e ZIP completo")
        print("   ✅ Sistema de tools avançadas")
        print("   ✅ Checkpoints para recuperação de estado")
        if GITHUB_ENTERPRISE_AVAILABLE:
            print("   ✅ GitHub Enterprise SSO")
        print("="*80)
        
        # Configurações do servidor
        if CONFIG_AVAILABLE:
            host = config.HOST
            port = config.PORT
            log_level = config.LOG_LEVEL.lower()
        else:
            host = "0.0.0.0"
            port = 8001
            log_level = "info"
        
        print("🔗 URLs de Acesso:")
        print(f"   🏠 Interface Principal: http://localhost:{port}")
        print(f"   📚 Documentação API:   http://localhost:{port}/docs")
        print(f"   ❤️  Health Check:      http://localhost:{port}/health")
        print(f"   📊 Modelos Disponíveis: http://localhost:{port}/api/models/available")
        if GITHUB_ENTERPRISE_AVAILABLE:
            print(f"   🔐 Login GitHub:       http://localhost:{port}/auth/login")
        print("="*80)
        print("🛠️ Configurações:")
        
        if os.environ.get("OPENAI_API_KEY"):
            print("   ✅ OpenAI API Key configurada")
        else:
            print("   ⚠️  OpenAI API Key não configurada")
            print("      Configure: export OPENAI_API_KEY=sk-...")
        
        if os.environ.get("GITHUB_TOKEN"):
            print("   ✅ GitHub Token configurado")
        else:
            print("   ⚠️  GitHub Token não configurado")
            print("      Para repos privados: export GITHUB_TOKEN=ghp_...")
        
        # Configurações GitHub Enterprise
        if GITHUB_ENTERPRISE_AVAILABLE:
            github_config = GitHubEnterpriseConfig()
            if github_config.is_configured():
                print("   ✅ GitHub Enterprise SSO configurado")
                print(f"      URL: {github_config.GITHUB_ENTERPRISE_URL}")
            else:
                print("   ⚠️  GitHub Enterprise SSO não configurado")
                print("      Configure: GITHUB_ENTERPRISE_CLIENT_ID e GITHUB_ENTERPRISE_CLIENT_SECRET")
        
        # Verificar Ollama
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, timeout=3)
            if result.returncode == 0:
                print("   ✅ Ollama disponível")
            else:
                print("   ⚠️  Ollama não está rodando")
                print("      Execute: ollama serve")
        except:
            print("   ⚠️  Ollama não instalado")
            print("      Instale em: https://ollama.ai/")
        
        print("="*80)
        
        # Integrar GitHub Enterprise SSO se disponível
        if GITHUB_ENTERPRISE_AVAILABLE:
            print("🔐 Integrando GitHub Enterprise SSO...")
            try:
                integration_success = integrate_github_enterprise_sso(app, app_state)
                if integration_success:
                    print("✅ GitHub Enterprise SSO integrado com sucesso")
                else:
                    print("⚠️ GitHub Enterprise SSO parcialmente integrado")
            except Exception as e:
                print(f"❌ Erro na integração SSO: {e}")
                print("   Continuando sem SSO...")
        
        print("🌟 Iniciando servidor web...")
        print(f"🎯 Acesse: http://localhost:{port}")
        if GITHUB_ENTERPRISE_AVAILABLE:
            github_config = GitHubEnterpriseConfig()
            if github_config.is_configured():
                print(f"🔐 Login SSO: http://localhost:{port}/auth/login")
        print("🔧 Pressione Ctrl+C para parar")
        print("="*80)
        
        # Iniciar servidor
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level=log_level,
            access_log=True
        )
        
    except KeyboardInterrupt:
        print("\n👋 Encerrando DocAgent LangGraph...")
        print("   Obrigado por usar o sistema!")
        return 0
    except Exception as e:
        logger.error(f"Erro crítico: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())

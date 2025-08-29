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

Autor: DocAgent Skyone v3.0 LangGraph
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
    from langgraph.prebuilt import ToolNode
    from langgraph.checkpoint.memory import MemorySaver
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
    from langchain_core.tools import tool
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
    from langchain_community.llms import Ollama
    from langchain_core.runnables import RunnableConfig
    LANGGRAPH_AVAILABLE = True
    print("✅ LangGraph e LangChain disponíveis")
except ImportError as e:
    LANGGRAPH_AVAILABLE = False
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
    repo_url: str
    model_provider: str = "ollama"  # "openai" ou "ollama"
    model_name: str = "qwen2.5:7b"
    max_files: int = 50
    deep_analysis: bool = True
    anonymous: bool = True

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
        """Anonimiza URL do repositório"""
        try:
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
    
    def analyze_code_files(self, max_files: int = 20) -> List[FileAnalysis]:
        """Analisa arquivos de código em detalhes"""
        try:
            analyses = []
            
            # Padrões de arquivos importantes
            important_patterns = [
                'main.py', 'app.py', 'index.js', 'server.py', 'api.py',
                'models.py', 'views.py', 'controllers.py', 'routes.py',
                'utils.py', 'helpers.js', 'config.py', 'settings.py'
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
        """Extrai funções do código"""
        functions = []
        try:
            if language == 'Python':
                pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
            elif language in ['JavaScript', 'TypeScript']:
                pattern = r'function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
            elif language == 'Java':
                pattern = r'(?:public|private|protected)?\s*(?:static)?\s*\w+\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
            else:
                return []
            
            matches = re.findall(pattern, content, re.MULTILINE)
            functions = list(set(matches)) if matches else []
        except Exception:
            pass
        
        return functions
    
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

# =============================================================================
# SISTEMA DE MODELOS LLM
# =============================================================================

class LLMManager:
    """Gerenciador de modelos LLM (OpenAI e Ollama)"""
    
    def __init__(self):
        self.current_config = None
        self.available_models = {
            "openai": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
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
                        model_name = line.split()[0]
                        models.append(model_name)
                self.available_models["ollama"] = models
                logger.info(f"Modelos Ollama encontrados: {models}")
            else:
                logger.warning("Ollama não está rodando")
        except Exception as e:
            logger.warning(f"Erro ao verificar Ollama: {e}")
    
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
        """Nó para clonar repositório"""
        try:
            repo_url = state["repo_url"]
            logger.info(f"Clonando repositório: {repo_url}")
            
            # Validar URL
            if not self._validate_github_url(repo_url):
                state["logs"].append("❌ URL inválida")
                state["error_count"] += 1
                return state
            
            # Preparar diretório
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
            logger.error(f"Erro no clone: {e}")
            state["logs"].append(f"❌ Erro no clone: {str(e)}")
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
            
            # Prompt para planejamento
            planning_prompt = ChatPromptTemplate.from_template("""
Você é um especialista em documentação técnica. Baseado na análise do repositório abaixo, 
crie um plano detalhado para documentação completa em formato JSON.

ANÁLISE DO REPOSITÓRIO:
{structure_analysis}

{code_analysis}

{dependencies}

Crie um plano JSON com exatamente 3 seções:
1. "Visão Geral do Projeto" - tecnologias, propósito, arquitetura
2. "Guia de Instalação" - pré-requisitos, instalação, configuração
3. "Relatório Técnico" - análise detalhada dos arquivos e código

Formato JSON obrigatório:
{{
  "overview": "Descrição geral do projeto baseada na análise",
  "sections": [
    {{
      "title": "Visão Geral do Projeto",
      "description": "Análise completa das tecnologias e arquitetura",
      "content_type": "overview"
    }},
    {{
      "title": "Guia de Instalação e Configuração", 
      "description": "Instruções detalhadas baseadas nas dependências encontradas",
      "content_type": "installation"
    }},
    {{
      "title": "Relatório Técnico dos Arquivos",
      "description": "Análise técnica detalhada de cada arquivo importante",
      "content_type": "technical"
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
                # Plano padrão
                state["documentation_plan"] = {
                    "overview": "Projeto analisado automaticamente",
                    "sections": [
                        {"title": "Visão Geral do Projeto", "description": "Análise geral", "content_type": "overview"},
                        {"title": "Guia de Instalação", "description": "Instruções de instalação", "content_type": "installation"},
                        {"title": "Relatório Técnico", "description": "Análise técnica", "content_type": "technical"}
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
    
    async def _generate_section_content(self, llm, section: Dict, state: DocumentationState, section_num: int) -> str:
        """Gera conteúdo de uma seção específica"""
        try:
            content_type = section.get("content_type", "general")
            repo_url = state["repo_url"]
            anonymous = state.get("anonymous", True)
            
            # Anonimizar URL se necessário
            final_url = self.anonymizer.anonymize_repo_url(repo_url) if anonymous else repo_url
            
            if content_type == "overview":
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
                return result.content
            else:
                return str(result)
                
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
    
    def _create_general_prompt(self, section: Dict, state: DocumentationState, final_url: str) -> str:
        """Cria prompt genérico"""
        return f"""
Crie documentação para: {section['title']}

Descrição: {section.get('description', 'Documentação geral')}

Repositório: {final_url}

Crie documentação útil e informativa em formato Markdown.
"""
    
    def _create_fallback_content(self, section: Dict, state: DocumentationState) -> str:
        """Cria conteúdo de fallback"""
        title = section.get('title', 'Documentação')
        repo_url = state.get("repo_url", "")
        
        return f"""# {title}

## 📋 Visão Geral

Esta seção documenta {title.lower()} do projeto.

## 🚀 Informações

- **Repositório:** {repo_url}
- **Gerado em:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
- **Sistema:** DocAgent LangGraph v3.0

## 📝 Observações

Esta documentação foi gerada automaticamente pelo DocAgent LangGraph.
Para informações mais detalhadas, consulte o código-fonte do projeto.

---
*Gerado pelo DocAgent LangGraph v3.0*
"""
    
    def _get_section_filename(self, title: str, index: int, anonymous: bool) -> str:
        """Gera nome do arquivo para seção"""
        suffix = "_anonimo" if anonymous else ""
        
        title_lower = title.lower()
        if "visão" in title_lower or "geral" in title_lower:
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
        debug=config.DEBUG
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
        description="Sistema de Documentação Automática com LangGraph, OpenAI e Ollama"
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
    """Página principal"""
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        logger.error(f"Erro ao carregar template: {e}")
        return HTMLResponse(content="<h1>DocAgent LangGraph</h1><p>Erro ao carregar interface</p>")

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
        
        doc_files = list(docs_dir.glob("*.md")) + list(docs_dir.glob("*.json"))
        
        if not doc_files:
            raise HTTPException(status_code=404, detail="Nenhum documento disponível")
        
        # Criar ZIP
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"docagent_langgraph_{timestamp}.zip"
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

class ModelSelectionRequest(BaseModel):
    """Requisição para configurar modelo"""
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

@app.get("/api/models/available")
async def get_available_models():
    """Lista todos os modelos disponíveis"""
    try:
        doc_agent = app_state["doc_agent"]
        available = doc_agent.llm_manager.list_available_models()
        
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
            "status": {
                "openai_configured": openai_configured,
                "ollama_status": ollama_status
            },
            "recommended": {
                "openai": "gpt-4o",
                "ollama": "qwen2.5:7b"
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
    """Verificação de saúde"""
    try:
        checks = {
            "langgraph_available": LANGGRAPH_AVAILABLE,
            "doc_agent_ready": app_state["doc_agent"] is not None,
            "github_fetcher_ready": app_state["github_fetcher"] is not None,
            "docs_directory": Path("docs").exists(),
            "workdir_exists": Path("workdir").exists()
        }
        
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
    """Função principal"""
    try:
        print("🚀 Iniciando DocAgent LangGraph v3.0")
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
        print("🌟 Iniciando servidor web...")
        print(f"🎯 Acesse: http://localhost:{port}")
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

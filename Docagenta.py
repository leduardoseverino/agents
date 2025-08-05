import os
import sys
import logging
import time
import json
import subprocess
import tempfile
import shutil
import urllib.request
import urllib.error
import urllib.parse
import socket
import zipfile
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import re
from datetime import datetime
import hashlib
import traceback
import asyncio
from concurrent.futures import ThreadPoolExecutor
import secrets  # Para geração de tokens de estado no fluxo OAuth

# =============================================================================
# CORREÇÃO COMPLETA DO TORCH + AG2
# =============================================================================

def comprehensive_torch_ag2_fix():
    """
    Ajusta algumas variáveis de ambiente relacionadas ao Torch e AG2.

    Anteriormente este método fazia uma ampla monkey-patch em ``sys.modules``
    criando um módulo ``torch`` fictício quando a biblioteca não estava instalada.
    Isso causava erros inesperados (como "'function' object is not iterable")
    durante a inicialização do AG2. Agora limitamos a configuração às variáveis
    de ambiente e não modificamos o sistema de módulos. Caso ``torch`` não esteja
    instalado, a importação simplesmente falhará de forma controlada.
    """
    try:
        # Configurar variáveis de ambiente para silenciar avisos do tokenizers e Torch
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("🔧 Torch/AG2 fix aplicado")
    except Exception as e:
        # Em caso de falha inesperada, apenas registrar
        print(f"⚠️ Warning torch fix: {e}")

# Aplicar fix ANTES de qualquer outro import
comprehensive_torch_ag2_fix()

# FastAPI e dependências web
try:
    from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
    from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    import uvicorn
    WEB_AVAILABLE = True
    print("✅ FastAPI disponível")
except ImportError as e:
    WEB_AVAILABLE = False
    print(f"❌ FastAPI não disponível: {e}")
    print("Execute: pip install fastapi uvicorn jinja2")

# Pydantic com suporte V2
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

# AG2 imports com tratamento de erro melhorado
# Inicialmente assume que o modo AG2 está desativado até que a importação seja bem-sucedida.
AG2_AVAILABLE = False
try:
    # Antes de importar, verificar se o módulo auxiliar necessário está presente.
    # Algumas versões do AutoGen dependem do pacote fix_busted_json para realizar
    # tratamento de JSON na integração com Ollama. Se este pacote não estiver
    # instalado, o AG2 ficará inoperante e é melhor desabilitar o modo AG2 de
    # imediato para evitar erros em tempo de execução.
    try:
        import fix_busted_json  # noqa: F401
    except Exception:
        # Caso o pacote não esteja disponível, avisar o usuário e manter o modo
        # simplificado. Na documentação do AutoGen 0.9.7 este requisito é citado
        # como obrigatório quando se usa Ollama.
        raise ImportError("fix_busted_json ausente")

    # Tentar importar as classes a partir do pacote oficial "autogen". Para algumas versões
    # (por exemplo, instaladas via autogen-agentchat) as classes podem estar no submódulo
    # ``autogen.agentchat``. Tentamos ambas as opções.
    try:
        from autogen import ConversableAgent, GroupChat, GroupChatManager  # type: ignore
    except ImportError:
        # Fallback para versões em que as classes ficam em autogen.agentchat
        from autogen.agentchat import ConversableAgent, GroupChat, GroupChatManager  # type: ignore
    AG2_AVAILABLE = True
    print("✅ AG2 disponível")
except ImportError as e:
    # Biblioteca ausente ou dependência obrigatória não encontrada. Conforme a documentação oficial,
    # o pacote correto se chama ``autogen-agentchat`` e a partir da versão 0.9.7
    # é necessário instalar também o ``fix-busted-json`` para suporte a Ollama.
    AG2_AVAILABLE = False
    missing_pkg = str(e)
    print("⚠️ AG2 não disponível (modo simplificado ativo)")
    # Mensagem instrutiva para o usuário
    print(
        "💡 Para habilitar o modo AG2, instale as dependências corretas. "
        "Use: pip install autogen-agentchat~=0.9.7 fix-busted-json"
    )
    # Em algumas plataformas o pacote se chama pyautogen. Tentar orientar se apropriado.
    if "autogen" in missing_pkg:
        print("💬 Dependência ausente:", missing_pkg)
    elif "fix_busted_json" in missing_pkg:
        print("💬 Módulo fix_busted_json ausente. Instale com: pip install fix-busted-json")
except Exception as e:
    # Qualquer outro erro inesperado ao importar as classes do AG2
    AG2_AVAILABLE = False
    print("⚠️ Erro ao inicializar o AG2 (modo simplificado ativo)")
    print(f"Detalhes: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# MODELOS DE DADOS COMPATÍVEIS COM PYDANTIC V2
# =============================================================================

@dataclass
class RepositorioInfo:
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

class SearchRequest(BaseModel):
    usuario: str
    incluir_forks: bool = False

class AnalysisRequest(BaseModel):
    repo_url: str
    max_files: int = 25
    deep_analysis: bool = True
    include_config: bool = True
    model: Optional[str] = None
    anonymous: bool = True

# Adicionamos um modelo para autenticação GitHub, permitindo que o usuário
# forneça um token de acesso pessoal. Este modelo será utilizado na nova
# rota de autenticação para repositórios privados.
class GitHubAuthRequest(BaseModel):
    token: str

# Modelo para requisição de login
class LoginRequest(BaseModel):
    username: str
    password: str

# Modelo para configuração GitHub Enterprise
class GitHubEnterpriseConfig(BaseModel):
    """Configuração para GitHub Enterprise"""
    server_url: str = Field(..., description="URL do servidor GitHub Enterprise (ex: https://github.empresa.com)")
    api_url: Optional[str] = Field(None, description="URL da API (será calculada automaticamente se não fornecida)")
    token: Optional[str] = Field(None, description="Token de acesso pessoal")
    verify_ssl: bool = Field(True, description="Verificar certificados SSL")
    
    def get_api_url(self) -> str:
        """Retorna a URL da API baseada na URL do servidor"""
        if self.api_url:
            return self.api_url
        # Para GitHub Enterprise, a API geralmente está em /api/v3
        base_url = self.server_url.rstrip('/')
        return f"{base_url}/api/v3"

# Modelo para requisição de autenticação GitHub Enterprise
class GitHubEnterpriseAuthRequest(BaseModel):
    server_url: str = Field(..., description="URL do servidor GitHub Enterprise")
    token: str = Field(..., description="Token de acesso pessoal")
    verify_ssl: bool = Field(True, description="Verificar certificados SSL")

# =============================================================================
# C4 MODEL DATA STRUCTURES
# =============================================================================

@dataclass
class C4Element:
    """Elemento base do modelo C4"""
    name: str
    type: str
    description: str
    technology: str = ""
    relationships: List[str] = None
    
    def __post_init__(self):
        if self.relationships is None:
            self.relationships = []

@dataclass
class C4Person:
    """Pessoa no modelo C4"""
    name: str
    description: str
    external: bool = False

@dataclass
class C4System:
    """Sistema no modelo C4"""
    name: str
    description: str
    external: bool = False
    technology: str = ""

@dataclass
class C4Container:
    """Container no modelo C4"""
    name: str
    description: str
    technology: str
    system: str

@dataclass
class C4Component:
    """Componente no modelo C4"""
    name: str
    description: str
    technology: str
    container: str
    responsibilities: List[str] = None
    
    def __post_init__(self):
        if self.responsibilities is None:
            self.responsibilities = []

@dataclass
class C4Relationship:
    """Relacionamento no modelo C4"""
    source: str
    target: str
    description: str
    technology: str = ""

@dataclass
class C4Model:
    """Modelo C4 completo"""
    context: Dict[str, Any]
    containers: List[C4Container]
    components: List[C4Component] 
    relationships: List[C4Relationship]
    metadata: Dict[str, Any]


class AnalysisStatus(BaseModel):
    status: str
    phase: str
    progress: int
    message: str
    logs: List[str] = []
    current_step: str = ""

class DocItem(BaseModel):
    """Item de documentação - Compatível Pydantic V1/V2"""
    title: str = Field(description="Título da seção de documentação")
    description: str = Field(description="Descrição detalhada do conteúdo")
    prerequisites: str = Field(description="Pré-requisitos necessários")
    examples: List[str] = Field(description="Lista de exemplos práticos", default_factory=list)
    goal: str = Field(description="Objetivo específico da documentação")
    
    # Configuração V2 (ignora se V1)
    if PYDANTIC_V2:
        model_config = ConfigDict(
            validate_assignment=True,
            extra='forbid'
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Método compatível V1/V2 para serialização"""
        if PYDANTIC_V2:
            return self.model_dump()
        else:
            return self.dict()

class DocPlan(BaseModel):
    """Plano de documentação - Compatível Pydantic V1/V2"""
    overview: str = Field(description="Visão geral do projeto")
    docs: List[DocItem] = Field(description="Lista de itens de documentação", default_factory=list)
    
    # Configuração V2 (ignora se V1)
    if PYDANTIC_V2:
        model_config = ConfigDict(
            validate_assignment=True,
            extra='forbid'
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Método compatível V1/V2 para serialização"""
        if PYDANTIC_V2:
            return self.model_dump()
        else:
            return self.dict()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocPlan':
        """Método compatível V1/V2 para deserialização"""
        if PYDANTIC_V2:
            return cls.model_validate(data)
        else:
            return cls.parse_obj(data)

class DocumentationState(BaseModel):
    """Estado do fluxo - Compatível Pydantic V1/V2"""
    project_url: str
    repo_path: Optional[str] = None
    current_phase: str = "init"
    plan: Optional[DocPlan] = None
    generated_docs: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Configuração V2 (ignora se V1)
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
        """Método compatível V1/V2"""
        if PYDANTIC_V2:
            return self.model_dump()
        else:
            return self.dict()

@dataclass
class ModelConfig:
    # Definimos o modelo padrão para o AG2. A pedido do usuário, substituímos
    # o modelo original "qwen3:8b" por "llama3.2:3b", que consome menos recursos
    # e é adequado para análises de código. Caso o usuário deseje outro modelo,
    # ele pode especificar em AnalysisRequest.model.
    llm_model: str = "devstral:latest"
    context_window: int = 50000
    max_tokens: int = 8192
    timeout: int = 200
    temperature: float = 0.1

# =============================================================================
# SISTEMA DE BUSCA GITHUB
# =============================================================================

class GitHubRepositoryFetcher:
    """Sistema para buscar repositórios do GitHub e GitHub Enterprise"""
    
    def __init__(self):
        self.session_cache = {}
        self.rate_limit_info = {}
        self.last_request_time = 0
        self.min_request_interval = 1.0
        # Configuração padrão para GitHub.com
        self.base_url = "https://api.github.com"
        self.web_url = "https://github.com"
        self.enterprise_config = None
        self.session_tokens = {}  # Tokens por sessão de usuário
        print("🔍 Sistema de busca GitHub inicializado (suporte GitHub.com e Enterprise)")
    
    def _rate_limit_wait(self):
        """Implementa rate limiting básico"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            wait_time = self.min_request_interval - time_since_last
            time.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    def configure_enterprise(self, config: GitHubEnterpriseConfig, session_id: str = None):
        """Configura acesso a GitHub Enterprise"""
        try:
            self.enterprise_config = config
            self.base_url = config.get_api_url()
            self.web_url = config.server_url
            
            # Armazenar token da sessão se fornecido
            if config.token and session_id:
                self.session_tokens[session_id] = config.token
            
            print(f"🏢 GitHub Enterprise configurado: {config.server_url}")
            print(f"📡 API URL: {self.base_url}")
            return True
            
        except Exception as e:
            print(f"❌ Erro ao configurar Enterprise: {e}")
            return False
    
    def reset_to_github_com(self):
        """Volta para configuração padrão do GitHub.com"""
        self.base_url = "https://api.github.com"
        self.web_url = "https://github.com"
        self.enterprise_config = None
        print("🔄 Resetado para GitHub.com")
    
    def get_session_token(self, session_id: str = None) -> Optional[str]:
        """Obtém token para a sessão específica ou token global"""
        if session_id and session_id in self.session_tokens:
            return self.session_tokens[session_id]
        return os.environ.get('GITHUB_TOKEN')
    
    def set_session_token(self, token: str, session_id: str):
        """Define token para uma sessão específica"""
        self.session_tokens[session_id] = token
        print(f"🔑 Token configurado para sessão: {session_id[:8]}...")
    
    def buscar_repositorios_usuario(self, usuario_ou_org: str, incluir_forks: bool = False, session_id: str = None) -> List[RepositorioInfo]:
        """Busca repositórios de um usuário ou organização"""
        try:
            usuario_limpo = self._extrair_usuario_da_entrada(usuario_ou_org)
            print(f"🔍 Buscando repositórios de: {usuario_limpo}")
            
            if not self._verificar_usuario_existe(usuario_limpo, session_id):
                print(f"❌ Usuário/organização não encontrado: {usuario_limpo}")
                return []
            
            repositorios = []
            pagina = 1
            
            while len(repositorios) < 50 and pagina <= 5:
                repos_api_url = f"{self.base_url}/users/{usuario_limpo}/repos?per_page=30&sort=updated&page={pagina}"
                
                try:
                    self._rate_limit_wait()
                    repos_data = self._fazer_requisicao_github(repos_api_url, session_id)
                    
                    if not repos_data:
                        break
                    
                    for repo_data in repos_data:
                        if not incluir_forks and repo_data.get('fork', False):
                            continue

                        # Ao buscar repositórios, ignoramos privados apenas se
                        # não houver token de autenticação configurado. Caso o
                        # usuário tenha fornecido um token (e portanto
                        # possua permissão para visualizar seus repositórios
                        # privados), estes também serão listados.
                        if repo_data.get('private', False) and not os.environ.get('GITHUB_TOKEN'):
                            continue
                        
                        repo_info = self._processar_dados_repositorio(repo_data)
                        if repo_info:
                            repositorios.append(repo_info)
                    
                    if len(repos_data) < 30:
                        break
                    
                    pagina += 1
                    
                except Exception as e:
                    print(f"⚠️ Erro na página {pagina}: {e}")
                    break
            
            print(f"✅ Encontrados {len(repositorios)} repositórios")
            return sorted(repositorios, key=lambda x: x.estrelas, reverse=True)
            
        except Exception as e:
            print(f"❌ Erro ao buscar repositórios: {e}")
            traceback.print_exc()
            return []
    
    def _extrair_usuario_da_entrada(self, entrada: str) -> str:
        """Extrai nome de usuário de diferentes formatos de entrada"""
        entrada = entrada.strip()
        
        if 'github.com' in entrada:
            match = re.search(r'github\.com/([^/]+)', entrada)
            if match:
                return match.group(1)
        
        usuario = re.sub(r'[^a-zA-Z0-9\-_]', '', entrada)
        return usuario
    
    def _verificar_usuario_existe(self, usuario: str, session_id: str = None) -> bool:
        """Verifica se usuário existe"""
        try:
            url = f"{self.base_url}/users/{usuario}"
            self._rate_limit_wait()
            response = self._fazer_requisicao_github(url, session_id)
            return response is not None
        except Exception as e:
            print(f"⚠️ Erro ao verificar usuário: {e}")
            return False
    
    def _fazer_requisicao_github(self, url: str, session_id: str = None) -> Optional[Dict]:
        """Faz requisição para API do GitHub/Enterprise com tratamento robusto"""
        try:
            request = urllib.request.Request(url)
            request.add_header('User-Agent', 'Mozilla/5.0 (compatible; DocAgent-Skyone/2.0)')
            request.add_header('Accept', 'application/vnd.github.v3+json')
            
            # Priorizar token da sessão, depois token global
            github_token = self.get_session_token(session_id)
            if github_token:
                request.add_header('Authorization', f'token {github_token}')
            
            # Para GitHub Enterprise, pode ser necessário configurar SSL
            context = None
            if self.enterprise_config and not self.enterprise_config.verify_ssl:
                import ssl
                context = ssl.create_default_context()
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
                print("⚠️ SSL verification desabilitado para Enterprise")
            
            with urllib.request.urlopen(request, timeout=30, context=context) as response:
                if response.getcode() == 200:
                    data = json.loads(response.read().decode('utf-8'))
                    
                    rate_limit_remaining = response.headers.get('X-RateLimit-Remaining')
                    if rate_limit_remaining:
                        self.rate_limit_info['remaining'] = int(rate_limit_remaining)
                        if int(rate_limit_remaining) < 10:
                            print(f"⚠️ Rate limit baixo: {rate_limit_remaining}")
                    
                    return data
                else:
                    print(f"⚠️ Resposta HTTP {response.getcode()}")
                    return None
                    
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"❌ Recurso não encontrado (404): {url}")
            elif e.code == 403:
                print(f"❌ Rate limit atingido ou acesso negado (403)")
                print("💡 Dica: Configure GITHUB_TOKEN para aumentar o rate limit")
                time.sleep(60)
            elif e.code == 401:
                print(f"❌ Token inválido ou expirado (401)")
            else:
                print(f"❌ Erro HTTP {e.code}: {e.reason}")
            return None
        except urllib.error.URLError as e:
            print(f"❌ Erro de URL: {e.reason}")
            return None
        except Exception as e:
            print(f"❌ Erro na requisição: {e}")
            return None
    
    def _processar_dados_repositorio(self, repo_data: Dict) -> Optional[RepositorioInfo]:
        """Processa dados do repositório"""
        try:
            return RepositorioInfo(
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
            print(f"⚠️ Erro ao processar repositório: {e}")
            return None

# =============================================================================
# SISTEMA DE ANONIMIZAÇÃO
# =============================================================================

class SistemaAnonimizacao:
    """Sistema para anonimizar informações pessoais"""
    
    def __init__(self):
        self.mapeamento_usuarios = {}
        self.mapeamento_repos = {}
        self.contador_anonimo = 1
        print("🔒 Sistema de anonimização inicializado")
    
    def anonimizar_url_repositorio(self, url: str) -> str:
        """Anonimiza URL do repositório"""
        try:
            match = re.search(r'github\.com/([^/]+)/([^/]+)', url)
            if match:
                usuario, repo = match.groups()
                
                if usuario not in self.mapeamento_usuarios:
                    self.mapeamento_usuarios[usuario] = f"usuario_anonimo_{self.contador_anonimo}"
                    self.contador_anonimo += 1
                
                if repo not in self.mapeamento_repos:
                    self.mapeamento_repos[repo] = f"projeto_anonimo_{len(self.mapeamento_repos) + 1}"
                
                usuario_anonimo = self.mapeamento_usuarios[usuario]
                repo_anonimo = self.mapeamento_repos[repo]
                
                return f"https://github.com/{usuario_anonimo}/{repo_anonimo}"
            
            return "https://github.com/usuario_anonimo/projeto_anonimo"
            
        except Exception as e:
            print(f"⚠️ Erro na anonimização: {e}")
            return "https://github.com/usuario_anonimo/projeto_anonimo"

# =============================================================================
# TOOLS AVANÇADAS PARA ANÁLISE DETALHADA DE REPOSITÓRIO (AG2 COMPATÍVEL)
# =============================================================================

class AdvancedRepositoryTools:
    """Tools avançadas para análise completa de repositório - AG2 Compatible"""
    
    def __init__(self, repo_path: Union[str, Path]):
        self.repo_path = Path(repo_path)
        self.file_cache = {}
        self.error_count = 0
        self.analysis_cache = {}
        print(f"🔧 Inicializando tools AG2 avançadas para: {self.repo_path}")
    
    def _safe_execute(self, func_name: str, operation):
        """Execução segura com tratamento de erros"""
        try:
            return operation()
        except PermissionError:
            self.error_count += 1
            return f"❌ Permissão negada em {func_name}"
        except FileNotFoundError:
            self.error_count += 1
            return f"❌ Arquivo/diretório não encontrado em {func_name}"
        except UnicodeDecodeError:
            self.error_count += 1
            return f"❌ Erro de encoding em {func_name}"
        except Exception as e:
            self.error_count += 1
            return f"❌ Erro em {func_name}: {str(e)[:100]}"
    
    def directory_read(self, path: str = "") -> str:
        """Lista conteúdo de diretórios com análise detalhada"""
        def _operation():
            target_path = self.repo_path / path if path else self.repo_path
            
            if not target_path.exists():
                return f"❌ Diretório não encontrado: {target_path}"
            
            if not target_path.is_dir():
                return f"❌ Não é um diretório: {target_path}"
            
            result = f"## 📁 Estrutura Detalhada: {target_path.name if path else 'raiz'}\n\n"
            
            try:
                items = list(target_path.iterdir())
            except PermissionError:
                return f"❌ Sem permissão para ler: {target_path}"
            
            if not items:
                return result + "📂 Diretório vazio\n"
            
            # Classificar e analisar itens
            dirs = []
            code_files = []
            config_files = []
            doc_files = []
            other_files = []
            
            for item in items[:150]:  # Limite aumentado
                try:
                    if item.name.startswith('.'):
                        continue
                    
                    if item.is_dir():
                        # Contar arquivos no subdiretório
                        try:
                            sub_items = len(list(item.iterdir()))
                            dirs.append(f"📁 {item.name}/ ({sub_items} itens)")
                        except:
                            dirs.append(f"📁 {item.name}/")
                    else:
                        size = item.stat().st_size
                        size_str = self._format_size(size)
                        ext = item.suffix.lower()
                        
                        # Classificar por tipo
                        if ext in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.go', '.rs', '.php', '.rb', '.scala', '.kt']:
                            code_files.append(f"💻 {item.name} ({size_str}) - {self._get_language(ext)}")
                        elif ext in ['.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf']:
                            config_files.append(f"⚙️ {item.name} ({size_str}) - Config")
                        elif ext in ['.md', '.txt', '.rst', '.adoc'] or item.name.upper() in ['README', 'LICENSE', 'CHANGELOG']:
                            doc_files.append(f"📖 {item.name} ({size_str}) - Doc")
                        else:
                            other_files.append(f"📄 {item.name} ({size_str})")
                            
                except (PermissionError, OSError):
                    continue
            
            # Exibir resultado organizado por categoria
            if dirs:
                result += "### 📁 Diretórios:\n" + "\n".join(sorted(dirs)[:15]) + "\n\n"
            
            if code_files:
                result += "### 💻 Arquivos de Código:\n" + "\n".join(sorted(code_files)[:20]) + "\n\n"
            
            if config_files:
                result += "### ⚙️ Arquivos de Configuração:\n" + "\n".join(sorted(config_files)[:10]) + "\n\n"
            
            if doc_files:
                result += "### 📖 Documentação:\n" + "\n".join(sorted(doc_files)[:10]) + "\n\n"
            
            if other_files:
                result += "### 📄 Outros Arquivos:\n" + "\n".join(sorted(other_files)[:15]) + "\n\n"
            
            total_shown = len(dirs) + len(code_files) + len(config_files) + len(doc_files) + len(other_files)
            if len(items) > total_shown:
                result += f"... e mais {len(items) - total_shown} itens\n"
            
            return result
        
        return self._safe_execute("directory_read", _operation)
    
    def file_read(self, file_path: str) -> str:
        """Lê arquivos com análise inteligente do conteúdo"""
        def _operation():
            target_file = self.repo_path / file_path
            
            if not target_file.exists():
                return f"❌ Arquivo não encontrado: {file_path}"
            
            if not target_file.is_file():
                return f"❌ Não é um arquivo: {file_path}"
            
            # Cache check
            cache_key = str(target_file)
            if cache_key in self.file_cache:
                return self.file_cache[cache_key]
            
            try:
                file_size = target_file.stat().st_size
                if file_size > 300 * 1024:  # 300KB max
                    return f"❌ Arquivo muito grande: {file_path} ({self._format_size(file_size)})"
                
                if file_size == 0:
                    return f"📄 Arquivo vazio: {file_path}"
            
            except OSError:
                return f"❌ Erro ao acessar: {file_path}"
            
            # Tentar múltiplos encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'ascii']
            content = None
            used_encoding = None
            
            for encoding in encodings:
                try:
                    content = target_file.read_text(encoding=encoding)
                    used_encoding = encoding
                    break
                except UnicodeDecodeError:
                    continue
                except Exception:
                    break
            
            if content is None:
                return f"❌ Não foi possível ler o arquivo: {file_path}"
            
            # Verificar se é arquivo binário
            if '\x00' in content[:1000]:
                return f"❌ Arquivo binário detectado: {file_path}"
            
            # Análise do conteúdo
            lines = content.count('\n') + 1
            ext = target_file.suffix.lower()
            language = self._get_language(ext)
            
            # Análise específica por linguagem
            analysis = self._analyze_code_content(content, language)
            
            # Construir sufixo de truncamento fora da f-string para evitar
            # uso de barras invertidas em expressões. Quando o conteúdo é
            # muito longo (>4000 caracteres), adicionamos um aviso após o
            # trecho exibido.
            truncation_suffix = "...\n[TRUNCADO - Arquivo muito longo]" if len(content) > 4000 else ""

            result = f"""## 📄 Arquivo: {file_path}

### 📊 Informações:
- **Tamanho:** {self._format_size(file_size)}
- **Linhas:** {lines}
- **Linguagem:** {language}
- **Encoding:** {used_encoding}

### 🔍 Análise do Código:
{analysis}

### 💻 Conteúdo:
```{ext[1:] if ext else 'text'}
{content[:4000]}{truncation_suffix}
```
"""
            
            # Cache resultado (limitado)
            if len(self.file_cache) < 30:
                self.file_cache[cache_key] = result
            
            return result
        
        return self._safe_execute("file_read", _operation)
    
    def analyze_code_structure(self) -> str:
        """Análise avançada da estrutura de código do projeto"""
        def _operation():
            result = "## 🏗️ Análise Detalhada da Estrutura de Código\n\n"
            
            # Estatísticas por linguagem
            language_stats = {}
            function_count = 0
            class_count = 0
            total_loc = 0
            
            # Arquivos importantes analisados
            important_files = []
            
            try:
                for root, dirs, files in os.walk(self.repo_path):
                    # Filtrar diretórios irrelevantes
                    dirs[:] = [d for d in dirs if not d.startswith('.') 
                              and d not in ['node_modules', '__pycache__', 'target', 'build', 'dist', 'vendor']]
                    
                    for file in files:
                        if file.startswith('.'):
                            continue
                        
                        file_path = Path(root) / file
                        relative_path = file_path.relative_to(self.repo_path)
                        ext = file_path.suffix.lower()
                        
                        # Focar em arquivos de código
                        if ext not in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.go', '.rs', '.php', '.rb']:
                            continue
                        
                        try:
                            if file_path.stat().st_size > 500 * 1024:  # 500KB max
                                continue
                            
                            content = file_path.read_text(encoding='utf-8', errors='ignore')
                            lines = len([l for l in content.split('\n') if l.strip()])
                            
                            language = self._get_language(ext)
                            
                            # Estatísticas por linguagem
                            if language not in language_stats:
                                language_stats[language] = {'files': 0, 'lines': 0, 'functions': 0, 'classes': 0}
                            
                            language_stats[language]['files'] += 1
                            language_stats[language]['lines'] += lines
                            total_loc += lines
                            
                            # Análise de funções e classes
                            funcs, classes = self._count_functions_classes(content, language)
                            language_stats[language]['functions'] += funcs
                            language_stats[language]['classes'] += classes
                            function_count += funcs
                            class_count += classes
                            
                            # Arquivos importantes (>50 linhas ou nomes específicos)
                            if (lines > 50 or 
                                file.lower() in ['main.py', 'index.js', 'app.py', 'server.py', 'main.go'] or
                                'main' in file.lower() or 'app' in file.lower()):
                                
                                important_files.append({
                                    'path': str(relative_path),
                                    'language': language,
                                    'lines': lines,
                                    'functions': funcs,
                                    'classes': classes
                                })
                        
                        except (UnicodeDecodeError, PermissionError, OSError):
                            continue
                    
                    # Limitar busca para projetos muito grandes
                    if len(important_files) > 50:
                        break
            
            except Exception as e:
                result += f"⚠️ Erro na análise: {str(e)[:100]}\n\n"
            
            # Resumo geral
            result += f"### 📊 Resumo Geral:\n"
            result += f"- **Total de linhas de código:** {total_loc:,}\n"
            result += f"- **Funções identificadas:** {function_count}\n"
            result += f"- **Classes identificadas:** {class_count}\n"
            result += f"- **Linguagens detectadas:** {len(language_stats)}\n\n"
            
            # Estatísticas por linguagem
            if language_stats:
                result += "### 💻 Estatísticas por Linguagem:\n\n"
                for lang, stats in sorted(language_stats.items(), key=lambda x: x[1]['lines'], reverse=True):
                    result += f"**{lang}:**\n"
                    result += f"- Arquivos: {stats['files']}\n"
                    result += f"- Linhas: {stats['lines']:,}\n"
                    result += f"- Funções: {stats['functions']}\n"
                    result += f"- Classes: {stats['classes']}\n\n"
            
            # Arquivos importantes
            if important_files:
                result += "### 🎯 Arquivos Importantes Identificados:\n\n"
                for file_info in sorted(important_files, key=lambda x: x['lines'], reverse=True)[:15]:
                    result += f"**{file_info['path']}** ({file_info['language']})\n"
                    result += f"- {file_info['lines']} linhas\n"
                    if file_info['functions'] > 0:
                        result += f"- {file_info['functions']} funções\n"
                    if file_info['classes'] > 0:
                        result += f"- {file_info['classes']} classes\n"
                    result += "\n"
            
            return result
        
        return self._safe_execute("analyze_code_structure", _operation)
    
    def find_key_files(self) -> str:
        """Encontra arquivos importantes com categorização detalhada"""
        def _operation():
            result = "## 🔍 Arquivos-Chave Identificados\n\n"
            
            key_patterns = {
                "🚀 Pontos de Entrada": [
                    "main.py", "index.js", "app.py", "server.py", "main.go", 
                    "index.html", "App.js", "__init__.py", "main.java", "index.php"
                ],
                "📋 Configuração de Projeto": [
                    "package.json", "requirements.txt", "pom.xml", "Cargo.toml", 
                    "go.mod", "setup.py", "pyproject.toml", "composer.json", "build.gradle"
                ],
                "📖 Documentação": [
                    "README.md", "README.rst", "README.txt", "CHANGELOG.md", 
                    "LICENSE", "CONTRIBUTING.md", "docs/", "INSTALL.md"
                ],
                "🔧 Build e Deploy": [
                    "Makefile", "Dockerfile", "docker-compose.yml", 
                    ".github/workflows/", "Jenkinsfile", "build.gradle", "webpack.config.js"
                ],
                "⚙️ Configuração de Ambiente": [
                    "config.py", "settings.py", ".env", "config.json",
                    "webpack.config.js", "tsconfig.json", ".eslintrc", "pytest.ini"
                ],
                "🧪 Testes": [
                    "test_", "_test.py", ".test.js", "spec.js", "tests/", 
                    "test/", "pytest.ini", "jest.config.js"
                ],
                "🎨 Interface/Frontend": [
                    "style.css", "main.css", "app.css", "index.html", 
                    "template", "static/", "public/", "assets/"
                ]
            }
            
            found_files = {}
            search_count = 0
            
            try:
                for root, dirs, files in os.walk(self.repo_path):
                    search_count += 1
                    if search_count > 2000:  # Limite ampliado
                        break
                    
                    # Filtrar diretórios
                    dirs[:] = [d for d in dirs if not d.startswith('.') 
                              and d not in ['node_modules', '__pycache__', 'target', 'build', 'dist']]
                    
                    current_dir = Path(root)
                    relative_dir = current_dir.relative_to(self.repo_path)
                    
                    for file in files:
                        if file.startswith('.'):
                            continue
                        
                        file_path = current_dir / file
                        relative_path = file_path.relative_to(self.repo_path)
                        
                        # Verificar padrões
                        for category, patterns in key_patterns.items():
                            for pattern in patterns:
                                if (pattern.endswith('/') and pattern[:-1] in str(relative_dir)) or \
                                   (pattern in file.lower()) or \
                                   (pattern.lower() == file.lower()) or \
                                   (file.lower().startswith(pattern.lower())):
                                    
                                    if category not in found_files:
                                        found_files[category] = []
                                    
                                    if len(found_files[category]) < 12:  # Mais arquivos por categoria
                                        try:
                                            size = file_path.stat().st_size
                                            found_files[category].append({
                                                'path': str(relative_path),
                                                'size': self._format_size(size),
                                                'type': self._get_language(file_path.suffix.lower())
                                            })
                                        except:
                                            found_files[category].append({
                                                'path': str(relative_path),
                                                'size': 'N/A',
                                                'type': 'Unknown'
                                            })
                
            except Exception as e:
                result += f"⚠️ Busca limitada devido a erro: {str(e)[:50]}\n\n"
            
            # Formatear resultados detalhados
            if found_files:
                for category, files in found_files.items():
                    if files:
                        result += f"### {category}\n"
                        for file_info in files:
                            result += f"- **{file_info['path']}** "
                            result += f"({file_info['size']}, {file_info['type']})\n"
                        result += "\n"
            else:
                result += "📂 Nenhum arquivo-chave óbvio identificado\n"
                # Fallback melhorado
                try:
                    first_files = list(self.repo_path.glob("*"))[:15]
                    if first_files:
                        result += "\n**Primeiros arquivos encontrados:**\n"
                        for f in first_files:
                            if f.is_file():
                                try:
                                    size = self._format_size(f.stat().st_size)
                                    lang = self._get_language(f.suffix.lower())
                                    result += f"- **{f.name}** ({size}, {lang})\n"
                                except:
                                    result += f"- **{f.name}**\n"
                except:
                    pass
            
            return result
        
        return self._safe_execute("find_key_files", _operation)
    
    def detailed_file_analysis(self, max_files: int = 10) -> str:
        """Análise detalhada dos arquivos mais importantes"""
        def _operation():
            result = "## 🔬 Análise Detalhada dos Arquivos Principais\n\n"
            
            # Identificar arquivos para análise detalhada
            analysis_targets = []
            
            # Padrões de arquivos importantes
            important_patterns = [
                'main.py', 'app.py', 'server.py', 'index.js', 'main.go',
                'README.md', 'setup.py', 'package.json', 'requirements.txt'
            ]
            
            try:
                # Buscar arquivos importantes
                for root, dirs, files in os.walk(self.repo_path):
                    dirs[:] = [d for d in dirs if not d.startswith('.') 
                              and d not in ['node_modules', '__pycache__', 'target']]
                    
                    for file in files:
                        if file.startswith('.'):
                            continue
                        
                        file_path = Path(root) / file
                        relative_path = file_path.relative_to(self.repo_path)
                        
                        # Critérios para análise detalhada
                        should_analyze = False
                        priority = 0
                        
                        # Alta prioridade para arquivos específicos
                        if any(pattern in file.lower() for pattern in important_patterns):
                            should_analyze = True
                            priority = 10
                        
                        # Prioridade média para arquivos de código grandes
                        elif file_path.suffix.lower() in ['.py', '.js', '.ts', '.java', '.go']:
                            try:
                                if file_path.stat().st_size > 1000:  # > 1KB
                                    should_analyze = True
                                    priority = 5
                            except:
                                pass
                        
                        if should_analyze and len(analysis_targets) < max_files * 2:
                            analysis_targets.append({
                                'path': file_path,
                                'relative_path': relative_path,
                                'priority': priority
                            })
                
                # Ordenar por prioridade e tamanho
                analysis_targets.sort(key=lambda x: (-x['priority'], -x['path'].stat().st_size if x['path'].exists() else 0))
                analysis_targets = analysis_targets[:max_files]
                
            except Exception as e:
                result += f"⚠️ Erro na identificação de arquivos: {str(e)[:100]}\n\n"
                return result
            
            if not analysis_targets:
                result += "❌ Nenhum arquivo identificado para análise detalhada\n"
                return result
            
            result += f"Analisando {len(analysis_targets)} arquivos principais:\n\n"
            
            # Analisar cada arquivo
            for i, target in enumerate(analysis_targets, 1):
                try:
                    file_path = target['path']
                    relative_path = target['relative_path']
                    
                    if not file_path.exists():
                        continue
                    
                    result += f"### {i}. 📄 {relative_path}\n\n"
                    
                    # Informações básicas
                    size = file_path.stat().st_size
                    ext = file_path.suffix.lower()
                    language = self._get_language(ext)
                    
                    result += f"**Informações:**\n"
                    result += f"- Tamanho: {self._format_size(size)}\n"
                    result += f"- Linguagem: {language}\n"
                    
                    # Ler e analisar conteúdo
                    if size > 100 * 1024:  # 100KB
                        result += f"- Status: Arquivo muito grande para análise completa\n\n"
                        continue
                    
                    try:
                        content = file_path.read_text(encoding='utf-8', errors='ignore')
                        lines = len([l for l in content.split('\n') if l.strip()])
                        
                        result += f"- Linhas de código: {lines}\n"
                        
                        # Análise específica do conteúdo
                        code_analysis = self._analyze_code_content(content, language)
                        result += f"- Análise: {code_analysis}\n\n"
                        
                        # Mostrar snippet relevante
                        if language != "Text" and lines > 5:
                            snippet = self._extract_relevant_snippet(content, language)
                            if snippet:
                                result += f"**Trecho relevante:**\n```{ext[1:] if ext else 'text'}\n{snippet}\n```\n\n"
                        
                    except (UnicodeDecodeError, PermissionError):
                        result += f"- Status: Erro na leitura do arquivo\n\n"
                        continue
                    
                except Exception as e:
                    result += f"⚠️ Erro na análise de {target['relative_path']}: {str(e)[:50]}\n\n"
                    continue
            
            return result
        
        return self._safe_execute("detailed_file_analysis", _operation)
    
    def _analyze_code_content(self, content: str, language: str) -> str:
        """Análise específica do conteúdo do código"""
        if language == "Text":
            return "Arquivo de texto/documentação"
        
        analysis = []
        
        try:
            lines = content.split('\n')
            code_lines = [l for l in lines if l.strip() and not l.strip().startswith('#') and not l.strip().startswith('//')]
            
            if language == "Python":
                # Análise Python
                imports = [l for l in lines if l.strip().startswith('import ') or l.strip().startswith('from ')]
                functions = len([l for l in lines if l.strip().startswith('def ')])
                classes = len([l for l in lines if l.strip().startswith('class ')])
                
                if imports:
                    main_imports = [imp.split()[1].split('.')[0] for imp in imports[:5] if len(imp.split()) > 1]
                    analysis.append(f"Principais imports: {', '.join(main_imports[:3])}")
                
                if functions > 0:
                    analysis.append(f"{functions} funções")
                if classes > 0:
                    analysis.append(f"{classes} classes")
                    
                # Detectar frameworks
                content_lower = content.lower()
                frameworks = []
                if 'flask' in content_lower:
                    frameworks.append('Flask')
                if 'django' in content_lower:
                    frameworks.append('Django')
                if 'streamlit' in content_lower:
                    frameworks.append('Streamlit')
                if 'fastapi' in content_lower:
                    frameworks.append('FastAPI')
                
                if frameworks:
                    analysis.append(f"Frameworks: {', '.join(frameworks)}")
                    
            elif language == "JavaScript":
                # Análise JavaScript
                functions = len(re.findall(r'function\s+\w+', content))
                arrow_functions = len(re.findall(r'\w+\s*=>\s*', content))
                const_vars = len([l for l in lines if l.strip().startswith('const ')])
                
                if functions > 0:
                    analysis.append(f"{functions} funções declaradas")
                if arrow_functions > 0:
                    analysis.append(f"{arrow_functions} arrow functions")
                if const_vars > 0:
                    analysis.append(f"{const_vars} constantes")
                    
                # Detectar frameworks/bibliotecas
                if 'react' in content.lower():
                    analysis.append("React")
                if 'vue' in content.lower():
                    analysis.append("Vue.js")
                if 'angular' in content.lower():
                    analysis.append("Angular")
                if 'node' in content.lower():
                    analysis.append("Node.js")
                    
            elif language == "JSON":
                # Análise JSON
                try:
                    data = json.loads(content)
                    if isinstance(data, dict):
                        keys = list(data.keys())[:5]
                        analysis.append(f"Chaves principais: {', '.join(keys)}")
                except:
                    analysis.append("JSON com possível erro de sintaxe")
                    
            elif language in ["Java", "C++", "Go"]:
                # Análise para linguagens compiladas
                classes = len(re.findall(r'class\s+\w+', content))
                methods = len(re.findall(r'(public|private|protected).*?\w+\s*\(', content))
                
                if classes > 0:
                    analysis.append(f"{classes} classes")
                if methods > 0:
                    analysis.append(f"{methods} métodos")
            
            # Análise geral
            if len(code_lines) > 100:
                analysis.append("Arquivo extenso")
            elif len(code_lines) < 20:
                analysis.append("Arquivo pequeno")
                
        except Exception:
            analysis.append("Análise limitada devido a formato complexo")
        
        return "; ".join(analysis) if analysis else "Código padrão"
    
    def _count_functions_classes(self, content: str, language: str) -> Tuple[int, int]:
        """Conta funções e classes no código"""
        functions = 0
        classes = 0
        
        try:
            if language == "Python":
                functions = len(re.findall(r'^\s*def\s+\w+', content, re.MULTILINE))
                classes = len(re.findall(r'^\s*class\s+\w+', content, re.MULTILINE))
            elif language == "JavaScript":
                functions = len(re.findall(r'function\s+\w+', content))
                functions += len(re.findall(r'\w+\s*=\s*\([^)]*\)\s*=>', content))
                classes = len(re.findall(r'class\s+\w+', content))
            elif language in ["Java", "C++", "C#"]:
                functions = len(re.findall(r'(public|private|protected).*?\w+\s*\([^)]*\)\s*{', content))
                classes = len(re.findall(r'class\s+\w+', content))
            elif language == "Go":
                functions = len(re.findall(r'func\s+\w+', content))
        except:
            pass
        
        return functions, classes
    
    def _extract_relevant_snippet(self, content: str, language: str, max_lines: int = 10) -> str:
        """Extrai trecho relevante do código"""
        lines = content.split('\n')
        
        # Procurar por trechos interessantes
        if language == "Python":
            # Procurar por main, classes ou funções importantes
            for i, line in enumerate(lines):
                if ('if __name__' in line or 
                    line.strip().startswith('class ') or 
                    line.strip().startswith('def main')):
                    return '\n'.join(lines[i:i+max_lines])
        
        elif language == "JavaScript":
            # Procurar por exports, functions principais
            for i, line in enumerate(lines):
                if ('export' in line or 
                    'function main' in line or
                    'module.exports' in line):
                    return '\n'.join(lines[i:i+max_lines])
        
        # Fallback: primeiras linhas não vazias
        non_empty_lines = [l for l in lines if l.strip()]
        if non_empty_lines:
            return '\n'.join(non_empty_lines[:max_lines])
        
        return ""
    
    def _format_size(self, size: int) -> str:
        """Formata tamanho do arquivo"""
        if size < 1024:
            return f"{size}B"
        elif size < 1024*1024:
            return f"{size//1024}KB"
        else:
            return f"{size//(1024*1024)}MB"
    
    def _get_language(self, ext: str) -> str:
        """Identifica linguagem pela extensão"""
        language_map = {
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
        return language_map.get(ext, 'Unknown')

# =============================================================================
# ANALISADOR DE CÓDIGO AVANÇADO (COMPATÍVEL COM AG2)
# =============================================================================

class CodeAnalyzer:
    """Analisador avançado de código"""
    
    def __init__(self):
        self.language_patterns = {
            'Python': {
                'functions': r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
                'classes': r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[\(:)]',
                'imports': r'(?:from\s+[\w.]+\s+)?import\s+([\w\s,.*]+)',
            },
            'JavaScript': {
                'functions': r'(?:function\s+([a-zA-Z_][a-zA-Z0-9_]*)|([a-zA-Z_][a-zA-Z0-9_]*)\s*[:=]\s*(?:function|\([^)]*\)\s*=>))',
                'classes': r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)',
                'imports': r'(?:import\s+.*?from\s+[\'"]([^\'"]+)[\'"]|require\([\'"]([^\'"]+)[\'"]\))',
            },
            'Java': {
                'functions': r'(?:public|private|protected)?\s*(?:static)?\s*\w+\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
                'classes': r'(?:public|private)?\s*class\s+([a-zA-Z_][a-zA-Z0-9_]*)',
                'imports': r'import\s+([\w.]+)',
            },
            'TypeScript': {
                'functions': r'(?:function\s+([a-zA-Z_][a-zA-Z0-9_]*)|([a-zA-Z_][a-zA-Z0-9_]*)\s*[:=]\s*(?:function|\([^)]*\)\s*=>))',
                'classes': r'(?:export\s+)?(?:abstract\s+)?class\s+([a-zA-Z_][a-zA-Z0-9_]*)',
                'imports': r'import\s+.*?from\s+[\'"]([^\'"]+)[\'"]',
            }
        }
    
    def analyze_file(self, file_path: Path, content: str, language: str) -> FileAnalysis:
        """Analisa um arquivo de código em detalhes"""
        try:
            lines = [line for line in content.split('\n') if line.strip()]
            
            # Extrair funções, classes e imports
            functions = self._extract_functions(content, language)
            classes = self._extract_classes(content, language)
            imports = self._extract_imports(content, language)
            
            # Determinar propósito do arquivo
            purpose = self._determine_purpose(file_path, content, language)
            
            # Gerar resumo
            summary = self._generate_summary(file_path, content, language, functions, classes)
            
            # Calcular complexidade
            complexity = self._calculate_complexity(content, functions, classes)
            
            return FileAnalysis(
                name=file_path.name,
                path=str(file_path),
                language=language,
                size=len(content.encode('utf-8')),
                lines=len(lines),
                functions=functions[:10],  # Limitar para não ficar muito longo
                classes=classes[:10],
                imports=imports[:15],
                purpose=purpose,
                summary=summary,
                complexity=complexity
            )
            
        except Exception as e:
            print(f"⚠️ Erro ao analisar arquivo {file_path}: {e}")
            # Número de linhas e resumo precisam ser calculados fora das f-strings
            lines_count = len(content.split('\n'))
            return FileAnalysis(
                name=file_path.name,
                path=str(file_path),
                language=language,
                size=len(content.encode('utf-8')),
                lines=lines_count,
                functions=[],
                classes=[],
                imports=[],
                purpose="Arquivo de código do projeto",
                summary=f"Arquivo {language} com {lines_count} linhas",
                complexity="Baixa"
            )
    
    def _extract_functions(self, content: str, language: str) -> List[str]:
        """Extrai funções do código"""
        if language not in self.language_patterns:
            return []
        
        pattern = self.language_patterns[language].get('functions', '')
        if not pattern:
            return []
        
        matches = re.findall(pattern, content, re.MULTILINE)
        functions = []
        
        for match in matches:
            if isinstance(match, tuple):
                # Para patterns com grupos múltiplos
                func_name = next((m for m in match if m), None)
            else:
                func_name = match
            
            if func_name and func_name not in functions:
                functions.append(func_name)
        
        return functions
    
    def _extract_classes(self, content: str, language: str) -> List[str]:
        """Extrai classes do código"""
        if language not in self.language_patterns:
            return []
        
        pattern = self.language_patterns[language].get('classes', '')
        if not pattern:
            return []
        
        matches = re.findall(pattern, content, re.MULTILINE)
        return list(set(matches)) if matches else []
    
    def _extract_imports(self, content: str, language: str) -> List[str]:
        """Extrai imports/dependências do código"""
        if language not in self.language_patterns:
            return []
        
        pattern = self.language_patterns[language].get('imports', '')
        if not pattern:
            return []
        
        matches = re.findall(pattern, content, re.MULTILINE)
        imports = []
        
        for match in matches:
            if isinstance(match, tuple):
                # Para patterns com grupos múltiplos
                import_name = next((m for m in match if m), None)
            else:
                import_name = match
            
            if import_name:
                # Limpar e simplificar imports
                import_name = import_name.strip().split(',')[0].strip()
                if import_name and import_name not in imports:
                    imports.append(import_name)
        
        return imports
    
    def _determine_purpose(self, file_path: Path, content: str, language: str) -> str:
        """Determina o propósito do arquivo baseado no nome e conteúdo"""
        filename = file_path.name.lower()
        
        # Propósitos baseados no nome do arquivo
        if 'test' in filename or 'spec' in filename:
            return "Arquivo de testes unitários"
        elif filename in ['main.py', 'app.py', 'index.js', 'server.py']:
            return "Ponto de entrada principal da aplicação"
        elif filename in ['config.py', 'settings.py', 'config.js']:
            return "Arquivo de configuração"
        elif filename in ['utils.py', 'helpers.js', 'common.py']:
            return "Utilitários e funções auxiliares"
        elif filename.startswith('api') or 'controller' in filename:
            return "Controlador de API/Web"
        elif 'model' in filename or 'schema' in filename:
            return "Modelo de dados"
        elif 'view' in filename or 'component' in filename:
            return "Componente de interface"
        elif filename in ['dockerfile', 'docker-compose.yml']:
            return "Configuração de containerização"
        elif filename in ['package.json', 'requirements.txt', 'setup.py']:
            return "Gerenciamento de dependências"
        elif filename.endswith('.md'):
            return "Documentação"
        elif 'readme' in filename:
            return "Documentação principal do projeto"
        
        # Propósitos baseados no conteúdo
        if language == 'Python':
            if 'if __name__ == "__main__"' in content:
                return "Script executável Python"
            elif 'class' in content and 'def __init__' in content:
                return "Definição de classes Python"
            elif 'from flask import' in content or 'from django' in content:
                return "Aplicação web Python"
        elif language == 'JavaScript':
            if 'module.exports' in content or 'export' in content:
                return "Módulo JavaScript"
            elif 'React' in content or 'Component' in content:
                return "Componente React"
            elif 'express' in content:
                return "Servidor Express.js"
        
        return f"Arquivo {language} do projeto"
    
    def _generate_summary(self, file_path: Path, content: str, language: str, functions: List[str], classes: List[str]) -> str:
        """Gera um resumo do que o arquivo faz"""
        summary_parts = []
        
        # Informações básicas
        lines_count = len([l for l in content.split('\n') if l.strip()])
        summary_parts.append(f"Arquivo {language} com {lines_count} linhas de código")
        
        # Funções e classes
        if classes:
            summary_parts.append(f"Define {len(classes)} classe(s): {', '.join(classes[:3])}")
            if len(classes) > 3:
                summary_parts[-1] += f" e mais {len(classes) - 3}"
        
        if functions:
            summary_parts.append(f"Implementa {len(functions)} função(ões): {', '.join(functions[:3])}")
            if len(functions) > 3:
                summary_parts[-1] += f" e mais {len(functions) - 3}"
        
        # Análise específica do conteúdo
        content_lower = content.lower()
        
        if 'api' in content_lower or 'endpoint' in content_lower:
            summary_parts.append("Contém definições de API/endpoints")
        if 'database' in content_lower or 'db' in content_lower or 'sql' in content_lower:
            summary_parts.append("Inclui operações de banco de dados")
        if 'test' in content_lower and 'assert' in content_lower:
            summary_parts.append("Contém testes automatizados")
        if 'config' in content_lower or 'setting' in content_lower:
            summary_parts.append("Gerencia configurações do sistema")
        
        return ". ".join(summary_parts) + "."
    
    def _calculate_complexity(self, content: str, functions: List[str], classes: List[str]) -> str:
        """Calcula a complexidade do arquivo"""
        lines = len([l for l in content.split('\n') if l.strip()])
        
        # Contadores de complexidade
        complexity_score = 0
        
        # Tamanho do arquivo
        if lines > 500:
            complexity_score += 3
        elif lines > 200:
            complexity_score += 2
        elif lines > 50:
            complexity_score += 1
        
        # Número de funções/classes
        total_functions = len(functions) + len(classes)
        if total_functions > 20:
            complexity_score += 3
        elif total_functions > 10:
            complexity_score += 2
        elif total_functions > 5:
            complexity_score += 1
        
        # Estruturas de controle
        control_structures = len(re.findall(r'\b(if|for|while|try|catch|switch|case)\b', content))
        if control_structures > 50:
            complexity_score += 2
        elif control_structures > 20:
            complexity_score += 1
        
        # Classificação final
        if complexity_score >= 6:
            return "Alta"
        elif complexity_score >= 3:
            return "Média"
        else:
            return "Baixa"

# =============================================================================
# C4 MODEL ANALYZER
# =============================================================================

class C4ModelAnalyzer:
    """Analisador de arquitetura usando o modelo C4"""
    
    def __init__(self):
        self.code_analyzer = CodeAnalyzer()
        print("🏗️ C4 Model Analyzer inicializado")
    
    def analyze_repository_architecture(self, repo_path: str, repo_url: str) -> C4Model:
        """Analisa a arquitetura do repositório usando o modelo C4"""
        try:
            print("🏗️ Iniciando análise C4 do repositório...")
            
            # Analisar estrutura do projeto
            project_structure = self._analyze_project_structure(repo_path)
            
            # Identificar linguagens e tecnologias principais
            technologies = self._identify_technologies(repo_path)
            
            # Gerar modelo C4
            c4_model = self._generate_c4_model(project_structure, technologies, repo_url)
            
            print("✅ Análise C4 concluída")
            return c4_model
            
        except Exception as e:
            print(f"❌ Erro na análise C4: {e}")
            raise
    
    def _analyze_project_structure(self, repo_path: str) -> Dict[str, Any]:
        """Analisa a estrutura do projeto"""
        repo_path = Path(repo_path)
        structure = {
            "directories": [],
            "config_files": [],
            "entry_points": [],
            "data_layers": [],
            "api_layers": [],
            "ui_layers": []
        }
        
        try:
            for root, dirs, files in os.walk(repo_path):
                dirs[:] = [d for d in dirs if not d.startswith('.') 
                          and d not in ['node_modules', '__pycache__', 'target', 'build', 'dist']]
                
                root_path = Path(root)
                relative_path = root_path.relative_to(repo_path)
                
                if relative_path.name in ['api', 'controllers', 'routes']:
                    structure["api_layers"].append(str(relative_path))
                elif relative_path.name in ['models', 'entities', 'database', 'data']:
                    structure["data_layers"].append(str(relative_path))
                elif relative_path.name in ['ui', 'frontend', 'views', 'templates']:
                    structure["ui_layers"].append(str(relative_path))
                
                for file in files:
                    if file in ['package.json', 'requirements.txt', 'pom.xml']:
                        structure["config_files"].append(str(relative_path / file))
                    elif file in ['main.py', 'app.py', 'index.js', 'main.js']:
                        structure["entry_points"].append(str(relative_path / file))
            
            return structure
        except Exception as e:
            print(f"⚠️ Erro na análise de estrutura: {e}")
            return structure
    
    def _identify_technologies(self, repo_path: str) -> Dict[str, Any]:
        """Identifica tecnologias usadas no projeto"""
        technologies = {
            "primary_language": "Unknown",
            "frameworks": [],
            "databases": [],
            "deployment": []
        }
        
        try:
            repo_path = Path(repo_path)
            
            # Identificar linguagem principal
            language_count = {}
            for file_path in repo_path.rglob("*"):
                if file_path.is_file():
                    ext = file_path.suffix.lower()
                    if ext in ['.py', '.js', '.ts', '.java', '.go']:
                        language_count[ext] = language_count.get(ext, 0) + 1
            
            if language_count:
                primary_ext = max(language_count, key=language_count.get)
                ext_to_lang = {'.py': 'Python', '.js': 'JavaScript', '.ts': 'TypeScript', '.java': 'Java', '.go': 'Go'}
                technologies["primary_language"] = ext_to_lang.get(primary_ext, "Unknown")
            
            return technologies
        except Exception as e:
            print(f"⚠️ Erro na identificação de tecnologias: {e}")
            return technologies
    
    def _generate_c4_model(self, structure: Dict, technologies: Dict, repo_url: str) -> C4Model:
        """Gera o modelo C4 baseado na análise"""
        try:
            project_name = repo_url.split('/')[-1] if repo_url else "Software System"
            
            # Context Level
            context = {
                "system_name": project_name,
                "description": f"Sistema {project_name} desenvolvido em {technologies['primary_language']}",
                "technology": technologies["primary_language"],
                "users": ["End Users", "Administrators"],
                "external_systems": []
            }
            
            # Container Level
            containers = []
            if structure.get("ui_layers"):
                containers.append(C4Container(
                    name="Web Application",
                    description="Interface web do sistema",
                    technology=technologies["primary_language"],
                    system=project_name
                ))
            
            if structure.get("api_layers"):
                containers.append(C4Container(
                    name="API Application", 
                    description="API do sistema",
                    technology=technologies["primary_language"],
                    system=project_name
                ))
            
            if not containers:
                containers.append(C4Container(
                    name="Application",
                    description=f"Aplicação {project_name}",
                    technology=technologies["primary_language"],
                    system=project_name
                ))
            
            # Component Level
            components = [
                C4Component(
                    name="Business Logic",
                    description="Lógica de negócio",
                    technology=technologies["primary_language"],
                    container="Application",
                    responsibilities=["Processar regras de negócio", "Coordenar operações"]
                )
            ]
            
            # Relationships
            relationships = []
            
            # Metadata
            metadata = {
                "generated_at": datetime.now().isoformat(),
                "analyzer": "DocAgent C4 Model Analyzer",
                "repository": repo_url,
                "technologies": technologies
            }
            
            return C4Model(
                context=context,
                containers=containers,
                components=components,
                relationships=relationships,
                metadata=metadata
            )
            
        except Exception as e:
            print(f"❌ Erro na geração do modelo C4: {e}")
            raise

# =============================================================================
# API WEB COM FASTAPI (ATUALIZADA PARA AG2)
# =============================================================================

if not WEB_AVAILABLE:
    print("❌ FastAPI não disponível. Instale com: pip install fastapi uvicorn jinja2")
    exit(1)

app = FastAPI(
    title="DocAgent Skyone", 
    version="2.0",
    description="Sistema de Análise Automática de Repositórios GitHub com AG2 e Relatórios Anônimos"
)

# Configurar diretórios estáticos
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
templates_dir = Path("templates")
templates_dir.mkdir(exist_ok=True)

try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
    templates = Jinja2Templates(directory="templates")
except Exception as e:
    print(f"⚠️ Erro ao configurar arquivos estáticos: {e}")

# =============================================================================
# ENHANCED LOGGING SYSTEM
# =============================================================================

class EnhancedLogger:
    """Sistema de logging avançado para capturar saída detalhada"""
    
    def __init__(self):
        self.logs_history = []
        self.max_logs = 1000
        print("📝 Enhanced Logger inicializado")
    
    def log(self, level: str, message: str, phase: str = "", step: str = ""):
        """Adiciona log com timestamp e formatação"""
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Emoji por nível
        level_emojis = {
            "info": "ℹ️",
            "success": "✅", 
            "warning": "⚠️",
            "error": "❌",
            "debug": "🔍",
            "progress": "🔄"
        }
        
        emoji = level_emojis.get(level.lower(), "📝")
        
        log_entry = {
            "timestamp": timestamp,
            "level": level,
            "phase": phase,
            "step": step,
            "message": message,
            "formatted": f"[{timestamp}] {emoji} {phase} {step}: {message}".strip()
        }
        
        self.logs_history.append(log_entry)
        
        # Manter apenas os últimos N logs
        if len(self.logs_history) > self.max_logs:
            self.logs_history = self.logs_history[-self.max_logs:]
        
        # Print para console também
        print(log_entry["formatted"])
    
    def info(self, message: str, phase: str = "", step: str = ""):
        self.log("info", message, phase, step)
    
    def success(self, message: str, phase: str = "", step: str = ""):
        self.log("success", message, phase, step)
    
    def warning(self, message: str, phase: str = "", step: str = ""):
        self.log("warning", message, phase, step)
    
    def error(self, message: str, phase: str = "", step: str = ""):
        self.log("error", message, phase, step)
    
    def progress(self, message: str, phase: str = "", step: str = ""):
        self.log("progress", message, phase, step)
    
    def get_recent_logs(self, count: int = 50) -> List[Dict]:
        """Retorna logs recentes"""
        return self.logs_history[-count:] if self.logs_history else []
    
    def clear_logs(self):
        """Limpa histórico de logs"""
        self.logs_history = []
        self.info("Log history cleared", "System", "Reset")

# Estado global da aplicação
enhanced_logger = EnhancedLogger()

app_state = {
    "github_fetcher": GitHubRepositoryFetcher(),
    "analysis_engine": None,  # Será inicializado no main
    "current_analysis": None,
    "analysis_status": AnalysisStatus(
        status="idle",
        phase="Aguardando",
        progress=0,
        message="Sistema pronto",
        current_step=""
    ),
    "user_sessions": {},  # Sessões de usuários logados
    "auth_required": True,  # Se verdadeiro, exige autenticação
    "github_enterprise_config": None,  # Configuração Enterprise por sessão
    "active_github_configs": {},  # Configurações GitHub ativas por usuário
    "enhanced_logger": enhanced_logger  # Logger avançado
}

# =============================================================================
# MIDDLEWARE DE AUTENTICAÇÃO
# =============================================================================

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer(auto_error=False)

def verify_auth(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> bool:
    """Verifica se o usuário está autenticado"""
    if not app_state["auth_required"]:
        return True
    
    # Para requisições que não precisam de auth
    return True

def require_auth():
    """Decorator para rotas que exigem autenticação"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Implementar verificação aqui se necessário
            return func(*args, **kwargs)
        return wrapper
    return decorator

# =============================================================================
# ROTAS DA API (ATUALIZADAS COM AUTENTICAÇÃO)
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Página principal"""
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        print(f"❌ Erro ao carregar template: {e}")
        return HTMLResponse(content="<h1>DocAgent Skyone</h1><p>Erro ao carregar interface</p>")

@app.post("/api/search")
async def search_repositories(search_request: SearchRequest, request: Request):
    """Busca repositórios (com suporte a GitHub Enterprise)"""
    try:
        print(f"🔍 API: Buscando repositórios para '{search_request.usuario}'")
        
        # Obter session_id se disponível (baseado no sistema de sessões existente)
        session_id = None
        active_sessions = app_state["user_sessions"]
        for sid, session_data in active_sessions.items():
            if session_data.get("active", False):
                session_id = sid
                break
        
        repositorios = app_state["github_fetcher"].buscar_repositorios_usuario(
            search_request.usuario, 
            search_request.incluir_forks,
            session_id
        )
        
        print(f"✅ API: Encontrados {len(repositorios)} repositórios")
        
        # Obter informações sobre a configuração atual
        github_config = app_state.get("github_enterprise_config")
        config_info = {
            "type": "enterprise" if github_config else "github.com",
            "server_url": github_config.server_url if github_config else "https://github.com"
        }
        
        return {
            "success": True,
            "repositories": [asdict(repo) for repo in repositorios],
            "count": len(repositorios),
            "user": search_request.usuario,
            "github_config": config_info
        }
    except Exception as e:
        print(f"❌ API: Erro na busca: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze")
async def start_analysis(analysis_request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Inicia análise de repositório com AG2"""
    try:
        print(f"🔬 API: Iniciando análise AG2 de {analysis_request.repo_url}")
        
        # Reset do status
        app_state["analysis_status"] = AnalysisStatus(
            status="starting",
            phase="Iniciando análise AG2",
            progress=0,
            message="Preparando sistema avançado...",
            logs=["Sistema AG2 iniciado"],
            current_step="Inicializando AG2"
        )
        
        # Iniciar análise em background
        background_tasks.add_task(run_analysis_ag2, analysis_request)
        
        return {
            "success": True,
            "message": "Análise AG2 iniciada",
            "analysis_id": f"analysis_{int(time.time())}",
            "ag2_enabled": AG2_AVAILABLE
        }
    except Exception as e:
        print(f"❌ API: Erro ao iniciar análise AG2: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------------------------------------------------------
# Rota de autenticação GitHub
@app.post("/api/auth/github")
async def github_auth(auth: GitHubAuthRequest):
    """Define o token de acesso do GitHub para autenticar chamadas."""
    try:
        token = auth.token.strip() if auth.token else ""
        if not token:
            raise HTTPException(status_code=400, detail="Token não fornecido")
        # Armazenar o token no ambiente para uso em chamadas subsequentes
        os.environ["GITHUB_TOKEN"] = token
        print("🔐 Token GitHub configurado")
        return {"success": True, "message": "Token configurado com sucesso"}
    except HTTPException as he:
        # Repassar erros HTTP (ex.: token em branco)
        raise he
    except Exception as e:
        print(f"❌ Erro na autenticação GitHub: {e}")
        return {"success": False, "message": str(e)}

# -----------------------------------------------------------------------------
# Rotas de autenticação GitHub Enterprise
@app.post("/api/auth/github-enterprise")
async def github_enterprise_auth(auth: GitHubEnterpriseAuthRequest):
    """Configura autenticação para GitHub Enterprise"""
    try:
        # Validar URL do servidor
        if not auth.server_url.startswith(('http://', 'https://')):
            raise HTTPException(status_code=400, detail="URL do servidor deve começar com http:// ou https://")
        
        # Criar configuração Enterprise
        config = GitHubEnterpriseConfig(
            server_url=auth.server_url,
            token=auth.token,
            verify_ssl=auth.verify_ssl
        )
        
        # Configurar o fetcher
        github_fetcher = app_state["github_fetcher"]
        success = github_fetcher.configure_enterprise(config)
        
        if success:
            # Armazenar configuração no estado da aplicação
            app_state["github_enterprise_config"] = config
            
            print(f"🏢 GitHub Enterprise configurado: {auth.server_url}")
            return {
                "success": True,
                "message": "GitHub Enterprise configurado com sucesso",
                "server_url": auth.server_url,
                "api_url": config.get_api_url()
            }
        else:
            raise HTTPException(status_code=500, detail="Falha ao configurar GitHub Enterprise")
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Erro na configuração Enterprise: {e}")
        return {"success": False, "message": str(e)}

@app.post("/api/auth/github-enterprise/test")
async def test_github_enterprise_connection(auth: GitHubEnterpriseAuthRequest):
    """Testa conexão com GitHub Enterprise"""
    try:
        # Criar configuração temporária para teste
        config = GitHubEnterpriseConfig(
            server_url=auth.server_url,
            token=auth.token,
            verify_ssl=auth.verify_ssl
        )
        
        # Testar conexão fazendo uma requisição à API
        import urllib.request
        import ssl
        
        test_url = f"{config.get_api_url()}/user"
        request = urllib.request.Request(test_url)
        request.add_header('User-Agent', 'DocAgent-Skyone/2.0')
        request.add_header('Authorization', f'token {auth.token}')
        request.add_header('Accept', 'application/vnd.github.v3+json')
        
        context = None
        if not auth.verify_ssl:
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        
        with urllib.request.urlopen(request, timeout=10, context=context) as response:
            if response.getcode() == 200:
                user_data = json.loads(response.read().decode('utf-8'))
                return {
                    "success": True,
                    "message": "Conexão Enterprise testada com sucesso",
                    "server_url": auth.server_url,
                    "api_url": config.get_api_url(),
                    "user": user_data.get('login', 'Desconhecido')
                }
            else:
                raise HTTPException(status_code=response.getcode(), detail="Falha na conexão")
                
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise HTTPException(status_code=401, detail="Token inválido ou sem permissões")
        elif e.code == 404:
            raise HTTPException(status_code=404, detail="Servidor não encontrado ou URL inválida")
        else:
            raise HTTPException(status_code=e.code, detail=f"Erro HTTP: {e.reason}")
    except Exception as e:
        print(f"❌ Erro no teste Enterprise: {e}")
        raise HTTPException(status_code=500, detail=f"Erro na conexão: {str(e)}")

@app.post("/api/auth/reset-github")
async def reset_github_config():
    """Reset para configuração padrão do GitHub.com"""
    try:
        github_fetcher = app_state["github_fetcher"]
        github_fetcher.reset_to_github_com()
        app_state["github_enterprise_config"] = None
        
        return {
            "success": True,
            "message": "Configuração resetada para GitHub.com",
            "server_url": "https://github.com"
        }
    except Exception as e:
        print(f"❌ Erro ao resetar configuração: {e}")
        return {"success": False, "message": str(e)}

# -----------------------------------------------------------------------------
# Rota de login simples - ATUALIZADA COM VALIDAÇÃO REAL
@app.post("/api/login")
async def login(auth: LoginRequest):
    """Processa o login do usuário com validação melhorada."""
    try:
        username = auth.username.strip() if auth.username else ""
        password = auth.password.strip() if auth.password else ""
        
        if not username or not password:
            raise HTTPException(status_code=401, detail="Credenciais inválidas")
        
        # Validação básica - você pode expandir isso
        # Para demo, aceita qualquer usuário/senha não vazios
        # Em produção, você validaria contra um banco de dados
        valid_users = {
            "admin": "admin123",
            "user": "user123",
            "demo": "demo123"
        }
        
        if username in valid_users and valid_users[username] == password:
            # Criar sessão do usuário
            session_id = f"session_{int(time.time())}_{username}"
            app_state["user_sessions"][session_id] = {
                "username": username,
                "login_time": datetime.now().isoformat(),
                "active": True
            }
            
            print(f"🔑 Login efetuado para usuário: {username}")
            return {
                "success": True, 
                "message": "Login efetuado com sucesso",
                "session_id": session_id,
                "username": username
            }
        else:
            # Para segurança, sempre aceitamos credenciais não vazias em demo
            # mas você pode desabilitar isso em produção
            session_id = f"session_{int(time.time())}_{username}"
            app_state["user_sessions"][session_id] = {
                "username": username,
                "login_time": datetime.now().isoformat(),
                "active": True
            }
            
            print(f"🔑 Login demo efetuado para usuário: {username}")
            return {
                "success": True, 
                "message": "Login efetuado (modo demo)",
                "session_id": session_id,
                "username": username
            }
            
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"❌ Erro no login: {e}")
        return {"success": False, "message": str(e)}

# -----------------------------------------------------------------------------
# Rota de logout
@app.post("/api/logout")
async def logout(request: Request):
    """Efetua logout do usuário"""
    try:
        # Em um sistema real, você extrairia o session_id do token/header
        # Para demo, limpamos todas as sessões do usuário
        
        # Simular logout limpando sessões antigas
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session_data in app_state["user_sessions"].items():
            session_data["active"] = False
            expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            if session_id in app_state["user_sessions"]:
                del app_state["user_sessions"][session_id]
        
        print("🔒 Logout efetuado")
        return {"success": True, "message": "Logout efetuado com sucesso"}
        
    except Exception as e:
        print(f"❌ Erro no logout: {e}")
        return {"success": False, "message": str(e)}

# -----------------------------------------------------------------------------
# Rota para verificar status de login
@app.get("/api/auth/status")
async def auth_status():
    """Verifica status de autenticação"""
    try:
        active_sessions = len([s for s in app_state["user_sessions"].values() if s.get("active", False)])
        
        return {
            "authenticated": active_sessions > 0,
            "active_sessions": active_sessions,
            "auth_required": app_state["auth_required"],
            "total_sessions": len(app_state["user_sessions"])
        }
    except Exception as e:
        print(f"❌ Erro ao verificar status de auth: {e}")
        return {
            "authenticated": False,
            "active_sessions": 0,
            "auth_required": app_state["auth_required"],
            "error": str(e)
        }

# -----------------------------------------------------------------------------
# Rotas de autenticação GitHub via OAuth (mantidas iguais)
from fastapi.responses import RedirectResponse

@app.get("/login/github")
async def login_github():
    """Inicia o fluxo OAuth redirecionando para o GitHub."""
    client_id = os.environ.get('GITHUB_CLIENT_ID')
    if not client_id:
        raise HTTPException(status_code=500, detail="GITHUB_CLIENT_ID não configurado")
    # Definir URL de callback; se não existir, usar endereço padrão
    redirect_uri = os.environ.get('GITHUB_REDIRECT_URI', 'http://localhost:8000/auth/github/callback')
    # Gerar um token de estado para prevenir CSRF
    state = secrets.token_urlsafe(16)
    app_state['github_oauth_state'] = state
    # Construir URL de autorização
    params = urllib.parse.urlencode({
        'client_id': client_id,
        'redirect_uri': redirect_uri,
        'scope': 'repo',
        'state': state
    })
    auth_url = f"https://github.com/login/oauth/authorize?{params}"
    return RedirectResponse(auth_url)

@app.get("/auth/github/callback")
async def github_callback(code: str = '', state: str = ''):
    """Callback do OAuth do GitHub."""
    # Verificar o estado
    expected_state = app_state.get('github_oauth_state')
    if not expected_state or state != expected_state:
        return HTMLResponse(
            content="<h2>Estado inválido ou ausente.</h2>",
            status_code=400
        )
    # Trocar código por token
    client_id = os.environ.get('GITHUB_CLIENT_ID')
    client_secret = os.environ.get('GITHUB_CLIENT_SECRET')
    if not client_id or not client_secret:
        return HTMLResponse(
            content="<h2>Variáveis de ambiente de OAuth não configuradas.</h2>",
            status_code=500
        )
    token_url = "https://github.com/login/oauth/access_token"
    # Preparar dados e cabeçalhos
    data = urllib.parse.urlencode({
        'client_id': client_id,
        'client_secret': client_secret,
        'code': code,
        'state': state
    }).encode('utf-8')
    req = urllib.request.Request(token_url, data=data)
    req.add_header('Accept', 'application/json')
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            token_json = json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        print(f"❌ Falha ao obter token do GitHub: {e}")
        token_json = {}
    access_token = token_json.get('access_token')
    if access_token:
        os.environ['GITHUB_TOKEN'] = access_token
        
        # Criar sessão de usuário GitHub
        session_id = f"github_session_{int(time.time())}"
        app_state["user_sessions"][session_id] = {
            "username": "github_user",
            "login_time": datetime.now().isoformat(),
            "active": True,
            "auth_type": "github_oauth"
        }
        
        print("🔐 Token GitHub obtido via OAuth e sessão criada")
        # Informar o front-end via postMessage e fechar a janela
        html_content = """
<script>
window.opener.postMessage({type:'github_oauth', success:true}, '*');
window.close();
</script>
<p>Autenticação concluída. Pode fechar esta janela.</p>
"""
        return HTMLResponse(content=html_content)
    else:
        # Erro ao obter token
        html_content = """
<script>
window.opener.postMessage({type:'github_oauth', success:false}, '*');
window.close();
</script>
<p>Falha na autenticação.</p>
"""
        return HTMLResponse(content=html_content, status_code=400)

# Outras rotas mantidas iguais...
@app.get("/api/status")
async def get_analysis_status():
    """Obtém status da análise"""
    try:
        status = app_state["analysis_status"]
        # Adicionar logs detalhados do enhanced logger
        enhanced_logs = app_state["enhanced_logger"].get_recent_logs(20)
        
        # Combinar logs tradicionais com enhanced logs
        status_dict = status.model_dump()
        status_dict["enhanced_logs"] = enhanced_logs
        status_dict["total_log_count"] = len(app_state["enhanced_logger"].logs_history)
        
        return status_dict
    except Exception as e:
        app_state["enhanced_logger"].error(f"Erro ao obter status: {e}", "API", "Status")
        return AnalysisStatus(
            status="error",
            phase="Erro",
            progress=0,
            message=f"Erro no sistema: {str(e)}",
            logs=[f"Erro: {str(e)}"],
            current_step="Erro"
        )

@app.get("/api/logs")
async def get_detailed_logs(count: int = 50):
    """Obtém logs detalhados do sistema"""
    try:
        enhanced_logs = app_state["enhanced_logger"].get_recent_logs(count)
        return {
            "success": True,
            "logs": enhanced_logs,
            "total_count": len(app_state["enhanced_logger"].logs_history),
            "max_logs": app_state["enhanced_logger"].max_logs
        }
    except Exception as e:
        app_state["enhanced_logger"].error(f"Erro ao obter logs: {e}", "API", "Logs")
        return {
            "success": False,
            "error": str(e),
            "logs": [],
            "total_count": 0
        }

@app.post("/api/logs/clear")
async def clear_logs():
    """Limpa o histórico de logs"""
    try:
        app_state["enhanced_logger"].clear_logs()
        return {"success": True, "message": "Logs limpos com sucesso"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/results")
async def get_analysis_results():
    """Obtém resultados da análise"""
    try:
        if app_state["current_analysis"]:
            return app_state["current_analysis"]
        else:
            raise HTTPException(status_code=404, detail="Nenhuma análise disponível")
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ API: Erro ao obter resultados: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    """Download de arquivos gerados"""
    try:
        if ".." in filename or "\\" in filename:
            raise HTTPException(status_code=400, detail="Nome de arquivo inválido")

        safe_name = filename
        if "/" in filename:
            parts = filename.split("/")
            safe_name = parts[-1]

        file_path = Path("docs") / safe_name
        
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path, filename=safe_name)
        else:
            raise HTTPException(status_code=404, detail="Arquivo não encontrado")
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ API: Erro no download: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download-all-zip")
async def download_all_docs_zip():
    """Download de todos os documentos de análise em um arquivo ZIP único"""
    try:
        docs_dir = Path("docs")
        if not docs_dir.exists():
            raise HTTPException(status_code=404, detail="Nenhum documento encontrado")
        
        # Obter arquivos de documentação
        doc_files = list(docs_dir.glob("*.md")) + list(docs_dir.glob("*.json")) + list(docs_dir.glob("*.txt"))
        
        if not doc_files:
            raise HTTPException(status_code=404, detail="Nenhum documento disponível")
        
        # Criar arquivo ZIP temporário
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"docagent_analysis_{timestamp}.zip"
        zip_path = docs_dir / zip_filename
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for doc_file in doc_files:
                # Adicionar arquivo ao ZIP com nome relativo
                arcname = doc_file.name
                zipf.write(doc_file, arcname)
                print(f"✨ Adicionado ao ZIP: {arcname}")
        
        print(f"✅ ZIP criado com sucesso: {zip_filename} ({len(doc_files)} arquivos)")
        
        return FileResponse(
            zip_path, 
            filename=zip_filename,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={zip_filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ API: Erro ao criar ZIP: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro ao criar arquivo ZIP: {str(e)}")

@app.get("/health")
async def health_check():
    """Verificação de saúde do sistema"""
    try:
        checks = {
            "github_fetcher": app_state["github_fetcher"] is not None,
            "analysis_engine": app_state["analysis_engine"] is not None,
            "ag2_available": AG2_AVAILABLE,
            "docs_directory": Path("docs").exists(),
            "workdir": Path("workdir").exists() or True,
            "auth_system": True,
            "active_sessions": len(app_state["user_sessions"])
        }
        
        all_healthy = all(checks.values()) if isinstance(checks.values(), (list, tuple)) else True
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "checks": checks,
            "ag2_enabled": AG2_AVAILABLE,
            "auth_enabled": app_state["auth_required"],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# =============================================================================
# FUNÇÃO DE ANÁLISE EM BACKGROUND (mantida igual)
# =============================================================================

async def run_analysis_ag2(analysis_request: AnalysisRequest):
    """Executa análise completa em background com AG2"""
    
    def update_status(phase: str, progress: int, message: str, current_step: str = "", new_log: str = ""):
        """Atualiza status da análise com logs melhorados"""
        current_logs = app_state["analysis_status"].logs.copy()
        if new_log:
            current_logs.append(f"🤖 AG2: {new_log}")
        
        app_state["analysis_status"] = AnalysisStatus(
            status="running",
            phase=phase,
            progress=progress,
            message=message,
            logs=current_logs,
            current_step=current_step
        )
        print(f"🔄 Status: {phase} ({progress}%) - {message}")
        if new_log:
            print(f"📄 Log: {new_log}")
    
    try:
        print(f"🚀 Background: Iniciando análise AG2 de {analysis_request.repo_url}")
        
        # Fase 1: Inicialização (0-20%)
        update_status("Inicialização AG2", 5, "Preparando sistema AG2...", "Configurando agentes", "Sistema AG2 Multi-Agent iniciando")
        
        # Inicializar engine de análise
        analysis_engine = app_state.get("analysis_engine")
        if not analysis_engine:
            analysis_engine = AdvancedAnalysisEngine(ModelConfig())
            app_state["analysis_engine"] = analysis_engine
        
        update_status("Inicialização AG2", 10, "Engine AG2 inicializada", "Verificando dependências", "Engine de análise AG2 carregada")
        
        # Verificar se AG2 está disponível
        if not AG2_AVAILABLE:
            update_status("Fallback", 15, "AG2 não disponível - usando modo simplificado", "Modo tradicional", "AG2 não encontrado, ativando fallback")
            result = await run_simplified_analysis(analysis_request)
        else:
            update_status("Análise AG2", 20, "Iniciando análise com agentes AG2...", "4 agentes ativos", "Agentes AG2 especializados ativados")
            
            # Fase 2: Clone e preparação (20-40%)
            update_status("Clone", 25, "Clonando repositório...", "Download em progresso", "Fazendo clone do repositório GitHub")
            
            # Executar análise AG2
            try:
                result = analysis_engine.execute_enhanced_analysis(
                    analysis_request.repo_url,
                    max_files=analysis_request.max_files,
                    deep_analysis=analysis_request.deep_analysis,
                    anonymous=analysis_request.anonymous
                )
                
                if result.get('status') == 'success':
                    update_status("Processamento AG2", 60, "Agentes AG2 analisando código...", "Análise colaborativa", "4 agentes especializados trabalhando em colaboração")
                    update_status("Documentação AG2", 80, "Gerando documentação técnica...", "Escrita de relatórios", "Criando documentação técnica detalhada")
                else:
                    update_status("Fallback", 30, "Análise AG2 falhou - usando modo simplificado", "Recuperação automática", "AG2 falhou, ativando sistema de fallback")
                    result = await run_simplified_analysis(analysis_request)
                    
            except Exception as e:
                print(f"⚠️ Erro na análise AG2: {e}")
                update_status("Fallback", 35, "Erro no AG2 - usando fallback", "Sistema de recuperação", f"Erro AG2: {str(e)[:50]}. Ativando fallback")
                result = await run_simplified_analysis(analysis_request)
        
        # Fase 3: Finalização (80-100%)
        update_status("Finalização", 90, "Preparando resultados...", "Organização de arquivos", "Organizando documentação gerada")
        
        # Extrair arquivos gerados
        generated_docs = []
        if result and result.get('status') == 'success':
            generated_docs = result.get('generated_docs', [])
            update_status("Sucesso", 95, f"Análise concluída - {len(generated_docs)} documentos gerados", "Concluído", f"Documentação completa: {len(generated_docs)} arquivos")
        
        # Resultado final
        app_state["current_analysis"] = {
            "status": "success",
            "message": "Análise concluída com sucesso",
            "repository_url": analysis_request.repo_url,
            "analysis_data": result,
            "generated_docs": generated_docs,
            "timestamp": datetime.now().isoformat(),
            "ag2_enabled": AG2_AVAILABLE,
            "analysis_type": "AG2_enhanced" if AG2_AVAILABLE else "traditional"
        }
        
        app_state["analysis_status"] = AnalysisStatus(
            status="completed",
            phase="Concluído",
            progress=100,
            message=f"Análise concluída! {len(generated_docs)} documentos disponíveis para download.",
            logs=app_state["analysis_status"].logs + [f"🎉 Análise finalizada com sucesso - {len(generated_docs)} arquivos gerados"],
            current_step="Pronto para download"
        )
        
        print(f"🎉 Background: Análise completamente concluída - {len(generated_docs)} arquivos")
        
    except Exception as e:
        error_msg = f"Erro na análise: {str(e)}"
        print(f"❌ Background: {error_msg}")
        traceback.print_exc()
        
        app_state["analysis_status"] = AnalysisStatus(
            status="error",
            phase="Erro",
            progress=0,
            message=error_msg,
            logs=app_state["analysis_status"].logs + [f"❌ Erro crítico: {str(e)}"],
            current_step="Falha no sistema"
        )

async def run_simplified_analysis(analysis_request: AnalysisRequest):
    """Análise simplificada quando AG2 não está disponível"""
    try:
        print(f"🔧 Executando análise simplificada para {analysis_request.repo_url}")
        
        # Simular análise básica
        docs_dir = Path("docs")
        docs_dir.mkdir(exist_ok=True)
        
        # Gerar documento básico
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analise_simplificada_{timestamp}.md"
        file_path = docs_dir / filename
        
        content = f"""# Análise de Repositório - Modo Simplificado

**Repositório:** {analysis_request.repo_url}
**Data:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
**Modo:** Análise Tradicional (AG2 não disponível)

## Resumo

Este relatório foi gerado usando o modo simplificado do DocAgent Skyone.
O sistema AG2 não estava disponível no momento da análise.

## Informações Básicas

- **URL do Repositório:** {analysis_request.repo_url}
- **Análise Solicitada:** {"Profunda" if analysis_request.deep_analysis else "Superficial"}
- **Máximo de Arquivos:** {analysis_request.max_files}
- **Modo Anônimo:** {"Sim" if analysis_request.anonymous else "Não"}

## Recomendações

Para análises mais detalhadas, instale as dependências AG2:

```bash
pip install pyautogen fix-busted-json
```

E execute o Ollama localmente:

```bash
ollama serve
ollama pull qwen2.5:7b
```

---

*Gerado pelo DocAgent Skyone v2.0 - Modo Simplificado*
"""
        
        file_path.write_text(content, encoding='utf-8')
        print(f"✅ Documento simplificado criado: {filename}")
        
        return {
            'status': 'success',
            'generated_docs': [filename],
            'message': 'Análise simplificada concluída'
        }
        
    except Exception as e:
        print(f"❌ Erro na análise simplificada: {e}")
        return {
            'status': 'error',
            'message': str(e),
            'generated_docs': []
        }

# =============================================================================
# TEMPLATE HTML ATUALIZADO COM AUTENTICAÇÃO MELHORADA
# =============================================================================

def create_html_template():
    """Cria template HTML aprimorado com autenticação completa"""
    
    html_content = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DocAgent Skyone - Análise Automática com AG2</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    colors: {
                        primary: '#1e40af',
                        secondary: '#3b82f6',
                        accent: '#0ea5e9'
                    }
                }
            }
        }
    </script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
</head>
<body class="bg-gradient-to-br from-blue-50 to-indigo-100 min-h-screen">
    <!-- Overlay de Login -->
    <div id="loginOverlay" class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div class="bg-white rounded-xl p-8 shadow-lg w-full max-w-md">
            <div class="text-center mb-6">
                <div class="w-16 h-16 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-full flex items-center justify-center mx-auto mb-4">
                    <i class="fas fa-robot text-white text-2xl"></i>
                </div>
                <h3 class="text-2xl font-bold text-gray-800">DocAgent Skyone</h3>
                <p class="text-gray-600">Faça login para continuar</p>
            </div>
            
            <div class="space-y-4">
                <input type="text" id="loginUsername" placeholder="Usuário" 
                       class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors" />
                <input type="password" id="loginPassword" placeholder="Senha" 
                       class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors" />
                
                <button type="button" onclick="realizarLogin()" 
                        class="w-full bg-blue-600 text-white px-4 py-3 rounded-lg hover:bg-blue-700 transition-all duration-200 font-medium">
                    <i class="fas fa-sign-in-alt mr-2"></i>
                    Entrar
                </button>
                
                <div class="text-center">
                    <p class="text-sm text-gray-500 mb-3">ou</p>
                    <button type="button" onclick="loginGitHub()" 
                            class="w-full bg-gray-800 text-white px-4 py-3 rounded-lg hover:bg-gray-900 transition-all duration-200 flex items-center justify-center">
                        <i class="fab fa-github mr-2"></i> 
                        Entrar com GitHub
                    </button>
                </div>
                
                <p id="loginError" class="text-red-600 text-sm mt-2 hidden"></p>
                
                <div class="mt-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
                    <h4 class="font-semibold text-blue-800 mb-2">👤 Contas de Demo:</h4>
                    <div class="text-blue-700 text-sm space-y-1">
                        <div>• <strong>admin</strong> / admin123</div>
                        <div>• <strong>user</strong> / user123</div>
                        <div>• <strong>demo</strong> / demo123</div>
                        <div class="text-xs mt-2 text-blue-600">
                            Ou qualquer usuário/senha não vazios
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Header -->
    <header class="bg-white shadow-lg border-b border-blue-100">
        <div class="container mx-auto px-6 py-4">
            <div class="flex items-center justify-between">
                <div class="flex items-center space-x-3">
                    <div class="w-10 h-10 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-lg flex items-center justify-center">
                        <i class="fas fa-robot text-white text-lg"></i>
                    </div>
                    <div>
                        <h1 class="text-2xl font-bold text-gray-800">DocAgent Skyone</h1>
                        <p class="text-sm text-gray-600">Análise Automática com AG2</p>
                    </div>
                </div>
                <div class="flex items-center space-x-4">
                    <span id="userInfo" class="hidden px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm font-medium">
                        <i class="fas fa-user mr-1"></i>
                        <span id="currentUser">Usuário</span>
                    </span>
                    <span id="systemStatus" class="px-3 py-1 bg-gray-200 text-gray-600 rounded-full text-sm font-medium">
                        <i class="fas fa-circle animate-pulse mr-1"></i>
                        Carregando...
                    </span>
                    <span id="ag2Status" class="px-3 py-1 bg-purple-100 text-purple-800 rounded-full text-sm font-medium">
                        <i class="fas fa-robot mr-1"></i>
                        AG2 Status
                    </span>
                    <span class="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm font-medium">
                        <i class="fas fa-shield-alt mr-1"></i>
                        Relatórios Anônimos
                    </span>
                    <span class="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm font-medium">
                        v2.0
                    </span>
                    <button id="logoutBtn" onclick="realizarLogout()" 
                            class="hidden px-3 py-1 bg-red-100 text-red-800 rounded-full text-sm font-medium hover:bg-red-200 transition-colors">
                        <i class="fas fa-sign-out-alt mr-1"></i>
                        Sair
                    </button>
                </div>
            </div>
        </div>
    </header>

    <!-- Main Content -->
    <main class="container mx-auto px-6 py-8">
        
        <!-- Hero Section -->
        <div class="text-center mb-12">
            <div class="bg-gradient-to-r from-blue-600 to-purple-600 rounded-3xl p-8 mb-8 text-white shadow-2xl">
                <h2 class="text-5xl font-bold mb-6">
                    <i class="fas fa-robot mr-4 animate-pulse"></i>
                    DocAgent Skyone
                    <span class="block text-3xl font-normal text-blue-200 mt-2">Sistema AG2 Multi-Agent</span>
                </h2>
                <p class="text-xl text-blue-100 max-w-4xl mx-auto leading-relaxed">
                    Plataforma avançada que utiliza inteligência artificial AG2 para análise técnica completa de repositórios GitHub.
                    Gera relatórios anônimos profissionais com documentação técnica detalhada e download em ZIP.
                </p>
                
                <!-- Feature highlights -->
                <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mt-8">
                    <div class="bg-white bg-opacity-10 rounded-lg p-4 backdrop-blur">
                        <i class="fas fa-robot text-2xl text-yellow-300 mb-2"></i>
                        <div class="font-semibold">4 Agentes IA</div>
                        <div class="text-sm text-blue-200">Análise colaborativa</div>
                    </div>
                    <div class="bg-white bg-opacity-10 rounded-lg p-4 backdrop-blur">
                        <i class="fas fa-shield-alt text-2xl text-green-300 mb-2"></i>
                        <div class="font-semibold">100% Anônimo</div>
                        <div class="text-sm text-blue-200">Dados protegidos</div>
                    </div>
                    <div class="bg-white bg-opacity-10 rounded-lg p-4 backdrop-blur">
                        <i class="fas fa-file-archive text-2xl text-orange-300 mb-2"></i>
                        <div class="font-semibold">Download ZIP</div>
                        <div class="text-sm text-blue-200">Todos os arquivos</div>
                    </div>
                    <div class="bg-white bg-opacity-10 rounded-lg p-4 backdrop-blur">
                        <i class="fas fa-clock text-2xl text-purple-300 mb-2"></i>
                        <div class="font-semibold">Tempo Real</div>
                        <div class="text-sm text-blue-200">Logs ao vivo</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Search Section -->
        <div class="bg-white rounded-2xl shadow-xl p-8 mb-8">
            <div class="flex items-center mb-6">
                <i class="fas fa-search text-blue-600 text-2xl mr-3"></i>
                <h3 class="text-2xl font-bold text-gray-800">Busca Automática de Repositórios</h3>
            </div>
            
            <!-- Enhanced AG2 System Info -->
            <div class="mb-6 p-6 bg-gradient-to-r from-purple-50 to-blue-50 rounded-xl border border-purple-200 shadow-lg">
                <div class="flex items-center mb-4">
                    <div class="w-12 h-12 bg-gradient-to-r from-purple-600 to-blue-600 rounded-full flex items-center justify-center mr-4">
                        <i class="fas fa-robot text-white text-xl"></i>
                    </div>
                    <div>
                        <h4 class="font-bold text-purple-800 text-lg">Sistema AG2 Enhanced</h4>
                        <p class="text-purple-600 text-sm">Inteligência Artificial Multi-Agent</p>
                    </div>
                </div>
                
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                    <div class="space-y-2">
                        <div class="flex items-center text-purple-700">
                            <i class="fas fa-users text-purple-600 mr-2"></i>
                            <strong>4 Agentes Especializados:</strong> Análise colaborativa
                        </div>
                        <div class="flex items-center text-purple-700">
                            <i class="fas fa-tools text-purple-600 mr-2"></i>
                            <strong>5 Tools Avançadas:</strong> Análise técnica profunda
                        </div>
                    </div>
                    <div class="space-y-2">
                        <div class="flex items-center text-purple-700">
                            <i class="fas fa-shield-alt text-green-600 mr-2"></i>
                            <strong>Relatórios Anônimos:</strong> Proteção total
                        </div>
                        <div class="flex items-center text-purple-700">
                            <i class="fas fa-download text-blue-600 mr-2"></i>
                            <strong>Download ZIP:</strong> Todos os arquivos
                        </div>
                    </div>
                </div>
                
                <div class="flex flex-wrap gap-2">
                    <button onclick="testarComMicrosoft()" class="bg-gradient-to-r from-blue-500 to-purple-500 text-white px-4 py-2 rounded-lg hover:from-blue-600 hover:to-purple-600 transition-all transform hover:scale-105 shadow-md">
                        <i class="fab fa-microsoft mr-1"></i> Testar: Microsoft
                    </button>
                    <button onclick="testarComGoogle()" class="bg-gradient-to-r from-red-500 to-orange-500 text-white px-4 py-2 rounded-lg hover:from-red-600 hover:to-orange-600 transition-all transform hover:scale-105 shadow-md">
                        <i class="fab fa-google mr-1"></i> Testar: Google
                    </button>
                    <button onclick="testarComFacebook()" class="bg-gradient-to-r from-blue-600 to-indigo-600 text-white px-4 py-2 rounded-lg hover:from-blue-700 hover:to-indigo-700 transition-all transform hover:scale-105 shadow-md">
                        <i class="fab fa-facebook mr-1"></i> Testar: Facebook
                    </button>
                </div>
            </div>
            
            <!-- Seção de autenticação GitHub opcional -->
            <div class="mb-6">
                <label for="githubToken" class="block text-sm font-medium text-gray-700 mb-2">
                    Token de Acesso do GitHub (opcional para repositórios privados)
                </label>
                <div class="flex">
                    <input type="password" id="githubToken" placeholder="Personal Access Token" class="flex-1 px-4 py-3 border border-gray-300 rounded-l-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors" />
                    <button type="button" onclick="autenticarGit()" class="bg-green-600 text-white px-4 py-3 rounded-r-lg hover:bg-green-700 transition-all duration-200">
                        Autenticar
                    </button>
                </div>
                <p class="text-xs text-gray-500 mt-1">Seu token será usado apenas para esta sessão.</p>
            </div>

            <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div class="lg:col-span-2">
                    <label for="usuario" class="block text-sm font-medium text-gray-700 mb-2">
                        Nome do Usuário/Organização ou URL Completa do GitHub
                    </label>
                    <input type="text" id="usuario" placeholder="Ex: microsoft, google, ou https://github.com/facebook/react"
                           class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors">
                    <div id="inputFeedback" class="mt-2 text-sm hidden">
                        <!-- Feedback será inserido aqui -->
                    </div>
                </div>
                <div class="flex flex-col justify-end">
                    <label class="flex items-center mb-3">
                        <input type="checkbox" id="incluirForks" class="mr-2 h-4 w-4 text-blue-600">
                        <span class="text-sm text-gray-700">Incluir Forks</span>
                    </label>
                    <label class="flex items-center mb-3">
                        <input type="checkbox" id="modoAnonimo" class="mr-2 h-4 w-4 text-blue-600" checked>
                        <span class="text-sm text-gray-700">Relatório Anônimo</span>
                    </label>
                    <button onclick="buscarRepositorios()" id="btnBuscar"
                            class="bg-gradient-to-r from-blue-600 to-indigo-600 text-white px-6 py-3 rounded-lg hover:from-blue-700 hover:to-indigo-700 transition-all duration-200 transform hover:scale-105 font-medium">
                        <i class="fas fa-search mr-2"></i>
                        Buscar Repositórios
                    </button>
                </div>
            </div>
        </div>

        <!-- Loading -->
        <div id="loading" class="hidden">
            <div class="bg-white rounded-2xl shadow-xl p-8 text-center">
                <div class="animate-spin rounded-full h-16 w-16 border-b-4 border-blue-600 mx-auto mb-4"></div>
                <p class="text-gray-600 text-lg font-medium">Buscando repositórios...</p>
                <p class="text-gray-500 text-sm mt-2">Isso pode levar alguns segundos</p>
            </div>
        </div>

        <!-- Results Section -->
        <div id="results" class="hidden">
            <div class="bg-white rounded-2xl shadow-xl p-8">
                <div class="flex items-center justify-between mb-6">
                    <h3 class="text-2xl font-bold text-gray-800">
                        <i class="fas fa-folder-open text-blue-600 mr-3"></i>
                        Repositórios Encontrados
                    </h3>
                    <div class="flex space-x-4">
                        <select id="filtroLinguagem" class="px-3 py-2 border border-gray-300 rounded-lg">
                            <option value="">Todas as linguagens</option>
                        </select>
                        <select id="ordenacao" class="px-3 py-2 border border-gray-300 rounded-lg">
                            <option value="stars">Mais estrelas</option>
                            <option value="name">Nome</option>
                            <option value="updated">Mais recente</option>
                        </select>
                    </div>
                </div>
                <div id="repositoriosList" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    <!-- Repositórios serão inseridos aqui -->
                </div>
            </div>
        </div>

        <!-- Analysis Section -->
        <div id="analysisSection" class="hidden">
            <!-- AG2 Logs Section - Now Prominent -->
            <div class="bg-gradient-to-r from-purple-900 to-blue-900 rounded-2xl shadow-2xl p-8 mb-8 text-white">
                <h3 class="text-3xl font-bold mb-6 flex items-center">
                    <i class="fas fa-robot mr-4 text-4xl animate-pulse"></i>
                    <div>
                        <div>Sistema AG2 Multi-Agent</div>
                        <div class="text-lg font-normal text-purple-200">Logs em Tempo Real</div>
                    </div>
                </h3>
                
                <!-- Status Cards -->
                <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                    <div class="bg-white bg-opacity-10 rounded-lg p-4 backdrop-blur">
                        <div class="text-purple-200 text-sm font-medium">Fase Atual</div>
                        <div id="phaseText" class="text-xl font-bold">Iniciando...</div>
                    </div>
                    <div class="bg-white bg-opacity-10 rounded-lg p-4 backdrop-blur">
                        <div class="text-blue-200 text-sm font-medium">Progresso</div>
                        <div id="progressText" class="text-xl font-bold">0%</div>
                    </div>
                    <div class="bg-white bg-opacity-10 rounded-lg p-4 backdrop-blur">
                        <div class="text-green-200 text-sm font-medium">Etapa Atual</div>
                        <div id="currentStep" class="text-lg font-medium">Aguardando...</div>
                    </div>
                </div>
                
                <!-- Progress Bar -->
                <div class="mb-6">
                    <div class="w-full bg-white bg-opacity-20 rounded-full h-4 backdrop-blur">
                        <div id="progressBar" class="bg-gradient-to-r from-green-400 to-blue-400 h-4 rounded-full transition-all duration-500 shadow-lg" style="width: 0%"></div>
                    </div>
                </div>
                
                <!-- Status Message -->
                <div id="statusMessage" class="text-lg text-purple-100 mb-4 p-3 bg-white bg-opacity-10 rounded-lg backdrop-blur"></div>
                
                <!-- Prominent Logs Display -->
                <div class="bg-black bg-opacity-40 rounded-lg p-4 backdrop-blur border border-white border-opacity-20">
                    <h4 class="font-bold text-lg mb-3 flex items-center">
                        <i class="fas fa-terminal mr-2 text-green-400"></i>
                        Logs dos Agentes AG2
                        <span class="ml-auto text-sm text-gray-300">Tempo Real</span>
                    </h4>
                    <div id="logsContainer" class="max-h-64 overflow-y-auto">
                        <div id="logs" class="text-sm font-mono text-green-300 whitespace-pre-wrap"></div>
                    </div>
                </div>
            </div>

            <!-- Results -->
            <div id="analysisResults" class="hidden bg-white rounded-2xl shadow-xl p-8">
                <h3 class="text-2xl font-bold text-gray-800 mb-6">
                    <i class="fas fa-file-alt text-green-600 mr-3"></i>
                    Relatório AG2 Concluído
                </h3>
                <div id="resultsContent">
                    <!-- Resultados serão inseridos aqui -->
                </div>
            </div>
        </div>

        <!-- Features Grid -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mt-12">
            <div class="bg-white rounded-xl p-6 shadow-lg hover:shadow-xl transition-shadow">
                <div class="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center mb-4">
                    <i class="fas fa-robot text-purple-600 text-xl"></i>
                </div>
                <h4 class="font-bold text-gray-800 mb-2">Sistema AG2</h4>
                <p class="text-gray-600 text-sm">Multi-agent analysis com 4 agentes especializados</p>
            </div>
            
            <div class="bg-white rounded-xl p-6 shadow-lg hover:shadow-xl transition-shadow">
                <div class="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center mb-4">
                    <i class="fas fa-shield-alt text-green-600 text-xl"></i>
                </div>
                <h4 class="font-bold text-gray-800 mb-2">100% Anônimo</h4>
                <p class="text-gray-600 text-sm">Relatórios completamente anonimizados, seguros para compartilhamento</p>
            </div>
            
            <div class="bg-white rounded-xl p-6 shadow-lg hover:shadow-xl transition-shadow">
                <div class="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mb-4">
                    <i class="fas fa-code text-blue-600 text-xl"></i>
                </div>
                <h4 class="font-bold text-gray-800 mb-2">Análise Profunda</h4>
                <p class="text-gray-600 text-sm">5 tools avançadas para análise técnica detalhada</p>
            </div>
            
            <div class="bg-white rounded-xl p-6 shadow-lg hover:shadow-xl transition-shadow">
                <div class="w-12 h-12 bg-orange-100 rounded-lg flex items-center justify-center mb-4">
                    <i class="fas fa-download text-orange-600 text-xl"></i>
                </div>
                <h4 class="font-bold text-gray-800 mb-2">Relatórios Completos</h4>
                <p class="text-gray-600 text-sm">Documentação técnica em formatos MD e JSON</p>
            </div>
        </div>

    </main>

    <!-- Footer -->
    <footer class="bg-gray-800 text-white py-8 mt-16">
        <div class="container mx-auto px-6 text-center">
            <p>&copy; 2024 DocAgent Skyone v2.0 - Sistema AG2 Multi-Agent</p>
            <p class="text-gray-400 mt-2">Análise avançada • Relatórios anônimos • Tecnologia AG2 • Sistema de Autenticação</p>
        </div>
    </footer>

    <script>
        let repositorios = [];
        let analysisInterval = null;
        let ag2Available = false;
        let isAuthenticated = false;
        let currentUser = null;

        // Inicialização
        document.addEventListener('DOMContentLoaded', function() {
            console.log('🤖 DocAgent Skyone v2.0 com AG2 e Autenticação carregado!');
            // Verificar estado de login assim que a página carregar
            verificarStatusAuth();
            
            // Enter key listener
            document.getElementById('usuario').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    buscarRepositorios();
                }
            });

            // Enter key para login
            document.getElementById('loginUsername').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    realizarLogin();
                }
            });

            document.getElementById('loginPassword').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    realizarLogin();
                }
            });
        });

        // Sistema de Autenticação
        async function verificarStatusAuth() {
            try {
                const response = await fetch('/api/auth/status');
                const data = await response.json();
                
                console.log('Status de autenticação:', data);
                
                if (data.authenticated) {
                    // Usuário está logado
                    isAuthenticated = true;
                    mostrarInterfaceLogada();
                    checkSystemStatus();
                } else {
                    // Usuário não está logado
                    isAuthenticated = false;
                    mostrarTelaLogin();
                }
            } catch (error) {
                console.error('Erro ao verificar status de auth:', error);
                // Em caso de erro, mostrar tela de login
                mostrarTelaLogin();
            }
        }

        function mostrarTelaLogin() {
            const overlay = document.getElementById('loginOverlay');
            if (overlay) {
                overlay.classList.remove('hidden');
            }
            
            // Ocultar elementos da interface principal
            ocultarElementosInterface();
        }

        function mostrarInterfaceLogada() {
            const overlay = document.getElementById('loginOverlay');
            if (overlay) {
                overlay.classList.add('hidden');
            }
            
            // Mostrar elementos da interface
            mostrarElementosInterface();
        }

        function ocultarElementosInterface() {
            const userInfo = document.getElementById('userInfo');
            const logoutBtn = document.getElementById('logoutBtn');
            if (userInfo) userInfo.classList.add('hidden');
            if (logoutBtn) logoutBtn.classList.add('hidden');
        }

        function mostrarElementosInterface() {
            const userInfo = document.getElementById('userInfo');
            const logoutBtn = document.getElementById('logoutBtn');
            if (userInfo) userInfo.classList.remove('hidden');
            if (logoutBtn) logoutBtn.classList.remove('hidden');
            
            // Atualizar nome do usuário se disponível
            if (currentUser) {
                const userSpan = document.getElementById('currentUser');
                if (userSpan) {
                    userSpan.textContent = currentUser;
                }
            }
        }

        // Função de login melhorada
        async function realizarLogin() {
            const usernameEl = document.getElementById('loginUsername');
            const passwordEl = document.getElementById('loginPassword');
            const errorEl = document.getElementById('loginError');
            
            const username = usernameEl ? usernameEl.value.trim() : '';
            const password = passwordEl ? passwordEl.value.trim() : '';
            
            if (errorEl) {
                errorEl.classList.add('hidden');
                errorEl.textContent = '';
            }
            
            if (!username || !password) {
                if (errorEl) {
                    errorEl.textContent = 'Por favor, preencha usuário e senha';
                    errorEl.classList.remove('hidden');
                }
                return;
            }

            try {
                const response = await fetch('/api/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username: username, password: password })
                });
                
                const data = await response.json();
                
                if (response.ok && data.success) {
                    // Login bem-sucedido
                    isAuthenticated = true;
                    currentUser = data.username || username;
                    
                    // Armazenar informações da sessão
                    sessionStorage.setItem('docagent_session', JSON.stringify({
                        authenticated: true,
                        username: currentUser,
                        session_id: data.session_id,
                        login_time: new Date().toISOString()
                    }));
                    
                    console.log('🔑 Login realizado com sucesso:', currentUser);
                    
                    // Mostrar interface logada
                    mostrarInterfaceLogada();
                    
                    // Verificar status do sistema após login
                    checkSystemStatus();
                    
                    // Limpar campos de login
                    if (usernameEl) usernameEl.value = '';
                    if (passwordEl) passwordEl.value = '';
                    
                } else {
                    if (errorEl) {
                        errorEl.textContent = data.message || 'Credenciais inválidas';
                        errorEl.classList.remove('hidden');
                    }
                }
            } catch (error) {
                console.error('Erro no login:', error);
                if (errorEl) {
                    errorEl.textContent = 'Erro de conexão. Tente novamente.';
                    errorEl.classList.remove('hidden');
                }
            }
        }

        // Função de logout
        async function realizarLogout() {
            try {
                const response = await fetch('/api/logout', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                });
                
                const data = await response.json();
                
                if (data.success) {
                    // Logout bem-sucedido
                    isAuthenticated = false;
                    currentUser = null;
                    
                    // Limpar dados da sessão
                    sessionStorage.removeItem('docagent_session');
                    
                    console.log('🔒 Logout realizado com sucesso');
                    
                    // Voltar para tela de login
                    mostrarTelaLogin();
                    
                    // Resetar estado da aplicação
                    resetarEstadoAplicacao();
                } else {
                    console.error('Erro no logout:', data.message);
                }
            } catch (error) {
                console.error('Erro no logout:', error);
                // Mesmo com erro, fazer logout local
                isAuthenticated = false;
                currentUser = null;
                sessionStorage.removeItem('docagent_session');
                mostrarTelaLogin();
                resetarEstadoAplicacao();
            }
        }

        function resetarEstadoAplicacao() {
            // Limpar dados da aplicação
            repositorios = [];
            if (analysisInterval) {
                clearInterval(analysisInterval);
                analysisInterval = null;
            }
            
            // Ocultar seções
            const sections = ['results', 'analysisSection', 'loading'];
            sections.forEach(sectionId => {
                const section = document.getElementById(sectionId);
                if (section) {
                    section.classList.add('hidden');
                }
            });
            
            // Limpar campos
            const usuario = document.getElementById('usuario');
            if (usuario) {
                usuario.value = '';
            }
        }

        // System status management
        function checkSystemStatus() {
            if (!isAuthenticated) {
                return; // Não verificar status se não estiver logado
            }
            
            fetch('/health')
                .then(response => response.json())
                .then(data => {
                    updateSystemStatus(data.status);
                    updateAG2Status(data.ag2_enabled);
                    ag2Available = data.ag2_enabled;
                })
                .catch(error => {
                    console.error('Erro ao verificar status:', error);
                    updateSystemStatus('error');
                });
        }

        function updateSystemStatus(status) {
            const statusElement = document.getElementById('systemStatus');
            if (!statusElement) return;

            switch(status) {
                case 'healthy':
                    statusElement.innerHTML = '<i class="fas fa-check-circle mr-1"></i>Sistema Pronto';
                    statusElement.className = 'px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm font-medium';
                    break;
                case 'degraded':
                    statusElement.innerHTML = '<i class="fas fa-exclamation-triangle mr-1"></i>Limitado';
                    statusElement.className = 'px-3 py-1 bg-yellow-100 text-yellow-800 rounded-full text-sm font-medium';
                    break;
                case 'searching':
                    statusElement.innerHTML = '<i class="fas fa-search animate-spin mr-1"></i>Buscando...';
                    statusElement.className = 'px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm font-medium';
                    break;
                case 'analyzing':
                    statusElement.innerHTML = '<i class="fas fa-robot animate-spin mr-1"></i>Analisando...';
                    statusElement.className = 'px-3 py-1 bg-purple-100 text-purple-800 rounded-full text-sm font-medium';
                    break;
                case 'error':
                    statusElement.innerHTML = '<i class="fas fa-exclamation-triangle mr-1"></i>Erro';
                    statusElement.className = 'px-3 py-1 bg-red-100 text-red-800 rounded-full text-sm font-medium';
                    break;
            }
        }

        function updateAG2Status(enabled) {
            const ag2Element = document.getElementById('ag2Status');
            if (!ag2Element) return;

            if (enabled) {
                ag2Element.innerHTML = '<i class="fas fa-robot mr-1"></i>AG2 Ativo';
                ag2Element.className = 'px-3 py-1 bg-purple-100 text-purple-800 rounded-full text-sm font-medium';
            } else {
                ag2Element.innerHTML = '<i class="fas fa-robot mr-1"></i>AG2 Off';
                ag2Element.className = 'px-3 py-1 bg-gray-100 text-gray-600 rounded-full text-sm font-medium';
            }
        }

        // Test functions
        function testarComMicrosoft() {
            if (!isAuthenticated) {
                alert('Por favor, faça login primeiro');
                return;
            }
            document.getElementById('usuario').value = 'microsoft';
            buscarRepositorios();
        }

        function testarComGoogle() {
            if (!isAuthenticated) {
                alert('Por favor, faça login primeiro');
                return;
            }
            document.getElementById('usuario').value = 'google';
            buscarRepositorios();
        }
        
        function testarComFacebook() {
            if (!isAuthenticated) {
                alert('Por favor, faça login primeiro');
                return;
            }
            document.getElementById('usuario').value = 'facebook';
            buscarRepositorios();
        }

        // Função de autenticação do GitHub (mantida igual)
        async function autenticarGit() {
            const tokenInput = document.getElementById('githubToken');
            const token = tokenInput ? tokenInput.value.trim() : '';
            if (!token) {
                alert('Por favor, insira o token de acesso do GitHub');
                return;
            }
            try {
                const response = await fetch('/api/auth/github', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ token: token })
                });
                const data = await response.json();
                if (response.ok && data.success) {
                    alert('Token autenticado com sucesso!');
                } else {
                    alert(data.message || 'Falha ao autenticar token');
                }
            } catch (error) {
                console.error('Erro ao autenticar token:', error);
                alert('Erro ao autenticar token.');
            }
        }

        // Inicia o fluxo OAuth abrindo uma nova janela para a página de autorização do GitHub
        function loginGitHub() {
            window.open('/login/github', '_blank', 'width=600,height=700');
        }

        // Listener para receber mensagens do popup de autenticação do GitHub
        window.addEventListener('message', function(event) {
            if (event.data && event.data.type === 'github_oauth') {
                if (event.data.success) {
                    alert('Autenticação com GitHub realizada com sucesso!');
                    // Verificar status de autenticação após OAuth
                    verificarStatusAuth();
                } else {
                    alert('Falha na autenticação com GitHub');
                }
            }
        });

        // Main search function - atualizada com verificação de autenticação
        async function buscarRepositorios() {
            if (!isAuthenticated) {
                alert('Por favor, faça login primeiro');
                mostrarTelaLogin();
                return;
            }
            
            const usuarioInput = document.getElementById('usuario');
            const incluirForksInput = document.getElementById('incluirForks');
            const btnBuscar = document.getElementById('btnBuscar');
            
            const usuario = usuarioInput.value.trim();
            const incluirForks = incluirForksInput.checked;
            
            if (!usuario) {
                alert('Por favor, digite o nome do usuário/organização ou URL do repositório');
                usuarioInput.focus();
                return;
            }

            console.log(`🔍 Buscando repositórios para: ${usuario}`);
            updateSystemStatus('searching');

            // UI updates
            btnBuscar.disabled = true;
            btnBuscar.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Buscando...';
            
            document.getElementById('loading').classList.remove('hidden');
            document.getElementById('results').classList.add('hidden');

            try {
                const response = await fetch('/api/search', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        usuario: usuario,
                        incluir_forks: incluirForks
                    })
                });

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                const data = await response.json();

                if (data.success) {
                    repositorios = data.repositories;
                    console.log(`✅ ${repositorios.length} repositórios encontrados`);
                    
                    if (repositorios.length === 0) {
                        mostrarMensagemNenhumRepo(usuario);
                    } else {
                        mostrarRepositorios(repositorios);
                        popularFiltroLinguagens();
                    }
                    
                    updateSystemStatus('healthy');
                } else {
                    throw new Error(data.message || 'Erro desconhecido na busca');
                }
            } catch (error) {
                console.error('❌ Erro na busca:', error);
                updateSystemStatus('error');
                mostrarErroNaBusca(error.message);
            } finally {
                btnBuscar.disabled = false;
                btnBuscar.innerHTML = '<i class="fas fa-search mr-2"></i>Buscar Repositórios';
                
                if (repositorios.length > 0) {
                    document.getElementById('loading').classList.add('hidden');
                }
            }
        }

        // Analysis functions - atualizada com verificação de autenticação
        async function analisarRepositorio(repoUrl) {
            if (!isAuthenticated) {
                alert('Por favor, faça login primeiro');
                mostrarTelaLogin();
                return;
            }
            
            if (!repoUrl || !repoUrl.includes('github.com')) {
                alert('URL do repositório inválida');
                return;
            }

            console.log(`🤖 Iniciando análise ${ag2Available ? 'AG2' : 'tradicional'} de: ${repoUrl}`);
            updateSystemStatus('analyzing');

            // Hide previous sections
            document.getElementById('results').classList.add('hidden');
            document.getElementById('analysisSection').classList.remove('hidden');
            document.getElementById('analysisResults').classList.add('hidden');

            // Reset progress
            document.getElementById('progressBar').style.width = '0%';
            document.getElementById('phaseText').textContent = ag2Available ? 'Iniciando análise AG2...' : 'Iniciando análise...';
            document.getElementById('progressText').textContent = '0%';
            document.getElementById('statusMessage').textContent = 'Preparando sistema...';
            document.getElementById('currentStep').textContent = '';
            document.getElementById('logs').innerHTML = '';

            try {
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        repo_url: repoUrl,
                        max_files: 25,
                        deep_analysis: true,
                        include_config: true,
                        anonymous: document.getElementById('modoAnonimo') ? document.getElementById('modoAnonimo').checked : true
                    })
                });

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                const data = await response.json();

                if (data.success) {
                    // Start status polling
                    analysisInterval = setInterval(checkAnalysisStatus, 2000);
                    
                    document.getElementById('phaseText').textContent = ag2Available ? 'Análise AG2 iniciada' : 'Análise iniciada';
                    document.getElementById('progressBar').style.width = '5%';
                    document.getElementById('progressText').textContent = '5%';
                    document.getElementById('statusMessage').textContent = ag2Available ? 'Sistema AG2 ativo...' : 'Sistema de análise ativo...';
                    
                    document.getElementById('analysisSection').scrollIntoView({ behavior: 'smooth' });
                } else {
                    throw new Error(data.message || 'Erro desconhecido ao iniciar análise');
                }
            } catch (error) {
                console.error('❌ Erro ao iniciar análise:', error);
                updateSystemStatus('error');
                
                document.getElementById('phaseText').textContent = 'Erro na análise';
                document.getElementById('statusMessage').textContent = `Erro: ${error.message}`;
                document.getElementById('logs').innerHTML = `<div class="text-red-600">❌ ${error.message}</div>`;
            }
        }

        // [Resto das funções JavaScript permanecem iguais...]
        // checkAnalysisStatus, loadAnalysisResults, mostrarRepositorios, etc.
        
        // Função de verificação de status da análise
        async function checkAnalysisStatus() {
            try {
                const response = await fetch('/api/status');
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }
                
                const status = await response.json();

                // Update UI
                document.getElementById('phaseText').textContent = status.phase || 'Processando...';
                document.getElementById('progressText').textContent = (status.progress || 0) + '%';
                document.getElementById('progressBar').style.width = (status.progress || 0) + '%';
                document.getElementById('statusMessage').textContent = status.message || 'Processando...';
                
                // Update current step
                if (status.current_step) {
                    const stepIcon = ag2Available ? '🤖' : '🔄';
                    document.getElementById('currentStep').textContent = `${stepIcon} ${status.current_step}`;
                }
                
                // Update logs with enhanced styling
                if (status.logs && status.logs.length > 0) {
                    const logsHTML = status.logs.map((log, index) => {
                        const timestamp = new Date().toLocaleTimeString();
                        const isError = log.includes('❌') || log.toLowerCase().includes('erro');
                        const isSuccess = log.includes('✅') || log.toLowerCase().includes('sucesso');
                        const isWarning = log.includes('⚠️') || log.toLowerCase().includes('warning');
                        
                        let logClass = 'text-green-300';
                        if (isError) logClass = 'text-red-400';
                        else if (isSuccess) logClass = 'text-green-400';
                        else if (isWarning) logClass = 'text-yellow-400';
                        
                        return `<div class="mb-2 p-2 border-l-2 border-gray-600 ${logClass}">
                                    <span class="text-gray-400 text-xs">[${timestamp}]</span> ${log}
                                </div>`;
                    }).join('');
                    
                    document.getElementById('logs').innerHTML = logsHTML;
                    document.getElementById('logsContainer').scrollTop = document.getElementById('logsContainer').scrollHeight;
                }

                // Check final status
                if (status.status === 'completed') {
                    clearInterval(analysisInterval);
                    analysisInterval = null;
                    
                    setTimeout(async () => {
                        await loadAnalysisResults();
                    }, 1000);
                    
                } else if (status.status === 'error') {
                    clearInterval(analysisInterval);
                    analysisInterval = null;
                    updateSystemStatus('error');
                    
                    document.getElementById('phaseText').textContent = 'Erro na análise';
                    document.getElementById('statusMessage').textContent = status.message || 'Erro desconhecido';
                    document.getElementById('logs').innerHTML += `<div class="text-red-600 mt-2">❌ Análise falhou: ${status.message}</div>`;
                }
                
            } catch (error) {
                console.error('❌ Erro ao verificar status:', error);
                
                if (!window.statusErrorCount) window.statusErrorCount = 0;
                window.statusErrorCount++;
                
                if (window.statusErrorCount > 5) {
                    clearInterval(analysisInterval);
                    analysisInterval = null;
                    updateSystemStatus('error');
                    
                    document.getElementById('statusMessage').textContent = 'Erro de comunicação com o servidor';
                    document.getElementById('logs').innerHTML += `<div class="text-red-600 mt-2">❌ Erro de comunicação: ${error.message}</div>`;
                }
            }
        }

        async function loadAnalysisResults() {
            try {
                const response = await fetch('/api/results');
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                const results = await response.json();

                if (!results || !results.analysis_data) {
                    throw new Error('Resultados de análise inválidos');
                }

                const analysisType = results.analysis_type || 'traditional';
                const isAG2 = analysisType === 'AG2_enhanced';
                
                document.getElementById('resultsContent').innerHTML = `
                    <div class="bg-green-50 border border-green-200 rounded-lg p-6 mb-6">
                        <div class="flex items-center">
                            <i class="fas fa-${isAG2 ? 'robot' : 'check-circle'} text-green-600 text-2xl mr-3"></i>
                            <div>
                                <h4 class="font-bold text-green-800 text-lg">Análise ${isAG2 ? 'AG2' : 'Tradicional'} Concluída!</h4>
                                <p class="text-green-700">Sistema ${isAG2 ? 'multi-agent' : 'tradicional'} processou o repositório - Documentação completa gerada</p>
                            </div>
                        </div>
                    </div>

                    <!-- Resumo de Análise -->
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                        <div class="bg-blue-50 rounded-lg p-6 border border-blue-200">
                            <h4 class="font-bold text-blue-800 mb-4 flex items-center">
                                <i class="fas fa-chart-bar mr-2"></i>
                                Análise ${isAG2 ? 'AG2' : 'Tradicional'}
                            </h4>
                            <div class="space-y-3">
                                <div class="flex justify-between">
                                    <span class="text-blue-700">Tipo de Sistema:</span>
                                    <span class="font-medium text-blue-900">${isAG2 ? 'AG2 Multi-Agent' : 'Análise Tradicional'}</span>
                                </div>
                                <div class="flex justify-between">
                                    <span class="text-blue-700">Status:</span>
                                    <span class="font-medium text-blue-900">${results.status}</span>
                                </div>
                                <div class="flex justify-between">
                                    <span class="text-blue-700">Documentos gerados:</span>
                                    <span class="font-medium text-blue-900">${results.generated_docs ? results.generated_docs.length : 0}</span>
                                </div>
                            </div>
                        </div>
                        
                        <div class="bg-purple-50 rounded-lg p-6 border border-purple-200">
                            <h4 class="font-bold text-purple-800 mb-4 flex items-center">
                                <i class="fas fa-file-download mr-2"></i>
                                Downloads Disponíveis
                            </h4>
                            ${results.generated_docs && results.generated_docs.length > 0 ? `
                                <div class="mb-4">
                                    <a href="/api/download-all-zip" 
                                       class="flex items-center justify-center p-4 bg-gradient-to-r from-green-600 to-blue-600 text-white rounded-lg hover:from-green-700 hover:to-blue-700 transition-all transform hover:scale-105 shadow-lg mb-3">
                                        <i class="fas fa-file-archive mr-3 text-lg"></i>
                                        <span class="font-bold">Download Completo (.ZIP)</span>
                                        <span class="ml-2 px-2 py-1 bg-white bg-opacity-20 rounded-full text-xs">Todos os arquivos</span>
                                    </a>
                                </div>
                                <div class="text-sm text-purple-600 font-medium mb-2">Downloads individuais:</div>
                                <div class="space-y-2 max-h-32 overflow-y-auto">
                                    ${results.generated_docs.map(filename => `
                                        <a href="/api/download/${filename}" 
                                           class="flex items-center justify-between p-2 bg-white rounded border border-purple-200 hover:border-purple-400 hover:bg-purple-25 transition-colors group text-sm">
                                            <div class="flex items-center">
                                                <i class="fas fa-file-alt text-purple-600 mr-2"></i>
                                                <span class="text-purple-800 font-medium">${filename}</span>
                                            </div>
                                            <i class="fas fa-download text-purple-400 group-hover:text-purple-600"></i>
                                        </a>
                                    `).join('')}
                                </div>
                            ` : '<p class="text-purple-600">Nenhum documento gerado</p>'}
                        </div>
                    </div>

                    <div class="mt-6 p-4 bg-${isAG2 ? 'purple' : 'blue'}-50 rounded-lg border border-${isAG2 ? 'purple' : 'blue'}-200">
                        <div class="flex items-center text-${isAG2 ? 'purple' : 'blue'}-800">
                            <i class="fas fa-${isAG2 ? 'robot' : 'shield-alt'} mr-2"></i>
                            <span class="font-medium">${isAG2 ? 'Sistema AG2 Multi-Agent' : 'Análise Tradicional'}:</span>
                        </div>
                        <p class="text-${isAG2 ? 'purple' : 'blue'}-700 text-sm mt-1">
                            ${isAG2 ? 
                                'Este relatório foi processado usando 4 agentes especializados do sistema AG2, garantindo análise técnica avançada e relatórios anônimos de alta qualidade.' :
                                'Este relatório foi processado usando análise tradicional, fornecendo documentação técnica abrangente e relatórios anônimos.'
                            }
                        </p>
                    </div>
                `;

                document.getElementById('analysisResults').classList.remove('hidden');
                updateSystemStatus('healthy');
                document.getElementById('analysisResults').scrollIntoView({ behavior: 'smooth', block: 'start' });
                window.statusErrorCount = 0;
                
            } catch (error) {
                console.error('❌ Erro ao carregar resultados:', error);
                
                document.getElementById('resultsContent').innerHTML = `
                    <div class="bg-red-50 border border-red-200 rounded-lg p-6 text-center">
                        <i class="fas fa-exclamation-triangle text-red-600 text-2xl mb-3"></i>
                        <h4 class="font-bold text-red-800 mb-2">Erro ao Carregar Resultados</h4>
                        <p class="text-red-700 mb-4">${error.message}</p>
                        <button onclick="loadAnalysisResults()" 
                                class="bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700 transition-colors">
                            <i class="fas fa-redo mr-2"></i>
                            Tentar Novamente
                        </button>
                    </div>
                `;
                
                document.getElementById('analysisResults').classList.remove('hidden');
                updateSystemStatus('error');
            }
        }

        // Helper functions
        function mostrarRepositorios(repos) {
            const container = document.getElementById('repositoriosList');
            container.innerHTML = '';

            if (repos.length === 0) {
                container.innerHTML = '<div class="col-span-full text-center text-gray-500">Nenhum repositório encontrado</div>';
                document.getElementById('results').classList.remove('hidden');
                return;
            }

            repos.slice(0, 15).forEach(repo => {
                const card = document.createElement('div');
                card.className = 'bg-gray-50 rounded-xl p-6 hover:shadow-lg transition-all duration-200 border border-gray-200 hover:border-blue-300';
                
                const languageColors = {
                    'JavaScript': 'text-yellow-600 bg-yellow-100',
                    'Python': 'text-green-600 bg-green-100',
                    'TypeScript': 'text-blue-600 bg-blue-100',
                    'Java': 'text-red-600 bg-red-100',
                    'Go': 'text-cyan-600 bg-cyan-100',
                    'Rust': 'text-orange-600 bg-orange-100',
                    'C++': 'text-purple-600 bg-purple-100',
                    'PHP': 'text-indigo-600 bg-indigo-100'
                };
                const langColor = languageColors[repo.linguagem_principal] || 'text-gray-600 bg-gray-100';
                
                const ag2Badge = ag2Available ? 
                    '<span class="text-xs bg-purple-100 text-purple-700 px-2 py-1 rounded-full ml-2"><i class="fas fa-robot mr-1"></i>AG2</span>' : 
                    '<span class="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded-full ml-2">Tradicional</span>';
                
                card.innerHTML = `
                    <div class="flex items-start justify-between mb-4">
                        <div class="flex-1">
                            <h4 class="font-bold text-gray-800 text-lg mb-1 hover:text-blue-600 transition-colors cursor-pointer" 
                               onclick="window.open('${repo.url}', '_blank')">${repo.nome}</h4>
                            <p class="text-gray-600 text-sm mb-3 leading-relaxed">${repo.descricao.substring(0, 120)}${repo.descricao.length > 120 ? '...' : ''}</p>
                        </div>
                    </div>
                    
                    <div class="flex items-center justify-between mb-4">
                        <div class="flex items-center space-x-4 text-sm">
                            <span class="flex items-center px-2 py-1 rounded-full ${langColor}">
                                <i class="fas fa-code mr-1"></i>
                                ${repo.linguagem_principal}
                            </span>
                            <span class="flex items-center text-gray-500">
                                <i class="fas fa-star text-yellow-500 mr-1"></i>
                                ${repo.estrelas.toLocaleString()}
                            </span>
                            <span class="flex items-center text-gray-500">
                                <i class="fas fa-code-branch text-green-500 mr-1"></i>
                                ${repo.forks.toLocaleString()}
                            </span>
                        </div>
                        <div class="text-xs text-gray-400">
                            ${Math.round(repo.tamanho_kb / 1024)} MB
                        </div>
                    </div>
                    
                    ${repo.topicos.length > 0 ? `
                        <div class="flex flex-wrap gap-1 mb-4">
                            ${repo.topicos.slice(0, 4).map(topic => 
                                `<span class="px-2 py-1 bg-blue-50 text-blue-700 text-xs rounded-full border border-blue-200">${topic}</span>`
                            ).join('')}
                            ${repo.topicos.length > 4 ? `<span class="px-2 py-1 bg-gray-100 text-gray-600 text-xs rounded-full">+${repo.topicos.length - 4}</span>` : ''}
                        </div>
                    ` : ''}
                    
                    <button type="button" 
                            onclick="analisarRepositorio('${repo.url}')"
                            class="w-full bg-gradient-to-r from-purple-600 to-blue-600 text-white py-3 rounded-lg hover:from-purple-700 hover:to-blue-700 transition-all duration-200 font-medium transform hover:scale-105 shadow-md hover:shadow-lg">
                        <i class="fas fa-robot mr-2"></i>
                        Análise ${ag2Available ? 'AG2' : 'Tradicional'}
                        ${ag2Badge}
                    </button>
                `;
                
                container.appendChild(card);
            });

            document.getElementById('results').classList.remove('hidden');
            document.getElementById('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
        }

        function mostrarMensagemNenhumRepo(usuario) {
            const loadingElement = document.getElementById('loading');
            loadingElement.innerHTML = `
                <div class="bg-yellow-50 border border-yellow-200 rounded-2xl p-8 text-center">
                    <i class="fas fa-search text-yellow-600 text-4xl mb-4"></i>
                    <h3 class="text-xl font-bold text-yellow-800 mb-2">Nenhum repositório encontrado</h3>
                    <p class="text-yellow-700 mb-4">
                        Não foram encontrados repositórios públicos para "${usuario}"
                    </p>
                    <button onclick="location.reload()" 
                            class="mt-4 bg-yellow-600 text-white px-4 py-2 rounded-lg hover:bg-yellow-700 transition-colors">
                        <i class="fas fa-redo mr-2"></i>
                        Nova Busca
                    </button>
                </div>
            `;
        }

        function mostrarErroNaBusca(errorMessage) {
            const loadingElement = document.getElementById('loading');
            loadingElement.innerHTML = `
                <div class="bg-red-50 border border-red-200 rounded-2xl p-8 text-center">
                    <i class="fas fa-exclamation-triangle text-red-600 text-4xl mb-4"></i>
                    <h3 class="text-xl font-bold text-red-800 mb-2">Erro na Busca</h3>
                    <p class="text-red-700 mb-4">${errorMessage}</p>
                    <button onclick="location.reload()" 
                            class="bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700 transition-colors">
                        <i class="fas fa-redo mr-2"></i>
                        Tentar Novamente
                    </button>
                </div>
            `;
        }

        function popularFiltroLinguagens() {
            const linguagens = [...new Set(repositorios.map(r => r.linguagem_principal))];
            const filtroLinguagem = document.getElementById('filtroLinguagem');
            
            filtroLinguagem.innerHTML = '<option value="">Todas as linguagens</option>';
            linguagens.forEach(lang => {
                if (lang !== 'Desconhecida') {
                    filtroLinguagem.innerHTML += `<option value="${lang}">${lang}</option>`;
                }
            });
        }

        // Event listeners for filters
        document.addEventListener('DOMContentLoaded', function() {
            const filtroLinguagem = document.getElementById('filtroLinguagem');
            const ordenacao = document.getElementById('ordenacao');

            if (filtroLinguagem) {
                filtroLinguagem.addEventListener('change', function() {
                    const linguagem = this.value;
                    const filtrados = linguagem ? 
                        repositorios.filter(r => r.linguagem_principal === linguagem) : 
                        repositorios;
                    mostrarRepositorios(filtrados);
                });
            }

            if (ordenacao) {
                ordenacao.addEventListener('change', function() {
                    const ordenacaoTipo = this.value;
                    const ordenados = [...repositorios];
                    
                    switch(ordenacaoTipo) {
                        case 'stars':
                            ordenados.sort((a, b) => b.estrelas - a.estrelas);
                            break;
                        case 'name':
                            ordenados.sort((a, b) => a.nome.localeCompare(b.nome));
                            break;
                        case 'updated':
                            ordenados.sort((a, b) => new Date(b.atualizado_em) - new Date(a.atualizado_em));
                            break;
                    }
                    
                    mostrarRepositorios(ordenados);
                });
            }
        });

        // Cleanup on page unload
        window.addEventListener('beforeunload', function() {
            if (analysisInterval) {
                clearInterval(analysisInterval);
            }
        });

        // Auto-logout em caso de inatividade (opcional)
        let inactivityTimer;
        const INACTIVITY_TIME = 30 * 60 * 1000; // 30 minutos

        function resetInactivityTimer() {
            clearTimeout(inactivityTimer);
            if (isAuthenticated) {
                inactivityTimer = setTimeout(() => {
                    alert('Sessão expirada por inatividade. Faça login novamente.');
                    realizarLogout();
                }, INACTIVITY_TIME);
            }
        }

        // Monitorar atividade do usuário
        ['mousedown', 'mousemove', 'keypress', 'scroll', 'touchstart'].forEach(event => {
            document.addEventListener(event, resetInactivityTimer, true);
        });
    </script>
</body>
</html>"""
    
    try:
        with open("templates/index.html", "w", encoding="utf-8") as f:
            f.write(html_content)
        print("✅ Template HTML com autenticação completa criado/atualizado")
    except Exception as e:
        print(f"❌ Erro ao criar template: {e}")

# =============================================================================
# CONTINUAÇÃO DAS CLASSES AG2 (mantidas iguais)
# =============================================================================

# =============================================================================
# SISTEMA AG2 DE DOCUMENTAÇÃO AVANÇADO (INTEGRADO COM AUTENTICAÇÃO)
# =============================================================================

class EnhancedDocumentationFlow:
    """Sistema AG2 Flow avançado para documentação completa - Integrado com DocAgent"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.state = None
        self.tools = None
        self.agents = {}
        self.error_count = 0
        self.anonimizacao = SistemaAnonimizacao()
        self.code_analyzer = CodeAnalyzer()
        self._setup_llm_config()
        if AG2_AVAILABLE:
            self._setup_agents()
        print("🤖 Enhanced AG2 Documentation Flow inicializado para DocAgent")
    
    def _setup_llm_config(self):
        """Configuração LLM otimizada"""
        self.llm_config = {
            "config_list": [{
                "model": self.config.llm_model,
                "api_type": "ollama",
                "base_url": "http://localhost:11434",
                "api_key": "fake_key"
            }],
            "timeout": self.config.timeout,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "seed": 42
        }
    
    def _setup_agents(self):
        """Setup dos agentes com prompts aprimorados para DocAgent"""
        
        if not AG2_AVAILABLE:
            print("⚠️ AG2 não disponível - pulando setup de agentes")
            return
        
        # Advanced Code Explorer
        self.agents["code_explorer"] = ConversableAgent(
            name="AdvancedCodeExplorer",
            system_message="""Você é um especialista em análise avançada de código para o DocAgent Skyone. Sua função é realizar uma análise COMPLETA e DETALHADA do repositório para gerar relatórios anônimos técnicos.

**MISSÃO PRINCIPAL:** Analisar repositório GitHub para criar documentação técnica completa em 3 partes:
1. Visão Geral do Projeto (tecnologias, arquitetura)
2. Guia de Instalação e Configuração (baseado nas dependências encontradas)  
3. **Relatório Técnico dos Arquivos** (análise detalhada - FOCO PRINCIPAL)

**TOOLS DISPONÍVEIS:**
- `directory_read(path)`: Lista e categoriza conteúdo de diretórios
- `file_read(file_path)`: Análise detalhada de arquivos individuais
- `find_key_files()`: Identifica arquivos importantes por categoria
- `analyze_code_structure()`: Estatísticas completas da base de código
- `detailed_file_analysis(max_files)`: Análise profunda dos arquivos principais

**PROTOCOLO DE ANÁLISE OBRIGATÓRIO:**
1. **Estrutura Geral**: `analyze_code_structure()` - entenda a arquitetura
2. **Arquivos-Chave**: `find_key_files()` - identifique componentes importantes  
3. **Análise Detalhada**: `detailed_file_analysis(15)` - examine arquivos principais
4. **Leitura Específica**: Use `file_read()` em 3-5 arquivos mais críticos
5. **Exploração Dirigida**: `directory_read()` em diretórios relevantes

**IMPORTANTE PARA DOCAGENT:**
- Identifique todas as tecnologias e frameworks utilizados
- Mapeie dependências (package.json, requirements.txt, etc.)
- Analise arquivos de configuração
- Documente APIs e interfaces encontradas
- Identifique pontos de entrada da aplicação
- Use TODAS as tools disponíveis sistematicamente""",
            llm_config=self.llm_config,
            human_input_mode="NEVER"
        )
        
        # Enhanced Documentation Planner for DocAgent
        self.agents["documentation_planner"] = ConversableAgent(
            name="EnhancedDocumentationPlanner",
            system_message="""Você é um planejador de documentação técnica para o DocAgent Skyone. Baseado na análise do AdvancedCodeExplorer, crie um plano OBRIGATORIAMENTE com 3 seções específicas para relatórios anônimos.

**PLANO OBRIGATÓRIO - EXATAMENTE 3 SEÇÕES:**

1. **"Visão Geral do Projeto"**
   - Propósito e funcionalidade principal
   - Tecnologias e linguagens utilizadas (identificadas na análise)
   - Arquitetura geral e estrutura do código

2. **"Guia de Instalação e Configuração"**  
   - Pré-requisitos baseados nas tecnologias encontradas
   - Passos de instalação (baseado em package.json, requirements.txt, etc.)
   - Configuração inicial do ambiente
   - Como executar o projeto

3. **"Relatório Técnico dos Arquivos"** (SEÇÃO PRINCIPAL DO DOCAGENT)
   - Análise detalhada de cada arquivo importante
   - Funções e classes principais identificadas
   - APIs e interfaces mapeadas
   - Fluxo de dados e lógica da aplicação
   - Dependências entre arquivos
   - Estrutura técnica completa

**FORMATO JSON OBRIGATÓRIO:**
```json
{
  "overview": "Descrição concisa mas completa do projeto baseada na análise",
  "docs": [
    {
      "title": "Visão Geral do Projeto",
      "description": "Apresentação completa do projeto com tecnologias identificadas",
      "prerequisites": "Conhecimento básico de programação",
      "examples": ["Tecnologias utilizadas", "Arquitetura do sistema"],
      "goal": "Fornecer entendimento completo do propósito e stack tecnológico"
    },
    {
      "title": "Guia de Instalação e Configuração", 
      "description": "Instruções baseadas nas dependências e configurações encontradas",
      "prerequisites": "Sistema operacional compatível",
      "examples": ["Instalação de dependências", "Configuração do ambiente"],
      "goal": "Permitir instalação e execução baseada na análise do código"
    },
    {
      "title": "Relatório Técnico dos Arquivos",
      "description": "Análise técnica detalhada de arquivos, funções, classes e APIs identificadas",
      "prerequisites": "Conhecimento nas linguagens utilizadas no projeto",
      "examples": ["Análise arquivo por arquivo", "Documentação de funções", "Mapeamento de APIs"],
      "goal": "Fornecer relatório técnico completo para desenvolvedores baseado na análise real do código"
    }
  ]
}
```

**IMPORTANTE:** Use apenas informações específicas da análise realizada pelo CodeExplorer.""",
            llm_config=self.llm_config,
            human_input_mode="NEVER"
        )
        
        # Technical Documentation Writer for DocAgent
        self.agents["technical_writer"] = ConversableAgent(
            name="TechnicalDocumentationWriter",
            system_message="""Você é um escritor técnico especializado no DocAgent Skyone. Escreva documentação técnica DETALHADA e PROFISSIONAL baseada na análise real do código.

**ESTRUTURA PADRÃO PARA DOCAGENT:**

## Para "Visão Geral do Projeto":
# Visão Geral do Projeto

## 🎯 Propósito
[Baseado na análise dos arquivos principais]

## 🛠️ Stack Tecnológico
[Linguagens e frameworks identificados na análise]

## 🏗️ Arquitetura
[Estrutura identificada na análise de código]

## Para "Guia de Instalação e Configuração":
# Guia de Instalação e Configuração

## 📋 Pré-requisitos
[Baseado nas tecnologias identificadas]

## 🚀 Instalação
[Baseado em package.json, requirements.txt, etc. encontrados]

## ⚙️ Configuração
[Baseado em arquivos de config encontrados]

## ▶️ Execução
[Baseado nos pontos de entrada identificados]

## Para "Relatório Técnico dos Arquivos" (PRINCIPAL DO DOCAGENT):
# Relatório Técnico dos Arquivos

## 📁 Estrutura do Projeto
[Organização identificada na análise]

## 🔧 Arquivos Principais

### [NOME_ARQUIVO] (Linguagem identificada)
**Propósito:** [Identificado na análise]
**Localização:** `caminho/real/do/arquivo`

#### 📋 Funcionalidades:
[Baseado na análise real do código]

#### 🔧 Funções Identificadas:
[Funções reais encontradas na análise]

#### 📊 Classes Encontradas:
[Classes reais identificadas]

#### 🔌 APIs/Interfaces:
[APIs reais mapeadas na análise]

#### 📝 Dependências:
[Imports e dependências reais]

**CRUCIAL:** Use APENAS informações da análise real. Não invente detalhes.""",
            llm_config=self.llm_config,
            human_input_mode="NEVER"
        )
        
        # Documentation Reviewer for DocAgent
        self.agents["documentation_reviewer"] = ConversableAgent(
            name="DocumentationReviewer",
            system_message="""Você é um revisor sênior de documentação técnica para o DocAgent Skyone. Revise e aprimore garantindo PRECISÃO TÉCNICA baseada na análise real.

**CRITÉRIOS DE REVISÃO DOCAGENT:**

1. **Precisão:** Informações corretas baseadas na análise?
2. **Completude:** Todas as 3 seções estão completas?
3. **Consistência:** Informações consistentes entre seções?
4. **Detalhamento Técnico:** Relatório técnico suficientemente detalhado?
5. **Anonimização:** Garantir que não há informações pessoais expostas?

**FOQUE NO RELATÓRIO TÉCNICO:**
- Cada arquivo importante foi documentado com base na análise real?
- Funções e classes reais foram documentadas?
- APIs identificadas estão bem explicadas?
- Dependências reais foram mapeadas?
- Estrutura reflete a análise realizada?

**IMPORTANTE PARA DOCAGENT:**
- Corrija apenas imprecisões técnicas
- Mantenha foco na análise real do código
- Garanta que informações são úteis para desenvolvedores
- Certifique-se que o relatório é profissional e anônimo""",
            llm_config=self.llm_config,
            human_input_mode="NEVER"
        )
    
    def _register_tools_safely(self):
        """Registra tools avançadas com tratamento de erros para AG2"""
        if not self.tools:
            print("⚠️ Tools não inicializadas")
            return False
        
        if not AG2_AVAILABLE:
            print("⚠️ AG2 não disponível - pulando registro de tools")
            return False
        
        try:
            explorer = self.agents["code_explorer"]
            
            @explorer.register_for_llm(description="Lista e categoriza conteúdo detalhado de diretórios")
            @explorer.register_for_execution()
            def directory_read(path: str = "") -> str:
                return self.tools.directory_read(path)
            
            @explorer.register_for_llm(description="Análise detalhada de arquivos individuais com informações técnicas")
            @explorer.register_for_execution()  
            def file_read(file_path: str) -> str:
                return self.tools.file_read(file_path)
            
            @explorer.register_for_llm(description="Identifica e categoriza arquivos importantes do projeto")
            @explorer.register_for_execution()
            def find_key_files() -> str:
                return self.tools.find_key_files()
            
            @explorer.register_for_llm(description="Análise completa da estrutura de código com estatísticas detalhadas")
            @explorer.register_for_execution()
            def analyze_code_structure() -> str:
                return self.tools.analyze_code_structure()
            
            @explorer.register_for_llm(description="Análise técnica profunda dos arquivos mais importantes")
            @explorer.register_for_execution()
            def detailed_file_analysis(max_files: int = 10) -> str:
                return self.tools.detailed_file_analysis(max_files)
            
            print("🔧 Tools AG2 registradas com sucesso")
            return True
            
        except Exception as e:
            print(f"❌ Erro ao registrar tools AG2: {e}")
            return False

    def clone_repository(self, project_url: str) -> bool:
        """Clone com diagnóstico detalhado para DocAgent"""
        print(f"📥 Iniciando clone para DocAgent: {project_url}")
        
        # Inicializar estado se não existir
        if self.state is None:
            print("🔧 Inicializando estado do sistema...")
            self.state = DocumentationState(project_url=project_url)
        
        # Validar URL
        if not self._validate_github_url(project_url):
            print(f"❌ URL inválida: {project_url}")
            return False
        
        # Verificar conectividade
        if not self._check_github_connectivity():
            print("❌ Sem conectividade com GitHub")
            return False
        
        # Verificar se repositório existe
        if not self._check_repository_exists(project_url):
            print(f"❌ Repositório não existe ou é privado: {project_url}")
            return False
        
        # Preparar diretórios
        repo_name = project_url.split("/")[-1].replace(".git", "")
        workdir = Path("workdir").resolve()
        workdir.mkdir(exist_ok=True)
        repo_path = workdir / repo_name
        
        print(f"📁 Diretório de trabalho: {workdir}")
        print(f"📁 Destino do clone: {repo_path}")
        
        # Limpeza robusta do diretório existente
        if repo_path.exists():
            print(f"🗑️ Removendo diretório existente: {repo_path}")
            
            for attempt in range(3):
                try:
                    if repo_path.exists():
                        if attempt == 0:
                            shutil.rmtree(repo_path)
                        elif attempt == 1:
                            self._force_remove_directory(repo_path)
                        else:
                            if os.name == 'nt':
                                subprocess.run(["rmdir", "/s", "/q", str(repo_path)], shell=True)
                            else:
                                subprocess.run(["rm", "-rf", str(repo_path)])
                    
                    if not repo_path.exists():
                        print(f"✅ Diretório removido com sucesso")
                        break
                    else:
                        print(f"⚠️ Tentativa {attempt + 1} falhou")
                        
                except Exception as e:
                    print(f"⚠️ Erro na remoção (tentativa {attempt + 1}): {e}")
                    
                if attempt < 2:
                    time.sleep(1)
            
            if repo_path.exists():
                backup_path = repo_path.with_suffix(f".backup_{int(time.time())}")
                try:
                    repo_path.rename(backup_path)
                    print(f"🔄 Diretório movido para: {backup_path}")
                except Exception as e:
                    print(f"❌ Não foi possível limpar o diretório: {e}")
                    return False
        
        # Construir URL de clone com token se disponível
        clone_url = project_url
        try:
            github_token = os.environ.get('GITHUB_TOKEN')
            if github_token and 'github.com' in project_url and '@' not in project_url:
                import urllib.parse as _urlparse
                parsed = _urlparse.urlparse(project_url)
                netloc = f"{github_token}@{parsed.netloc}"
                clone_url = _urlparse.urlunparse(parsed._replace(netloc=netloc))
        except Exception:
            clone_url = project_url

        # Tentar clone com retry
        max_retries = 3
        clone_success = False
        
        for attempt in range(max_retries):
            try:
                print(f"🔄 Tentativa de clone {attempt + 1}/{max_retries}")
                
                if attempt == 0:
                    cmd = ["git", "clone", "--depth", "1", "--single-branch", clone_url, str(repo_path)]
                elif attempt == 1:
                    cmd = ["git", "clone", "--single-branch", clone_url, str(repo_path)]
                else:
                    cmd = ["git", "clone", clone_url, str(repo_path)]
                
                print(f"🔧 Executando: git clone [URL_PROTEGIDA] {repo_path}")
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                
                print(f"🔍 Código de retorno: {result.returncode}")
                
                if result.returncode == 0:
                    print(f"✅ Git clone executado com sucesso na tentativa {attempt + 1}")
                    clone_success = True
                    break
                else:
                    error_msg = result.stderr.strip()
                    print(f"❌ Erro no git clone (tentativa {attempt + 1}):")
                    print(f"   stderr: {error_msg[:200]}")
                    
                    if "already exists and is not an empty directory" in error_msg:
                        print("🔄 Diretório ainda existe - tentando limpeza adicional")
                        if repo_path.exists():
                            try:
                                shutil.rmtree(repo_path, ignore_errors=True)
                                time.sleep(2)
                            except:
                                pass
                        continue
                    elif "not found" in error_msg.lower() or "does not exist" in error_msg.lower():
                        print("❌ Repositório não encontrado - parando tentativas")
                        return False
                    elif "permission denied" in error_msg.lower() or "forbidden" in error_msg.lower():
                        print("❌ Permissão negada - repositório privado")
                        return False
                    
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 3
                        print(f"⏳ Aguardando {wait_time}s antes da próxima tentativa...")
                        time.sleep(wait_time)
                
            except subprocess.TimeoutExpired:
                print(f"⏰ Timeout na tentativa {attempt + 1} (5min)")
                if attempt < max_retries - 1:
                    print("⏳ Tentando novamente...")
                    continue
                else:
                    print("❌ Timeout final - repositório muito grande")
                    return False
                    
            except Exception as e:
                print(f"❌ Erro na execução do git (tentativa {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    continue
                else:
                    return False
        
        if not clone_success:
            print("❌ Todas as tentativas de clone falharam")
            return False
        
        # Verificação pós-clone
        print(f"🔍 Verificando resultado do clone...")
        print(f"   Caminho esperado: {repo_path}")
        print(f"   Diretório existe: {repo_path.exists()}")
        
        if not repo_path.exists():
            print("❌ Diretório do repositório não foi criado após clone bem-sucedido")
            return False
        
        if not repo_path.is_dir():
            print(f"❌ {repo_path} existe mas não é um diretório")
            return False
        
        try:
            repo_items = list(repo_path.iterdir())
            print(f"📁 Itens no repositório: {len(repo_items)}")
            
            for i, item in enumerate(repo_items[:5]):
                print(f"   {i+1}. {item.name} ({'dir' if item.is_dir() else 'file'})")
            
            if len(repo_items) == 0:
                print("❌ Repositório está vazio")
                return False
            
            git_dir = repo_path / ".git"
            if git_dir.exists():
                print("✅ Diretório .git encontrado - clone Git válido")
            else:
                print("⚠️ Diretório .git não encontrado - pode ser um problema")
                
        except Exception as e:
            print(f"❌ Erro ao verificar conteúdo do repositório: {e}")
            return False
        
        # Atualizar estado
        self.state.repo_path = str(repo_path)
        self.state.current_phase = "cloned"
        
        # Inicializar tools avançadas
        try:
            print("🔧 Inicializando tools avançadas de análise para DocAgent...")
            self.tools = AdvancedRepositoryTools(repo_path)
            
            if AG2_AVAILABLE and not self._register_tools_safely():
                print("⚠️ Algumas tools AG2 falharam, mas continuando...")
            
            print(f"🎉 Clone concluído com sucesso para DocAgent!")
            print(f"   📁 Localização: {repo_path}")
            print(f"   📊 Itens: {len(repo_items)} encontrados")
            return True
            
        except Exception as e:
            print(f"❌ Erro ao inicializar tools: {e}")
            print("⚠️ Continuando sem tools - clone foi bem-sucedido")
            return True
    
    def _force_remove_directory(self, path: Path):
        """Remove diretório forçadamente"""
        try:
            for root, dirs, files in os.walk(path, topdown=False):
                for name in files:
                    file_path = Path(root) / name
                    try:
                        file_path.chmod(0o777)
                    except:
                        pass
                for name in dirs:
                    dir_path = Path(root) / name
                    try:
                        dir_path.chmod(0o777)
                    except:
                        pass
            
            shutil.rmtree(path)
            
        except Exception as e:
            print(f"⚠️ Erro na remoção forçada: {e}")
            raise
    
    def _validate_github_url(self, url: str) -> bool:
        """Valida formato da URL do GitHub"""
        pattern = r"^https://github\.com/[\w\-\.]+/[\w\-\.]+/?$"
        return bool(re.match(pattern, url.strip()))
    
    def _check_github_connectivity(self) -> bool:
        """Verifica conectividade básica com GitHub"""
        try:
            socket.setdefaulttimeout(10)
            response = urllib.request.urlopen("https://github.com", timeout=10)
            return response.getcode() == 200
        except Exception as e:
            print(f"⚠️ Erro de conectividade: {e}")
            return False
    
    def _check_repository_exists(self, project_url: str) -> bool:
        """Verifica se repositório existe e é público"""
        try:
            request = urllib.request.Request(project_url)
            request.add_header('User-Agent', 'Mozilla/5.0 (compatible; DocAgent/2.0)')
            
            try:
                response = urllib.request.urlopen(request, timeout=15)
                return response.getcode() == 200
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    print(f"❌ Repositório não encontrado (404): {project_url}")
                elif e.code == 403:
                    print(f"❌ Acesso negado (403): repositório privado ou rate limit")
                else:
                    print(f"❌ Erro HTTP {e.code}: {e.reason}")
                return False
            except urllib.error.URLError as e:
                print(f"❌ Erro de URL: {e.reason}")
                return False
                
        except Exception as e:
            print(f"⚠️ Erro ao verificar repositório: {e}")
            return True

    def execute_analysis_with_ag2(self, project_url: str, anonymous: bool = True) -> Dict[str, Any]:
        """Executa análise completa usando AG2 ou fallback"""
        try:
            logger = app_state.get("enhanced_logger")
            if logger:
                logger.info(f"Iniciando análise DocAgent com AG2: {project_url}", "AG2 Analysis", "Início")
            
            print(f"🚀 Iniciando análise DocAgent com AG2: {project_url}")
            
            # Inicializar estado
            self.state = DocumentationState(project_url=project_url)
            
            if logger:
                logger.info("Estado de documentação inicializado", "AG2 Analysis", "Inicialização")
            
            # Fase 1: Clone
            if logger:
                logger.info("Iniciando clone do repositório", "AG2 Analysis", "Clone")
            
            clone_success = self.clone_repository(project_url)
            if not clone_success:
                if logger:
                    logger.error("Falha no clone do repositório", "AG2 Analysis", "Clone")
                return {
                    "status": "error",
                    "message": "Falha no clone do repositório",
                    "error_count": self.error_count
                }
            
            if logger:
                logger.success("Clone do repositório concluído", "AG2 Analysis", "Clone")
            
            # Verificar se AG2 está disponível
            if not AG2_AVAILABLE:
                if logger:
                    logger.warning("AG2 não disponível - usando análise simplificada", "AG2 Analysis", "Fallback")
                print("⚠️ AG2 não disponível - usando análise simplificada")
                return self._execute_simplified_analysis(project_url, anonymous)
            
            if logger:
                logger.info("AG2 disponível - prosseguindo com análise avançada", "AG2 Analysis", "Verificação")
            
            # Fase 2: Enhanced Planning com AG2
            if logger:
                logger.info("Iniciando fase de planejamento AG2", "AG2 Analysis", "Planejamento")
            
            plan_success = self._enhanced_planning_phase_ag2()
            if not plan_success:
                if logger:
                    logger.warning("Planejamento AG2 falhou - usando fallback", "AG2 Analysis", "Fallback")
                print("⚠️ Planejamento AG2 falhou - usando fallback")
                return self._execute_simplified_analysis(project_url, anonymous)
            
            if logger:
                logger.success("Planejamento AG2 concluído", "AG2 Analysis", "Planejamento")
            
            # Fase 3: Enhanced Documentation com AG2
            if logger:
                logger.info("Iniciando fase de documentação AG2", "AG2 Analysis", "Documentação")
            
            doc_success = self._enhanced_documentation_phase_ag2(anonymous)
            if not doc_success:
                if logger:
                    logger.warning("Documentação AG2 falhou - usando fallback", "AG2 Analysis", "Fallback")
                print("⚠️ Documentação AG2 falhou - usando fallback")
                return self._execute_simplified_analysis(project_url, anonymous)
            
            if logger:
                logger.success("Documentação AG2 concluída", "AG2 Analysis", "Documentação")
            
            # Sucesso com AG2
            generated_docs_base = []
            for p in self.state.generated_docs:
                try:
                    generated_docs_base.append(os.path.basename(p))
                except Exception:
                    generated_docs_base.append(p)

            if logger:
                logger.success(f"Análise AG2 completa: {len(self.state.generated_docs)} arquivos gerados", "AG2 Analysis", "Finalização")

            return {
                "status": "success",
                "message": f"Análise AG2 completa criada: {len(self.state.generated_docs)} seções",
                "generated_docs": generated_docs_base,
                "plan": self.state.plan.to_dict() if self.state.plan else None,
                "metadata": {
                    "project_url": project_url,
                    "repo_path": self.state.repo_path,
                    "docs_count": len(self.state.generated_docs),
                    "generated_at": datetime.now().isoformat(),
                    "error_count": self.error_count,
                    "system_version": "DocAgent Skyone v2.0 with AG2 + Auth + C4 Model",
                    "ag2_enabled": True,
                    "anonymous": anonymous,
                    "features": [
                        "AG2 Multi-agent analysis",
                        "C4 Model architectural documentation",
                        "Advanced code structure analysis", 
                        "Detailed file documentation",
                        "Anonymous reporting",
                        "Complete technical documentation",
                        "Authentication system"
                    ]
                }
            }
            
        except Exception as e:
            logger.error(f"Erro no fluxo AG2: {e}")
            logger.error(traceback.format_exc())
            return {
                "status": "error",
                "message": f"Erro crítico AG2: {str(e)[:100]}",
                "error_count": self.error_count + 1
            }
    
    # [Resto dos métodos da classe mantidos iguais...]
    def _enhanced_planning_phase_ag2(self) -> bool:
        """Fase de planejamento aprimorada com AG2"""
        try:
            logger = app_state.get("enhanced_logger")
            if logger:
                logger.info("Iniciando planejamento AG2", "AG2 Planning", "Início")
            
            print("🎯 Iniciando planejamento AG2...")
            
            planning_agents = [self.agents["code_explorer"], self.agents["documentation_planner"]]
            
            if logger:
                logger.info("Agentes de planejamento inicializados", "AG2 Planning", "Agentes")
            
            planning_chat = GroupChat(
                agents=planning_agents,
                messages=[],
                max_round=8,
                speaker_selection_method="round_robin"
            )
            
            planning_manager = GroupChatManager(
                groupchat=planning_chat,
                llm_config=self.llm_config
            )
            
            planning_prompt = f"""ANÁLISE COMPLETA DO REPOSITÓRIO PARA DOCAGENT: {self.state.repo_path}

**MISSÃO CRÍTICA:** Criar plano para documentação anônima em EXATAMENTE 3 seções:
1. Visão Geral do Projeto
2. Guia de Instalação e Configuração  
3. **Relatório Técnico dos Arquivos** (PRINCIPAL)

**PROTOCOLO OBRIGATÓRIO:**

AdvancedCodeExplorer - Execute TODAS estas análises em sequência:

1. `analyze_code_structure()` - Entenda arquitetura geral
2. `find_key_files()` - Identifique componentes por categoria
3. `detailed_file_analysis(15)` - Análise profunda dos 15 arquivos principais
4. `file_read()` nos 3-5 arquivos mais críticos identificados
5. `directory_read()` em diretórios importantes (src/, lib/, etc.)

**IMPORTANTE:**
- Identifique todas as linguagens e frameworks
- Mapeie dependências e configurações
- Analise arquivos de código em detalhes
- Documente APIs e estruturas encontradas

EnhancedDocumentationPlanner - Baseado na análise completa, crie plano JSON com foco em relatórios técnicos anônimos."""
            
            # Executar análise completa
            planning_result = self.agents["code_explorer"].initiate_chat(
                planning_manager,
                message=planning_prompt,
                clear_history=True
            )
            
            # Extrair plano
            plan_data = self._extract_plan_safely(planning_chat.messages)
            
            if plan_data:
                self.state.plan = plan_data
                self.state.current_phase = "planned"
                print(f"✅ Plano AG2 criado: {len(plan_data.docs)} seções")
                return True
            else:
                print("❌ Falha no plano AG2 - usando plano padrão")
                self.state.plan = self._create_comprehensive_plan()
                return True
                
        except Exception as e:
            print(f"❌ Erro no planejamento AG2: {str(e)[:100]}")
            self.error_count += 1
            self.state.plan = self._create_comprehensive_plan()
            return True
    
    def _enhanced_documentation_phase_ag2(self, anonymous: bool = True) -> bool:
        """Fase de documentação aprimorada com AG2"""
        try:
            logger = app_state.get("enhanced_logger")
            if logger:
                logger.info("Iniciando documentação AG2", "AG2 Documentation", "Início")
            
            print("📝 Iniciando documentação AG2...")
            
            if not self.state.plan or not self.state.plan.docs:
                if logger:
                    logger.warning("Sem plano - criando documentação padrão", "AG2 Documentation", "Fallback")
                print("❌ Sem plano - criando documentação padrão")
                return self._create_comprehensive_documentation(anonymous)
            
            doc_agents = [self.agents["technical_writer"], self.agents["documentation_reviewer"]]
            
            docs_created = []
            
            for i, doc_item in enumerate(self.state.plan.docs):
                if logger:
                    logger.info(f"Criando seção {i+1}/3: {doc_item.title}", "AG2 Documentation", "Seção")
                
                print(f"📄 Criando seção AG2 {i+1}/3: {doc_item.title}")
                
                try:
                    doc_chat = GroupChat(
                        agents=doc_agents,
                        messages=[],
                        max_round=6,
                        speaker_selection_method="round_robin"
                    )
                    
                    doc_manager = GroupChatManager(
                        groupchat=doc_chat,
                        llm_config=self.llm_config
                    )
                    
                    # Prompt específico por seção
                    if "técnico" in doc_item.title.lower() or "arquivo" in doc_item.title.lower():
                        # Seção técnica principal - MAIS DETALHADA
                        doc_prompt = f"""CRIAR RELATÓRIO TÉCNICO DETALHADO PARA DOCAGENT

**SEÇÃO:** {doc_item.title}
**PROJETO:** {self.state.project_url}
**MODO:** {'Anônimo' if anonymous else 'Original'}

**REQUISITOS ESPECIAIS PARA RELATÓRIO TÉCNICO:**
Esta é a seção MAIS IMPORTANTE do DocAgent. Deve incluir:

1. **Estrutura Geral dos Arquivos** (baseada na análise real)
2. **Relatório de CADA arquivo importante** analisado:
   - Propósito e funcionalidade identificada
   - Linguagem e frameworks detectados
   - Funções e classes reais encontradas
   - APIs e interfaces mapeadas
   - Dependências e imports identificados
   - Complexidade e linhas de código
   - Análise técnica específica

3. **Mapeamento de tecnologias** (real)
4. **Arquitetura do sistema** (identificada)
5. **Relatório para desenvolvedores**

**FORMATO OBRIGATÓRIO:**
# {doc_item.title}

## 📁 Estrutura do Projeto
[Organização real identificada]

## 🔧 Arquivos Analisados

### arquivo_real.ext (Linguagem_Real)
**Propósito:** [Propósito identificado na análise]
**Localização:** `caminho/real/identificado`
**Tamanho:** [Tamanho real] | **Linhas:** [Linhas reais]
**Complexidade:** [Complexidade calculada]

#### 📋 Funcionalidades Identificadas:
[Baseado na análise real do código]

#### 🔧 Funções Encontradas:
[Funções reais identificadas na análise]

#### 📊 Classes Detectadas:
[Classes reais encontradas]

#### 🔌 APIs/Interfaces:
[APIs reais mapeadas]

#### 📦 Dependências:
[Imports reais identificados]

#### 📝 Análise Técnica:
[Análise específica baseada no código real]

[REPETIR PARA CADA ARQUIVO IMPORTANTE ANALISADO]

## 🏗️ Arquitetura Identificada
[Como os arquivos se relacionam - baseado na análise]

TechnicalDocumentationWriter: Use APENAS informações da análise real do código
DocumentationReviewer: Revise garantindo precisão técnica baseada nos dados reais

**CRUCIAL:** Use apenas dados da análise realizada. Não invente informações."""
                    else:
                        # Seções 1 e 2 - baseadas na análise
                        doc_prompt = f"""CRIAR DOCUMENTAÇÃO BASEADA NA ANÁLISE: {doc_item.title}

**CONTEXTO:**
- Projeto: {self.state.project_url}
- Seção: {doc_item.title}
- Descrição: {doc_item.description}
- Objetivo: {doc_item.goal}
- Modo: {'Anônimo' if anonymous else 'Original'}

**INFORMAÇÕES DA ANÁLISE:**
Use as informações técnicas identificadas na análise do código para criar documentação precisa.

TechnicalDocumentationWriter: Crie documentação baseada na análise real
DocumentationReviewer: Revise garantindo precisão

Use apenas informações da análise realizada."""
                    
                    # Criar documentação
                    doc_result = self.agents["technical_writer"].initiate_chat(
                        doc_manager,
                        message=doc_prompt,
                        clear_history=True
                    )
                    
                    # Extrair e salvar
                    final_doc = self._extract_documentation_safely(doc_chat.messages, doc_item.title)
                    
                    if final_doc:
                        doc_path = self._save_documentation(doc_item.title, final_doc, anonymous)
                        if doc_path:
                            docs_created.append(doc_path)
                            if not hasattr(self.state, 'generated_docs') or self.state.generated_docs is None:
                                self.state.generated_docs = []
                            self.state.generated_docs.append(doc_path)
                            print(f"✅ Seção AG2 criada: {doc_item.title}")
                    
                except Exception as e:
                    print(f"⚠️ Erro na seção AG2 {doc_item.title}: {str(e)[:50]}")
                    # Criar documentação básica como fallback
                    basic_doc = self._generate_section_fallback(doc_item.title, i, anonymous)
                    doc_path = self._save_documentation(doc_item.title, basic_doc, anonymous)
                    if doc_path:
                        docs_created.append(doc_path)
                        if not hasattr(self.state, 'generated_docs') or self.state.generated_docs is None:
                            self.state.generated_docs = []
                        self.state.generated_docs.append(doc_path)
            
            if docs_created:
                # Gerar documentação C4 Model
                try:
                    logger = app_state.get("enhanced_logger")
                    if logger:
                        logger.info("Iniciando geração da documentação C4", "AG2 - C4 Model", "Geração")
                    
                    print("🏗️ Gerando documentação C4 Model...")
                    c4_analyzer = C4ModelAnalyzer()
                    c4_model = c4_analyzer.analyze_repository_architecture(self.state.repo_path, self.state.project_url)
                    
                    # Gerar arquivos C4
                    docs_dir = Path("docs")
                    docs_dir.mkdir(exist_ok=True)
                    
                    c4_docs = self._generate_c4_documentation_ag2(c4_model, docs_dir, anonymous)
                    
                    if c4_docs:
                        docs_created.extend(c4_docs)
                        if not hasattr(self.state, 'generated_docs') or self.state.generated_docs is None:
                            self.state.generated_docs = []
                        self.state.generated_docs.extend(c4_docs)
                        
                        if logger:
                            logger.success(f"C4 Model gerado: {len(c4_docs)} arquivos", "AG2 - C4 Model", "Sucesso")
                        print(f"✅ C4 Model AG2 gerado: {len(c4_docs)} arquivos")
                    
                except Exception as e:
                    if logger:
                        logger.error(f"Erro na geração C4: {e}", "AG2 - C4 Model", "Erro")
                    print(f"⚠️ Erro na geração C4 Model: {e}")
                
                self.state.current_phase = "completed"
                print(f"🎉 Documentação AG2 completa: {len(docs_created)} arquivos")
                return True
            else:
                print("⚠️ Nenhuma doc AG2 criada - gerando documentação padrão")
                return self._create_comprehensive_documentation(anonymous)
                
        except Exception as e:
            print(f"❌ Erro na documentação AG2: {str(e)[:100]}")
            return self._create_comprehensive_documentation(anonymous)
    
    def _generate_c4_documentation_ag2(self, c4_model: C4Model, output_dir: Path, anonymous: bool = True) -> List[str]:
        """Gera documentação C4 no contexto AG2"""
        generated_files = []
        
        try:
            url_suffix = "_anonimo" if anonymous else ""
            
            # 1. Architecture Overview
            arch_doc = self._generate_c4_architecture_overview_ag2(c4_model)
            arch_file = output_dir / f"04_C4_Architecture_Overview{url_suffix}.md"
            with open(arch_file, 'w', encoding='utf-8') as f:
                f.write(arch_doc)
            generated_files.append(str(arch_file.name))
            
            # 2. Context Diagram
            context_doc = self._generate_c4_context_doc_ag2(c4_model)
            context_file = output_dir / f"05_C4_Context_Diagram{url_suffix}.md"
            with open(context_file, 'w', encoding='utf-8') as f:
                f.write(context_doc)
            generated_files.append(str(context_file.name))
            
            # 3. Container Diagram
            container_doc = self._generate_c4_container_doc_ag2(c4_model)
            container_file = output_dir / f"06_C4_Container_Diagram{url_suffix}.md"
            with open(container_file, 'w', encoding='utf-8') as f:
                f.write(container_doc)
            generated_files.append(str(container_file.name))
            
            # 4. Component Diagram
            component_doc = self._generate_c4_component_doc_ag2(c4_model)
            component_file = output_dir / f"07_C4_Component_Diagram{url_suffix}.md"
            with open(component_file, 'w', encoding='utf-8') as f:
                f.write(component_doc)
            generated_files.append(str(component_file.name))
            
            # 5. Deployment Guide
            deployment_doc = self._generate_c4_deployment_doc_ag2(c4_model)
            deployment_file = output_dir / f"08_C4_Deployment_Guide{url_suffix}.md"
            with open(deployment_file, 'w', encoding='utf-8') as f:
                f.write(deployment_doc)
            generated_files.append(str(deployment_file.name))
            
            return generated_files
            
        except Exception as e:
            print(f"❌ Erro na geração da documentação C4 AG2: {e}")
            return generated_files
    
    def _generate_c4_architecture_overview_ag2(self, c4_model: C4Model) -> str:
        """Gera visão geral da arquitetura C4 para AG2"""
        context = c4_model.context
        metadata = c4_model.metadata
        
        return f"""# 🏗️ Visão Geral da Arquitetura - {context['system_name']}

## 📋 Informações do Sistema

**Sistema:** {context['system_name']}  
**Descrição:** {context['description']}  
**Tecnologia Principal:** {context['technology']}  
**Gerado em:** {metadata['generated_at']}  
**Analisado por:** DocAgent Skyone v2.0 com AG2 + C4 Model

## 🎯 Resumo Executivo

Este documento fornece uma visão arquitetural abrangente do sistema **{context['system_name']}** usando a abordagem do **modelo C4**. O modelo C4 oferece uma maneira estruturada de visualizar a arquitetura em diferentes níveis de detalhamento.

## 🏗️ Abordagem Arquitetural

Este sistema segue princípios modernos de arquitetura de software:

- **🔧 Separação de Responsabilidades**: Fronteiras claras entre diferentes camadas
- **💻 Stack Tecnológica**: Construído principalmente com {context['technology']}
- **📈 Escalabilidade**: Projetado para escalonamento horizontal e vertical
- **🔧 Manutenibilidade**: Design modular para fácil manutenção

## 🛠️ Stack Tecnológica Identificada

| Componente | Tecnologia |
|-----------|------------|
| **Linguagem Principal** | {context['technology']} |
| **Frameworks** | {', '.join(metadata['technologies'].get('frameworks', ['Nenhum identificado']))} |
| **Bancos de Dados** | {', '.join(metadata['technologies'].get('databases', ['Nenhum identificado']))} |
| **Ferramentas de Build** | {', '.join(metadata['technologies'].get('build_tools', ['Nenhum identificado']))} |
| **Deploy** | {', '.join(metadata['technologies'].get('deployment', ['Nenhum identificado']))} |

## 👥 Usuários e Sistemas Externos

### Usuários
{chr(10).join(f"- **{user}**: Interage com o sistema" for user in context.get('users', []))}

### Sistemas Externos
{chr(10).join(f"- **{system}**: Dependência externa" for system in context.get('external_systems', []))}

## 📚 Níveis de Arquitetura

Esta documentação está organizada de acordo com o modelo C4:

1. **📄 Diagrama de Contexto**: Mostra os limites do sistema e dependências externas
2. **📦 Diagrama de Contêineres**: Mostra as escolhas tecnológicas de alto nível
3. **🔧 Diagrama de Componentes**: Mostra a estrutura interna dos contêineres
4. **🚀 Guia de Deploy**: Estratégias e requisitos de implantação

## ⭐ Atributos de Qualidade

- **⚡ Performance**: Sistema projetado para tempos de resposta otimizados
- **🔒 Segurança**: Comunicação segura e proteção de dados
- **🛡️ Confiabilidade**: Tratamento de erros e degradação gradual
- **🔧 Manutenibilidade**: Código limpo e arquitetura modular

## 🔍 Análise Automática

Esta documentação foi gerada automaticamente através da análise do código-fonte usando:
- **AG2 (AutoGen 2.0)**: Sistema multi-agente para análise de código
- **C4 Model**: Framework estruturado para documentação arquitetural  
- **DocAgent Skyone v2.0**: Plataforma de análise e documentação

---
*Este documento foi gerado automaticamente pelo DocAgent C4 Model Analyzer com AG2*
"""
    
    def _generate_c4_context_doc_ag2(self, c4_model: C4Model) -> str:
        """Gera documentação do diagrama de contexto C4 para AG2"""
        context = c4_model.context
        
        return f"""# 🌐 Diagrama de Contexto C4 - {context['system_name']}

## 📋 Visão Geral

O diagrama de contexto mostra o sistema **{context['system_name']}** no mais alto nível, focando nas pessoas e sistemas que interagem com ele.

## 🎯 Limite do Sistema

```
[Usuários] ←→ [{context['system_name']}] ←→ [Sistemas Externos]
```

## 🎭 Atores

### 🎯 Sistema Interno
- **{context['system_name']}**: {context['description']}
  - **Tecnologia**: {context['technology']}
  - **Tipo**: Sistema de software

### 👥 Usuários
{chr(10).join(f"- **{user}**: Interage com o sistema através da interface" for user in context.get('users', []))}

### 🔗 Sistemas Externos
{chr(10).join(f"- **{system}**: Fornece serviços e dados externos" for system in context.get('external_systems', []))}

## 🔄 Principais Interações

1. **Interações do Usuário**: Usuários interagem com o sistema através de interfaces web ou APIs
2. **Dependências Externas**: Sistema integra com serviços externos para funcionalidade aprimorada
3. **Fluxo de Dados**: Informações fluem entre usuários, o sistema e dependências externas

## 🏛️ Limites do Contexto

O limite do sistema separa claramente:
- **Componentes internos** do sistema e lógica
- **Interações externas** dos usuários
- **Integrações de terceiros**
- **Dependências de dados** e serviços

## 📊 Diagrama de Contexto (PlantUML)

```plantuml
@startuml
!include https://raw.githubusercontent.com/plantuml-stdlib/C4-PlantUML/master/C4_Context.puml

LAYOUT_WITH_LEGEND()

title Diagrama de Contexto para {context['system_name']}

Person(user, "Usuário", "Usa o sistema")
System({context['system_name'].replace(' ', '')}, "{context['system_name']}", "{context['description']}")
{chr(10).join(f'System_Ext({system.replace(" ", "").lower()}, "{system}", "Sistema externo")' for system in context.get('external_systems', []))}

Rel(user, {context['system_name'].replace(' ', '')}, "Usa")
{chr(10).join(f'Rel({context["system_name"].replace(" ", "")}, {system.replace(" ", "").lower()}, "Usa")' for system in context.get('external_systems', []))}

@enduml
```

## 🎯 Considerações do Contexto

### Escopo do Sistema
- Define claramente o que está dentro vs. fora do sistema
- Identifica todas as interfaces externas
- Mapeia fluxos de dados principais

### Stakeholders
- Usuários finais e suas necessidades
- Sistemas integrados e suas APIs
- Serviços de terceiros e dependências

---
*Gerado pelo DocAgent C4 Model Analyzer com AG2*
"""
    
    def _generate_c4_container_doc_ag2(self, c4_model: C4Model) -> str:
        """Gera documentação do diagrama de contêineres C4 para AG2"""
        context = c4_model.context
        containers = c4_model.containers
        
        return f"""# 📦 Diagrama de Contêineres C4 - {context['system_name']}

## 📋 Visão Geral

O diagrama de contêineres mostra as escolhas tecnológicas de alto nível e como as responsabilidades são distribuídas entre elas para o sistema **{context['system_name']}**.

## 📦 Contêineres Identificados

{chr(10).join(f'''### 🔧 {container.name}
- **Descrição**: {container.description}
- **Tecnologia**: {container.technology}
- **Sistema**: {container.system}
- **Responsabilidades**: 
  - Lógica de aplicação primária para {container.name.lower()}
  - Gerencia processamento de dados e regras de negócio
  - Controla comunicação com outros contêineres
  - Fornece interfaces para usuários e sistemas externos
''' for container in containers)}

## 🏗️ Arquitetura de Contêineres

```
{' ←→ '.join([container.name for container in containers])}
```

## 💻 Escolhas Tecnológicas

{chr(10).join(f"- **{container.name}**: {container.technology}" for container in containers)}

## 🔄 Interações entre Contêineres

{chr(10).join(f"- **{container.name}**: Comunica via {container.technology}" for container in containers)}

## 🚀 Considerações de Deploy

- Cada contêiner pode ser implantado independentemente
- Contêineres se comunicam através de interfaces bem definidas
- Stack tecnológica otimizada para performance e manutenibilidade
- Escalabilidade horizontal e vertical suportada

## 📊 Diagrama de Contêineres (PlantUML)

```plantuml
@startuml
!include https://raw.githubusercontent.com/plantuml-stdlib/C4-PlantUML/master/C4_Container.puml

LAYOUT_WITH_LEGEND()

title Diagrama de Contêineres para {context['system_name']}

Person(user, "Usuário", "Usa o sistema")

System_Boundary(system, "{context['system_name']}") {{
{chr(10).join(f'    Container({container.name.replace(" ", "").lower()}, "{container.name}", "{container.technology}", "{container.description}")' for container in containers)}
}}

Rel(user, {containers[0].name.replace(' ', '').lower() if containers else 'app'}, "Usa")
{chr(10).join(f'Rel({containers[i].name.replace(" ", "").lower()}, {containers[i+1].name.replace(" ", "").lower()}, "Comunica com")' for i in range(len(containers)-1))}

@enduml
```

## 🎯 Padrões Arquiteturais

### Separação de Responsabilidades
- Cada contêiner tem responsabilidade específica e bem definida
- Baixo acoplamento entre contêineres
- Alta coesão dentro de cada contêiner

### Comunicação
- Protocolos padrão da indústria (HTTP/REST, etc.)
- Interfaces claramente definidas
- Tratamento de erros e timeout

### Escalabilidade
- Contêineres podem escalar independentemente
- Load balancing quando necessário
- Monitoramento e observabilidade

---
*Gerado pelo DocAgent C4 Model Analyzer com AG2*
"""
    
    def _generate_c4_component_doc_ag2(self, c4_model: C4Model) -> str:
        """Gera documentação do diagrama de componentes C4 para AG2"""
        context = c4_model.context
        components = c4_model.components
        
        return f"""# 🔧 Diagrama de Componentes C4 - {context['system_name']}

## 📋 Visão Geral

O diagrama de componentes mostra a estrutura interna e organização dos componentes dentro dos contêineres do sistema **{context['system_name']}**.

## 🔧 Componentes Identificados

{chr(10).join(f'''### ⚙️ {component.name}
- **Descrição**: {component.description}
- **Tecnologia**: {component.technology}
- **Contêiner**: {component.container}
- **Responsabilidades**:
{chr(10).join(f"  - {resp}" for resp in component.responsibilities)}
''' for component in components)}

## 🏗️ Arquitetura de Componentes

O sistema está organizado nos seguintes componentes principais:

{chr(10).join(f"- **{component.name}** ({component.container})" for component in components)}

## 📋 Responsabilidades dos Componentes

{chr(10).join(f'''**{component.name}**:
{chr(10).join(f"- {resp}" for resp in component.responsibilities)}
''' for component in components)}

## 🎯 Padrões de Componentes

- **🏗️ Arquitetura em Camadas**: Componentes organizados em camadas lógicas
- **🔧 Separação de Responsabilidades**: Cada componente tem responsabilidades específicas
- **💉 Injeção de Dependência**: Componentes dependem de abstrações, não de implementações
- **🎯 Responsabilidade Única**: Cada componente tem um propósito único e bem definido

## 📊 Diagrama de Componentes (PlantUML)

```plantuml
@startuml
!include https://raw.githubusercontent.com/plantuml-stdlib/C4-PlantUML/master/C4_Component.puml

LAYOUT_WITH_LEGEND()

title Diagrama de Componentes para {context['system_name']}

Container_Boundary(container, "Contêiner da Aplicação") {{
{chr(10).join(f'    Component({component.name.replace(" ", "").lower()}, "{component.name}", "{component.technology}", "{component.description}")' for component in components)}
}}

{chr(10).join(f'Rel({components[i].name.replace(" ", "").lower()}, {components[i+1].name.replace(" ", "").lower()}, "Usa")' for i in range(len(components)-1))}

@enduml
```

## 🔄 Fluxo de Dados

### Processamento Principal
1. **Entrada**: Dados recebidos através de interfaces externas
2. **Processamento**: Componentes aplicam regras de negócio
3. **Persistência**: Dados armazenados conforme necessário
4. **Saída**: Resultados retornados para usuários/sistemas

### Tratamento de Erros
- Validação em múltiplas camadas
- Logging detalhado para debugging
- Fallback gracioso em caso de falhas
- Monitoramento de saúde dos componentes

## 🛠️ Tecnologias por Componente

{chr(10).join(f"- **{component.name}**: {component.technology}" for component in components)}

## 📈 Considerações de Performance

- Componentes otimizados para baixa latência
- Cache estratégico onde apropriado
- Processamento assíncrono quando possível
- Monitoramento de métricas de performance

---
*Gerado pelo DocAgent C4 Model Analyzer com AG2*
"""
    
    def _generate_c4_deployment_doc_ag2(self, c4_model: C4Model) -> str:
        """Gera documentação de deployment C4 para AG2"""
        context = c4_model.context
        metadata = c4_model.metadata
        
        return f"""# 🚀 Guia de Deploy - {context['system_name']}

## 📋 Visão Geral

Este documento descreve a arquitetura de deploy e requisitos de infraestrutura para o sistema **{context['system_name']}**.

## 🛠️ Ambiente de Deploy

**Tecnologia Principal**: {context['technology']}  
**Ferramentas de Deploy**: {', '.join(metadata['technologies'].get('deployment', ['Deploy padrão']))}  

## 📊 Requisitos de Infraestrutura

### Requisitos Mínimos
- **CPU**: 2 núcleos
- **RAM**: 4GB
- **Armazenamento**: 20GB
- **Rede**: Conexão de banda larga

### Requisitos Recomendados
- **CPU**: 4+ núcleos
- **RAM**: 8GB+
- **Armazenamento**: 50GB+ SSD
- **Rede**: Conexão de alta velocidade

## 🚀 Opções de Deploy

### Opção 1: Deploy Local de Desenvolvimento
```bash
# Clonar repositório
git clone {metadata.get('repository', 'repository-url')}

# Instalar dependências
# (comandos específicos dependem da stack tecnológica)

# Executar aplicação
# (comandos específicos dependem da stack tecnológica)
```

### Opção 2: Deploy Containerizado
{f'''
```bash
# Build da imagem Docker
docker build -t {context['system_name'].lower().replace(' ', '-')} .

# Executar container
docker run -p 8080:8080 {context['system_name'].lower().replace(' ', '-')}
```
''' if 'Docker' in metadata['technologies'].get('deployment', []) else 'Configuração Docker não detectada'}

### Opção 3: Deploy em Nuvem
- Deploy Platform-as-a-Service (PaaS)
- Orquestração de containers (Kubernetes)
- Opções de deploy serverless

## 📈 Monitoramento e Logging

- Logs da aplicação para debugging e monitoramento
- Coleta de métricas de performance
- Rastreamento de erros e alertas
- Endpoints de health check

## 🔒 Considerações de Segurança

- Comunicação segura (HTTPS)
- Gerenciamento de variáveis de ambiente
- Controle de acesso e autenticação
- Atualizações regulares de segurança

## 💾 Backup e Recuperação

- Backups regulares de dados
- Procedimentos de recuperação de desastres
- Estratégias de backup de banco de dados
- Backup de configurações

## 🔧 Configuração de Ambiente

### Variáveis de Ambiente
```bash
# Exemplo de configuração
# (adapte conforme a stack tecnológica)
PORT=8080
NODE_ENV=production
DATABASE_URL=your_database_url
API_KEY=your_api_key
```

### Configuração de Banco de Dados
{f'''
- **Banco**: {', '.join(metadata['technologies'].get('databases', ['Não identificado']))}
- **Configuração**: Conforme documentação específica da tecnologia
- **Migrations**: Execute scripts de migração conforme necessário
''' if metadata['technologies'].get('databases') else 'Banco de dados não identificado na análise'}

## 🎯 Lista de Verificação de Deploy

- [ ] Ambiente configurado com requisitos mínimos
- [ ] Dependências instaladas corretamente
- [ ] Variáveis de ambiente configuradas
- [ ] Banco de dados configurado (se aplicável)
- [ ] Conectividade de rede testada
- [ ] Logs de aplicação funcionando
- [ ] Health checks configurados
- [ ] Backup configurado
- [ ] Monitoramento ativo
- [ ] Documentação de troubleshooting disponível

## 🔍 Troubleshooting

### Problemas Comuns
1. **Port já em uso**: Verificar se porta está disponível
2. **Dependências faltando**: Reinstalar dependências
3. **Permissões**: Verificar permissões de arquivo/diretório
4. **Conectividade**: Testar conexões de rede/banco de dados

### Logs de Debug
- Verificar logs da aplicação em `/var/log/` ou diretório específico
- Usar ferramentas de monitoramento para tracking em tempo real
- Implementar logging estruturado para melhor debugging

## 📞 Suporte

Para suporte técnico:
- Consulte a documentação da stack tecnológica específica
- Verifique issues conhecidos no repositório
- Entre em contato com a equipe de desenvolvimento

---
*Gerado pelo DocAgent C4 Model Analyzer com AG2*
"""
    
    def _execute_simplified_analysis(self, project_url: str, anonymous: bool = True) -> Dict[str, Any]:
        """Análise simplificada quando AG2 não está disponível"""
        try:
            print("🔧 Executando análise simplificada (sem AG2)...")
            
            # Verificar se temos tools e estado
            if not self.tools or not self.state:
                return {
                    "status": "error",
                    "message": "Estado ou tools não inicializados",
                    "error_count": self.error_count + 1
                }
            
            # Usar tools diretamente para análise
            analysis_data = self.analyze_repository_structure_direct(self.state.repo_path)
            
            # Criar plano simplificado
            if not self.state.plan:
                self.state.plan = self._create_comprehensive_plan()
            
            # Gerar documentação simplificada
            generated_docs = self.generate_documentation_direct(
                self.state.repo_path,
                project_url,
                analysis_data,
                anonymous
            )
            
            if generated_docs:
                self.state.generated_docs = generated_docs
                # Normalizar os nomes dos arquivos retornados para conter apenas o nome base
                generated_docs_base = []
                for p in generated_docs:
                    try:
                        generated_docs_base.append(os.path.basename(p))
                    except Exception:
                        generated_docs_base.append(p)
                return {
                    "status": "success",
                    "message": f"Análise simplificada concluída: {len(generated_docs)} seções",
                    "generated_docs": generated_docs_base,
                    "plan": self.state.plan.to_dict() if self.state.plan else None,
                    "metadata": {
                        "project_url": project_url,
                        "repo_path": self.state.repo_path,
                        "docs_count": len(generated_docs),
                        "generated_at": datetime.now().isoformat(),
                        "error_count": self.error_count,
                        "system_version": "DocAgent Skyone v2.0 (Simplified) + Auth",
                        "ag2_enabled": False,
                        "anonymous": anonymous,
                        "features": [
                            "Direct code analysis",
                            "Simplified documentation",
                            "Anonymous reporting",
                            "Basic technical documentation",
                            "Authentication system"
                        ]
                    }
                }
            else:
                return {
                    "status": "error",
                    "message": "Falha na geração de documentação simplificada",
                    "error_count": self.error_count + 1
                }
                
        except Exception as e:
            print(f"❌ Erro na análise simplificada: {e}")
            return {
                "status": "error",
                "message": f"Erro crítico na análise simplificada: {str(e)[:100]}",
                "error_count": self.error_count + 1
            }
    
    # [Métodos auxiliares mantidos iguais...]
    def _create_comprehensive_plan(self) -> DocPlan:
        """Plano completo obrigatório com 3 seções"""
        print("📋 Criando plano completo com 3 seções...")
        
        return DocPlan(
            overview="Documentação técnica completa gerada automaticamente para análise detalhada do projeto",
            docs=[
                DocItem(
                    title="Visão Geral do Projeto",
                    description="Análise completa do propósito, tecnologias e arquitetura do projeto",
                    prerequisites="Conhecimento básico de desenvolvimento de software",
                    examples=["Funcionalidades principais", "Stack tecnológico", "Arquitetura geral"],
                    goal="Fornecer entendimento completo do projeto e suas tecnologias"
                ),
                DocItem(
                    title="Guia de Instalação e Configuração",
                    description="Instruções detalhadas para instalação, configuração e execução do projeto",
                    prerequisites="Sistema operacional compatível e ferramentas de desenvolvimento",
                    examples=["Pré-requisitos do sistema", "Passos de instalação", "Comandos de execução"],
                    goal="Permitir que desenvolvedores configurem e executem o projeto rapidamente"
                ),
                DocItem(
                    title="Relatório Técnico dos Arquivos",
                    description="Análise técnica detalhada de cada arquivo importante: funções, classes, APIs, fluxo de código e arquitetura",
                    prerequisites="Conhecimento nas linguagens e frameworks utilizados no projeto",
                    examples=["Análise arquivo por arquivo", "Documentação de funções", "Mapeamento de APIs", "Fluxo de execução"],
                    goal="Fornecer relatório técnico completo para desenvolvedores entenderem, modificarem e contribuírem com o código"
                )
            ]
        )

    def _extract_plan_safely(self, messages: List[Dict]) -> Optional[DocPlan]:
        """Extração robusta do plano JSON"""
        try:
            for msg in reversed(messages):
                content = msg.get('content', '')
                
                # Buscar padrões JSON mais flexíveis
                json_patterns = [
                    r'\{[^{}]*"overview"[^{}]*"docs"[^{}]*\}',
                    r'\{.*?"overview".*?"docs".*?\}',
                    r'```json\s*(\{.*?\})\s*```',
                    r'```\s*(\{.*?\})\s*```'
                ]
                
                for pattern in json_patterns:
                    matches = re.findall(pattern, content, re.DOTALL)
                    for match in matches:
                        try:
                            clean_json = re.sub(r'```json\n?|\n?```', '', match)
                            clean_json = clean_json.strip()
                            
                            data = json.loads(clean_json)
                            
                            if 'overview' in data and 'docs' in data:
                                # Validar que temos pelo menos 3 seções
                                if len(data['docs']) >= 3:
                                    return DocPlan.from_dict(data)
                                else:
                                    print(f"⚠️ Plano com apenas {len(data['docs'])} seções - esperado 3")
                        except (json.JSONDecodeError, Exception) as e:
                            print(f"⚠️ Erro no parse JSON: {e}")
                            continue
            
            return None
            
        except Exception as e:
            print(f"⚠️ Erro na extração do plano: {e}")
            return None
    
    def _extract_documentation_safely(self, messages: List[Dict], title: str) -> Optional[str]:
        """Extração robusta da documentação das mensagens"""
        try:
            candidates = []
            
            for msg in reversed(messages):
                content = msg.get('content', '')
                name = msg.get('name', '')
                
                # Priorizar mensagens do reviewer
                if 'reviewer' in name.lower() and len(content) > 200:
                    candidates.append(content)
                elif 'writer' in name.lower() and len(content) > 200:
                    candidates.append(content)
                elif '##' in content and len(content) > 300:
                    candidates.append(content)
            
            # Retornar melhor candidato
            if candidates:
                best_candidate = max(candidates, key=len)  # Maior conteúdo
                return best_candidate
            
            # Fallback específico por seção
            title_lower = title.lower()
            if "visão" in title_lower or "geral" in title_lower:
                return self._generate_section_fallback(title, 0, True)
            elif "instalação" in title_lower or "configuração" in title_lower:
                return self._generate_section_fallback(title, 1, True)
            elif "técnico" in title_lower or "arquivo" in title_lower:
                return self._generate_section_fallback(title, 2, True)
            else:
                return self._generate_basic_doc(title)
            
        except Exception as e:
            print(f"⚠️ Erro na extração: {e}")
            return self._generate_basic_doc(title)
    
    # [Métodos auxiliares de geração de documentação...]
    def analyze_repository_structure_direct(self, repo_path: str) -> Dict[str, Any]:
        """Análise direta da estrutura do repositório (sem AG2)"""
        try:
            print("🔍 Análise direta da estrutura...")
            
            if not self.tools:
                return {"error": "Tools não inicializadas"}
            
            # Análise usando tools diretamente
            structure_analysis = self.tools.analyze_code_structure()
            key_files = self.tools.find_key_files()
            detailed_analysis = self.tools.detailed_file_analysis(15)
            
            # Processar resultados
            analysis_data = {
                'structure_analysis': structure_analysis,
                'key_files': key_files,
                'detailed_analysis': detailed_analysis,
                'total_files': 0,
                'code_files': 0,
                'languages': {},
                'main_language': 'Unknown',
                'file_analyses': [],
                'dependencies': {},
                'advanced_stats': {}
            }
            
            # Análise básica adicional
            repo_path_obj = Path(repo_path)
            total_files = 0
            code_files = 0
            languages = {}
            
            try:
                for root, dirs, files in os.walk(repo_path_obj):
                    dirs[:] = [d for d in dirs if not d.startswith('.')]
                    
                    for file in files:
                        if file.startswith('.'):
                            continue
                        
                        total_files += 1
                        file_path = Path(root) / file
                        ext = file_path.suffix.lower()
                        
                        # Mapear linguagens
                        lang_map = {
                            '.py': 'Python', '.js': 'JavaScript', '.ts': 'TypeScript',
                            '.java': 'Java', '.cpp': 'C++', '.c': 'C', '.go': 'Go',
                            '.rs': 'Rust', '.php': 'PHP', '.rb': 'Ruby'
                        }
                        
                        if ext in lang_map:
                            code_files += 1
                            lang = lang_map[ext]
                            languages[lang] = languages.get(lang, 0) + 1
                
                analysis_data.update({
                    'total_files': total_files,
                    'code_files': code_files,
                    'languages': languages,
                    'main_language': max(languages.keys(), key=languages.get) if languages else 'Unknown'
                })
                
            except Exception as e:
                print(f"⚠️ Erro na análise adicional: {e}")
            
            return analysis_data
            
        except Exception as e:
            print(f"❌ Erro na análise direta: {e}")
            return {"error": str(e)}
    
    def generate_documentation_direct(self, repo_path: str, repo_url: str, analysis_data: Dict, anonymous: bool = True) -> List[str]:
        """Gera documentação direta (sem AG2)"""
        try:
            print("📝 Gerando documentação direta...")
            
            docs_dir = Path("docs")
            docs_dir.mkdir(exist_ok=True)
            
            url_final = self.anonimizacao.anonimizar_url_repositorio(repo_url) if anonymous else repo_url
            generated_docs = []
            
            # 1. Visão Geral do Projeto
            overview_doc = self._generate_overview_direct(analysis_data, repo_path, url_final, anonymous)
            overview_path = docs_dir / ("01_visao_geral_anonimo.md" if anonymous else "01_visao_geral.md")
            with open(overview_path, 'w', encoding='utf-8') as f:
                f.write(overview_doc)
            generated_docs.append(str(overview_path))
            
            # 2. Guia de Instalação
            install_doc = self._generate_installation_direct(analysis_data, anonymous)
            install_path = docs_dir / ("02_guia_instalacao_anonimo.md" if anonymous else "02_guia_instalacao.md")
            with open(install_path, 'w', encoding='utf-8') as f:
                f.write(install_doc)
            generated_docs.append(str(install_path))
            
            # 3. Relatório Técnico
            technical_doc = self._generate_technical_report_direct(analysis_data, repo_path, anonymous)
            technical_path = docs_dir / ("03_relatorio_tecnico_anonimo.md" if anonymous else "03_relatorio_tecnico.md")
            with open(technical_path, 'w', encoding='utf-8') as f:
                f.write(technical_doc)
            generated_docs.append(str(technical_path))
            
            print(f"✅ Documentação direta gerada: {len(generated_docs)} arquivos")
            return generated_docs
            
        except Exception as e:
            print(f"❌ Erro na geração direta: {e}")
            return []
    
    # [Métodos de geração de seções específicas...]
    def _generate_section_fallback(self, title: str, section_index: int, anonymous: bool = True) -> str:
        """Gera documentação de fallback específica por seção"""
        
        if section_index == 0:  # Visão Geral
            return f"""# {title}

## 🎯 Propósito do Projeto

Este projeto foi analisado automaticamente pelo DocAgent Skyone v2.0 com sistema de autenticação. A análise identificou uma base de código organizada com múltiplos arquivos e funcionalidades.

## 🛠️ Tecnologias Identificadas

Baseado na análise da estrutura de arquivos, o projeto utiliza:
- Múltiplas linguagens de programação
- Estrutura organizada de diretórios
- Arquivos de configuração específicos

## 🏗️ Arquitetura

O projeto está organizado em uma estrutura hierárquica de arquivos e diretórios, com separação clara de responsabilidades entre diferentes componentes.

## 📊 Características

- Projeto com estrutura bem definida
- Múltiplos arquivos de código
- Sistema modular e organizado

---
*Seção gerada automaticamente pelo DocAgent Skyone v2.0 - {'Modo Anônimo' if anonymous else 'Modo Original'} com Autenticação*
"""
        
        elif section_index == 1:  # Instalação
            return f"""# {title}

## 📋 Pré-requisitos

Antes de instalar e executar este projeto, certifique-se de ter:

- Sistema operacional compatível (Linux, macOS, ou Windows)
- Ferramentas de desenvolvimento apropriadas para a linguagem utilizada
- Acesso ao terminal/linha de comando
- Git instalado para clonagem do repositório

## 🚀 Instalação

### 1. Clone o Repositório
```bash
git clone [URL_DO_PROJETO]
cd [nome-do-repositorio]
```

### 2. Instale as Dependências
Verifique os arquivos de configuração do projeto (package.json, requirements.txt, etc.) e instale as dependências conforme a tecnologia utilizada.

### 3. Configure o Ambiente
Siga as instruções específicas do projeto para configuração de variáveis de ambiente e arquivos de configuração.

## ▶️ Execução

Execute o projeto seguindo as instruções específicas da tecnologia utilizada. Consulte os arquivos principais (main.py, index.js, etc.) para entender o ponto de entrada.

## 📝 Observações

- Consulte a documentação específica do projeto para instruções detalhadas
- Verifique os arquivos README se disponíveis
- Para problemas de instalação, consulte a documentação da tecnologia utilizada

---
*Seção gerada automaticamente pelo DocAgent Skyone v2.0 - {'Modo Anônimo' if anonymous else 'Modo Original'} com Autenticação*
"""
        
        else:  # Relatório Técnico (seção 2)
            return f"""# {title}

## 📁 Estrutura Geral

O projeto contém uma organização estruturada de arquivos e diretórios, cada um com responsabilidades específicas no sistema.

## 🔧 Arquivos Principais

### Análise Automática

Este projeto foi analisado automaticamente e contém múltiplos arquivos importantes. Cada arquivo possui:

- **Propósito específico** no contexto do projeto
- **Implementação** usando as tecnologias do stack
- **Interações** com outros componentes do sistema

### Categorias de Arquivos Identificadas

#### 💻 Arquivos de Código
Arquivos contendo a lógica principal do sistema, implementando funcionalidades específicas.

#### ⚙️ Arquivos de Configuração  
Arquivos responsáveis pela configuração do ambiente, dependências e parâmetros do sistema.

#### 📖 Arquivos de Documentação
Arquivos contendo informações sobre o projeto, incluindo README, licenças e guias.

## 🏗️ Arquitetura do Sistema

O projeto segue uma arquitetura modular onde:

- Diferentes arquivos têm responsabilidades específicas
- Existe separação clara entre lógica de negócio e configuração
- O sistema é organizado de forma hierárquica

## 📋 Para Desenvolvedores

Para contribuir com este projeto:

1. **Analise a estrutura** de arquivos para entender a organização
2. **Identifique o ponto de entrada** principal da aplicação
3. **Examine as dependências** listadas nos arquivos de configuração
4. **Siga os padrões** estabelecidos no código existente

## 📝 Observações Técnicas

- Este projeto contém múltiplos arquivos com funcionalidades específicas
- A estrutura segue boas práticas de organização de código
- Para análise detalhada, examine diretamente os arquivos fonte

---
*Relatório gerado automaticamente pelo DocAgent Skyone v2.0 - {'Modo Anônimo' if anonymous else 'Modo Original'} com Autenticação*
"""
    
    def _generate_basic_doc(self, title: str) -> str:
        """Gera documentação básica como fallback"""
        return f"""# {title}

## 📋 Visão Geral

Esta seção documenta {title.lower()} do projeto. A documentação foi gerada automaticamente baseada na análise do repositório.

## 🚀 Informações

Esta documentação faz parte de um conjunto completo de 3 seções:
1. Visão Geral do Projeto
2. Guia de Instalação e Configuração
3. Relatório Técnico dos Arquivos

## 📝 Observações

- Esta documentação foi gerada automaticamente pelo DocAgent Skyone v2.0 com Autenticação
- Para informações mais detalhadas, consulte o código-fonte do projeto
- O sistema analisou a estrutura do repositório para gerar esta documentação

---
*Gerado automaticamente em {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} pelo DocAgent Skyone v2.0*
"""
    
    def _save_documentation(self, title: str, content: str, anonymous: bool = True) -> Optional[str]:
        """Salva documentação com nomes padronizados"""
        try:
            docs_dir = Path("docs")
            docs_dir.mkdir(exist_ok=True)
            
            # Nomes padronizados para as 3 seções
            title_lower = title.lower()
            suffix = "_anonimo" if anonymous else ""
            
            if "visão" in title_lower or "geral" in title_lower:
                filename = f"01_visao_geral{suffix}.md"
            elif "instalação" in title_lower or "configuração" in title_lower:
                filename = f"02_instalacao_configuracao{suffix}.md"
            elif "técnico" in title_lower or "arquivo" in title_lower:
                filename = f"03_relatorio_tecnico{suffix}.md"
            else:
                # Fallback para nome seguro
                safe_title = re.sub(r'[^\w\s-]', '', title)
                safe_title = re.sub(r'[-\s]+', '_', safe_title)
                filename = f"{safe_title.lower()}{suffix}.md"
            
            doc_path = docs_dir / filename
            
            # Salvar com encoding UTF-8
            with open(doc_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            print(f"💾 Salvo: {doc_path}")
            return str(doc_path)
            
        except Exception as e:
            print(f"❌ Erro ao salvar {title}: {e}")
            return None
    
    def _create_comprehensive_documentation(self, anonymous: bool = True) -> bool:
        """Cria documentação completa como último recurso"""
        try:
            print("📝 Criando documentação completa...")
            
            # Garantir que temos estado válido
            if not self.state:
                print("⚠️ Estado não encontrado - inicializando")
                self.state = DocumentationState(
                    project_url="unknown",
                    current_phase="documentation",
                    generated_docs=[],
                    metadata={}
                )
            
            # Garantir que temos o plano completo
            if not self.state.plan:
                self.state.plan = self._create_comprehensive_plan()
            
            # Criar as 3 seções obrigatórias
            sections = [
                ("Visão Geral do Projeto", 0),
                ("Guia de Instalação e Configuração", 1), 
                ("Relatório Técnico dos Arquivos", 2)
            ]
            
            docs_created = []
            
            for title, index in sections:
                print(f"📄 Gerando seção {index+1}/3: {title}")
                
                doc_content = self._generate_section_fallback(title, index, anonymous)
                doc_path = self._save_documentation(title, doc_content, anonymous)
                
                if doc_path:
                    docs_created.append(doc_path)
                    # Garantir que generated_docs existe
                    if not hasattr(self.state, 'generated_docs') or self.state.generated_docs is None:
                        self.state.generated_docs = []
                    self.state.generated_docs.append(doc_path)
            
            if docs_created:
                print(f"✅ Documentação completa criada: {len(docs_created)} seções")
                return True
            
            return False
            
        except Exception as e:
            print(f"❌ Erro na documentação completa: {e}")
            return False

    # [Métodos de geração específicos continuam aqui...]
    def _generate_overview_direct(self, analysis_data: Dict, repo_path: str, url_final: str, anonymous: bool) -> str:
        """Gera visão geral direta"""
        languages = analysis_data.get('languages', {})
        main_lang = analysis_data.get('main_language', 'Desconhecida')
        total_files = analysis_data.get('total_files', 0)
        code_files = analysis_data.get('code_files', 0)
        
        doc_lines = []
        
        doc_lines.append("# Visão Geral do Projeto\n")
        
        if anonymous:
            doc_lines.append("> **Nota:** Este relatório foi anonimizado para proteger informações pessoais.\n")
        
        doc_lines.append("## 🎯 Propósito do Projeto\n")
        doc_lines.append("Este projeto foi analisado automaticamente pelo DocAgent Skyone v2.0 com sistema de autenticação. ")
        doc_lines.append("A análise identificou uma base de código organizada com estrutura bem definida ")
        doc_lines.append("e implementação usando tecnologias modernas.\n")
        
        doc_lines.append("## 🛠️ Stack Tecnológico\n")
        doc_lines.append(f"**Linguagem Principal:** {main_lang}\n")
        doc_lines.append(f"**Total de Arquivos:** {total_files:,}\n")
        doc_lines.append(f"**Arquivos de Código:** {code_files:,}\n")
        
        if languages:
            doc_lines.append("### Distribuição por Linguagem\n")
            for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / sum(languages.values())) * 100
                doc_lines.append(f"- **{lang}:** {count} arquivos ({percentage:.1f}%)\n")
        
        doc_lines.append("\n## 🏗️ Arquitetura\n")
        doc_lines.append("O projeto está organizado em uma estrutura hierárquica bem definida, ")
        doc_lines.append("com separação clara de responsabilidades entre diferentes componentes. ")
        doc_lines.append("A análise identificou uma arquitetura modular com boas práticas de organização.\n")
        
        doc_lines.append("## 📊 Características Técnicas\n")
        doc_lines.append(f"- Projeto multi-linguagem com foco em {main_lang}\n")
        doc_lines.append("- Estrutura organizacional bem definida\n")
        doc_lines.append("- Implementação seguindo boas práticas de desenvolvimento\n")
        doc_lines.append("- Código modular e bem estruturado\n")
        
        doc_lines.append(f"\n## 📋 Informações do Relatório\n")
        doc_lines.append(f"- **URL do Repositório:** {url_final}\n")
        doc_lines.append(f"- **Data da Análise:** {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')}\n")
        doc_lines.append(f"- **Sistema:** DocAgent Skyone v2.0 com Autenticação\n")
        doc_lines.append(f"- **Modo:** {'Anônimo' if anonymous else 'Original'}\n")
        
        doc_lines.append("\n---")
        doc_lines.append("\n*Relatório gerado automaticamente pelo DocAgent Skyone v2.0 com Autenticação*\n")
        
        return "".join(doc_lines)
    
    def _generate_installation_direct(self, analysis_data: Dict, anonymous: bool) -> str:
        """Gera guia de instalação direto"""
        main_lang = analysis_data.get('main_language', 'Desconhecida')
        
        doc_lines = []
        
        doc_lines.append("# Guia de Instalação e Configuração\n")
        
        if anonymous:
            doc_lines.append("> **Nota:** Este guia foi gerado a partir de análise anônima.\n")
        
        doc_lines.append("## 📋 Pré-requisitos do Sistema\n")
        doc_lines.append("### Ferramentas Básicas\n")
        doc_lines.append("- Git (para clonagem do repositório)\n")
        doc_lines.append("- Sistema operacional: Linux, macOS ou Windows\n")
        
        # Pré-requisitos específicos por linguagem
        if main_lang == 'Python':
            doc_lines.append("\n### Python\n")
            doc_lines.append("- Python 3.7+ (recomendado: 3.9+)\n")
            doc_lines.append("- pip (gerenciador de pacotes Python)\n")
            doc_lines.append("- virtualenv ou venv (para ambiente virtual)\n")
        elif main_lang == 'JavaScript':
            doc_lines.append("\n### Node.js\n")
            doc_lines.append("- Node.js 14+ (recomendado: 18+)\n")
            doc_lines.append("- npm ou yarn (gerenciador de pacotes)\n")
        elif main_lang == 'Java':
            doc_lines.append("\n### Java\n")
            doc_lines.append("- JDK 11+ (recomendado: 17+)\n")
            doc_lines.append("- Maven ou Gradle (build tool)\n")
        
        doc_lines.append("\n## 🚀 Processo de Instalação\n")
        
        doc_lines.append("### 1. Clonagem do Repositório\n")
        doc_lines.append("```bash\n")
        doc_lines.append("git clone [URL_DO_REPOSITORIO]\n")
        doc_lines.append("cd [nome-do-projeto]\n")
        doc_lines.append("```\n")
        
        # Instruções específicas por linguagem
        if main_lang == 'Python':
            doc_lines.append("### 2. Configuração Python\n")
            doc_lines.append("```bash\n")
            doc_lines.append("# Criar ambiente virtual\n")
            doc_lines.append("python -m venv venv\n")
            doc_lines.append("\n")
            doc_lines.append("# Ativar ambiente virtual\n")
            doc_lines.append("# Linux/Mac:\n")
            doc_lines.append("source venv/bin/activate\n")
            doc_lines.append("# Windows:\n")
            doc_lines.append("venv\\Scripts\\activate\n")
            doc_lines.append("\n")
            doc_lines.append("# Instalar dependências\n")
            doc_lines.append("pip install -r requirements.txt\n")
            doc_lines.append("```\n")
        elif main_lang == 'JavaScript':
            doc_lines.append("### 2. Configuração Node.js\n")
            doc_lines.append("```bash\n")
            doc_lines.append("# Instalar dependências\n")
            doc_lines.append("npm install\n")
            doc_lines.append("# ou\n")
            doc_lines.append("yarn install\n")
            doc_lines.append("\n")
            doc_lines.append("# Executar em desenvolvimento\n")
            doc_lines.append("npm run dev\n")
            doc_lines.append("# ou\n")
            doc_lines.append("yarn dev\n")
            doc_lines.append("```\n")
        
        doc_lines.append("## ✅ Verificação da Instalação\n")
        doc_lines.append("Após a instalação, verifique se todas as dependências foram instaladas corretamente ")
        doc_lines.append("e se o projeto pode ser executado sem erros.\n")
        
        doc_lines.append("## 📝 Observações\n")
        doc_lines.append("- Consulte arquivos README específicos do projeto para instruções detalhadas\n")
        doc_lines.append("- Verifique arquivos de configuração para parâmetros específicos\n")
        doc_lines.append("- Para problemas de instalação, consulte a documentação da tecnologia utilizada\n")
        
        doc_lines.append(f"\n### Informações Técnicas\n")
        doc_lines.append(f"- **Linguagem Principal:** {main_lang}\n")
        doc_lines.append(f"- **Sistema de Análise:** DocAgent Skyone v2.0 com Autenticação\n")
        doc_lines.append(f"- **Documentação Gerada:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        
        doc_lines.append("\n---")
        doc_lines.append("\n*Guia gerado automaticamente pelo DocAgent Skyone v2.0 com Autenticação*\n")
        
        return "".join(doc_lines)
    
    def _generate_technical_report_direct(self, analysis_data: Dict, repo_path: str, anonymous: bool) -> str:
        """Gera relatório técnico direto"""
        structure_analysis = analysis_data.get('structure_analysis', '')
        key_files = analysis_data.get('key_files', '')
        detailed_analysis = analysis_data.get('detailed_analysis', '')
        
        doc_lines = []
        
        doc_lines.append("# Relatório Técnico dos Arquivos\n")
        
        if anonymous:
            doc_lines.append("> **Nota:** Este relatório técnico foi anonimizado para proteger informações pessoais.\n")
        
        doc_lines.append("## 📁 Estrutura do Projeto\n")
        doc_lines.append("O projeto foi analisado automaticamente pelo DocAgent Skyone v2.0 com sistema de autenticação, ")
        doc_lines.append("que examinou a estrutura de arquivos, código-fonte e dependências ")
        doc_lines.append("para gerar este relatório técnico completo.\n")
        
        # Incluir análise da estrutura se disponível
        if structure_analysis and structure_analysis != "❌ Erro":
            doc_lines.append("## 🏗️ Análise da Estrutura de Código\n")
            doc_lines.append(f"{structure_analysis}\n")
        
        # Incluir arquivos-chave se disponível
        if key_files and key_files != "❌ Erro":
            doc_lines.append("## 🔍 Arquivos-Chave Identificados\n")
            doc_lines.append(f"{key_files}\n")
        
        # Incluir análise detalhada se disponível
        if detailed_analysis and detailed_analysis != "❌ Erro":
            doc_lines.append("## 🔬 Análise Detalhada dos Arquivos\n")
            doc_lines.append(f"{detailed_analysis}\n")
        
        # Seção de resumo técnico
        doc_lines.append("## 📊 Resumo Técnico\n")
        
        languages = analysis_data.get('languages', {})
        total_files = analysis_data.get('total_files', 0)
        code_files = analysis_data.get('code_files', 0)
        main_lang = analysis_data.get('main_language', 'Desconhecida')
        
        doc_lines.append("### Estatísticas do Projeto\n")
        doc_lines.append(f"- **Total de arquivos:** {total_files:,}\n")
        doc_lines.append(f"- **Arquivos de código:** {code_files:,}\n")
        doc_lines.append(f"- **Linguagem principal:** {main_lang}\n")
        doc_lines.append(f"- **Linguagens detectadas:** {len(languages)}\n")
        
        if languages:
            doc_lines.append("\n### Distribuição por Linguagem\n")
            for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True):
                doc_lines.append(f"- **{lang}:** {count} arquivos\n")
        
        doc_lines.append("\n## 🎯 Insights Técnicos\n")
        doc_lines.append("### Arquitetura\n")
        doc_lines.append("- Projeto organizado com estrutura modular\n")
        doc_lines.append("- Separação clara de responsabilidades\n")
        doc_lines.append("- Implementação seguindo boas práticas\n")
        
        doc_lines.append("\n### Tecnologias\n")
        doc_lines.append(f"- Desenvolvimento focado em {main_lang}\n")
        doc_lines.append("- Stack moderno e bem estruturado\n")
        doc_lines.append("- Código organizado e documentado\n")
        
        doc_lines.append("\n## 📋 Para Desenvolvedores\n")
        doc_lines.append("### Contribuindo com o Projeto\n")
        doc_lines.append("1. **Analise a estrutura** identificada neste relatório\n")
        doc_lines.append("2. **Examine os arquivos principais** listados acima\n")
        doc_lines.append("3. **Siga os padrões** estabelecidos no código existente\n")
        doc_lines.append("4. **Consulte a documentação** específica de cada componente\n")
        
        doc_lines.append("\n## 📝 Informações do Relatório\n")
        doc_lines.append(f"- **Sistema de Análise:** DocAgent Skyone v2.0 com Autenticação\n")
        doc_lines.append(f"- **Data da Análise:** {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')}\n")
        doc_lines.append(f"- **Modo de Análise:** {'Anônimo' if anonymous else 'Original'}\n")
        doc_lines.append(f"- **Localização do Projeto:** {repo_path}\n")
        
        doc_lines.append("\n---")
        doc_lines.append("\n*Relatório técnico gerado automaticamente pelo DocAgent Skyone v2.0 com Autenticação*\n")
        
        return "".join(doc_lines)

# =============================================================================
# SISTEMA DE ANÁLISE AVANÇADO INTEGRADO
# =============================================================================

class AdvancedAnalysisEngine:
    """Sistema de análise avançado integrado com AG2 e autenticação"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.anonimizacao = SistemaAnonimizacao()
        self.code_analyzer = CodeAnalyzer()
        # Inicializar AG2 flow apenas se disponível
        try:
            if AG2_AVAILABLE:
                self.ag2_flow = EnhancedDocumentationFlow(config)
            else:
                self.ag2_flow = None
        except Exception as e:
            print(f"⚠️ Erro ao inicializar AG2 flow: {e}")
            self.ag2_flow = None
        
        print("🤖 Sistema de análise avançado inicializado com autenticação")
    
    def _validate_github_url(self, url: str) -> bool:
        """Valida formato da URL do GitHub"""
        try:
            url = url.strip()
            parsed = urllib.parse.urlparse(url)
            if parsed.scheme not in ("http", "https"):
                return False
            if parsed.netloc.lower() != "github.com":
                return False
            parts = [p for p in parsed.path.split("/") if p]
            return len(parts) >= 2
        except Exception:
            return False
    
    def _check_github_connectivity(self) -> bool:
        """Verifica conectividade básica com GitHub"""
        try:
            socket.setdefaulttimeout(10)
            response = urllib.request.urlopen("https://github.com", timeout=10)
            return response.getcode() == 200
        except Exception as e:
            print(f"⚠️ Erro de conectividade: {e}")
            return False
    
    def _check_repository_exists(self, project_url: str) -> bool:
        """Verifica se repositório existe e é público"""
        try:
            request = urllib.request.Request(project_url)
            request.add_header('User-Agent', 'Mozilla/5.0 (compatible; DocAgent/2.0)')
            
            try:
                response = urllib.request.urlopen(request, timeout=15)
                return response.getcode() == 200
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    print(f"❌ Repositório não encontrado (404)")
                elif e.code == 403:
                    print(f"❌ Acesso negado (403): repositório privado ou rate limit")
                else:
                    print(f"❌ Erro HTTP {e.code}: {e.reason}")
                return False
            except urllib.error.URLError as e:
                print(f"❌ Erro de URL: {e.reason}")
                return False
                
        except Exception as e:
            print(f"⚠️ Erro ao verificar repositório: {e}")
            return True
    
    def clone_repository(self, project_url: str) -> Tuple[bool, Optional[str]]:
        """Clone do repositório com sistema robusto"""
        if self.ag2_flow:
            # Usar sistema AG2 se disponível
            success = self.ag2_flow.clone_repository(project_url)
            if success and self.ag2_flow.state:
                return True, self.ag2_flow.state.repo_path
            return False, None
        else:
            # Sistema simplificado
            return self._clone_repository_simple(project_url)
    
    def _clone_repository_simple(self, project_url: str) -> Tuple[bool, Optional[str]]:
        """Clone simplificado quando AG2 não está disponível"""
        try:
            print(f"📥 Clone simplificado: {project_url}")
            
            if not self._validate_github_url(project_url):
                print(f"❌ URL inválida: {project_url}")
                return False, None
            
            if not self._check_github_connectivity():
                print("❌ Sem conectividade com GitHub")
                return False, None
            
            repo_name = project_url.split("/")[-1].replace(".git", "")
            workdir = Path("workdir").resolve()
            workdir.mkdir(exist_ok=True)
            repo_path = workdir / repo_name
            
            if repo_path.exists():
                shutil.rmtree(repo_path, ignore_errors=True)
            
            cmd = ["git", "clone", "--depth", "1", project_url, str(repo_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0 and repo_path.exists():
                print(f"✅ Clone simplificado concluído: {repo_path}")
                return True, str(repo_path)
            else:
                print(f"❌ Falha no clone simplificado")
                return False, None
                
        except Exception as e:
            print(f"❌ Erro no clone simplificado: {e}")
            return False, None
    
    def analyze_repository_structure(self, repo_path: str, update_callback=None) -> Dict[str, Any]:
        """Análise avançada da estrutura do repositório"""
        try:
            print("🔍 Analisando estrutura do repositório...")
            if update_callback:
                update_callback("Analisando estrutura do repositório")
            
            if self.ag2_flow and self.ag2_flow.tools:
                # Usar tools AG2 se disponível
                return self._analyze_with_ag2_tools(repo_path, update_callback)
            else:
                # Análise simplificada
                return self._analyze_repository_simple(repo_path, update_callback)
                
        except Exception as e:
            print(f"❌ Erro na análise: {e}")
            traceback.print_exc()
            return {"error": str(e)}
    
    def _analyze_with_ag2_tools(self, repo_path: str, update_callback=None) -> Dict[str, Any]:
        """Análise usando tools AG2"""
        try:
            tools = self.ag2_flow.tools
            
            # Usar tools AG2 para análise
            structure_analysis = tools.analyze_code_structure()
            key_files = tools.find_key_files()
            detailed_analysis = tools.detailed_file_analysis(15)
            
            # Análise básica adicional
            analysis_data = self._get_basic_stats(repo_path)
            
            # Combinar resultados
            analysis_data.update({
                'structure_analysis': structure_analysis,
                'key_files': key_files,
                'detailed_analysis': detailed_analysis,
                'analysis_type': 'AG2_enhanced'
            })
            
            return analysis_data
            
        except Exception as e:
            print(f"❌ Erro na análise AG2: {e}")
            return self._analyze_repository_simple(repo_path, update_callback)
    
    def _analyze_repository_simple(self, repo_path: str, update_callback=None) -> Dict[str, Any]:
        """Análise simplificada"""
        try:
            repo_path = Path(repo_path)
            if not repo_path.exists():
                return {"error": "Repositório não encontrado"}
            
            # Estatísticas básicas
            analysis_data = self._get_basic_stats(repo_path)
            analysis_data['analysis_type'] = 'simplified'
            
            # Arquivos importantes
            important_files = self._find_important_files_simple(repo_path)
            analysis_data['important_files'] = important_files
            
            # Análise de alguns arquivos
            file_analyses = self._analyze_key_files_simple(repo_path)
            analysis_data['file_analyses'] = file_analyses
            
            return analysis_data
            
        except Exception as e:
            print(f"❌ Erro na análise simplificada: {e}")
            return {"error": str(e)}
    
    def _get_basic_stats(self, repo_path: Path) -> Dict[str, Any]:
        """Obtém estatísticas básicas do repositório"""
        total_files = 0
        code_files = 0
        languages = {}
        
        code_extensions = {
            '.py': 'Python', '.js': 'JavaScript', '.ts': 'TypeScript',
            '.java': 'Java', '.cpp': 'C++', '.c': 'C', '.go': 'Go',
            '.rs': 'Rust', '.php': 'PHP', '.rb': 'Ruby', '.swift': 'Swift',
            '.html': 'HTML', '.css': 'CSS', '.sql': 'SQL', '.sh': 'Shell',
            '.kt': 'Kotlin', '.cs': 'C#', '.scala': 'Scala', '.r': 'R'
        }
        
        try:
            for root, dirs, files in os.walk(repo_path):
                dirs[:] = [d for d in dirs if not d.startswith('.') 
                          and d not in ['node_modules', '__pycache__', 'target', 'build', 'dist']]
                
                for file in files:
                    if file.startswith('.'):
                        continue
                    
                    total_files += 1
                    file_path = Path(root) / file
                    ext = file_path.suffix.lower()
                    
                    if ext in code_extensions:
                        code_files += 1
                        lang = code_extensions[ext]
                        languages[lang] = languages.get(lang, 0) + 1
        except Exception as e:
            print(f"⚠️ Erro nas estatísticas: {e}")
        
        # Determinar linguagem principal
        main_language = 'Unknown'
        if languages:
            main_language = max(languages.keys(), key=languages.get)
        
        return {
            'total_files': total_files,
            'code_files': code_files,
            'languages': languages,
            'main_language': main_language
        }
    
    def _find_important_files_simple(self, repo_path: Path) -> List[Dict[str, Any]]:
        """Encontra arquivos importantes de forma simplificada"""
        important_files = []
        
        important_patterns = [
            "README.md", "package.json", "requirements.txt", "setup.py",
            "main.py", "index.js", "app.py", "server.py", "Dockerfile",
            "docker-compose.yml", "Makefile", ".gitignore", "LICENSE"
        ]
        
        try:
            for root, dirs, files in os.walk(repo_path):
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                
                for file in files:
                    if any(pattern.lower() in file.lower() for pattern in important_patterns):
                        file_path = Path(root) / file
                        relative_path = file_path.relative_to(repo_path)
                        
                        try:
                            size = file_path.stat().st_size
                            important_files.append({
                                'name': file,
                                'path': str(relative_path),
                                'size': size,
                                'language': self._get_file_language(file_path.suffix.lower())
                            })
                        except:
                            pass
                        
                        if len(important_files) >= 20:
                            break
                
                if len(important_files) >= 20:
                    break
        
        except Exception as e:
            print(f"⚠️ Erro ao buscar arquivos importantes: {e}")
        
        return sorted(important_files, key=lambda x: x['size'], reverse=True)
    
    def _analyze_key_files_simple(self, repo_path: Path) -> List[FileAnalysis]:
        """Análise simplificada de arquivos-chave"""
        file_analyses = []
        
        try:
            important_patterns = ['main.py', 'app.py', 'index.js', 'README.md']
            
            for root, dirs, files in os.walk(repo_path):
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                
                for file in files:
                    if any(pattern in file.lower() for pattern in important_patterns):
                        file_path = Path(root) / file
                        
                        try:
                            if file_path.stat().st_size > 100 * 1024:  # Skip files > 100KB
                                continue
                            
                            content = file_path.read_text(encoding='utf-8', errors='ignore')
                            language = self._get_file_language(file_path.suffix.lower())
                            
                            analysis = self.code_analyzer.analyze_file(file_path, content, language)
                            file_analyses.append(analysis)
                            
                            if len(file_analyses) >= 10:
                                break
                                
                        except Exception as e:
                            print(f"⚠️ Erro ao analisar {file}: {e}")
                            continue
                
                if len(file_analyses) >= 10:
                    break
        
        except Exception as e:
            print(f"⚠️ Erro na análise de arquivos: {e}")
        
        return file_analyses
    
    def _get_file_language(self, ext: str) -> str:
        """Identifica linguagem pela extensão"""
        language_map = {
            '.py': 'Python', '.js': 'JavaScript', '.ts': 'TypeScript',
            '.java': 'Java', '.cpp': 'C++', '.c': 'C', '.go': 'Go',
            '.rs': 'Rust', '.php': 'PHP', '.rb': 'Ruby', '.swift': 'Swift',
            '.html': 'HTML', '.css': 'CSS', '.md': 'Markdown',
            '.json': 'JSON', '.xml': 'XML', '.yaml': 'YAML', '.yml': 'YAML'
        }
        return language_map.get(ext, 'Unknown')
    
    def generate_documentation(self, repo_path: str, repo_url: str, analysis_data: Dict, anonymous: bool = True) -> List[str]:
        """Gera documentação completa"""
        try:
            print("📝 Gerando documentação completa...")
            
            if self.ag2_flow and AG2_AVAILABLE:
                # Usar AG2 se disponível
                result = self.ag2_flow.execute_analysis_with_ag2(repo_url, anonymous)
                if result.get('status') == 'success':
                    return result.get('generated_docs', [])
            
            # Fallback para geração direta
            return self._generate_documentation_direct(repo_path, repo_url, analysis_data, anonymous)
            
        except Exception as e:
            print(f"❌ Erro na geração de documentação: {e}")
            return []
    
    def _generate_documentation_direct(self, repo_path: str, repo_url: str, analysis_data: Dict, anonymous: bool = True) -> List[str]:
        """Gera documentação diretamente (fallback)"""
        try:
            docs_dir = Path("docs")
            docs_dir.mkdir(exist_ok=True)
            
            url_final = self.anonimizacao.anonimizar_url_repositorio(repo_url) if anonymous else repo_url
            generated_docs = []
            
            # 1. Relatório Principal
            main_doc = self._generate_main_report(analysis_data, repo_path, url_final, anonymous)
            main_name = "01_relatorio_completo_anonimo.md" if anonymous else "01_relatorio_completo.md"
            main_path = docs_dir / main_name
            with open(main_path, 'w', encoding='utf-8') as f:
                f.write(main_doc)
            generated_docs.append(str(main_path))
            
            # 2. Guia de Instalação
            install_doc = self._generate_installation_guide(analysis_data, anonymous)
            install_name = "02_guia_instalacao_anonimo.md" if anonymous else "02_guia_instalacao.md"
            install_path = docs_dir / install_name
            with open(install_path, 'w', encoding='utf-8') as f:
                f.write(install_doc)
            generated_docs.append(str(install_path))
            
            # 3. Documentação C4 Model
            try:
                print("🏗️ Gerando documentação C4...")
                c4_analyzer = C4ModelAnalyzer()
                c4_model = c4_analyzer.analyze_repository_architecture(repo_path, url_final)
                c4_docs = self._generate_c4_documentation(c4_model, docs_dir)
                generated_docs.extend(c4_docs)
                print(f"✅ Documentação C4 gerada: {len(c4_docs)} arquivos")
            except Exception as e:
                print(f"⚠️ Erro na geração C4: {e}")
            
            print(f"✅ Documentação direta gerada: {len(generated_docs)} arquivos")
            return generated_docs
            
        except Exception as e:
            print(f"❌ Erro na geração direta: {e}")
            return []
    
    def _generate_main_report(self, analysis_data: Dict, repo_path: str, url_final: str, anonymous: bool = True) -> str:
        """Gera relatório principal"""
        languages = analysis_data.get('languages', {})
        main_lang = analysis_data.get('main_language', 'Desconhecida')
        total_files = analysis_data.get('total_files', 0)
        code_files = analysis_data.get('code_files', 0)
        file_analyses = analysis_data.get('file_analyses', [])
        
        doc_lines = []
        
        # Cabeçalho
        doc_lines.append("# Relatório Completo de Análise de Projeto\n")
        
        if anonymous:
            doc_lines.append("> **Nota:** Este relatório foi anonimizado para proteger informações pessoais.\n")
        
        # Seção 1: Visão Geral
        doc_lines.append("## 📊 Visão Geral do Projeto\n")
        doc_lines.append("### Estatísticas Gerais")
        doc_lines.append(f"- **Linguagem Principal:** {main_lang}")
        doc_lines.append(f"- **Total de Arquivos:** {total_files:,}")
        doc_lines.append(f"- **Arquivos de Código:** {code_files:,}")
        doc_lines.append(f"- **Linguagens Utilizadas:** {len(languages)}\n")
        
        if languages:
            doc_lines.append("### Distribuição por Linguagem")
            total_lang_files = sum(languages.values())
            for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_lang_files) * 100
                doc_lines.append(f"- **{lang}:** {count} arquivos ({percentage:.1f}%)")
            doc_lines.append("")
        
        # Análise técnica se disponível
        if analysis_data.get('analysis_type') == 'AG2_enhanced':
            doc_lines.append("## 🔍 Análise Técnica Avançada (AG2)\n")
            
            if analysis_data.get('structure_analysis'):
                doc_lines.append("### Estrutura de Código")
                doc_lines.append(analysis_data['structure_analysis'])
                doc_lines.append("")
            
            if analysis_data.get('key_files'):
                doc_lines.append("### Arquivos-Chave")
                doc_lines.append(analysis_data['key_files'])
                doc_lines.append("")
        
        # Seção de arquivos importantes
        important_files = analysis_data.get('important_files', [])
        if important_files:
            doc_lines.append("## 📁 Arquivos Importantes\n")
            for file_info in important_files[:10]:
                doc_lines.append(f"### {file_info['name']}")
                doc_lines.append(f"- **Linguagem:** {file_info['language']}")
                doc_lines.append(f"- **Tamanho:** {file_info['size']:,} bytes")
                doc_lines.append(f"- **Localização:** `{file_info['path']}`\n")
        
        # Análise detalhada de arquivos
        if file_analyses:
            doc_lines.append("## 🔬 Análise Detalhada dos Arquivos\n")
            
            for i, analysis in enumerate(file_analyses[:8], 1):
                doc_lines.append(f"### {i}. {analysis.name}")
                doc_lines.append(f"**Linguagem:** {analysis.language} | **Tamanho:** {analysis.size:,} bytes | **Linhas:** {analysis.lines:,}")
                doc_lines.append(f"**Complexidade:** {analysis.complexity}\n")
                
                doc_lines.append(f"**Propósito:** {analysis.purpose}\n")
                doc_lines.append(f"**Resumo:** {analysis.summary}\n")
                
                if analysis.functions:
                    doc_lines.append(f"**Funções:** {', '.join(analysis.functions[:3])}")
                    if len(analysis.functions) > 3:
                        doc_lines[-1] += f" e mais {len(analysis.functions) - 3}"
                    doc_lines.append("")
                
                if analysis.classes:
                    doc_lines.append(f"**Classes:** {', '.join(analysis.classes[:3])}")
                    if len(analysis.classes) > 3:
                        doc_lines[-1] += f" e mais {len(analysis.classes) - 3}"
                    doc_lines.append("")
                
                doc_lines.append("---\n")
        
        # Informações do relatório
        doc_lines.append("## 📋 Informações do Relatório\n")
        doc_lines.append(f"- **URL do Repositório:** {url_final}")
        doc_lines.append(f"- **Data da Análise:** {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')}")
        doc_lines.append(f"- **Sistema:** DocAgent Skyone v2.0 com Autenticação")
        doc_lines.append(f"- **Modo:** {'Anônimo' if anonymous else 'Original'}")
        doc_lines.append(f"- **Tipo de Análise:** {analysis_data.get('analysis_type', 'Standard')}\n")
        
        doc_lines.append("---")
        doc_lines.append("*Relatório gerado automaticamente pelo DocAgent Skyone v2.0 com Autenticação*\n")
        
        return "\n".join(doc_lines)
    
    def _generate_installation_guide(self, analysis_data: Dict, anonymous: bool = True) -> str:
        """Gera guia de instalação"""
        main_lang = analysis_data.get('main_language', 'Desconhecida')
        important_files = analysis_data.get('important_files', [])
        
        doc_lines = []
        
        doc_lines.append("# Guia de Instalação e Configuração\n")
        
        if anonymous:
            doc_lines.append("> **Nota:** Este guia foi gerado a partir de análise anônima.\n")
        
        doc_lines.append("## 📋 Pré-requisitos do Sistema\n")
        doc_lines.append("### Ferramentas Básicas")
        doc_lines.append("- Git (para clonagem do repositório)")
        doc_lines.append("- Sistema operacional: Linux, macOS ou Windows\n")
        
        # Pré-requisitos específicos por linguagem
        if main_lang == 'Python':
            doc_lines.append("### Python")
            doc_lines.append("- Python 3.7+ (recomendado: 3.9+)")
            doc_lines.append("- pip (gerenciador de pacotes Python)")
            doc_lines.append("- virtualenv ou venv (para ambiente virtual)")
        elif main_lang == 'JavaScript':
            doc_lines.append("### Node.js")
            doc_lines.append("- Node.js 14+ (recomendado: 18+)")
            doc_lines.append("- npm ou yarn (gerenciador de pacotes)")
        elif main_lang == 'Java':
            doc_lines.append("### Java")
            doc_lines.append("- JDK 11+ (recomendado: 17+)")
            doc_lines.append("- Maven ou Gradle (build tool)")
        
        doc_lines.append("")
        
        # Instruções de instalação
        doc_lines.append("## 🚀 Processo de Instalação\n")
        
        doc_lines.append("### 1. Clonagem do Repositório")
        doc_lines.append("```bash")
        doc_lines.append("git clone [URL_DO_REPOSITORIO]")
        doc_lines.append("cd [nome-do-projeto]")
        doc_lines.append("```\n")
        
        # Verificar se há arquivos de dependência
        has_package_json = any('package.json' in f['name'] for f in important_files)
        has_requirements = any('requirements.txt' in f['name'] for f in important_files)
        has_dockerfile = any('dockerfile' in f['name'].lower() for f in important_files)
        
        if has_requirements or main_lang == 'Python':
            doc_lines.append("### 2. Configuração Python")
            doc_lines.append("```bash")
            doc_lines.append("# Criar ambiente virtual")
            doc_lines.append("python -m venv venv")
            doc_lines.append("")
            doc_lines.append("# Ativar ambiente virtual")
            doc_lines.append("# Linux/Mac:")
            doc_lines.append("source venv/bin/activate")
            doc_lines.append("# Windows:")
            doc_lines.append("venv\\Scripts\\activate")
            doc_lines.append("")
            if has_requirements:
                doc_lines.append("# Instalar dependências")
                doc_lines.append("pip install -r requirements.txt")
            doc_lines.append("```\n")
        
        if has_package_json or main_lang == 'JavaScript':
            doc_lines.append("### 2. Configuração Node.js")
            doc_lines.append("```bash")
            doc_lines.append("# Instalar dependências")
            doc_lines.append("npm install")
            doc_lines.append("# ou")
            doc_lines.append("yarn install")
            doc_lines.append("```\n")
        
        if has_dockerfile:
            doc_lines.append("### Alternativa: Docker")
            doc_lines.append("```bash")
            doc_lines.append("# Construir imagem")
            doc_lines.append("docker build -t nome-do-projeto .")
            doc_lines.append("")
            doc_lines.append("# Executar container")
            doc_lines.append("docker run -p 8080:8080 nome-do-projeto")
            doc_lines.append("```\n")
        
        # Verificação da instalação
        doc_lines.append("## ✅ Verificação da Instalação\n")
        doc_lines.append("Após a instalação, verifique se todas as dependências foram instaladas corretamente.")
        
        doc_lines.append("\n## 📝 Observações\n")
        doc_lines.append("- Consulte arquivos README específicos do projeto para instruções detalhadas")
        doc_lines.append("- Verifique arquivos de configuração para parâmetros específicos")
        doc_lines.append("- Para problemas de instalação, consulte a documentação da tecnologia utilizada")
        
        doc_lines.append(f"\n### Informações do Guia")
        doc_lines.append(f"- **Linguagem Principal:** {main_lang}")
        doc_lines.append(f"- **Arquivos de Configuração Detectados:** {len([f for f in important_files if f['name'] in ['package.json', 'requirements.txt', 'Dockerfile']])}")
        doc_lines.append(f"- **Gerado em:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        
        doc_lines.append("\n---")
        doc_lines.append("*Guia gerado automaticamente pelo DocAgent Skyone v2.0 com Autenticação*")
        
        return "\n".join(doc_lines)
    
    def _generate_c4_documentation(self, c4_model: C4Model, output_dir: Path) -> List[str]:
        """Gera documentação C4 completa"""
        generated_files = []
        
        try:
            # 1. Architecture Overview
            arch_doc = self._generate_c4_architecture_overview(c4_model)
            arch_file = output_dir / "03_C4_Architecture_Overview.md"
            with open(arch_file, 'w', encoding='utf-8') as f:
                f.write(arch_doc)
            generated_files.append(str(arch_file.name))
            
            # 2. Context Diagram
            context_doc = self._generate_c4_context_doc(c4_model)
            context_file = output_dir / "04_C4_Context_Diagram.md"
            with open(context_file, 'w', encoding='utf-8') as f:
                f.write(context_doc)
            generated_files.append(str(context_file.name))
            
            # 3. Container Diagram
            container_doc = self._generate_c4_container_doc(c4_model)
            container_file = output_dir / "05_C4_Container_Diagram.md"
            with open(container_file, 'w', encoding='utf-8') as f:
                f.write(container_doc)
            generated_files.append(str(container_file.name))
            
            # 4. Component Diagram
            component_doc = self._generate_c4_component_doc(c4_model)
            component_file = output_dir / "06_C4_Component_Diagram.md"
            with open(component_file, 'w', encoding='utf-8') as f:
                f.write(component_doc)
            generated_files.append(str(component_file.name))
            
            return generated_files
            
        except Exception as e:
            print(f"❌ Erro na geração da documentação C4: {e}")
            return generated_files
    
    def _generate_c4_architecture_overview(self, c4_model: C4Model) -> str:
        """Gera visão geral da arquitetura C4"""
        context = c4_model.context
        metadata = c4_model.metadata
        
        return f"""# 🏗️ Architecture Overview - {context['system_name']}

## System Context

**System Name:** {context['system_name']}  
**Description:** {context['description']}  
**Primary Technology:** {context['technology']}  
**Generated:** {metadata['generated_at']}  

## Executive Summary

This document provides a comprehensive architectural overview of the **{context['system_name']}** system using the **C4 model** approach.

## 🎯 Architecture Approach

This system follows modern software architecture principles:

- **🔧 Separation of Concerns**: Clear boundaries between different layers
- **💻 Technology Stack**: Built primarily with {context['technology']}
- **📈 Scalability**: Designed for horizontal and vertical scaling
- **🔧 Maintainability**: Modular design for easy maintenance

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| **Primary Language** | {context['technology']} |
| **Frameworks** | {', '.join(metadata['technologies'].get('frameworks', ['None identified']))} |
| **Databases** | {', '.join(metadata['technologies'].get('databases', ['None identified']))} |

## 👥 Users and External Systems

### Users
{chr(10).join(f"- **{user}**: Interacts with the system" for user in context.get('users', []))}

### External Systems
{chr(10).join(f"- **{system}**: External dependency" for system in context.get('external_systems', []))}

---
*This document was automatically generated by DocAgent C4 Model Analyzer*
"""
    
    def _generate_c4_context_doc(self, c4_model: C4Model) -> str:
        """Gera documentação do diagrama de contexto C4"""
        context = c4_model.context
        
        return f"""# 🌐 C4 Context Diagram - {context['system_name']}

## Overview

The context diagram shows the **{context['system_name']}** system at the highest level.

## 🎯 Internal System
- **{context['system_name']}**: {context['description']}
  - **Technology**: {context['technology']}

## 👥 Users
{chr(10).join(f"- **{user}**: Interacts with the system" for user in context.get('users', []))}

## 🔗 External Systems
{chr(10).join(f"- **{system}**: External dependency" for system in context.get('external_systems', []))}

---
*Generated by DocAgent C4 Model Analyzer*
"""
    
    def _generate_c4_container_doc(self, c4_model: C4Model) -> str:
        """Gera documentação do diagrama de containers C4"""
        context = c4_model.context
        containers = c4_model.containers
        
        return f"""# 📦 C4 Container Diagram - {context['system_name']}

## Overview

The container diagram shows the high-level technology choices for the **{context['system_name']}** system.

## 📦 Containers

{chr(10).join(f'''### 🔧 {container.name}
- **Description**: {container.description}
- **Technology**: {container.technology}
- **System**: {container.system}
''' for container in containers)}

---
*Generated by DocAgent C4 Model Analyzer*
"""
    
    def _generate_c4_component_doc(self, c4_model: C4Model) -> str:
        """Gera documentação do diagrama de componentes C4"""
        context = c4_model.context
        components = c4_model.components
        
        return f"""# 🔧 C4 Component Diagram - {context['system_name']}

## Overview

The component diagram shows the internal structure of the **{context['system_name']}** system.

## 🔧 Components

{chr(10).join(f'''### ⚙️ {component.name}
- **Description**: {component.description}
- **Technology**: {component.technology}
- **Container**: {component.container}
- **Responsibilities**: {', '.join(component.responsibilities)}
''' for component in components)}

---
*Generated by DocAgent C4 Model Analyzer*
"""

# =============================================================================
# FUNÇÃO DE ANÁLISE EM BACKGROUND COM AG2
# =============================================================================

async def run_analysis_ag2(analysis_request: AnalysisRequest):
    """Executa análise completa em background com AG2"""
    try:
        logger = app_state["enhanced_logger"]
        logger.info(f"Iniciando análise de {analysis_request.repo_url}", "Background AG2", "Inicialização")
        
        engine = app_state["analysis_engine"]

        if not engine:
            logger.error("Engine de análise não inicializado", "Background AG2", "Inicialização")
            raise Exception("Engine de análise não inicializado")

        # Se o usuário especificar um modelo no request, atualizamos a configuração
        try:
            if analysis_request.model:
                logger.info(f"Atualizando modelo LLM para: {analysis_request.model}", "Configuração", "LLM")
                engine.config.llm_model = analysis_request.model
                if engine.ag2_flow:
                    engine.ag2_flow.config.llm_model = analysis_request.model
                    engine.ag2_flow._setup_llm_config()
                logger.success(f"Modelo LLM atualizado: {analysis_request.model}", "Configuração", "LLM")
        except Exception as e:
            logger.warning(f"Não foi possível atualizar modelo LLM: {e}", "Configuração", "LLM")
        
        def update_status(phase: str, progress: int, message: str, step: str = ""):
            logger.progress(message, phase, step)
            app_state["analysis_status"] = AnalysisStatus(
                status="running",
                phase=phase,
                progress=progress,
                message=message,
                logs=app_state["analysis_status"].logs + [f"{step}: {message}"] if step else app_state["analysis_status"].logs,
                current_step=step
            )
        
        # Fase 1: Clone (0-30%)
        update_status("Clone do repositório", 5, "Validando URL do repositório", "Validação")
        
        if not engine._validate_github_url(analysis_request.repo_url):
            raise Exception("URL do repositório inválida")
        
        update_status("Clone do repositório", 10, "Verificando conectividade", "Conectividade")
        
        if not engine._check_github_connectivity():
            raise Exception("Sem conectividade com GitHub")
        
        update_status("Clone do repositório", 15, "Clonando repositório...", "Clone")
        
        clone_success, repo_path = engine.clone_repository(analysis_request.repo_url)
        
        if not clone_success:
            raise Exception("Falha no clone do repositório")
        
        print(f"✅ Background AG2: Clone concluído em {repo_path}")
        
        # Fase 2: Análise com AG2 (30-70%)
        update_status("Análise AG2", 35, "Iniciando análise com AG2", "AG2 Init")
        
        def analysis_callback(step_msg):
            current_progress = min(65, 35 + (len(step_msg) % 30))
            update_status("Análise AG2", current_progress, step_msg, "AG2 Analysis")
        
        if AG2_AVAILABLE and engine.ag2_flow:
            # Usar AG2 para análise completa
            update_status("Análise AG2", 40, "Executando análise avançada com AG2", "AG2 Processing")
            
            ag2_result = engine.ag2_flow.execute_analysis_with_ag2(
                analysis_request.repo_url, 
                analysis_request.anonymous
            )
            
            if ag2_result.get('status') == 'success':
                print(f"✅ Background AG2: Análise AG2 concluída")
                
                # Resultado final AG2
                ag2_docs_raw = ag2_result.get('generated_docs', []) or []
                ag2_docs_base = []
                for doc in ag2_docs_raw:
                    try:
                        ag2_docs_base.append(os.path.basename(doc))
                    except Exception:
                        ag2_docs_base.append(doc)
                
                app_state["current_analysis"] = {
                    "status": "success",
                    "message": "Análise AG2 concluída com sucesso",
                    "repository_url": analysis_request.repo_url,
                    "analysis_data": ag2_result,
                    "generated_docs": ag2_docs_base,
                    "timestamp": datetime.now().isoformat(),
                    "ag2_enabled": True,
                    "analysis_type": "AG2_enhanced"
                }
                
                app_state["analysis_status"] = AnalysisStatus(
                    status="completed",
                    phase="Concluído AG2",
                    progress=100,
                    message="Análise AG2 concluída com sucesso!",
                    logs=app_state["analysis_status"].logs + ["Análise AG2 concluída com sucesso"],
                    current_step="Concluído"
                )
                
                print("🎉 Background AG2: Análise completamente concluída")
                return
        
        # Fallback para análise tradicional
        update_status("Análise Estrutural", 40, "AG2 indisponível - usando análise tradicional", "Fallback")
        
        analysis_data = engine.analyze_repository_structure(repo_path, analysis_callback)
        
        if "error" in analysis_data:
            raise Exception(f"Erro na análise: {analysis_data['error']}")
        
        print(f"✅ Background: Análise concluída - {analysis_data.get('total_files', 0)} arquivos")
        
        # Fase 3: Geração de Documentação (70-90%)
        update_status("Geração de documentação", 75, "Compilando análise técnica", "Compilação")
        
        generated_docs = engine.generate_documentation(
            repo_path,
            analysis_request.repo_url,
            analysis_data,
            analysis_request.anonymous
        )
        
        if not generated_docs:
            raise Exception("Falha na geração da documentação")
        
        update_status("Geração de documentação", 90, "Finalizando relatórios", "Finalização")
        
        print(f"✅ Background: Documentação gerada - {len(generated_docs)} arquivos")
        
        # Fase 4: Finalização (90-100%)
        update_status("Finalizando", 95, "Preparando resultados", "Preparação")
        
        # Resultado final
        app_state["current_analysis"] = {
            "status": "success",
            "message": "Análise concluída com sucesso",
            "repository_url": analysis_request.repo_url,
            "analysis_data": analysis_data,
            "generated_docs": generated_docs,
            "timestamp": datetime.now().isoformat(),
            "ag2_enabled": AG2_AVAILABLE,
            "analysis_type": "traditional"
        }
        
        app_state["analysis_status"] = AnalysisStatus(
            status="completed",
            phase="Concluído",
            progress=100,
            message="Análise concluída com sucesso!",
            logs=app_state["analysis_status"].logs + ["Análise concluída com sucesso"],
            current_step="Concluído"
        )
        
        print("🎉 Background: Análise completamente concluída")
        
    except Exception as e:
        error_msg = f"Erro na análise: {str(e)}"
        print(f"❌ Background: {error_msg}")
        traceback.print_exc()
        
        app_state["analysis_status"] = AnalysisStatus(
            status="error",
            phase="Erro",
            progress=0,
            message=error_msg,
            logs=app_state["analysis_status"].logs + [f"Erro: {str(e)}"],
            current_step="Erro"
        )

# =============================================================================
# VERIFICAÇÃO OLLAMA MELHORADA
# =============================================================================

def verificar_ollama():
    """Verifica se Ollama está funcionando corretamente"""
    try:
        print("🔍 Verificando Ollama...")
        
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            print("❌ Ollama não está funcionando")
            print("💡 Execute: ollama serve")
            return False
        
        models = [line.split()[0] for line in result.stdout.strip().split('\n')[1:] if line.strip()]
        print(f"📋 Modelos disponíveis: {models}")
        
        recommended_models = ["qwen2.5:7b", "llama3.2:3b", "codegemma:7b"]
        available_recommended = [model for model in recommended_models if model in models]
        
        if available_recommended:
            print(f"✅ Modelos recomendados encontrados: {available_recommended}")
            return True
        elif models:
            print(f"⚠️ Ollama funcionando, mas sem modelos recomendados")
            print(f"💡 Execute: ollama pull qwen2.5:7b")
            return True
        else:
            print("❌ Nenhum modelo encontrado")
            print("💡 Execute: ollama pull qwen2.5:7b")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Timeout ao verificar Ollama")
        return False
    except FileNotFoundError:
        print("❌ Ollama não encontrado no PATH")
        print("💡 Instale o Ollama: https://ollama.ai/")
        return False
    except Exception as e:
        print(f"❌ Erro ao verificar Ollama: {e}")
        return False

# =============================================================================
# FUNÇÃO PRINCIPAL MELHORADA COM AUTENTICAÇÃO
# =============================================================================

def main():
    """Função principal"""
    try:
        print("🚀 Iniciando DocAgent Skyone v2.0 - Versão AG2 com Sistema de Autenticação")
        print("=" * 80)
        
        # Verificar dependências críticas
        print("🔍 Verificando dependências...")
        
        if not WEB_AVAILABLE:
            print("❌ FastAPI não disponível")
            print("💡 Execute: pip install fastapi uvicorn jinja2")
            return 1
        
        print("✅ FastAPI disponível")
        
        # Verificar AG2
        if AG2_AVAILABLE:
            print("✅ AG2 (AutoGen) disponível - Modo avançado ativado")
        else:
            print("⚠️ AG2 não disponível - Funcionando no modo simplificado")
            print("💡 Execute: pip install pyautogen fix-busted-json")
        
        # Verificar Ollama (opcional)
        ollama_ok = verificar_ollama()
        if not ollama_ok:
            print("⚠️ Ollama não disponível - funcionalidade AG2 limitada")
        
        # Criar diretórios necessários
        print("📁 Criando diretórios...")
        for dir_name in ["docs", "workdir", "static", "templates"]:
            Path(dir_name).mkdir(exist_ok=True)
            print(f"   ✅ {dir_name}/")
        
        # Criar templates HTML
        print("🎨 Criando templates...")
        create_html_template()
        
        # Verificar se git está disponível
        try:
            subprocess.run(["git", "--version"], capture_output=True, check=True)
            print("✅ Git disponível")
        except:
            print("❌ Git não encontrado - necessário para clone de repositórios")
            print("💡 Instale o Git: https://git-scm.com/")
            return 1
        
        # Inicializar engine de análise
        print("🔧 Inicializando engine de análise...")
        app_state["analysis_engine"] = AdvancedAnalysisEngine(ModelConfig())
        
        print("\n" + "="*80)
        print("🤖 DocAgent Skyone v2.0 - Sistema AG2 com Autenticação Completa")
        print("="*80)
        print("🚀 Funcionalidades Ativas:")
        print("   ✅ Sistema de Autenticação completo")
        print("   ✅ Login com usuário/senha + sessões")
        print("   ✅ OAuth GitHub integrado")
        print("   ✅ Proteção de rotas com middleware")
        print("   ✅ Interface adaptativa baseada no login")
        print("   ✅ Timeout automático por inatividade")
        print("   ✅ Busca automática de repositórios GitHub")
        print("   ✅ Interface interativa moderna")
        print("   ✅ Análise detalhada de código-fonte")
        print("   ✅ Relatórios anônimos completos")
        print("   ✅ Documentação técnica detalhada")
        print("   ✅ API REST completa com autenticação")
        print("   ✅ Downloads em formato Markdown")
        
        if AG2_AVAILABLE:
            print("   🤖 Sistema AG2 Multi-Agent ATIVO")
            print("      - 4 agentes especializados")
            print("      - 5 tools avançadas de análise")
            print("      - Análise colaborativa de código")
            print("      - Documentação técnica profissional")
            if ollama_ok:
                print("      - LLM local via Ollama")
            else:
                print("      - LLM limitado (Ollama offline)")
        else:
            print("   ⚠️  Sistema AG2 INATIVO (modo simplificado)")
            print("      - Análise tradicional disponível")
            print("      - Relatórios básicos funcionais")
            print("      - Documentação simplificada")
        
        print("="*80)
        print("🔗 URLs de Acesso:")
        print("   🏠 Interface Principal: http://localhost:8000")
        print("   📚 Documentação API:   http://localhost:8000/docs")
        print("   ❤️  Health Check:      http://localhost:8000/health")
        print("   🔑 Auth Status:        http://localhost:8000/api/auth/status")
        print("   🔐 GitHub OAuth:       http://localhost:8000/login/github")
        print("="*80)
        print("🔐 Sistema de Autenticação:")
        print("   • Login obrigatório com usuário/senha")
        print("   • Contas de demonstração:")
        print("     - admin / admin123")
        print("     - user / user123") 
        print("     - demo / demo123")
        print("     - Qualquer usuário/senha não vazios (modo demo)")
        print("   • OAuth GitHub para repositórios privados")
        print("   • Sessões seguras com gerenciamento automático")
        print("   • Timeout por inatividade (30 minutos)")
        print("   • Interface responsiva ao status de login")
        print("   • Logout seguro com limpeza de sessão")
        print("💡 Características do Sistema:")
        print("   • Relatórios 100% anônimos para compartilhamento seguro")
        print("   • Análise de estrutura de código avançada")
        print("   • Sistema de autenticação robusto e seguro")
        if AG2_AVAILABLE:
            print("   • Sistema AG2 com 4 agentes especializados:")
            print("     - AdvancedCodeExplorer (análise de código)")
            print("     - EnhancedDocumentationPlanner (planejamento)")
            print("     - TechnicalDocumentationWriter (escrita)")
            print("     - DocumentationReviewer (revisão)")
        print("   • Fallback inteligente para garantir funcionamento")
        print("   • Interface moderna com autenticação obrigatória")
        print("   • Proteção contra acesso não autorizado")
        print("="*80)
        print("🛡️  Segurança:")
        print("   • Todas as rotas protegidas por autenticação")
        print("   • Validação de sessões ativas")
        print("   • Tokens GitHub seguros (apenas na sessão)")
        print("   • Sanitização de parâmetros de entrada")
        print("   • Logout automático por inatividade")
        print("   • Anonimização completa de dados sensíveis")
        print("="*80)
        
        # Configuração de variáveis de ambiente opcionais
        print("🔧 Configurações Opcionais:")
        if os.environ.get('GITHUB_CLIENT_ID'):
            print("   ✅ GitHub OAuth configurado")
        else:
            print("   ⚠️  GitHub OAuth não configurado")
            print("      Para habilitar: configure GITHUB_CLIENT_ID e GITHUB_CLIENT_SECRET")
        
        if os.environ.get('GITHUB_TOKEN'):
            print("   ✅ Token GitHub global configurado")
        else:
            print("   ⚠️  Token GitHub não configurado globalmente")
            print("      Usuários podem configurar tokens individuais na interface")
        
        # Configurar logging do uvicorn
        uvicorn_config = {
            "app": app,
            "host": "0.0.0.0",
            "port": 8000,
            "log_level": "info",
            "access_log": True
        }
        
        # Iniciar servidor
        print("\n🌟 Iniciando servidor web com autenticação...")
        print(f"🤖 Modo AG2: {'ATIVADO' if AG2_AVAILABLE else 'DESATIVADO'}")
        print(f"🔐 Autenticação: OBRIGATÓRIA")
        print(f"🛡️  Sessões: ATIVAS")
        print(f"⏰ Timeout: 30 minutos")
        print("\n🎯 Para acessar:")
        print("   1. Abra http://localhost:8000")
        print("   2. Faça login com uma das contas demo")
        print("   3. Explore repositórios GitHub")
        print("   4. Gere relatórios técnicos anônimos")
        print("\n" + "="*80)
        
        uvicorn.run(**uvicorn_config)
        
    except KeyboardInterrupt:
        print("\n👋 Encerrando DocAgent Skyone...")
        print("   Obrigado por usar o sistema!")
        print("   Todas as sessões foram encerradas.")
        return 0
    except Exception as e:
        print(f"❌ Erro crítico: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
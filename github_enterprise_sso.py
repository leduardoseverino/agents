#!/usr/bin/env python3
"""
GitHub Enterprise SSO Integration for DocAgent LangGraph
========================================================

Adiciona autenticação SSO do GitHub Enterprise ao sistema existente
sem alterar a funcionalidade principal.
"""

import os
import json
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List
from urllib.parse import urlencode, parse_qs, urlparse
import jwt

try:
    from fastapi import HTTPException, Depends, Request, Response
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.responses import RedirectResponse
    import httpx
    OAUTH_AVAILABLE = True
except ImportError:
    OAUTH_AVAILABLE = False
    print("⚠️ OAuth dependencies not available. Install: pip install httpx pyjwt")

# =============================================================================
# CONFIGURAÇÃO GITHUB ENTERPRISE SSO
# =============================================================================

class GitHubEnterpriseConfig:
    """Configuração para GitHub Enterprise SSO"""
    
    def __init__(self):
        # GitHub Enterprise URLs
        self.GITHUB_ENTERPRISE_URL = os.environ.get("GITHUB_ENTERPRISE_URL", "https://github.company.com")
        self.GITHUB_API_URL = f"{self.GITHUB_ENTERPRISE_URL}/api/v3"
        
        # OAuth App Configuration
        self.CLIENT_ID = os.environ.get("GITHUB_ENTERPRISE_CLIENT_ID")
        self.CLIENT_SECRET = os.environ.get("GITHUB_ENTERPRISE_CLIENT_SECRET")
        self.REDIRECT_URI = os.environ.get("GITHUB_ENTERPRISE_REDIRECT_URI", "http://localhost:8001/auth/github/callback")
        
        # JWT Configuration
        self.JWT_SECRET = os.environ.get("JWT_SECRET", secrets.token_urlsafe(32))
        self.JWT_ALGORITHM = "HS256"
        self.JWT_EXPIRATION_HOURS = 24
        
        # Session Configuration
        self.SESSION_COOKIE_NAME = "docagent_session"
        self.SESSION_EXPIRATION_HOURS = 24
        
        # Scopes necessários
        self.OAUTH_SCOPES = ["repo", "user:email", "read:org"]
        
        print(f"🔧 GitHub Enterprise configurado: {self.GITHUB_ENTERPRISE_URL}")
        
        if not self.CLIENT_ID or not self.CLIENT_SECRET:
            print("⚠️ CLIENT_ID e CLIENT_SECRET não configurados")
            print("   Configure: GITHUB_ENTERPRISE_CLIENT_ID e GITHUB_ENTERPRISE_CLIENT_SECRET")
    
    def is_configured(self) -> bool:
        """Verifica se OAuth está configurado"""
        return bool(self.CLIENT_ID and self.CLIENT_SECRET)

# =============================================================================
# SISTEMA DE AUTENTICAÇÃO OAUTH
# =============================================================================

class GitHubEnterpriseAuth:
    """Sistema de autenticação SSO para GitHub Enterprise"""
    
    def __init__(self, config: GitHubEnterpriseConfig):
        self.config = config
        self.active_sessions: Dict[str, Dict] = {}
        self.oauth_states: Dict[str, Dict] = {}
        print("🔐 Sistema SSO GitHub Enterprise inicializado")
    
    def generate_oauth_url(self, state: str = None) -> str:
        """Gera URL para iniciar OAuth flow"""
        if not self.config.is_configured():
            raise HTTPException(status_code=500, detail="GitHub Enterprise OAuth não configurado")
        
        if not state:
            state = secrets.token_urlsafe(32)
        
        # Armazenar state para validação
        self.oauth_states[state] = {
            "created_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(minutes=10)
        }
        
        params = {
            "client_id": self.config.CLIENT_ID,
            "redirect_uri": self.config.REDIRECT_URI,
            "scope": " ".join(self.config.OAUTH_SCOPES),
            "state": state,
            "allow_signup": "false"  # Apenas usuários existentes do Enterprise
        }
        
        oauth_url = f"{self.config.GITHUB_ENTERPRISE_URL}/login/oauth/authorize?{urlencode(params)}"
        print(f"🔗 OAuth URL gerada: {oauth_url}")
        return oauth_url, state
    
    async def handle_oauth_callback(self, code: str, state: str) -> Dict[str, Any]:
        """Processa callback do OAuth"""
        try:
            # Validar state
            if state not in self.oauth_states:
                raise HTTPException(status_code=400, detail="State inválido")
            
            state_info = self.oauth_states[state]
            if datetime.now() > state_info["expires_at"]:
                raise HTTPException(status_code=400, detail="State expirado")
            
            # Trocar code por access token
            token_data = await self._exchange_code_for_token(code)
            access_token = token_data["access_token"]
            
            # Obter informações do usuário
            user_info = await self._get_user_info(access_token)
            
            # Criar sessão
            session_data = {
                "user_id": user_info["id"],
                "username": user_info["login"],
                "email": user_info.get("email"),
                "name": user_info.get("name"),
                "avatar_url": user_info.get("avatar_url"),
                "access_token": access_token,
                "created_at": datetime.now(),
                "expires_at": datetime.now() + timedelta(hours=self.config.SESSION_EXPIRATION_HOURS),
                "github_enterprise_url": self.config.GITHUB_ENTERPRISE_URL
            }
            
            # Gerar JWT token
            jwt_token = self._generate_jwt_token(session_data)
            
            # Armazenar sessão
            session_id = secrets.token_urlsafe(32)
            self.active_sessions[session_id] = session_data
            
            # Limpar state usado
            del self.oauth_states[state]
            
            print(f"✅ Usuário autenticado: {user_info['login']}")
            
            return {
                "session_id": session_id,
                "jwt_token": jwt_token,
                "user": {
                    "username": user_info["login"],
                    "name": user_info.get("name"),
                    "email": user_info.get("email"),
                    "avatar_url": user_info.get("avatar_url")
                }
            }
            
        except Exception as e:
            print(f"❌ Erro no OAuth callback: {e}")
            raise HTTPException(status_code=400, detail=f"Erro na autenticação: {str(e)}")
    
    async def _exchange_code_for_token(self, code: str) -> Dict[str, Any]:
        """Troca authorization code por access token"""
        token_url = f"{self.config.GITHUB_ENTERPRISE_URL}/login/oauth/access_token"
        
        data = {
            "client_id": self.config.CLIENT_ID,
            "client_secret": self.config.CLIENT_SECRET,
            "code": code,
            "redirect_uri": self.config.REDIRECT_URI
        }
        
        headers = {
            "Accept": "application/json",
            "User-Agent": "DocAgent-LangGraph/3.0"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(token_url, data=data, headers=headers)
            response.raise_for_status()
            return response.json()
    
    async def _get_user_info(self, access_token: str) -> Dict[str, Any]:
        """Obtém informações do usuário autenticado"""
        user_url = f"{self.config.GITHUB_API_URL}/user"
        
        headers = {
            "Authorization": f"token {access_token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "DocAgent-LangGraph/3.0"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(user_url, headers=headers)
            response.raise_for_status()
            return response.json()
    
    def _generate_jwt_token(self, session_data: Dict) -> str:
        """Gera JWT token para a sessão"""
        payload = {
            "user_id": session_data["user_id"],
            "username": session_data["username"],
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(hours=self.config.JWT_EXPIRATION_HOURS)
        }
        
        return jwt.encode(payload, self.config.JWT_SECRET, algorithm=self.config.JWT_ALGORITHM)
    
    def validate_jwt_token(self, token: str) -> Optional[Dict]:
        """Valida JWT token"""
        try:
            payload = jwt.decode(token, self.config.JWT_SECRET, algorithms=[self.config.JWT_ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Obtém sessão ativa"""
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        
        # Verificar expiração
        if datetime.now() > session["expires_at"]:
            del self.active_sessions[session_id]
            return None
        
        return session
    
    def logout(self, session_id: str) -> bool:
        """Encerra sessão"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            return True
        return False
    
    def cleanup_expired_sessions(self):
        """Remove sessões expiradas"""
        now = datetime.now()
        expired_sessions = [
            sid for sid, session in self.active_sessions.items()
            if now > session["expires_at"]
        ]
        
        for sid in expired_sessions:
            del self.active_sessions[sid]
        
        # Limpar states expirados
        expired_states = [
            state for state, state_info in self.oauth_states.items()
            if now > state_info["expires_at"]
        ]
        
        for state in expired_states:
            del self.oauth_states[state]

# =============================================================================
# GITHUB REPOSITORY FETCHER COM SSO
# =============================================================================

class GitHubEnterpriseRepositoryFetcher:
    """Fetcher de repositórios com autenticação SSO"""
    
    def __init__(self, config: GitHubEnterpriseConfig):
        self.config = config
        print("🔍 GitHub Enterprise Repository Fetcher inicializado")
    
    async def search_user_repositories(self, access_token: str, include_forks: bool = False, include_private: bool = True) -> List[Dict]:
        """Busca repositórios do usuário autenticado"""
        try:
            headers = {
                "Authorization": f"token {access_token}",
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "DocAgent-LangGraph/3.0"
            }
            
            repositories = []
            page = 1
            
            while len(repositories) < 100 and page <= 10:  # Limite de 100 repos, 10 páginas
                repos_url = f"{self.config.GITHUB_API_URL}/user/repos"
                params = {
                    "per_page": 50,
                    "page": page,
                    "sort": "updated",
                    "type": "all" if include_private else "public"
                }
                
                async with httpx.AsyncClient() as client:
                    response = await client.get(repos_url, headers=headers, params=params)
                    response.raise_for_status()
                    repos_data = response.json()
                
                if not repos_data:
                    break
                
                for repo in repos_data:
                    # Filtrar forks se necessário
                    if not include_forks and repo.get("fork", False):
                        continue
                    
                    repo_info = {
                        "id": repo["id"],
                        "name": repo["name"],
                        "full_name": repo["full_name"],
                        "description": (repo.get("description") or "Sem descrição")[:200],
                        "url": repo["html_url"],
                        "clone_url": repo["clone_url"],
                        "ssh_url": repo["ssh_url"],
                        "language": repo.get("language") or "Desconhecida",
                        "stars": repo.get("stargazers_count", 0),
                        "forks": repo.get("forks_count", 0),
                        "size_kb": repo.get("size", 0),
                        "updated_at": repo.get("updated_at", ""),
                        "topics": repo.get("topics", []),
                        "private": repo.get("private", False),
                        "owner": {
                            "login": repo["owner"]["login"],
                            "avatar_url": repo["owner"]["avatar_url"]
                        }
                    }
                    repositories.append(repo_info)
                
                if len(repos_data) < 50:
                    break
                
                page += 1
            
            print(f"✅ Encontrados {len(repositories)} repositórios")
            return sorted(repositories, key=lambda x: x["stars"], reverse=True)
            
        except Exception as e:
            print(f"❌ Erro ao buscar repositórios: {e}")
            raise HTTPException(status_code=500, detail=f"Erro ao buscar repositórios: {str(e)}")
    
    async def search_organization_repositories(self, access_token: str, org_name: str, include_forks: bool = False) -> List[Dict]:
        """Busca repositórios de uma organização"""
        try:
            headers = {
                "Authorization": f"token {access_token}",
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "DocAgent-LangGraph/3.0"
            }
            
            repositories = []
            page = 1
            
            while len(repositories) < 100 and page <= 10:
                repos_url = f"{self.config.GITHUB_API_URL}/orgs/{org_name}/repos"
                params = {
                    "per_page": 50,
                    "page": page,
                    "sort": "updated",
                    "type": "all"
                }
                
                async with httpx.AsyncClient() as client:
                    response = await client.get(repos_url, headers=headers, params=params)
                    response.raise_for_status()
                    repos_data = response.json()
                
                if not repos_data:
                    break
                
                for repo in repos_data:
                    if not include_forks and repo.get("fork", False):
                        continue
                    
                    repo_info = {
                        "id": repo["id"],
                        "name": repo["name"],
                        "full_name": repo["full_name"],
                        "description": (repo.get("description") or "Sem descrição")[:200],
                        "url": repo["html_url"],
                        "clone_url": repo["clone_url"],
                        "ssh_url": repo["ssh_url"],
                        "language": repo.get("language") or "Desconhecida",
                        "stars": repo.get("stargazers_count", 0),
                        "forks": repo.get("forks_count", 0),
                        "size_kb": repo.get("size", 0),
                        "updated_at": repo.get("updated_at", ""),
                        "topics": repo.get("topics", []),
                        "private": repo.get("private", False),
                        "owner": {
                            "login": repo["owner"]["login"],
                            "avatar_url": repo["owner"]["avatar_url"]
                        }
                    }
                    repositories.append(repo_info)
                
                if len(repos_data) < 50:
                    break
                
                page += 1
            
            print(f"✅ Encontrados {len(repositories)} repositórios da organização {org_name}")
            return sorted(repositories, key=lambda x: x["stars"], reverse=True)
            
        except Exception as e:
            print(f"❌ Erro ao buscar repositórios da organização: {e}")
            raise HTTPException(status_code=500, detail=f"Erro ao buscar repositórios: {str(e)}")

# =============================================================================
# MIDDLEWARE DE AUTENTICAÇÃO
# =============================================================================

class AuthenticationMiddleware:
    """Middleware para verificar autenticação"""
    
    def __init__(self, auth_system: GitHubEnterpriseAuth):
        self.auth_system = auth_system
    
    async def get_current_user(self, request: Request) -> Optional[Dict]:
        """Obtém usuário atual da requisição"""
        # Verificar cookie de sessão
        session_id = request.cookies.get(self.auth_system.config.SESSION_COOKIE_NAME)
        if session_id:
            session = self.auth_system.get_session(session_id)
            if session:
                return session
        
        # Verificar header Authorization
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            payload = self.auth_system.validate_jwt_token(token)
            if payload:
                # Buscar sessão completa
                for session_id, session in self.auth_system.active_sessions.items():
                    if session["user_id"] == payload["user_id"]:
                        return session
        
        return None
    
    async def require_auth(self, request: Request) -> Dict:
        """Requer autenticação válida"""
        user = await self.get_current_user(request)
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Autenticação necessária",
                headers={"WWW-Authenticate": "Bearer"}
            )
        return user

# =============================================================================
# INTEGRAÇÃO COM SISTEMA EXISTENTE
# =============================================================================

def integrate_github_enterprise_sso(app, app_state):
    """Integra GitHub Enterprise SSO ao sistema existente"""
    
    if not OAUTH_AVAILABLE:
        print("⚠️ OAuth não disponível - SSO desabilitado")
        return
    
    # Configurar sistema de autenticação
    github_config = GitHubEnterpriseConfig()
    auth_system = GitHubEnterpriseAuth(github_config)
    repo_fetcher = GitHubEnterpriseRepositoryFetcher(github_config)
    auth_middleware = AuthenticationMiddleware(auth_system)
    
    # Adicionar ao estado da aplicação
    app_state["github_enterprise"] = {
        "config": github_config,
        "auth": auth_system,
        "repo_fetcher": repo_fetcher,
        "middleware": auth_middleware
    }
    
    # Rotas de autenticação
    @app.get("/auth/login")
    async def initiate_github_login():
        """Inicia processo de login SSO"""
        if not github_config.is_configured():
            raise HTTPException(status_code=500, detail="GitHub Enterprise OAuth não configurado")
        
        oauth_url, state = auth_system.generate_oauth_url()
        return {"login_url": oauth_url, "state": state}
    
    @app.get("/auth/github/callback")
    async def github_oauth_callback(code: str, state: str, response: Response):
        """Callback do GitHub OAuth"""
        try:
            auth_result = await auth_system.handle_oauth_callback(code, state)
            
            # Configurar cookie de sessão
            response.set_cookie(
                key=github_config.SESSION_COOKIE_NAME,
                value=auth_result["session_id"],
                max_age=github_config.SESSION_EXPIRATION_HOURS * 3600,
                httponly=True,
                secure=True,
                samesite="lax"
            )
            
            # Redirecionar para a aplicação
            redirect_response = RedirectResponse(url="/?login=success")
            redirect_response.set_cookie(
                key=github_config.SESSION_COOKIE_NAME,
                value=auth_result["session_id"],
                max_age=github_config.SESSION_EXPIRATION_HOURS * 3600,
                httponly=True,
                secure=False,  # Para desenvolvimento local
                samesite="lax"
            )
            
            return redirect_response
            
        except Exception as e:
            return RedirectResponse(url=f"/?login=error&message={str(e)}")
    
    @app.post("/auth/logout")
    async def logout(request: Request, response: Response):
        """Logout do usuário"""
        session_id = request.cookies.get(github_config.SESSION_COOKIE_NAME)
        if session_id:
            auth_system.logout(session_id)
        
        response.delete_cookie(github_config.SESSION_COOKIE_NAME)
        return {"message": "Logout realizado"}
    
    @app.get("/api/auth/user")
    async def get_current_user_info(request: Request):
        """Obtém informações do usuário autenticado"""
        user = await auth_middleware.get_current_user(request)
        if not user:
            raise HTTPException(status_code=401, detail="Não autenticado")
        
        return {
            "authenticated": True,
            "user": {
                "username": user["username"],
                "name": user.get("name"),
                "email": user.get("email"),
                "avatar_url": user.get("avatar_url")
            },
            "github_enterprise_url": user.get("github_enterprise_url")
        }
    
    @app.get("/api/repositories/user")
    async def get_user_repositories(
        request: Request,
        include_forks: bool = False,
        include_private: bool = True
    ):
        """Lista repositórios do usuário autenticado"""
        user = await auth_middleware.require_auth(request)
        
        repositories = await repo_fetcher.search_user_repositories(
            user["access_token"],
            include_forks=include_forks,
            include_private=include_private
        )
        
        return {
            "success": True,
            "repositories": repositories,
            "count": len(repositories),
            "user": user["username"]
        }
    
    @app.get("/api/repositories/org/{org_name}")
    async def get_organization_repositories(
        org_name: str,
        request: Request,
        include_forks: bool = False
    ):
        """Lista repositórios de uma organização"""
        user = await auth_middleware.require_auth(request)
        
        repositories = await repo_fetcher.search_organization_repositories(
            user["access_token"],
            org_name,
            include_forks=include_forks
        )
        
        return {
            "success": True,
            "repositories": repositories,
            "count": len(repositories),
            "organization": org_name
        }
    
    @app.get("/api/auth/config")
    async def get_auth_config():
        """Obtém configuração de autenticação"""
        return {
            "github_enterprise_enabled": github_config.is_configured(),
            "github_enterprise_url": github_config.GITHUB_ENTERPRISE_URL,
            "login_required": github_config.is_configured()
        }
    
    # Tarefa de limpeza de sessões expiradas
    @app.on_event("startup")
    async def setup_cleanup_task():
        import asyncio
        
        async def cleanup_sessions():
            while True:
                try:
                    auth_system.cleanup_expired_sessions()
                    await asyncio.sleep(3600)  # Limpar a cada hora
                except Exception as e:
                    print(f"❌ Erro na limpeza de sessões: {e}")
                    await asyncio.sleep(300)  # Tentar novamente em 5 minutos
        
        asyncio.create_task(cleanup_sessions())
    
    print("✅ GitHub Enterprise SSO integrado ao sistema")
    return True

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def setup_environment_variables():
    """Configura variáveis de ambiente para desenvolvimento"""
    env_vars = {
        "GITHUB_ENTERPRISE_URL": "https://github.company.com",
        "GITHUB_ENTERPRISE_CLIENT_ID": "your_client_id_here",
        "GITHUB_ENTERPRISE_CLIENT_SECRET": "your_client_secret_here",
        "GITHUB_ENTERPRISE_REDIRECT_URI": "http://localhost:8001/auth/github/callback",
        "JWT_SECRET": secrets.token_urlsafe(32)
    }
    
    print("\n" + "="*80)
    print("🔧 CONFIGURAÇÃO GITHUB ENTERPRISE SSO")
    print("="*80)
    print("Para habilitar o SSO, configure as seguintes variáveis de ambiente:")
    print()
    
    for var, example in env_vars.items():
        current_value = os.environ.get(var)
        status = "✅ Configurado" if current_value else "❌ Não configurado"
        print(f"{var}: {status}")
        if not current_value:
            print(f"  export {var}=\"{example}\"")
        print()
    
    print("="*80)
    print("📝 Passos para configurar GitHub Enterprise OAuth:")
    print("1. Acesse GitHub Enterprise Admin > Applications > OAuth Apps")
    print("2. Crie uma nova OAuth App com:")
    print("   - Homepage URL: http://localhost:8001")
    print("   - Callback URL: http://localhost:8001/auth/github/callback")
    print("3. Configure as variáveis de ambiente com Client ID e Secret")
    print("4. Reinicie o DocAgent")
    print("="*80)

if __name__ == "__main__":
    setup_environment_variables()
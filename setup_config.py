#!/usr/bin/env python3
"""
Setup Config - Configuração Dinâmica para DocAgent LangGraph
===========================================================

Este script configura automaticamente as variáveis de ambiente
e corrige os erros de configuração do DocAgent LangGraph.
"""

import os
import sys
import secrets
import subprocess
from pathlib import Path
from dotenv import load_dotenv

class DocAgentConfigurator:
    """Configurador automático para DocAgent LangGraph"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.env_file = self.base_dir / ".env"
        self.env_example = self.base_dir / "env.example"
        
    def print_status(self, message: str, status: str = "info"):
        """Imprime mensagem com status colorido"""
        colors = {
            "info": "\033[0;34m",     # Azul
            "success": "\033[0;32m",  # Verde
            "warning": "\033[1;33m",  # Amarelo
            "error": "\033[0;31m",    # Vermelho
            "reset": "\033[0m"        # Reset
        }
        
        symbols = {
            "info": "🔧",
            "success": "✅",
            "warning": "⚠️",
            "error": "❌"
        }
        
        color = colors.get(status, colors["info"])
        symbol = symbols.get(status, "•")
        reset = colors["reset"]
        
        print(f"{color}{symbol} {message}{reset}")
    
    def check_dependencies(self):
        """Verifica dependências necessárias"""
        self.print_status("Verificando dependências...", "info")
        
        dependencies = {
            "python3": ["python3", "--version"],
            "git": ["git", "--version"],
            "ollama": ["ollama", "list"],
            "pip": ["pip", "--version"]
        }
        
        missing = []
        
        for name, cmd in dependencies.items():
            try:
                result = subprocess.run(cmd, capture_output=True, timeout=5)
                if result.returncode == 0:
                    self.print_status(f"{name} disponível", "success")
                else:
                    missing.append(name)
                    self.print_status(f"{name} não disponível", "warning")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                missing.append(name)
                self.print_status(f"{name} não encontrado", "error")
        
        if "python3" in missing or "pip" in missing:
            self.print_status("Python3 e pip são obrigatórios!", "error")
            sys.exit(1)
        
        return missing
    
    def install_python_dotenv(self):
        """Instala python-dotenv se necessário"""
        try:
            import dotenv
            self.print_status("python-dotenv já instalado", "success")
            return True
        except ImportError:
            self.print_status("Instalando python-dotenv...", "info")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "python-dotenv"], 
                             check=True, capture_output=True)
                self.print_status("python-dotenv instalado com sucesso", "success")
                return True
            except subprocess.CalledProcessError as e:
                self.print_status(f"Erro ao instalar python-dotenv: {e}", "error")
                return False
    
    def generate_jwt_secret(self):
        """Gera uma chave JWT segura"""
        return secrets.token_urlsafe(32)
    
    def create_env_file(self):
        """Cria arquivo .env baseado no template"""
        if self.env_file.exists():
            self.print_status("Arquivo .env já existe - fazendo backup", "warning")
            backup_file = self.env_file.with_suffix(".env.backup")
            self.env_file.rename(backup_file)
            self.print_status(f"Backup criado: {backup_file}", "info")
        
        # Valores padrão para configuração
        config_vars = {
            "HOST": "0.0.0.0",
            "PORT": "8001",
            "DEBUG": "false",
            "LOG_LEVEL": "INFO",
            "ENVIRONMENT": "development",
            "DEFAULT_OLLAMA_MODEL": "qwen2.5:7b",
            "DEFAULT_OPENAI_MODEL": "gpt-4o",
            "OLLAMA_BASE_URL": "http://localhost:11434",
            "GITHUB_ENTERPRISE_URL": "https://github.company.com",
            "GITHUB_ENTERPRISE_CLIENT_ID": "your_client_id_here",
            "GITHUB_ENTERPRISE_CLIENT_SECRET": "your_client_secret_here",
            "GITHUB_ENTERPRISE_REDIRECT_URI": "http://localhost:8001/auth/github/callback",
            "JWT_SECRET": self.generate_jwt_secret(),
            "MAX_FILES_ANALYSIS": "50",
            "MAX_FILE_SIZE_KB": "500",
            "ANALYSIS_TIMEOUT": "600",
            "ENABLE_CACHE": "true",
            "CACHE_TTL": "3600",
            "ENABLE_CORS": "true",
            "ALLOWED_ORIGINS": "*",
            "GITHUB_RATE_LIMIT": "60"
        }
        
        # Criar conteúdo do arquivo .env
        env_content = ["# DocAgent LangGraph - Configurações de Ambiente"]
        env_content.append("# Gerado automaticamente pelo setup_config.py")
        env_content.append(f"# Data: {os.popen('date').read().strip()}")
        env_content.append("")
        
        env_content.append("# === CONFIGURAÇÕES DO SERVIDOR ===")
        for key in ["HOST", "PORT", "DEBUG", "LOG_LEVEL", "ENVIRONMENT"]:
            env_content.append(f"{key}={config_vars[key]}")
        env_content.append("")
        
        env_content.append("# === CONFIGURAÇÕES DE MODELOS ===")
        for key in ["DEFAULT_OLLAMA_MODEL", "DEFAULT_OPENAI_MODEL", "OLLAMA_BASE_URL"]:
            env_content.append(f"{key}={config_vars[key]}")
        env_content.append("")
        
        env_content.append("# === GITHUB ENTERPRISE SSO ===")
        env_content.append("# CONFIGURE ESTES VALORES PARA HABILITAR SSO")
        for key in ["GITHUB_ENTERPRISE_URL", "GITHUB_ENTERPRISE_CLIENT_ID", 
                   "GITHUB_ENTERPRISE_CLIENT_SECRET", "GITHUB_ENTERPRISE_REDIRECT_URI"]:
            env_content.append(f"{key}={config_vars[key]}")
        env_content.append("")
        
        env_content.append("# === SEGURANÇA ===")
        env_content.append(f"JWT_SECRET={config_vars['JWT_SECRET']}")
        env_content.append("")
        
        env_content.append("# === APIs OPCIONAIS ===")
        env_content.append("# Descomente e configure se necessário")
        env_content.append("#OPENAI_API_KEY=sk-your-openai-api-key-here")
        env_content.append("#GITHUB_TOKEN=ghp_your-github-token-here")
        env_content.append("")
        
        env_content.append("# === OUTRAS CONFIGURAÇÕES ===")
        for key in ["MAX_FILES_ANALYSIS", "MAX_FILE_SIZE_KB", "ANALYSIS_TIMEOUT",
                   "ENABLE_CACHE", "CACHE_TTL", "ENABLE_CORS", "ALLOWED_ORIGINS", "GITHUB_RATE_LIMIT"]:
            env_content.append(f"{key}={config_vars[key]}")
        
        # Escrever arquivo
        self.env_file.write_text("\n".join(env_content))
        self.print_status(f"Arquivo .env criado: {self.env_file}", "success")
    
    def create_directories(self):
        """Cria diretórios necessários"""
        directories = ["docs", "workdir", "static", "templates", "logs"]
        
        for dir_name in directories:
            dir_path = self.base_dir / dir_name
            dir_path.mkdir(exist_ok=True)
            self.print_status(f"Diretório criado/verificado: {dir_name}", "success")
    
    def update_html_template(self):
        """Atualiza template HTML para configuração dinâmica"""
        template_file = self.base_dir / "templates" / "index.html"
        
        if not template_file.exists():
            self.print_status("Criando template HTML básico...", "info")
            
            html_content = '''<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DocAgent LangGraph v3.0</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
        .header { text-align: center; margin-bottom: 30px; }
        .status-card { background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px; padding: 15px; margin: 10px 0; }
        .status-ok { border-left: 4px solid #28a745; }
        .status-warning { border-left: 4px solid #ffc107; }
        .status-error { border-left: 4px solid #dc3545; }
        .config-section { margin: 20px 0; }
        .btn { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
        .btn:hover { background: #0056b3; }
        .btn-success { background: #28a745; }
        .btn-warning { background: #ffc107; }
        .btn-danger { background: #dc3545; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 DocAgent LangGraph v3.0</h1>
            <p>Sistema de Documentação Automática com IA</p>
        </div>
        
        <div id="config-status">
            <h2>📊 Status da Configuração</h2>
            <div id="status-cards"></div>
        </div>
        
        <div class="config-section">
            <h2>🔧 Configuração GitHub Enterprise SSO</h2>
            <div class="status-card status-warning">
                <h3>⚠️ SSO não configurado</h3>
                <p>Para habilitar autenticação SSO:</p>
                <ol>
                    <li>Configure as variáveis no arquivo .env:
                        <ul>
                            <li>GITHUB_ENTERPRISE_CLIENT_ID</li>
                            <li>GITHUB_ENTERPRISE_CLIENT_SECRET</li>
                        </ul>
                    </li>
                    <li>Reinicie o DocAgent</li>
                </ol>
                <button class="btn btn-warning" onclick="showSSOInstructions()">
                    📝 Ver Instruções Completas
                </button>
            </div>
        </div>
        
        <div class="config-section">
            <h2>🔑 APIs Opcionais</h2>
            <div class="status-card">
                <h4>OpenAI API</h4>
                <p>Configure OPENAI_API_KEY no .env para usar modelos GPT</p>
                <button class="btn" onclick="testOpenAI()">🧪 Testar OpenAI</button>
            </div>
            <div class="status-card">
                <h4>GitHub Token</h4>
                <p>Configure GITHUB_TOKEN no .env para acessar repositórios privados</p>
                <button class="btn" onclick="testGitHub()">🧪 Testar GitHub</button>
            </div>
        </div>
        
        <div class="config-section">
            <h2>🚀 Ações</h2>
            <button class="btn btn-success" onclick="startAnalysis()">
                📊 Iniciar Análise
            </button>
            <button class="btn" onclick="refreshStatus()">
                🔄 Atualizar Status
            </button>
            <button class="btn btn-warning" onclick="downloadLogs()">
                📋 Download Logs
            </button>
        </div>
    </div>
    
    <script>
        // Configuração dinâmica via variáveis do servidor
        const CONFIG = {
            github_enterprise_url: "{{ config.github_enterprise_url }}",
            github_enterprise_enabled: {{ config.github_enterprise_enabled | tojson }},
            openai_configured: {{ config.openai_configured | tojson }},
            github_token_configured: {{ config.github_token_configured | tojson }},
            ollama_available: {{ config.ollama_available | tojson }}
        };
        
        function updateStatusCards() {
            const statusContainer = document.getElementById('status-cards');
            statusContainer.innerHTML = '';
            
            const statuses = [
                {
                    name: 'GitHub Enterprise SSO',
                    status: CONFIG.github_enterprise_enabled ? 'ok' : 'warning',
                    message: CONFIG.github_enterprise_enabled ? 'Configurado' : 'Não configurado'
                },
                {
                    name: 'OpenAI API',
                    status: CONFIG.openai_configured ? 'ok' : 'warning',
                    message: CONFIG.openai_configured ? 'Configurado' : 'Não configurado'
                },
                {
                    name: 'GitHub Token',
                    status: CONFIG.github_token_configured ? 'ok' : 'warning',
                    message: CONFIG.github_token_configured ? 'Configurado' : 'Não configurado'
                },
                {
                    name: 'Ollama',
                    status: CONFIG.ollama_available ? 'ok' : 'error',
                    message: CONFIG.ollama_available ? 'Disponível' : 'Não disponível'
                }
            ];
            
            statuses.forEach(item => {
                const card = document.createElement('div');
                card.className = `status-card status-${item.status}`;
                card.innerHTML = `<strong>${item.name}:</strong> ${item.message}`;
                statusContainer.appendChild(card);
            });
        }
        
        function showSSOInstructions() {
            alert(`Instruções para configurar GitHub Enterprise SSO:

1. Acesse GitHub Enterprise Admin > Applications > OAuth Apps
2. Crie uma nova OAuth App com:
   - Homepage URL: http://localhost:8001
   - Callback URL: http://localhost:8001/auth/github/callback
3. Copie Client ID e Client Secret
4. Edite o arquivo .env:
   GITHUB_ENTERPRISE_CLIENT_ID=seu_client_id
   GITHUB_ENTERPRISE_CLIENT_SECRET=seu_client_secret
5. Reinicie o DocAgent`);
        }
        
        async function testOpenAI() {
            try {
                const response = await fetch('/api/test/openai');
                const result = await response.json();
                alert(result.message);
            } catch (error) {
                alert('Erro ao testar OpenAI: ' + error.message);
            }
        }
        
        async function testGitHub() {
            try {
                const response = await fetch('/api/test/github');
                const result = await response.json();
                alert(result.message);
            } catch (error) {
                alert('Erro ao testar GitHub: ' + error.message);
            }
        }
        
        async function startAnalysis() {
            window.location.href = '/analyze';
        }
        
        async function refreshStatus() {
            window.location.reload();
        }
        
        async function downloadLogs() {
            window.open('/api/logs/download', '_blank');
        }
        
        // Inicializar interface
        document.addEventListener('DOMContentLoaded', function() {
            updateStatusCards();
        });
    </script>
</body>
</html>'''
            
            template_file.write_text(html_content)
            self.print_status("Template HTML criado com configuração dinâmica", "success")
        else:
            self.print_status("Template HTML já existe", "info")
    
    def run_setup(self):
        """Executa configuração completa"""
        print("🚀 DocAgent LangGraph - Setup Automático")
        print("=" * 50)
        
        # Verificar dependências
        missing_deps = self.check_dependencies()
        
        # Instalar python-dotenv
        if not self.install_python_dotenv():
            self.print_status("Não foi possível instalar python-dotenv", "error")
            return False
        
        # Criar diretórios
        self.create_directories()
        
        # Criar arquivo .env
        self.create_env_file()
        
        # Atualizar template HTML
        self.update_html_template()
        
        print("\n" + "=" * 50)
        self.print_status("Setup concluído com sucesso!", "success")
        print("\n📋 PRÓXIMOS PASSOS:")
        print("1. Edite o arquivo .env e configure:")
        print("   - GITHUB_ENTERPRISE_CLIENT_ID")
        print("   - GITHUB_ENTERPRISE_CLIENT_SECRET")
        print("   - OPENAI_API_KEY (opcional)")
        print("   - GITHUB_TOKEN (opcional)")
        print("\n2. Execute o DocAgent:")
        print("   python3 Docagent_Skyone.py")
        print("\n3. Acesse: http://localhost:8001")
        
        return True

if __name__ == "__main__":
    configurator = DocAgentConfigurator()
    success = configurator.run_setup()
    sys.exit(0 if success else 1)

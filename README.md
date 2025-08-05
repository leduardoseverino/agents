# 🚀 DocAgent Skyone v2.0

**Plataforma Avançada de Análise e Documentação Automática de Repositórios**

DocAgent Skyone v2.0 é uma solução completa para análise automatizada de repositórios de código, gerando documentação técnica abrangente usando tecnologias de ponta como AG2 (AutoGen 2.0) e modelo C4.

## ✨ Principais Funcionalidades

### 🤖 **AG2 Multi-Agent Analysis**
- Sistema multi-agente baseado em AutoGen 2.0
- Análise colaborativa entre agentes especializados
- Documentação técnica detalhada e precisa
- Suporte a múltiplas linguagens de programação

### 🏗️ **C4 Model Documentation**
- Documentação arquitetural seguindo o padrão C4
- Diagramas de Contexto, Contêiner, Componente e Deploy
- Integração com PlantUML para visualizações profissionais
- Análise automática de tecnologias e dependências

### 🔐 **Sistema de Autenticação**
- Autenticação local com usuário/senha
- Integração OAuth com GitHub
- Suporte completo ao GitHub Enterprise
- Gestão de sessões e tokens seguros

### 📊 **Interface Web Moderna**
- Dashboard responsivo com Tailwind CSS
- Análise em tempo real com logs detalhados
- Download automático de documentação
- Interface intuitiva e amigável

### 🌐 **GitHub Integration**
- Suporte ao GitHub.com e GitHub Enterprise
- Clonagem automática de repositórios
- Análise de repositórios públicos e privados
- Configuração flexível de tokens de acesso

## 🛠️ **Tecnologias**

- **Backend**: Python 3.8+, FastAPI, Pydantic
- **AI/ML**: AutoGen 2.0 (AG2), OpenAI GPT
- **Frontend**: HTML5, Tailwind CSS, JavaScript
- **Documentation**: Markdown, PlantUML, C4 Model
- **Integration**: GitHub API, OAuth 2.0

## 📋 **Pré-requisitos**

- Python 3.8 ou superior
- Git instalado no sistema
- Conexão com internet para APIs
- (Opcional) Token do OpenAI para funcionalidades AG2
- (Opcional) Token do GitHub para repositórios privados

## 🚀 **Instalação**

### 1. Clone o Repositório
```bash
git clone https://github.com/seu-usuario/docagent-skyone.git
cd docagent-skyone
```

### 2. Instale as Dependências
```bash
pip install -r requirements.txt
```

### 3. Configure as Variáveis de Ambiente (Opcional)
```bash
# Para funcionalidades AG2
export OPENAI_API_KEY="sua_api_key_openai"

# Para repositórios privados do GitHub
export GITHUB_TOKEN="seu_token_github"

# Para Ollama (alternativa local)
export OLLAMA_MODEL="llama2"
```

### 4. Execute a Aplicação
```bash
python Docagenta.py
```

### 5. Acesse a Interface Web
Abra seu navegador e acesse: `http://localhost:8000`

## 📖 **Como Usar**

### 1. **Autenticação**
- Faça login com credenciais locais (admin/admin por padrão)
- Ou configure autenticação GitHub OAuth
- Para GitHub Enterprise, configure a URL do servidor

### 2. **Análise de Repositório**
- Insira a URL do repositório GitHub
- Escolha entre análise com ou sem AG2
- Configure tokens de acesso se necessário
- Inicie a análise e acompanhe o progresso

### 3. **Documentação Gerada**
A análise produz os seguintes arquivos:

#### **Documentação Tradicional**
- `01_relatorio_completo.md` - Análise técnica detalhada
- `02_guia_instalacao.md` - Guia de instalação e uso
- `03_documentacao_api.md` - Documentação de APIs (se aplicável)

#### **Documentação C4 Model**
- `04_C4_Architecture_Overview.md` - Visão geral da arquitetura
- `05_C4_Context_Diagram.md` - Diagrama de contexto
- `06_C4_Container_Diagram.md` - Diagrama de contêineres
- `07_C4_Component_Diagram.md` - Diagrama de componentes
- `08_C4_Deployment_Guide.md` - Guia de deploy

### 4. **Download dos Arquivos**
- Download individual de cada arquivo
- Download em lote (ZIP) de toda a documentação
- Suporte a modo anônimo para relatórios sem identificação

## ⚙️ **Configuração**

### **Autenticação Local**
Edite as credenciais padrão no código:
```python
VALID_USERS = {
    "admin": "admin",  # usuário: senha
    "user": "password"
}
```

### **GitHub Enterprise**
Configure através da interface web:
1. Acesse "Configuração GitHub"
2. Selecione "GitHub Enterprise"
3. Insira URL do servidor e token
4. Teste a conexão

### **AG2 Configuration**
O sistema detecta automaticamente:
- OpenAI API (via variável de ambiente)
- Ollama (se instalado localmente)
- Azure OpenAI (configuração manual)

## 🔌 **API Endpoints**

### **Autenticação**
- `POST /api/auth/login` - Login local
- `GET /api/auth/status` - Status de autenticação
- `POST /api/auth/logout` - Logout

### **GitHub Integration**
- `POST /api/auth/github` - Configurar token GitHub
- `POST /api/auth/github-enterprise` - Configurar GitHub Enterprise
- `POST /api/search` - Buscar repositórios

### **Análise**
- `POST /api/analyze` - Iniciar análise
- `GET /api/status` - Status da análise
- `GET /api/results` - Resultados da análise

### **Downloads**
- `GET /api/download/{filename}` - Download de arquivo específico
- `GET /api/download-all` - Download de todos os arquivos (ZIP)

### **Logs**
- `GET /api/logs` - Obter logs detalhados
- `POST /api/logs/clear` - Limpar logs

## 🏗️ **Arquitetura do Sistema**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   External      │
│   (Web UI)      │◄──►│   (FastAPI)     │◄──►│   (GitHub API)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                       ┌─────────────────┐
                       │   AG2 Agents    │
                       │   (Analysis)    │
                       └─────────────────┘
                                │
                       ┌─────────────────┐
                       │   C4 Analyzer   │
                       │   (Architecture)│
                       └─────────────────┘
```

## 📁 **Estrutura do Projeto**

```
docagent-skyone/
├── Docagenta.py              # Aplicação principal
├── templates/
│   └── index.html           # Interface web
├── static/                  # Arquivos estáticos
├── docs/                    # Documentação gerada
├── workdir/                 # Diretório de trabalho
├── requirements.txt         # Dependências Python
└── README.md               # Este arquivo
```

## 🔧 **Desenvolvimento**

### **Executar em Modo Debug**
```bash
python Docagenta.py --debug
```

### **Executar Testes**
```bash
python -m pytest tests/
```

### **Linting**
```bash
flake8 Docagenta.py
black Docagenta.py
```

## 🤝 **Contribuindo**

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📝 **Licença**

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 🐛 **Reportar Bugs**

Encontrou um bug? Abra uma [issue](https://github.com/seu-usuario/docagent-skyone/issues) com:
- Descrição detalhada do problema
- Passos para reproduzir
- Ambiente (SO, Python version, etc.)
- Logs de erro (se aplicável)

## 💡 **Roadmap**

### **Próximas Funcionalidades**
- [ ] Suporte a GitLab e Bitbucket
- [ ] Análise de dependências e vulnerabilidades
- [ ] Integração com ferramentas de CI/CD
- [ ] API GraphQL
- [ ] Análise de performance de código
- [ ] Geração de testes automatizados
- [ ] Suporte a mais modelos de LLM

### **Melhorias Planejadas**
- [ ] Cache inteligente de análises
- [ ] Análise incremental
- [ ] Dashboard de métricas
- [ ] Notificações em tempo real
- [ ] Exportação para outros formatos (PDF, DOCX)

## 📞 **Suporte**

- **Documentação**: [Wiki do Projeto](https://github.com/seu-usuario/docagent-skyone/wiki)
- **Issues**: [GitHub Issues](https://github.com/seu-usuario/docagent-skyone/issues)
- **Discussões**: [GitHub Discussions](https://github.com/seu-usuario/docagent-skyone/discussions)

## 🏆 **Créditos**

- **AutoGen Framework**: Microsoft Research
- **C4 Model**: Simon Brown
- **FastAPI**: Sebastian Ramirez
- **Tailwind CSS**: Tailwind Labs

## 📊 **Status do Projeto**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

---

**DocAgent Skyone v2.0** - Transformando código em documentação profissional automaticamente 🚀

*Desenvolvido com ❤️ para a comunidade de desenvolvedores*
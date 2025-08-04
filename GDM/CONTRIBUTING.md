# Contributing to AG2 Enhanced Documentation Flow

Thank you for your interest in contributing to AG2 Enhanced Documentation Flow! This document provides guidelines and information for contributors.

## 🚀 Quick Start

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a feature branch** from `main`
4. **Make your changes** with tests
5. **Push to your fork** and submit a pull request

## 🛠️ Development Setup

### Prerequisites

- Python 3.8+ with pip
- Git
- [Ollama](https://ollama.ai) with `devstral:latest` model
- GitHub OAuth App (see [GITHUB_SETUP.md](GITHUB_SETUP.md))

### Local Development

```bash
# Clone your fork
git clone https://github.com/yourusername/ag2-enhanced-documentation-flow.git
cd ag2-enhanced-documentation-flow

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install black flake8 mypy pytest pytest-asyncio

# Set up environment variables
cp .env.example .env
# Edit .env with your GitHub OAuth credentials

# Run the application
python start_web.py
```

### Docker Development

```bash
# Build and run with Docker Compose
docker-compose up --build

# Run with Ollama service
docker-compose --profile ollama up --build
```

## 📋 Code Standards

### Python Code Style

We use **Black** for code formatting and **flake8** for linting:

```bash
# Format code
black .

# Check linting
flake8 .

# Type checking
mypy .
```

### Code Structure

```
app.py              # FastAPI application with OAuth
frontend.html       # Single-page web application  
agents.py           # AG2 documentation agents
config.py           # Configuration models
tools.py            # Analysis and generation tools
start_web.py        # Application launcher
```

### Naming Conventions

- **Files**: snake_case
- **Classes**: PascalCase
- **Functions/Variables**: snake_case
- **Constants**: UPPER_SNAKE_CASE

## 🧪 Testing

```bash
# Run basic import tests
python -c "import config, agents, tools"

# Test FastAPI app loads
python -c "from app import app; print('FastAPI app loads successfully')"

# Run with pytest (when tests are added)
pytest
```

## 📝 Pull Request Process

### Before Submitting

1. **Test your changes** thoroughly
2. **Format code** with Black
3. **Check linting** with flake8
4. **Update documentation** if needed
5. **Write descriptive commit messages**

### PR Requirements

- [ ] **Descriptive title** and description
- [ ] **Reference any related issues**
- [ ] **Include tests** for new features
- [ ] **Update documentation** if needed
- [ ] **Follow code style guidelines**
- [ ] **Pass all CI checks**

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Manual testing completed
- [ ] Added new tests for features

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

## 🚦 Issue Guidelines

### Bug Reports

Include:
- **Clear description** of the issue
- **Steps to reproduce**
- **Expected vs actual behavior**
- **Environment details** (OS, Python version, etc.)
- **Error messages/logs**

### Feature Requests

Include:
- **Clear description** of the feature
- **Use case and motivation**
- **Proposed implementation** (if any)
- **Additional context**

## 🔧 Adding New Features

### New Analysis Tools

1. Add tool functions to `tools.py`
2. Update configuration in `config.py`
3. Integrate with agents in `agents.py`
4. Add UI components in `frontend.html`

### New GitHub Features

1. Add API endpoints in `app.py`
2. Update authentication models
3. Add frontend interface
4. Test with both GitHub.com and Enterprise

### New Documentation Formats

1. Add generation logic to `agents.py`
2. Update download endpoints in `app.py`
3. Add download buttons to frontend
4. Test output formatting

## 🌐 Architecture

### Backend (FastAPI)

```
app.py
├── Authentication (OAuth2)
├── Repository Management
├── Analysis Orchestration
└── File Downloads
```

### Frontend (HTML/JS)

```
frontend.html
├── Authentication UI
├── Repository Selection
├── Analysis Configuration  
└── Progress Tracking
```

### AI Agents (AG2)

```
agents.py
├── DocumentationFlow (Main orchestrator)
├── AnalysisAgent (Code analysis)
├── WriterAgent (Documentation generation)
└── ReviewAgent (Quality control)
```

## 📚 Resources

- **[AG2 Documentation](https://ag2ai.github.io/ag2/)**
- **[FastAPI Documentation](https://fastapi.tiangolo.com/)**
- **[GitHub API Documentation](https://docs.github.com/en/rest)**
- **[Ollama Documentation](https://ollama.ai/docs)**

## 🤝 Community

- **Questions**: [GitHub Discussions](https://github.com/yourusername/ag2-enhanced-documentation-flow/discussions)
- **Issues**: [GitHub Issues](https://github.com/yourusername/ag2-enhanced-documentation-flow/issues)
- **Security**: Email maintainers for security issues

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to AG2 Enhanced Documentation Flow! 🚀
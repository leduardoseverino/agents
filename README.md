# 🚀 AG2 Enhanced Documentation Flow

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![GitHub OAuth](https://img.shields.io/badge/GitHub-OAuth-black.svg)](https://docs.github.com/en/developers/apps/building-oauth-apps)

An advanced technical documentation system powered by **AG2 (AutoGen)** that analyzes GitHub repositories and generates comprehensive, professional documentation using state-of-the-art AI agents.

## ✨ Key Features

🔐 **GitHub Authentication**
- Full OAuth integration with GitHub.com and GitHub Enterprise
- Secure access to public and private repositories
- Repository browsing and selection interface

🤖 **AI-Powered Analysis**
- Multi-agent documentation system using AG2/AutoGen
- Intelligent code analysis with Ollama integration
- Support for multiple programming languages

📚 **Comprehensive Documentation**
- Project overview and architecture analysis
- Installation and configuration guides
- Technical documentation with API references
- Automated README generation

🌐 **Modern Web Interface**
- Responsive design with real-time progress tracking
- Live log streaming during analysis
- Multiple download formats (Markdown, JSON)
- User-friendly repository selection

## 🏗️ Architecture

```
ag2-enhanced-documentation-flow/
├── app.py                    # FastAPI backend with GitHub OAuth
├── frontend.html            # Modern web interface
├── agents.py                # AG2 multi-agent system
├── config.py                # Configuration management
├── tools.py                 # Analysis tools and utilities
├── start_web.py             # Application launcher
├── requirements.txt         # Python dependencies
├── GITHUB_SETUP.md         # OAuth setup guide
└── README.md               # This file
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** with pip
- **[Ollama](https://ollama.ai)** installed with `devstral:latest` model
- **GitHub OAuth App** (see [Setup Guide](GITHUB_SETUP.md))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ag2-enhanced-documentation-flow.git
   cd ag2-enhanced-documentation-flow
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Ollama and pull the model**
   ```bash
   # Install Ollama from https://ollama.ai
   ollama pull devstral:latest
   ```

4. **Set up GitHub OAuth** (see [GITHUB_SETUP.md](GITHUB_SETUP.md))
   ```bash
   # Create .env file
   GITHUB_CLIENT_ID=your_client_id_here
   GITHUB_CLIENT_SECRET=your_client_secret_here
   ```

5. **Start the application**
   ```bash
   python start_web.py
   ```

6. **Open your browser** to `http://localhost:8000`

## 📖 Usage

### Web Interface

1. **Authenticate with GitHub**
   - Choose GitHub.com or GitHub Enterprise
   - Login with your GitHub credentials
   - Authorize the application

2. **Select Repository**
   - Browse your accessible repositories
   - Use search to find specific repos
   - Click to select for analysis

3. **Configure Analysis**
   - Choose analysis mode (Complete/Smart/Limited)
   - Configure options (deep analysis, config files, etc.)
   - Start the enhanced analysis

4. **Download Results**
   - Project overview documentation
   - Installation and configuration guide
   - Technical documentation
   - Complete documentation bundle

### Command Line (Alternative)

```bash
# Direct repository analysis
python app.py --url https://github.com/user/repo

# Custom configuration
python app.py --url https://github.com/user/repo --mode smart --no-config
```

## 🔧 Configuration

### Analysis Modes

- **Complete Analysis**: Analyzes all files in the repository
- **Smart Analysis**: Focuses on important files only
- **Limited Analysis**: Analyzes top 20 most relevant files

### GitHub Enterprise Setup

For GitHub Enterprise Server:
1. Select "GitHub Enterprise" in the web interface
2. Enter your enterprise server URL (e.g., `https://github.company.com`)
3. Configure OAuth app on your enterprise instance

### Environment Variables

```bash
# Required
GITHUB_CLIENT_ID=your_oauth_app_client_id
GITHUB_CLIENT_SECRET=your_oauth_app_client_secret

# Optional
LOG_LEVEL=INFO
HOST=127.0.0.1
PORT=8000
```

## 🛠️ Development

### Project Structure

```
├── app.py              # FastAPI application with OAuth
├── frontend.html       # Single-page web application
├── agents.py           # AG2 documentation agents
├── config.py           # Configuration models
├── tools.py            # Analysis and generation tools
├── start_web.py        # Application launcher
└── static/             # Static assets (if any)
```

### Running in Development Mode

```bash
# Start with auto-reload
python start_web.py --reload

# Or use uvicorn directly
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

### Adding New Analysis Features

1. Add new tools in `tools.py`
2. Create new agents in `agents.py`
3. Update configuration in `config.py`
4. Extend the web interface in `frontend.html`

## 📚 API Documentation

Once running, visit:
- **Interactive API Docs**: `http://localhost:8000/docs`
- **ReDoc Documentation**: `http://localhost:8000/redoc`

### Key Endpoints

- `POST /api/auth/login` - GitHub OAuth login
- `GET /api/repos` - Get user repositories
- `POST /api/analyze` - Start repository analysis
- `GET /api/status` - Check analysis progress
- `GET /api/download/{filename}` - Download results

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[AG2 (AutoGen)](https://github.com/ag2ai/ag2)** - Multi-agent framework
- **[Ollama](https://ollama.ai)** - Local LLM inference
- **[FastAPI](https://fastapi.tiangolo.com/)** - Modern web framework
- **[GitHub API](https://docs.github.com/en/rest)** - Repository access

## 📞 Support

- **Documentation**: [GITHUB_SETUP.md](GITHUB_SETUP.md)
- **Issues**: [GitHub Issues](https://github.com/yourusername/ag2-enhanced-documentation-flow/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/ag2-enhanced-documentation-flow/discussions)

---

<div align="center">
  <strong>Built with ❤️ using AG2, FastAPI, and modern web technologies</strong>
</div>
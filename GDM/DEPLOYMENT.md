# 🚀 Deployment Guide for AG2 Enhanced Documentation Flow

This guide covers different deployment options for the AG2 Enhanced Documentation Flow project.

## 📋 Prerequisites

Before deploying, ensure you have:

- **GitHub OAuth App** configured (see [GITHUB_SETUP.md](GITHUB_SETUP.md))
- **Ollama** server with `devstral:latest` model
- **Environment variables** properly set

## 🐳 Docker Deployment

### Quick Start with Docker Compose

```bash
# Clone the repository
git clone https://github.com/yourusername/ag2-enhanced-documentation-flow.git
cd ag2-enhanced-documentation-flow

# Create environment file
cp .env.example .env
# Edit .env with your GitHub OAuth credentials

# Deploy with Docker Compose
docker-compose up -d

# With Ollama service included
docker-compose --profile ollama up -d
```

### Manual Docker Build

```bash
# Build the image
docker build -t ag2-documentation-flow .

# Run the container
docker run -d \
  --name ag2-docs \
  -p 8000:8000 \
  -e GITHUB_CLIENT_ID=your_client_id \
  -e GITHUB_CLIENT_SECRET=your_client_secret \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/docs:/app/docs \
  ag2-documentation-flow
```

## ☁️ Cloud Platform Deployment

### Heroku

1. **Create Heroku app**
   ```bash
   heroku create your-app-name
   ```

2. **Set environment variables**
   ```bash
   heroku config:set GITHUB_CLIENT_ID=your_client_id
   heroku config:set GITHUB_CLIENT_SECRET=your_client_secret
   ```

3. **Deploy**
   ```bash
   git push heroku main
   ```

### Railway

1. **Connect GitHub repository** to Railway
2. **Set environment variables** in Railway dashboard
3. **Deploy automatically** on push to main branch

### DigitalOcean App Platform

1. **Create new app** from GitHub repository
2. **Configure environment variables**
3. **Set build command**: `pip install -r requirements.txt`
4. **Set run command**: `python app.py --host 0.0.0.0 --port $PORT`

## 🖥️ Local Development

### Standard Python Setup

```bash
# Clone and setup
git clone https://github.com/yourusername/ag2-enhanced-documentation-flow.git
cd ag2-enhanced-documentation-flow
pip install -r requirements.txt

# Install Ollama
# Visit https://ollama.ai and follow installation instructions
ollama pull devstral:latest

# Set environment variables
export GITHUB_CLIENT_ID=your_client_id
export GITHUB_CLIENT_SECRET=your_client_secret

# Run the application
python start_web.py
```

### Development Mode

```bash
# Run with auto-reload
python start_web.py --reload

# Or use uvicorn directly
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

## 🔧 Configuration

### Environment Variables

```bash
# Required
GITHUB_CLIENT_ID=your_oauth_app_client_id
GITHUB_CLIENT_SECRET=your_oauth_app_client_secret

# Optional
LOG_LEVEL=INFO                    # Logging level
HOST=127.0.0.1                    # Server host
PORT=8000                         # Server port
OLLAMA_BASE_URL=http://localhost:11434  # Ollama server URL
```

### GitHub OAuth Configuration

Update your GitHub OAuth app settings for production:

- **Homepage URL**: `https://yourdomain.com`
- **Authorization callback URL**: `https://yourdomain.com/api/auth/callback`

### Ollama Setup

For production deployments:

1. **Install Ollama** on your server
2. **Pull the model**: `ollama pull devstral:latest`
3. **Configure base URL** if running on different server

## 🔒 Security Considerations

### Production Checklist

- [ ] **HTTPS enabled** for OAuth security
- [ ] **Environment variables** securely stored
- [ ] **GitHub OAuth app** configured for production URLs
- [ ] **Rate limiting** configured if needed
- [ ] **Monitoring** and logging set up
- [ ] **Backup strategy** for generated documentation

### Security Headers

For production deployments, consider adding security headers:

```python
# Add to app.py
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware

app.add_middleware(HTTPSRedirectMiddleware)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["yourdomain.com", "*.yourdomain.com"])
```

## 📊 Monitoring

### Health Checks

The application provides a health check endpoint:

```bash
curl https://yourdomain.com/health
```

### Logging

Logs are written to:
- **Console**: Application logs
- **Files**: `/app/logs/` in container

### Metrics

Consider integrating with monitoring services:
- **Application Performance**: New Relic, DataDog
- **Infrastructure**: Prometheus + Grafana
- **Uptime**: UptimeRobot, Pingdom

## 🚨 Troubleshooting

### Common Issues

**Issue**: "GitHub authentication failed"
**Solution**: Verify OAuth app configuration and credentials

**Issue**: "Ollama connection failed"  
**Solution**: Ensure Ollama is running and accessible

**Issue**: "Port already in use"
**Solution**: Change port in environment variables or stop conflicting service

**Issue**: "Permission denied" in Docker
**Solution**: Check file permissions and user configuration

### Debug Mode

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
python start_web.py
```

## 📱 Mobile Considerations

The web interface is responsive and works on mobile devices. For better mobile experience:

- **Viewport meta tag** is included
- **Touch-friendly** buttons and inputs
- **Responsive design** adapts to screen size

## 🔄 Updates and Maintenance

### Updating the Application

```bash
# Pull latest changes
git pull origin main

# Rebuild Docker image
docker-compose build

# Restart services
docker-compose up -d
```

### Model Updates

```bash
# Update Ollama model
ollama pull devstral:latest

# Restart application to use updated model
docker-compose restart app
```

---

For more deployment options and configurations, check the [GitHub repository](https://github.com/yourusername/ag2-enhanced-documentation-flow) or open an issue for support.
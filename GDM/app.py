"""
AG2 Enhanced Documentation Flow - FastAPI Backend
=================================================

Clean FastAPI backend with proper progress tracking and file downloads.
"""

import os
import json
import asyncio
import logging
import secrets
import httpx
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field
import uvicorn

# Import our organized modules
from config import ModelConfig, DocumentationState, DocPlan, DocItem
from agents import EnhancedDocumentationFlow

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app setup
app = FastAPI(
    title="AG2 Enhanced Documentation Flow",
    description="Advanced technical documentation system",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GitHub OAuth Configuration
GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET")

# Global state management
class ApplicationState:
    def __init__(self):
        self.current_analysis = None
        self.analysis_status = {
            "status": "idle",
            "phase": "",
            "progress": 0,
            "message": "Ready for analysis",
            "logs": [],
            "current_step": "",
            "results": None
        }
        self.logs = []
        
        # GitHub authentication state
        self.oauth_states = {}  # Store OAuth states
        self.user_sessions = {}  # Store user authentication sessions
        self.github_clients = {}  # Store GitHub API clients per user
    
    def update_status(self, status: str = None, phase: str = None, 
                     progress: int = None, message: str = None, 
                     current_step: str = None):
        if status is not None:
            self.analysis_status["status"] = status
        if phase is not None:
            self.analysis_status["phase"] = phase
        if progress is not None:
            self.analysis_status["progress"] = progress
        if message is not None:
            self.analysis_status["message"] = message
        if current_step is not None:
            self.analysis_status["current_step"] = current_step
    
    def add_log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        self.analysis_status["logs"] = self.logs[-20:]  # Keep last 20 logs
        logger.info(message)
    
    def clear_logs(self):
        self.logs = []
        self.analysis_status["logs"] = []

# Global application state
app_state = ApplicationState()

# Pydantic models
class AnalysisRequest(BaseModel):
    url: str = Field(description="GitHub repository URL")
    model: str = Field(default="devstral:latest", description="LLM model to use")
    analysis_mode: str = Field(default="all", description="Analysis mode: all, smart, or limited")
    include_config: bool = Field(default=True, description="Include config files")
    deep_analysis: bool = Field(default=True, description="Deep function analysis")
    fast_review: bool = Field(default=True, description="Fast review mode")
    anonymous: bool = Field(default=True, description="Anonymous mode")

class StatusResponse(BaseModel):
    status: str
    phase: str
    progress: int
    message: str
    logs: List[str] = []
    current_step: str = ""

# GitHub Authentication Models
class GitHubLoginRequest(BaseModel):
    github_type: str = Field(description="'github.com' or 'enterprise'")
    base_url: str = Field(description="GitHub base URL")

class GitHubLoginResponse(BaseModel):
    auth_url: str = Field(description="OAuth authorization URL")
    state: str = Field(description="OAuth state parameter")

class GitHubUser(BaseModel):
    id: int
    login: str
    name: Optional[str]
    email: Optional[str]
    avatar_url: str
    html_url: str

class AuthStatusResponse(BaseModel):
    authenticated: bool
    user: Optional[GitHubUser] = None
    github_type: Optional[str] = None

class GitHubRepository(BaseModel):
    id: int
    name: str
    full_name: str
    description: Optional[str]
    html_url: str
    private: bool
    fork: bool
    language: Optional[str]
    stargazers_count: int
    forks_count: int
    updated_at: str

# Routes
@app.get("/health")
async def health_check():
    """Health check endpoint for Docker and monitoring"""
    return {"status": "healthy", "service": "AG2 Enhanced Documentation Flow"}

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main frontend"""
    try:
        frontend_path = Path("frontend.html")
        if frontend_path.exists():
            return frontend_path.read_text(encoding="utf-8")
        else:
            return """
            <html>
                <body>
                    <h1>AG2 Enhanced Documentation Flow</h1>
                    <p>Frontend file not found. Please ensure frontend.html exists.</p>
                </body>
            </html>
            """
    except Exception as e:
        logger.error(f"Error serving frontend: {e}")
        return HTMLResponse(f"<html><body><h1>Error: {e}</h1></body></html>")

@app.post("/api/analyze")
async def start_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Start repository analysis"""
    try:
        # Check if analysis is already running
        if app_state.analysis_status["status"] == "running":
            raise HTTPException(status_code=400, detail="Analysis already in progress")
        
        # Validate URL
        if not request.url.startswith("https://github.com/"):
            raise HTTPException(status_code=400, detail="Invalid GitHub URL")
        
        # Reset state
        app_state.clear_logs()
        app_state.update_status(
            status="running",
            phase="initialization",
            progress=0,
            message="Initializing analysis...",
            current_step="Starting enhanced documentation flow"
        )
        
        app_state.add_log("🚀 Starting AG2 Enhanced Documentation Flow")
        app_state.add_log(f"📊 Repository: {request.url}")
        app_state.add_log(f"⚙️ Configuration: {request.analysis_mode} mode, Fast review: {request.fast_review}")
        
        # Start analysis in background
        background_tasks.add_task(run_analysis, request)
        
        return {"status": "started", "message": "Analysis started successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """Get current analysis status"""
    return StatusResponse(**app_state.analysis_status)

@app.get("/api/results")
async def get_results():
    """Get analysis results"""
    if app_state.analysis_status["results"] is None:
        raise HTTPException(status_code=404, detail="No results available")
    
    return app_state.analysis_status["results"]

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    """Download generated documentation files"""
    try:
        # Security check - only allow specific file patterns
        allowed_patterns = [
            "01_project_overview.md",
            "02_installation_configuration.md", 
            "03_technical_documentation.md",
            "complete_documentation.md",
            "analysis_metadata.json"
        ]
        
        if filename not in allowed_patterns:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Check docs directory
        docs_dir = Path("docs")
        file_path = docs_dir / filename
        
        # Special handling for complete documentation
        if filename == "complete_documentation.md":
            file_path = await create_complete_documentation()
        elif filename == "analysis_metadata.json":
            file_path = await create_metadata_file()
        
        if not file_path or not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type='application/octet-stream'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading file {filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }

# Background analysis function
async def run_analysis(request: AnalysisRequest):
    """Run the complete analysis workflow"""
    try:
        app_state.add_log("🔧 Initializing enhanced flow system...")
        
        # Create configuration
        config = ModelConfig()
        config.llm_model = request.model
        config.fast_review_mode = request.fast_review
        
        # Initialize flow system with analysis mode
        flow_system = EnhancedDocumentationFlow(config, request.analysis_mode)
        
        # Phase 1: Clone repository
        app_state.update_status(
            phase="cloning",
            progress=10,
            message="Cloning repository...",
            current_step="Downloading repository"
        )
        app_state.add_log("📥 Starting repository clone...")
        
        clone_success = flow_system.clone_repository(request.url)
        if not clone_success:
            raise Exception("Failed to clone repository")
        
        app_state.add_log("✅ Repository cloned successfully")
        
        # Phase 2: Analysis and Planning
        app_state.update_status(
            phase="analysis",
            progress=30,
            message="Analyzing code structure...",
            current_step="Examining files and structure"
        )
        app_state.add_log("🔬 Starting code analysis...")
        
        plan_success = flow_system.enhanced_planning_phase()
        if not plan_success:
            app_state.add_log("⚠️ Planning had issues, using fallback")
        else:
            app_state.add_log("✅ Analysis plan created")
        
        # Phase 3: Documentation Generation
        app_state.update_status(
            phase="documentation",
            progress=60,
            message="Generating technical documentation...",
            current_step="Creating documentation sections"
        )
        app_state.add_log("📝 Starting documentation generation...")
        
        doc_success = flow_system.enhanced_documentation_phase()
        if not doc_success:
            app_state.add_log("⚠️ Documentation had issues, using fallback")
        else:
            app_state.add_log("✅ Documentation generated successfully")
        
        # Phase 4: Finalization
        app_state.update_status(
            phase="finalization",
            progress=90,
            message="Finalizing results...",
            current_step="Preparing downloads and results"
        )
        
        # Prepare results
        # Calculate actual files analyzed based on mode
        files_analyzed = "all" if request.analysis_mode == "all" else "smart selection" if request.analysis_mode == "smart" else "limited (20)"
        
        results = {
            "status": "success" if doc_success else "partial",
            "message": "Documentation generated successfully",
            "docs_count": len(flow_system.state.generated_docs) if flow_system.state else 0,
            "files_analyzed": files_analyzed,
            "error_count": flow_system.error_count if flow_system else 0,
            "generated_docs": flow_system.state.generated_docs if flow_system.state else [],
            "metadata": {
                "project_url": request.url,
                "repo_path": flow_system.state.repo_path if flow_system.state else None,
                "generated_at": datetime.now().isoformat(),
                "system_version": "Enhanced AG2 Flow v2.0",
                "configuration": {
                    "analysis_mode": request.analysis_mode,
                    "fast_review": request.fast_review,
                    "deep_analysis": request.deep_analysis,
                    "include_config": request.include_config,
                    "anonymous": request.anonymous
                }
            }
        }
        
        # Store results
        app_state.analysis_status["results"] = results
        
        # Complete
        app_state.update_status(
            status="completed",
            progress=100,
            message="Analysis completed successfully!",
            current_step="Ready for download"
        )
        app_state.add_log("🎉 Enhanced documentation flow completed!")
        app_state.add_log(f"📊 Generated {results['docs_count']} documentation sections")
        
    except Exception as e:
        error_msg = f"Analysis failed: {str(e)}"
        app_state.add_log(f"❌ {error_msg}")
        app_state.update_status(
            status="error",
            message=error_msg,
            current_step="Analysis failed"
        )
        logger.error(f"Analysis error: {e}")

async def create_complete_documentation() -> Optional[Path]:
    """Create complete documentation file"""
    try:
        docs_dir = Path("docs")
        if not docs_dir.exists():
            return None
        
        complete_path = docs_dir / "complete_documentation.md"
        
        # Combine all documentation files
        section_files = [
            "01_project_overview.md",
            "02_installation_configuration.md",
            "03_technical_documentation.md"
        ]
        
        content = f"""# Complete Technical Documentation

Generated by AG2 Enhanced Documentation Flow v2.0  
Date: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}

---

"""
        
        for section_file in section_files:
            section_path = docs_dir / section_file
            if section_path.exists():
                section_content = section_path.read_text(encoding="utf-8")
                content += f"{section_content}\n\n---\n\n"
        
        content += f"""
## Generation Information

- **System:** AG2 Enhanced Documentation Flow v2.0
- **Generated:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
- **Status:** {app_state.analysis_status.get('status', 'unknown')}

---
*Documentation generated automatically by AG2 Enhanced Documentation Flow*
"""
        
        complete_path.write_text(content, encoding="utf-8")
        return complete_path
        
    except Exception as e:
        logger.error(f"Error creating complete documentation: {e}")
        return None

async def create_metadata_file() -> Optional[Path]:
    """Create metadata JSON file"""
    try:
        docs_dir = Path("docs")
        docs_dir.mkdir(exist_ok=True)
        
        metadata_path = docs_dir / "analysis_metadata.json"
        
        metadata = {
            "system": "AG2 Enhanced Documentation Flow v2.0",
            "analysis_status": app_state.analysis_status,
            "logs": app_state.logs,
            "generated_at": datetime.now().isoformat(),
            "version": "2.0.0"
        }
        
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        
        return metadata_path
        
    except Exception as e:
        logger.error(f"Error creating metadata file: {e}")
        return None

# Helper function to get session ID from request
def get_session_id(request: Request) -> str:
    """Get or create a session ID for the user"""
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = secrets.token_urlsafe(32)
    return session_id

def get_github_client(session_id: str) -> Optional[httpx.AsyncClient]:
    """Get GitHub API client for authenticated user"""
    if session_id in app_state.user_sessions:
        user_data = app_state.user_sessions[session_id]
        if "access_token" in user_data:
            headers = {
                "Authorization": f"token {user_data['access_token']}",
                "Accept": "application/vnd.github.v3+json"
            }
            if user_data.get("github_type") == "enterprise":
                base_url = user_data["base_url"].rstrip("/") + "/api/v3"
            else:
                base_url = "https://api.github.com"
            
            return httpx.AsyncClient(base_url=base_url, headers=headers)
    return None

# GitHub Authentication Routes
@app.post("/api/auth/login", response_model=GitHubLoginResponse)
async def github_login(request: GitHubLoginRequest, request_obj: Request):
    """Initiate GitHub OAuth login"""
    try:
        session_id = get_session_id(request_obj)
        state = secrets.token_urlsafe(32)
        
        # Store OAuth state and configuration
        app_state.oauth_states[state] = {
            "session_id": session_id,
            "github_type": request.github_type,
            "base_url": request.base_url
        }
        
        # Determine OAuth URLs based on GitHub type
        if request.github_type == "enterprise":
            base_url = request.base_url.rstrip("/")
            auth_url = f"{base_url}/login/oauth/authorize"
        else:
            auth_url = "https://github.com/login/oauth/authorize"
        
        # Create OAuth URL
        oauth_params = {
            "client_id": GITHUB_CLIENT_ID,
            "redirect_uri": f"{request_obj.base_url}api/auth/callback",
            "scope": "repo,user:email",
            "state": state
        }
        
        oauth_url = f"{auth_url}?" + "&".join([f"{k}={v}" for k, v in oauth_params.items()])
        
        response = GitHubLoginResponse(auth_url=oauth_url, state=state)
        return response
        
    except Exception as e:
        logger.error(f"GitHub login error: {e}")
        raise HTTPException(status_code=500, detail=f"Login initiation failed: {str(e)}")

@app.get("/api/auth/callback")
async def github_callback(code: str, state: str, request: Request):
    """Handle GitHub OAuth callback"""
    try:
        # Validate state
        if state not in app_state.oauth_states:
            raise HTTPException(status_code=400, detail="Invalid OAuth state")
        
        oauth_data = app_state.oauth_states.pop(state)
        session_id = oauth_data["session_id"]
        github_type = oauth_data["github_type"]
        base_url = oauth_data["base_url"]
        
        # Exchange code for access token
        if github_type == "enterprise":
            token_url = f"{base_url.rstrip('/')}/login/oauth/access_token"
            api_base = f"{base_url.rstrip('/')}/api/v3"
        else:
            token_url = "https://github.com/login/oauth/access_token"
            api_base = "https://api.github.com"
        
        async with httpx.AsyncClient() as client:
            # Get access token
            token_response = await client.post(token_url, data={
                "client_id": GITHUB_CLIENT_ID,
                "client_secret": GITHUB_CLIENT_SECRET,
                "code": code
            }, headers={"Accept": "application/json"})
            
            token_data = token_response.json()
            
            if "access_token" not in token_data:
                raise HTTPException(status_code=400, detail="Failed to get access token")
            
            access_token = token_data["access_token"]
            
            # Get user information
            user_response = await client.get(f"{api_base}/user", headers={
                "Authorization": f"token {access_token}",
                "Accept": "application/vnd.github.v3+json"
            })
            
            user_data = user_response.json()
            
            # Store user session
            app_state.user_sessions[session_id] = {
                "access_token": access_token,
                "user": user_data,
                "github_type": github_type,
                "base_url": base_url
            }
        
        # Redirect to main page
        response = RedirectResponse(url="/", status_code=302)
        response.set_cookie("session_id", session_id, max_age=86400)  # 24 hours
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"GitHub callback error: {e}")
        raise HTTPException(status_code=500, detail=f"Authentication failed: {str(e)}")

@app.get("/api/auth/status", response_model=AuthStatusResponse)
async def auth_status(request: Request):
    """Check authentication status"""
    session_id = request.cookies.get("session_id")
    
    if session_id and session_id in app_state.user_sessions:
        user_data = app_state.user_sessions[session_id]["user"]
        github_user = GitHubUser(**user_data)
        github_type = app_state.user_sessions[session_id].get("github_type", "github.com")
        return AuthStatusResponse(authenticated=True, user=github_user, github_type=github_type)
    
    return AuthStatusResponse(authenticated=False)

@app.post("/api/auth/logout")
async def github_logout(request: Request):
    """Logout user"""
    session_id = request.cookies.get("session_id")
    
    if session_id and session_id in app_state.user_sessions:
        del app_state.user_sessions[session_id]
    
    response = JSONResponse({"message": "Logged out successfully"})
    response.delete_cookie("session_id")
    return response

@app.get("/api/repos", response_model=List[GitHubRepository])
async def get_repositories(request: Request):
    """Get user's repositories"""
    session_id = request.cookies.get("session_id")
    
    if not session_id or session_id not in app_state.user_sessions:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    github_client = get_github_client(session_id)
    if not github_client:
        raise HTTPException(status_code=401, detail="Invalid authentication")
    
    try:
        async with github_client as client:
            # Get user's repositories (including organizations)
            repos_response = await client.get("/user/repos", params={
                "type": "all",
                "sort": "updated",
                "per_page": 100
            })
            
            if repos_response.status_code != 200:
                raise HTTPException(status_code=repos_response.status_code, detail="Failed to fetch repositories")
            
            repos_data = repos_response.json()
            repositories = [GitHubRepository(**repo) for repo in repos_data]
            
            return repositories
            
    except Exception as e:
        logger.error(f"Error fetching repositories: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch repositories: {str(e)}")

# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AG2 Enhanced Documentation Flow")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    print("🚀 AG2 Enhanced Documentation Flow")
    print("=" * 50)
    print(f"🌐 Server starting on http://{args.host}:{args.port}")
    print("💡 Open your browser to access the interface")
    print("💡 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )
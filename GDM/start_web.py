#!/usr/bin/env python3
"""
AG2 Enhanced Documentation Flow - Web Interface Launcher
========================================================

Launch the modern web interface with FastAPI backend.
"""

import sys
import subprocess
import os
import webbrowser
import time
from pathlib import Path

def check_requirements():
    """Check if all requirements are met"""
    print("🔍 Checking requirements for web interface...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    
    print(f"✅ Python {sys.version.split()[0]}")
    
    # Ensure we're working from the script's directory
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    print(f"📁 Working directory: {script_dir}")
    
    # Check if we're in the right directory
    required_files = ["app.py", "frontend.html", "config.py", "agents.py", "tools.py"]
    for file in required_files:
        file_path = script_dir / file
        if not file_path.exists():
            print(f"❌ Required file not found: {file} (looked in {file_path})")
            return False
    
    print("✅ Project files found")
    
    # Check critical imports
    try:
        import fastapi
        print("✅ FastAPI available")
    except ImportError:
        print("❌ FastAPI not installed")
        print("💡 Install with: pip install -r requirements.txt")
        return False
    
    try:
        import uvicorn
        print("✅ Uvicorn available")
    except ImportError:
        print("❌ Uvicorn not installed")
        print("💡 Install with: pip install -r requirements.txt")
        return False
    
    try:
        import autogen
        print("✅ AG2 available")
    except ImportError:
        print("❌ AG2 (pyautogen) not installed")
        print("💡 Install with: pip install pyautogen")
        return False
    
    # Check Ollama
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("✅ Ollama available")
        else:
            print("⚠️ Ollama may not be working properly")
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("❌ Ollama not found")
        print("💡 Install from: https://ollama.ai")
        return False
    
    # Check devstral model
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0 and "devstral:latest" in result.stdout:
            print("✅ devstral:latest model available")
        else:
            print("❌ devstral:latest model not found")
            print("💡 Install with: ollama pull devstral:latest")
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("❌ Cannot check Ollama models")
        return False
    
    return True

def main():
    """Main startup function"""
    print("🚀 AG2 Enhanced Documentation Flow - Web Interface")
    print("=" * 60)
    
    if not check_requirements():
        print("\n❌ Requirements not met. Please fix the issues above.")
        return 1
    
    print("\n✅ All requirements met!")
    print("🚀 Starting Enhanced Documentation Flow Web Interface...")
    
    # Set default host and port
    host = "127.0.0.1"
    port = 8000
    
    # Check if port is available
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex((host, port))
    sock.close()
    
    if result == 0:
        print(f"⚠️ Port {port} is already in use, trying port {port + 1}")
        port = port + 1
    
    print(f"\n🌐 Server will start on http://{host}:{port}")
    print("💡 The web interface will open automatically in your browser")
    print("💡 Press Ctrl+C to stop the server")
    print("-" * 60)
    
    try:
        # Start the server using absolute path
        script_dir = Path(__file__).parent.absolute()
        app_path = script_dir / "app.py"
        process = subprocess.Popen([
            sys.executable, str(app_path),
            "--host", host,
            "--port", str(port)
        ])
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Open browser
        url = f"http://{host}:{port}"
        print(f"🌐 Opening browser at {url}")
        webbrowser.open(url)
        
        # Wait for the process
        process.wait()
        
    except KeyboardInterrupt:
        print("\n\n👋 Enhanced Documentation Flow stopped")
        if 'process' in locals():
            process.terminate()
        return 0
    except Exception as e:
        print(f"\n❌ Error launching web interface: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
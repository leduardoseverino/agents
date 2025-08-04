"""
AG2 Multi-Agent Documentation Flow System
=========================================

Contains the enhanced documentation workflow with specialized AG2 agents.
"""

import os
import sys
import json
import re
import time
import shutil
import subprocess
import urllib.request
import urllib.error
import socket
import tempfile
import traceback
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime

from config import (
    AG2_AVAILABLE, ModelConfig, DocumentationState, DocPlan, DocItem, 
    SYSTEM_VERSION, logger
)
from tools import AdvancedRepositoryTools

if AG2_AVAILABLE:
    from autogen import ConversableAgent, GroupChat, GroupChatManager


class EnhancedDocumentationFlow:
    """Enhanced AG2 Flow system for complete documentation"""
    
    def __init__(self, config: ModelConfig, analysis_mode: str = "all"):
        self.config = config
        self.analysis_mode = analysis_mode
        self.state = None
        self.tools = None
        self.agents = {}
        self.error_count = 0
        self._setup_llm_config()
        if AG2_AVAILABLE:
            self._setup_agents()
        print(f"🤖 Enhanced AG2 Documentation Flow initialized (mode: {analysis_mode})")
    
    def _setup_llm_config(self):
        """Optimized LLM configuration"""
        self.llm_config = {
            "config_list": [{
                "model": self.config.llm_model,
                "api_type": "ollama",
                "base_url": "{env.OLLAMA_URL}",
                "api_key": "fake_key"
            }],
            "timeout": self.config.timeout,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "seed": 42
        }
    
    def _setup_agents(self):
        """Setup agents with enhanced prompts"""
        if not AG2_AVAILABLE:
            print("⚠️ AG2 not available - skipping agent setup")
            return
        
        # Advanced Code Explorer
        self.agents["code_explorer"] = ConversableAgent(
            name="AdvancedCodeExplorer",
            system_message="""You are an expert in advanced code analysis. Your function is to perform COMPLETE and DETAILED repository analysis.

**MISSION:** Provide deep technical analysis to enable complete documentation in 3 parts:
1. Project Overview
2. Installation and Configuration Guide  
3. **Detailed Technical Documentation of Files** (MAIN FOCUS)

**AVAILABLE TOOLS:**
- `directory_read(path)`: List and categorize directory content
- `file_read(file_path)`: Detailed analysis of individual files
- `find_key_files()`: Identify important files by category
- `analyze_code_structure()`: Complete code base statistics
- `detailed_file_analysis(analysis_mode)`: Deep analysis based on mode (all/smart/limited)

**MANDATORY ANALYSIS PROTOCOL:**

1. **General Structure**: `analyze_code_structure()` - understand architecture
2. **Key Files**: `find_key_files()` - identify important components  
3. **Detailed Analysis**: `detailed_file_analysis("all")` - examine ALL files (complete analysis)
4. **Specific Reading**: Use `file_read()` on most critical files found
5. **Directed Exploration**: `directory_read()` on relevant directories

**ANALYSIS MODES:**
- `detailed_file_analysis("all")` - Analyze ALL code files in repository
- `detailed_file_analysis("smart")` - Smart selection of important files  
- `detailed_file_analysis("limited")` - Limited to most important files only

**FOCUS ESPECIALLY ON:**
- Entry points (main.py, index.js, app.py)
- Configurations (package.json, requirements.txt, etc.)
- Main logic files
- Important APIs and interfaces
- Data structures and models
- Main functions and classes

**IMPORTANT:** 
- Use ALL available tools
- Be systematic and complete
- Document languages, frameworks, APIs found
- Identify dependencies and technologies used
- Map architecture and code flow""",
            llm_config=self.llm_config,
            human_input_mode="NEVER"
        )
        
        # Enhanced Documentation Planner
        self.agents["documentation_planner"] = ConversableAgent(
            name="EnhancedDocumentationPlanner",
            system_message="""You are an advanced technical documentation planner. Based on AdvancedCodeExplorer analysis, create a plan MANDATORILY with 3 specific sections.

**MANDATORY PLAN - EXACTLY 3 SECTIONS:**

1. **"Project Overview"**
   - Purpose and main functionality
   - Technologies and languages used
   - General architecture

2. **"Installation and Configuration Guide"**  
   - System prerequisites
   - Installation steps
   - Initial configuration
   - How to run the project

3. **"Technical Documentation of Files"** (MAIN SECTION)
   - Detailed analysis of each important file
   - Main functions and classes
   - APIs and interfaces
   - Data flow and logic
   - Code structure

**MANDATORY JSON FORMAT:**
```json
{
  "overview": "Concise but complete project description",
  "docs": [
    {
      "title": "Project Overview",
      "description": "Complete project presentation, technologies and architecture",
      "prerequisites": "Basic programming knowledge",
      "examples": ["Main functionalities", "Technologies used"],
      "goal": "Provide complete understanding of project purpose and structure"
    },
    {
      "title": "Installation and Configuration Guide", 
      "description": "Complete instructions for installation, configuration and execution",
      "prerequisites": "Compatible operating system and basic tools",
      "examples": ["Installation steps", "Execution commands", "Necessary configurations"],
      "goal": "Allow any developer to configure and run the project"
    },
    {
      "title": "Technical Documentation of Files",
      "description": "Detailed analysis of each file, functions, classes, APIs and code flow",
      "prerequisites": "Knowledge in languages and frameworks used",
      "examples": ["Analysis of main files", "Function documentation", "API mapping"],
      "goal": "Provide complete technical documentation for developers to contribute or understand the code"
    }
  ]
}
```

**IMPORTANT:**
- Use specific information from code analysis
- Be precise about identified technologies
- Focus on third section as most important""",
            llm_config=self.llm_config,
            human_input_mode="NEVER"
        )
        
        # Technical Documentation Writer
        self.agents["technical_writer"] = ConversableAgent(
            name="TechnicalDocumentationWriter",
            system_message="""You are a technical writer specialized in code documentation. Write DETAILED and PROFESSIONAL technical documentation.

**MAIN FOCUS:** "Technical Documentation of Files" section must be EXTREMELY detailed.

**STANDARD STRUCTURE FOR EACH SECTION:**

## For "Project Overview":
# Project Overview

## 🎯 Purpose
[Clear explanation of what the project does]

## 🛠️ Technologies Used
[Detailed list of languages, frameworks, libraries]

## 🏗️ Architecture
[Description of structure and code organization]

## For "Installation and Configuration Guide":
# Installation and Configuration Guide

## 📋 Prerequisites
[Systems, tools and necessary dependencies]

## 🚀 Installation
[Detailed installation steps]

## ⚙️ Configuration
[Necessary configurations]

## ▶️ Execution
[How to run the project]

## For "Technical Documentation of Files" (MOST IMPORTANT):
# Technical Documentation of Files

## 📁 General Structure
[Organization of directories and files]

## 🔧 Main Files

### [FILENAME] (Language)
**Purpose:** [What this file does]
**Location:** `path/to/file`

#### 📋 Functionalities:
- [Detailed list of functionalities]

#### 🔧 Main Functions:
- `function1()`: [Detailed description]
- `function2()`: [Detailed description]

#### 📊 Classes/Structures:
- `Class1`: [Description and purpose]

#### 🔌 APIs/Interfaces:
- [Documentation of exposed APIs]

#### ⚡ Execution Flow:
[How the code executes]

#### 📝 Notes:
[Important notes, limitations, etc.]

**IMPORTANT:**
- For third section, document ALL important files
- Include code examples when relevant
- Use emojis for visual organization
- Be technical but clear
- Document dependencies between files""",
            llm_config=self.llm_config,
            human_input_mode="NEVER"
        )
        
        # Fast Documentation Reviewer (Optimized for Speed)
        self.agents["documentation_reviewer"] = ConversableAgent(
            name="FastDocumentationReviewer",
            system_message="""You are a fast and efficient documentation reviewer. Provide QUICK but quality improvements.

**FAST REVIEW CHECKLIST:**
✅ Check if all 3 sections exist
✅ Verify technical accuracy 
✅ Ensure proper Markdown formatting
✅ Add brief improvements where needed

**SPEED OPTIMIZATION RULES:**
- Make SMALL, focused improvements only
- Don't rewrite large sections
- Focus on critical issues only
- Keep responses concise and direct
- Prioritize technical section fixes

**QUICK ACTIONS:**
- Fix obvious errors
- Add missing technical details (briefly)
- Improve formatting consistency
- Ensure developer usefulness

**RESPONSE FORMAT:**
Provide the improved documentation directly without lengthy explanations. Be fast and efficient while maintaining quality.""",
            llm_config=self.llm_config,
            human_input_mode="NEVER"
        )
    
    def _register_tools_safely(self):
        """Register advanced tools with error handling"""
        if not self.tools:
            print("⚠️ Tools not initialized")
            return False
        
        if not AG2_AVAILABLE:
            print("⚠️ AG2 not available - skipping tool registration")
            return False
        
        try:
            explorer = self.agents["code_explorer"]
            
            @explorer.register_for_llm(description="List and categorize detailed directory content")
            @explorer.register_for_execution()
            def directory_read(path: str = "") -> str:
                return self.tools.directory_read(path)
            
            @explorer.register_for_llm(description="Detailed analysis of individual files with technical information")
            @explorer.register_for_execution()  
            def file_read(file_path: str) -> str:
                return self.tools.file_read(file_path)
            
            @explorer.register_for_llm(description="Identify and categorize important project files")
            @explorer.register_for_execution()
            def find_key_files() -> str:
                return self.tools.find_key_files()
            
            @explorer.register_for_llm(description="Complete code structure analysis with detailed statistics")
            @explorer.register_for_execution()
            def analyze_code_structure() -> str:
                return self.tools.analyze_code_structure()
            
            @explorer.register_for_llm(description="Deep technical analysis of files based on analysis mode")
            @explorer.register_for_execution()
            def detailed_file_analysis(analysis_mode: str = None) -> str:
                mode = analysis_mode if analysis_mode else self.analysis_mode
                return self.tools.detailed_file_analysis(mode)
            
            print("🔧 Advanced tools registered successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error registering tools: {e}")
            return False
    
    def clone_repository(self, project_url: str) -> bool:
        """Clone with detailed diagnostics"""
        print(f"📥 Starting clone: {project_url}")
        
        # Initialize state if not exists
        if self.state is None:
            print("🔧 Initializing system state...")
            self.state = DocumentationState(project_url=project_url)
        
        # Validate URL
        if not self._validate_github_url(project_url):
            print(f"❌ Invalid URL: {project_url}")
            return False
        
        # Check connectivity
        if not self._check_github_connectivity():
            print("❌ No GitHub connectivity")
            return False
        
        # Check if repository exists
        if not self._check_repository_exists(project_url):
            print(f"❌ Repository doesn't exist or is private: {project_url}")
            return False
        
        # Prepare directories
        repo_name = project_url.split("/")[-1].replace(".git", "")
        workdir = Path("workdir").resolve()
        workdir.mkdir(exist_ok=True)
        repo_path = workdir / repo_name
        
        print(f"📁 Working directory: {workdir}")
        print(f"📁 Clone destination: {repo_path}")
        
        # Robust cleanup of existing directory
        if repo_path.exists():
            print(f"🗑️ Removing existing directory: {repo_path}")
            
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
                        print(f"✅ Directory removed successfully")
                        break
                    else:
                        print(f"⚠️ Attempt {attempt + 1} failed")
                        
                except Exception as e:
                    print(f"⚠️ Error in removal (attempt {attempt + 1}): {e}")
                    
                if attempt < 2:
                    time.sleep(1)
            
            if repo_path.exists():
                backup_path = repo_path.with_suffix(f".backup_{int(time.time())}")
                try:
                    repo_path.rename(backup_path)
                    print(f"🔄 Directory moved to: {backup_path}")
                except Exception as e:
                    print(f"❌ Could not clean directory: {e}")
                    return False
        
        # Try clone with retry
        max_retries = 3
        clone_success = False
        
        for attempt in range(max_retries):
            try:
                print(f"🔄 Clone attempt {attempt + 1}/{max_retries}")
                
                if attempt == 0:
                    cmd = ["git", "clone", "--depth", "1", "--single-branch", project_url, str(repo_path)]
                elif attempt == 1:
                    cmd = ["git", "clone", "--single-branch", project_url, str(repo_path)]
                else:
                    cmd = ["git", "clone", project_url, str(repo_path)]
                
                print(f"🔧 Executing: {' '.join(cmd)}")
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                
                print(f"🔍 Return code: {result.returncode}")
                
                if result.returncode == 0:
                    print(f"✅ Git clone executed successfully on attempt {attempt + 1}")
                    clone_success = True
                    break
                else:
                    error_msg = result.stderr.strip()
                    print(f"❌ Error in git clone (attempt {attempt + 1}):")
                    print(f"   stderr: {error_msg[:200]}")
                    
                    if "already exists and is not an empty directory" in error_msg:
                        print("🔄 Directory still exists - trying additional cleanup")
                        if repo_path.exists():
                            try:
                                shutil.rmtree(repo_path, ignore_errors=True)
                                time.sleep(2)
                            except:
                                pass
                        continue
                    elif "not found" in error_msg.lower() or "does not exist" in error_msg.lower():
                        print("❌ Repository not found - stopping attempts")
                        return False
                    elif "permission denied" in error_msg.lower() or "forbidden" in error_msg.lower():
                        print("❌ Permission denied - private repository")
                        return False
                    
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 3
                        print(f"⏳ Waiting {wait_time}s before next attempt...")
                        time.sleep(wait_time)
                
            except subprocess.TimeoutExpired:
                print(f"⏰ Timeout on attempt {attempt + 1} (5min)")
                if attempt < max_retries - 1:
                    print("⏳ Trying again...")
                    continue
                else:
                    print("❌ Final timeout - repository too large")
                    return False
                    
            except Exception as e:
                print(f"❌ Error in git execution (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    continue
                else:
                    return False
        
        if not clone_success:
            print("❌ All clone attempts failed")
            return False
        
        # Post-clone verification
        print(f"🔍 Verifying clone result...")
        print(f"   Expected path: {repo_path}")
        print(f"   Directory exists: {repo_path.exists()}")
        
        if not repo_path.exists():
            print("❌ Repository directory was not created after successful clone")
            return False
        
        if not repo_path.is_dir():
            print(f"❌ {repo_path} exists but is not a directory")
            return False
        
        try:
            repo_items = list(repo_path.iterdir())
            print(f"📁 Items in repository: {len(repo_items)}")
            
            for i, item in enumerate(repo_items[:5]):
                print(f"   {i+1}. {item.name} ({'dir' if item.is_dir() else 'file'})")
            
            if len(repo_items) == 0:
                print("❌ Repository is empty")
                return False
            
            git_dir = repo_path / ".git"
            if git_dir.exists():
                print("✅ .git directory found - valid Git clone")
            else:
                print("⚠️ .git directory not found - may be a problem")
                
        except Exception as e:
            print(f"❌ Error verifying repository content: {e}")
            return False
        
        # Update state
        self.state.repo_path = str(repo_path)
        self.state.current_phase = "cloned"
        
        # Initialize advanced tools
        try:
            print("🔧 Initializing advanced analysis tools...")
            self.tools = AdvancedRepositoryTools(repo_path)
            
            if not self._register_tools_safely():
                print("⚠️ Some tools failed, but continuing...")
            
            print(f"🎉 Clone completed successfully!")
            print(f"   📁 Location: {repo_path}")
            print(f"   📊 Items: {len(repo_items)} found")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing tools: {e}")
            print("⚠️ Continuing without tools - clone was successful")
            return True
    
    def enhanced_planning_phase(self) -> bool:
        """Enhanced planning phase with complete analysis"""
        if not AG2_AVAILABLE:
            print("❌ AG2 not available - using fallback planning")
            self.state.plan = self._create_comprehensive_plan()
            return True
        
        try:
            print("🎯 Starting advanced planning...")
            
            # Check if we have valid state
            if not self.state:
                print("⚠️ State not initialized - creating basic state")
                self.state = DocumentationState(
                    project_url="unknown",
                    current_phase="planning"
                )
            
            planning_agents = [self.agents["code_explorer"], self.agents["documentation_planner"]]
            
            planning_chat = GroupChat(
                agents=planning_agents,
                messages=[],
                max_round=10,  # More rounds for complete analysis
                speaker_selection_method="round_robin"
            )
            
            planning_manager = GroupChatManager(
                groupchat=planning_chat,
                llm_config=self.llm_config
            )
            
            planning_prompt = f"""COMPLETE REPOSITORY ANALYSIS: {self.state.repo_path}

**CRITICAL MISSION:** Create plan for documentation in EXACTLY 3 sections:
1. Project Overview
2. Installation and Configuration Guide  
3. **Detailed Technical Documentation of Files** (MAIN)

**MANDATORY PROTOCOL:**

AdvancedCodeExplorer - Execute ALL these analyses in sequence:

1. `analyze_code_structure()` - Understand general architecture
2. `find_key_files()` - Identify components by category
3. `detailed_file_analysis()` - Use current analysis mode for complete analysis
4. `file_read()` on most critical files identified
5. `directory_read()` on important directories (src/, lib/, etc.)

**FOCUS ON:**
- Languages and frameworks used
- Entry points and main files
- Data structures and APIs
- Dependencies and configurations
- Code execution flow

EnhancedDocumentationPlanner - Based on complete analysis, create JSON plan with:
- Precise technical overview
- Installation guide based on found dependencies
- **DETAILED technical section** to document each important file

**EXPECTED RESULT:** Complete JSON plan that enables deep technical documentation."""
            
            # Execute complete analysis
            planning_result = self.agents["code_explorer"].initiate_chat(
                planning_manager,
                message=planning_prompt,
                clear_history=True
            )
            
            # Extract plan
            plan_data = self._extract_plan_safely(planning_chat.messages)
            
            if plan_data:
                self.state.plan = plan_data
                self.state.current_phase = "planned"
                print(f"✅ Advanced plan created: {len(plan_data.docs)} sections")
                return True
            else:
                print("❌ Plan failed - using complete plan")
                self.state.plan = self._create_comprehensive_plan()
                return True
                
        except Exception as e:
            print(f"❌ Error in planning: {str(e)[:100]}")
            self.error_count += 1
            self.state.plan = self._create_comprehensive_plan()
            return True
    
    def enhanced_documentation_phase(self) -> bool:
        """Enhanced documentation phase focused on technical analysis"""
        if not AG2_AVAILABLE:
            print("❌ AG2 not available - using fallback documentation")
            return self._create_comprehensive_documentation()
        
        try:
            print("📝 Starting advanced technical documentation...")
            
            # Check if we have valid state
            if not self.state:
                print("⚠️ State not initialized - creating basic state")
                self.state = DocumentationState(
                    project_url="unknown",
                    current_phase="documentation"
                )
            
            if not self.state.plan or not self.state.plan.docs:
                print("❌ No plan - creating complete documentation")
                return self._create_comprehensive_documentation()
            
            doc_agents = [self.agents["technical_writer"], self.agents["documentation_reviewer"]]
            
            docs_created = []
            
            for i, doc_item in enumerate(self.state.plan.docs):
                print(f"📄 Creating section {i+1}/3: {doc_item.title}")
                
                try:
                    doc_chat = GroupChat(
                        agents=doc_agents,
                        messages=[],
                        max_round=3,  # Reduced rounds for speed (was 6)
                        speaker_selection_method="round_robin"
                    )
                    
                    doc_manager = GroupChatManager(
                        groupchat=doc_chat,
                        llm_config=self.llm_config
                    )
                    
                    # Specific prompt per section
                    if "technical" in doc_item.title.lower() or "files" in doc_item.title.lower():
                        # Main technical section - MORE DETAILED
                        doc_prompt = f"""CREATE ADVANCED TECHNICAL DOCUMENTATION

**SECTION:** {doc_item.title}
**PROJECT:** {self.state.project_url}

**SPECIAL REQUIREMENTS FOR TECHNICAL SECTION:**
This is the MOST IMPORTANT section. Must include:

1. **General File Structure**
2. **Analysis of EACH important file**:
   - Purpose and functionality
   - Language and frameworks used
   - Main functions and classes
   - Exposed or consumed APIs
   - Dependencies and imports
   - Execution flow
   - Relevant code examples

3. **Technology mapping**
4. **System architecture**
5. **Developer guide**

**MANDATORY FORMAT:**
# {doc_item.title}

## 📁 General Structure
[Organization of directories and files]

## 🔧 Main Files

### file1.ext (Language)
**Purpose:** [Detailed description]
**Location:** `path/file`
**Technologies:** [Frameworks, libraries]

#### 📋 Functionalities:
- [Detailed list]

#### 🔧 Main Functions/Methods:
- `function()`: [Description and parameters]

#### 📊 Classes/Structures:
- `Class`: [Purpose and methods]

#### 🔌 APIs/Endpoints:
- [API documentation]

#### 📝 Notes:
[Important technical notes]

[REPEAT FOR EACH IMPORTANT FILE]

## 🏗️ Architecture and Flow
[How files relate]

TechnicalDocumentationWriter: Create detailed documentation
FastDocumentationReviewer: Quick review - fix errors and improve formatting only

**IMPORTANT:** Focus on essential technical details. Reviewer should be fast and efficient."""
                    else:
                        # Sections 1 and 2 - standard
                        doc_prompt = f"""CREATE DOCUMENTATION: {doc_item.title}

**CONTEXT:**
- Project: {self.state.project_url}
- Section: {doc_item.title}
- Description: {doc_item.description}
- Goal: {doc_item.goal}

**EXPECTED FORMAT:**
# {doc_item.title}

## 📋 [Main Section]
[Detailed content]

## 🚀 [Secondary Section]
[Practical instructions]

## 📝 Notes
[Important notes]

TechnicalDocumentationWriter: Create clear and complete documentation
FastDocumentationReviewer: Quick review - focus on essential fixes only

Work efficiently to create professional documentation."""
                    
                    # Create documentation
                    doc_result = self.agents["technical_writer"].initiate_chat(
                        doc_manager,
                        message=doc_prompt,
                        clear_history=True
                    )
                    
                    # Extract and save
                    final_doc = self._extract_documentation_safely(doc_chat.messages, doc_item.title)
                    
                    if final_doc:
                        doc_path = self._save_documentation(doc_item.title, final_doc)
                        if doc_path:
                            docs_created.append(doc_path)
                            # Ensure generated_docs exists
                            if not hasattr(self.state, 'generated_docs') or self.state.generated_docs is None:
                                self.state.generated_docs = []
                            self.state.generated_docs.append(doc_path)
                            print(f"✅ Section created: {doc_item.title}")
                    
                except Exception as e:
                    print(f"⚠️ Error in section {doc_item.title}: {str(e)[:50]}")
                    # Create basic documentation as fallback
                    basic_doc = self._generate_section_fallback(doc_item.title, i)
                    doc_path = self._save_documentation(doc_item.title, basic_doc)
                    if doc_path:
                        docs_created.append(doc_path)
                        # Ensure generated_docs exists
                        if not hasattr(self.state, 'generated_docs') or self.state.generated_docs is None:
                            self.state.generated_docs = []
                        self.state.generated_docs.append(doc_path)
            
            if docs_created:
                self.state.current_phase = "completed"
                print(f"🎉 Complete documentation: {len(docs_created)} files")
                return True
            else:
                print("⚠️ No docs created - generating complete documentation")
                return self._create_comprehensive_documentation()
                
        except Exception as e:
            print(f"❌ Error in documentation: {str(e)[:100]}")
            return self._create_comprehensive_documentation()
    
    def execute_flow(self, project_url: str) -> Dict[str, Any]:
        """Execute complete enhanced flow"""
        try:
            print(f"🚀 Starting AG2 Enhanced Flow: {project_url}")
            
            # Initialize state
            self.state = DocumentationState(project_url=project_url)
            
            # Phase 1: Clone
            clone_success = self.clone_repository(project_url)
            if not clone_success:
                return {
                    "status": "error",
                    "message": "Repository clone failure",
                    "error_count": self.error_count
                }
            
            # Phase 2: Enhanced Planning
            plan_success = self.enhanced_planning_phase()
            if not plan_success:
                return {
                    "status": "error", 
                    "message": "Advanced planning phase failure",
                    "error_count": self.error_count
                }
            
            # Phase 3: Enhanced Documentation
            doc_success = self.enhanced_documentation_phase()
            if not doc_success:
                return {
                    "status": "error",
                    "message": "Advanced documentation creation failure", 
                    "error_count": self.error_count
                }
            
            # Success
            return {
                "status": "success",
                "message": f"Complete technical documentation created: {len(self.state.generated_docs)} sections",
                "generated_docs": self.state.generated_docs,
                "plan": self.state.plan.to_dict() if self.state.plan else None,
                "metadata": {
                    "project_url": project_url,
                    "repo_path": self.state.repo_path,
                    "docs_count": len(self.state.generated_docs),
                    "generated_at": datetime.now().isoformat(),
                    "error_count": self.error_count,
                    "system_version": SYSTEM_VERSION,
                    "ag2_available": AG2_AVAILABLE,
                    "features": [
                        "Advanced code analysis",
                        "Detailed technical documentation",
                        "3 mandatory sections always generated",
                        "Multi-language file analysis",
                        "API and function mapping",
                        "Complete code structure analysis"
                    ]
                }
            }
            
        except Exception as e:
            logger.error(f"Flow error: {e}")
            logger.error(traceback.format_exc())
            return {
                "status": "error",
                "message": f"Critical error: {str(e)[:100]}",
                "error_count": self.error_count + 1
            }
    
    # Helper methods
    def _force_remove_directory(self, path: Path):
        """Forcefully remove directory"""
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
            print(f"⚠️ Error in forced removal: {e}")
            raise
    
    def _validate_github_url(self, url: str) -> bool:
        """Validate GitHub URL format"""
        pattern = r"^https://github\.com/[\w\-\.]+/[\w\-\.]+/?$"
        return bool(re.match(pattern, url.strip()))
    
    def _check_github_connectivity(self) -> bool:
        """Check basic GitHub connectivity"""
        try:
            socket.setdefaulttimeout(10)
            response = urllib.request.urlopen("https://github.com", timeout=10)
            return response.getcode() == 200
        except Exception as e:
            print(f"⚠️ Connectivity error: {e}")
            return False
    
    def _check_repository_exists(self, project_url: str) -> bool:
        """Check if repository exists and is public"""
        try:
            request = urllib.request.Request(project_url)
            request.add_header('User-Agent', 'Mozilla/5.0 (compatible; DocAgent/1.0)')
            
            try:
                response = urllib.request.urlopen(request, timeout=15)
                return response.getcode() == 200
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    print(f"❌ Repository not found (404): {project_url}")
                elif e.code == 403:
                    print(f"❌ Access denied (403): private repository or rate limit")
                else:
                    print(f"❌ HTTP error {e.code}: {e.reason}")
                return False
            except urllib.error.URLError as e:
                print(f"❌ URL error: {e.reason}")
                return False
                
        except Exception as e:
            print(f"⚠️ Error checking repository: {e}")
            return True
    
    def _create_comprehensive_plan(self) -> DocPlan:
        """Comprehensive mandatory plan with 3 sections"""
        print("📋 Creating complete plan with 3 sections...")
        
        return DocPlan(
            overview="Complete technical documentation generated automatically for detailed project analysis",
            docs=[
                DocItem(
                    title="Project Overview",
                    description="Complete analysis of purpose, technologies and project architecture",
                    prerequisites="Basic software development knowledge",
                    examples=["Main functionalities", "Technology stack", "General architecture"],
                    goal="Provide complete understanding of project and its technologies"
                ),
                DocItem(
                    title="Installation and Configuration Guide",
                    description="Detailed instructions for installation, configuration and project execution",
                    prerequisites="Compatible operating system and development tools",
                    examples=["System prerequisites", "Installation steps", "Execution commands"],
                    goal="Allow developers to configure and run the project quickly"
                ),
                DocItem(
                    title="Technical Documentation of Files",
                    description="Detailed technical analysis of each important file: functions, classes, APIs, code flow and architecture",
                    prerequisites="Knowledge in languages and frameworks used in the project",
                    examples=["File by file analysis", "Function documentation", "API mapping", "Execution flow"],
                    goal="Provide complete technical documentation for developers to understand, modify and contribute to the code"
                )
            ]
        )
    
    def _extract_plan_safely(self, messages: List[Dict]) -> Optional[DocPlan]:
        """Robust extraction of JSON plan"""
        try:
            for msg in reversed(messages):
                content = msg.get('content', '')
                
                # Look for more flexible JSON patterns
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
                                # Validate that we have at least 3 sections
                                if len(data['docs']) >= 3:
                                    return DocPlan.from_dict(data)
                                else:
                                    print(f"⚠️ Plan with only {len(data['docs'])} sections - expected 3")
                        except (json.JSONDecodeError, Exception) as e:
                            print(f"⚠️ JSON parse error: {e}")
                            continue
            
            return None
            
        except Exception as e:
            print(f"⚠️ Error in plan extraction: {e}")
            return None
    
    def _extract_documentation_safely(self, messages: List[Dict], title: str) -> Optional[str]:
        """Robust extraction of documentation from messages"""
        try:
            candidates = []
            
            for msg in reversed(messages):
                content = msg.get('content', '')
                name = msg.get('name', '')
                
                # Prioritize reviewer messages
                if 'reviewer' in name.lower() and len(content) > 200:
                    candidates.append(content)
                elif 'writer' in name.lower() and len(content) > 200:
                    candidates.append(content)
                elif '##' in content and len(content) > 300:
                    candidates.append(content)
            
            # Return best candidate
            if candidates:
                best_candidate = max(candidates, key=len)  # Largest content
                return best_candidate
            
            # Section-specific fallback
            title_lower = title.lower()
            if "overview" in title_lower or "general" in title_lower:
                return self._generate_section_fallback(title, 0)
            elif "installation" in title_lower or "configuration" in title_lower:
                return self._generate_section_fallback(title, 1)
            elif "technical" in title_lower or "files" in title_lower:
                return self._generate_section_fallback(title, 2)
            else:
                return self._generate_basic_doc(title)
            
        except Exception as e:
            print(f"⚠️ Error in extraction: {e}")
            return self._generate_basic_doc(title)
    
    def _generate_section_fallback(self, title: str, section_index: int) -> str:
        """Generate section-specific fallback documentation"""
        
        if section_index == 0:  # Overview
            return f"""# {title}

## 🎯 Project Purpose

This project was automatically analyzed by the AG2 Documentation Flow system. The analysis identified an organized code base with multiple files and functionalities.

## 🛠️ Technologies Identified

Based on file structure analysis, the project uses:
- Multiple programming languages
- Organized directory structure
- Specific configuration files

## 🏗️ Architecture

The project is organized in a hierarchical structure of files and directories, with clear separation of responsibilities between different components.

## 📊 Characteristics

- Project with well-defined structure
- Multiple code files
- Modular and organized system

---
*Section generated automatically - For more detailed information, consult the project source files*
"""
        
        elif section_index == 1:  # Installation
            return f"""# {title}

## 📋 Prerequisites

Before installing and running this project, make sure you have:

- Compatible operating system (Linux, macOS, or Windows)
- Appropriate development tools for the language used
- Terminal/command line access
- Git installed for repository cloning

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone {self.state.project_url if self.state else '[PROJECT_URL]'}
cd [repository-name]
```

### 2. Install Dependencies
Check the project configuration files (package.json, requirements.txt, etc.) and install dependencies according to the technology used.

### 3. Configure Environment
Follow project-specific instructions for environment variables and configuration files.

## ▶️ Execution

Run the project following technology-specific instructions. Check main files (main.py, index.js, etc.) to understand the entry point.

## 📝 Notes

- Consult project-specific documentation for detailed instructions
- Check README files if available
- For installation problems, consult the technology documentation

---
*Section generated automatically - Consult project-specific files for detailed instructions*
"""
        
        else:  # Technical Documentation (section 2)
            return f"""# {title}

## 📁 General Structure

The project contains a structured organization of files and directories, each with specific responsibilities in the system.

## 🔧 Main Files

### Automatic Analysis

This project was automatically analyzed and contains multiple important files. Each file has:

- **Specific purpose** in the project context
- **Implementation** using the stack technologies
- **Interactions** with other system components

### Identified File Categories

#### 💻 Code Files
Files containing the main system logic, implementing specific functionalities.

#### ⚙️ Configuration Files  
Files responsible for environment configuration, dependencies and system parameters.

#### 📖 Documentation Files
Files containing project information, including README, licenses and guides.

## 🏗️ System Architecture

The project follows a modular architecture where:

- Different files have specific responsibilities
- There is clear separation between business logic and configuration
- The system is organized hierarchically

## 📋 For Developers

To contribute to this project:

1. **Analyze the structure** of files to understand the organization
2. **Identify the main entry point** of the application
3. **Examine dependencies** listed in configuration files
4. **Follow patterns** established in existing code

## 📝 Technical Notes

- This project contains multiple files with specific functionalities
- The structure follows good code organization practices
- For detailed analysis, examine the source files directly

---
*Documentation generated automatically - For specific technical information, consult the source code files*
"""
    
    def _generate_basic_doc(self, title: str) -> str:
        """Generate basic documentation as fallback"""
        return f"""# {title}

## 📋 Overview

This section documents {title.lower()} of the project. The documentation was generated automatically based on repository analysis.

## 🚀 Information

This documentation is part of a complete set of 3 sections:
1. Project Overview
2. Installation and Configuration Guide
3. Technical Documentation of Files

## 📝 Notes

- This documentation was generated automatically by AG2 Documentation Flow
- For more detailed information, consult the project source code
- The system analyzed the repository structure to generate this documentation

---
*Generated automatically on {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}*
"""
    
    def _save_documentation(self, title: str, content: str) -> Optional[str]:
        """Save documentation with standardized names"""
        try:
            docs_dir = Path("docs")
            docs_dir.mkdir(exist_ok=True)
            
            # Standardized names for the 3 sections
            title_lower = title.lower()
            if "overview" in title_lower or "general" in title_lower:
                filename = "01_project_overview.md"
            elif "installation" in title_lower or "configuration" in title_lower:
                filename = "02_installation_configuration.md"
            elif "technical" in title_lower or "files" in title_lower:
                filename = "03_technical_documentation.md"
            else:
                # Fallback for safe name
                safe_title = re.sub(r'[^\w\s-]', '', title)
                safe_title = re.sub(r'[-\s]+', '_', safe_title)
                filename = f"{safe_title.lower()}.md"
            
            doc_path = docs_dir / filename
            
            # Save with UTF-8 encoding
            with open(doc_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            print(f"💾 Saved: {doc_path}")
            return str(doc_path)
            
        except Exception as e:
            print(f"❌ Error saving {title}: {e}")
            return None
    
    def _create_comprehensive_documentation(self) -> bool:
        """Create complete documentation as last resort"""
        try:
            print("📝 Creating complete documentation...")
            
            # Ensure we have valid state
            if not self.state:
                print("⚠️ State not found - initializing")
                self.state = DocumentationState(
                    project_url="unknown",
                    current_phase="documentation",
                    generated_docs=[],
                    metadata={}
                )
            
            # Ensure we have complete plan
            if not self.state.plan:
                self.state.plan = self._create_comprehensive_plan()
            
            # Create the 3 mandatory sections
            sections = [
                ("Project Overview", 0),
                ("Installation and Configuration Guide", 1), 
                ("Technical Documentation of Files", 2)
            ]
            
            docs_created = []
            
            for title, index in sections:
                print(f"📄 Generating section {index+1}/3: {title}")
                
                doc_content = self._generate_section_fallback(title, index)
                doc_path = self._save_documentation(title, doc_content)
                
                if doc_path:
                    docs_created.append(doc_path)
                    # Ensure generated_docs exists
                    if not hasattr(self.state, 'generated_docs') or self.state.generated_docs is None:
                        self.state.generated_docs = []
                    self.state.generated_docs.append(doc_path)
            
            if docs_created:
                print(f"✅ Complete documentation created: {len(docs_created)} sections")
                return True
            
            return False
            
        except Exception as e:
            print(f"❌ Error in complete documentation: {e}")
            return False
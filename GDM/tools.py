"""
Advanced Repository Analysis Tools for AG2 Enhanced Documentation Flow
=======================================================================

Provides comprehensive tools for analyzing repository structure, code, and files.
"""

import os
import re
import json
import time
from typing import Union, Tuple, Dict, List, Any
from pathlib import Path
from config import SUPPORTED_LANGUAGES, KEY_FILE_PATTERNS


class AdvancedRepositoryTools:
    """Advanced tools for comprehensive repository analysis"""
    
    def __init__(self, repo_path: Union[str, Path]):
        self.repo_path = Path(repo_path)
        self.file_cache = {}
        self.error_count = 0
        self.analysis_cache = {}
        print(f"🔧 Initializing advanced tools for: {self.repo_path}")
    
    def _safe_execute(self, func_name: str, operation):
        """Safe execution with error handling"""
        try:
            return operation()
        except PermissionError:
            self.error_count += 1
            return f"❌ Permission denied in {func_name}"
        except FileNotFoundError:
            self.error_count += 1
            return f"❌ File/directory not found in {func_name}"
        except UnicodeDecodeError:
            self.error_count += 1
            return f"❌ Encoding error in {func_name}"
        except Exception as e:
            self.error_count += 1
            return f"❌ Error in {func_name}: {str(e)[:100]}"
    
    def directory_read(self, path: str = "") -> str:
        """List directory contents with detailed analysis"""
        def _operation():
            target_path = self.repo_path / path if path else self.repo_path
            
            if not target_path.exists():
                return f"❌ Directory not found: {target_path}"
            
            if not target_path.is_dir():
                return f"❌ Not a directory: {target_path}"
            
            result = f"## 📁 Detailed Structure: {target_path.name if path else 'root'}\n\n"
            
            try:
                items = list(target_path.iterdir())
            except PermissionError:
                return f"❌ No permission to read: {target_path}"
            
            if not items:
                return result + "📂 Empty directory\n"
            
            # Classify and analyze items
            dirs = []
            code_files = []
            config_files = []
            doc_files = []
            other_files = []
            
            for item in items[:150]:  # Increased limit
                try:
                    if item.name.startswith('.'):
                        continue
                    
                    if item.is_dir():
                        # Count files in subdirectory
                        try:
                            sub_items = len(list(item.iterdir()))
                            dirs.append(f"📁 {item.name}/ ({sub_items} items)")
                        except:
                            dirs.append(f"📁 {item.name}/")
                    else:
                        size = item.stat().st_size
                        size_str = self._format_size(size)
                        ext = item.suffix.lower()
                        
                        # Classify by type
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
            
            # Display organized result by category
            if dirs:
                result += "### 📁 Directories:\n" + "\n".join(sorted(dirs)[:15]) + "\n\n"
            
            if code_files:
                result += "### 💻 Code Files:\n" + "\n".join(sorted(code_files)[:20]) + "\n\n"
            
            if config_files:
                result += "### ⚙️ Configuration Files:\n" + "\n".join(sorted(config_files)[:10]) + "\n\n"
            
            if doc_files:
                result += "### 📖 Documentation:\n" + "\n".join(sorted(doc_files)[:10]) + "\n\n"
            
            if other_files:
                result += "### 📄 Other Files:\n" + "\n".join(sorted(other_files)[:15]) + "\n\n"
            
            total_shown = len(dirs) + len(code_files) + len(config_files) + len(doc_files) + len(other_files)
            if len(items) > total_shown:
                result += f"... and {len(items) - total_shown} more items\n"
            
            return result
        
        return self._safe_execute("directory_read", _operation)
    
    def file_read(self, file_path: str) -> str:
        """Read files with intelligent content analysis"""
        def _operation():
            target_file = self.repo_path / file_path
            
            if not target_file.exists():
                return f"❌ File not found: {file_path}"
            
            if not target_file.is_file():
                return f"❌ Not a file: {file_path}"
            
            # Cache check
            cache_key = str(target_file)
            if cache_key in self.file_cache:
                return self.file_cache[cache_key]
            
            try:
                file_size = target_file.stat().st_size
                if file_size > 300 * 1024:  # 300KB max
                    return f"❌ File too large: {file_path} ({self._format_size(file_size)})"
                
                if file_size == 0:
                    return f"📄 Empty file: {file_path}"
            
            except OSError:
                return f"❌ Error accessing: {file_path}"
            
            # Try multiple encodings
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
                return f"❌ Could not read file: {file_path}"
            
            # Check if binary file
            if '\x00' in content[:1000]:
                return f"❌ Binary file detected: {file_path}"
            
            # Content analysis
            lines = content.count('\n') + 1
            ext = target_file.suffix.lower()
            language = self._get_language(ext)
            
            # Language-specific analysis
            analysis = self._analyze_code_content(content, language)
            
            result = f"""## 📄 File: {file_path}

### 📊 Information:
- **Size:** {self._format_size(file_size)}
- **Lines:** {lines}
- **Language:** {language}
- **Encoding:** {used_encoding}

### 🔍 Code Analysis:
{analysis}

### 💻 Content:
```{ext[1:] if ext else 'text'}
{content[:4000]}{'...\n[TRUNCATED - File too long]' if len(content) > 4000 else ''}
```
"""
            
            # Cache result (limited)
            if len(self.file_cache) < 30:
                self.file_cache[cache_key] = result
            
            return result
        
        return self._safe_execute("file_read", _operation)
    
    def analyze_code_structure(self) -> str:
        """Advanced analysis of project code structure"""
        def _operation():
            result = "## 🏗️ Detailed Code Structure Analysis\n\n"
            
            # Statistics by language
            language_stats = {}
            function_count = 0
            class_count = 0
            total_loc = 0
            
            # Important files analyzed
            important_files = []
            
            try:
                for root, dirs, files in os.walk(self.repo_path):
                    # Filter irrelevant directories
                    dirs[:] = [d for d in dirs if not d.startswith('.') 
                              and d not in ['node_modules', '__pycache__', 'target', 'build', 'dist', 'vendor']]
                    
                    for file in files:
                        if file.startswith('.'):
                            continue
                        
                        file_path = Path(root) / file
                        relative_path = file_path.relative_to(self.repo_path)
                        ext = file_path.suffix.lower()
                        
                        # Focus on code files
                        if ext not in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.go', '.rs', '.php', '.rb']:
                            continue
                        
                        try:
                            if file_path.stat().st_size > 500 * 1024:  # 500KB max
                                continue
                            
                            content = file_path.read_text(encoding='utf-8', errors='ignore')
                            lines = len([l for l in content.split('\n') if l.strip()])
                            
                            language = self._get_language(ext)
                            
                            # Statistics by language
                            if language not in language_stats:
                                language_stats[language] = {'files': 0, 'lines': 0, 'functions': 0, 'classes': 0}
                            
                            language_stats[language]['files'] += 1
                            language_stats[language]['lines'] += lines
                            total_loc += lines
                            
                            # Function and class analysis
                            funcs, classes = self._count_functions_classes(content, language)
                            language_stats[language]['functions'] += funcs
                            language_stats[language]['classes'] += classes
                            function_count += funcs
                            class_count += classes
                            
                            # Important files (>50 lines or specific names)
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
                    
                    # Limit search for very large projects
                    if len(important_files) > 50:
                        break
            
            except Exception as e:
                result += f"⚠️ Error in analysis: {str(e)[:100]}\n\n"
            
            # General summary
            result += f"### 📊 General Summary:\n"
            result += f"- **Total lines of code:** {total_loc:,}\n"
            result += f"- **Functions identified:** {function_count}\n"
            result += f"- **Classes identified:** {class_count}\n"
            result += f"- **Languages detected:** {len(language_stats)}\n\n"
            
            # Statistics by language
            if language_stats:
                result += "### 💻 Statistics by Language:\n\n"
                for lang, stats in sorted(language_stats.items(), key=lambda x: x[1]['lines'], reverse=True):
                    result += f"**{lang}:**\n"
                    result += f"- Files: {stats['files']}\n"
                    result += f"- Lines: {stats['lines']:,}\n"
                    result += f"- Functions: {stats['functions']}\n"
                    result += f"- Classes: {stats['classes']}\n\n"
            
            # Important files
            if important_files:
                result += "### 🎯 Important Files Identified:\n\n"
                for file_info in sorted(important_files, key=lambda x: x['lines'], reverse=True)[:15]:
                    result += f"**{file_info['path']}** ({file_info['language']})\n"
                    result += f"- {file_info['lines']} lines\n"
                    if file_info['functions'] > 0:
                        result += f"- {file_info['functions']} functions\n"
                    if file_info['classes'] > 0:
                        result += f"- {file_info['classes']} classes\n"
                    result += "\n"
            
            return result
        
        return self._safe_execute("analyze_code_structure", _operation)
    
    def find_key_files(self) -> str:
        """Find important files with detailed categorization"""
        def _operation():
            result = "## 🔍 Key Files Identified\n\n"
            
            found_files = {}
            search_count = 0
            
            try:
                for root, dirs, files in os.walk(self.repo_path):
                    search_count += 1
                    if search_count > 2000:  # Expanded limit
                        break
                    
                    # Filter directories
                    dirs[:] = [d for d in dirs if not d.startswith('.') 
                              and d not in ['node_modules', '__pycache__', 'target', 'build', 'dist']]
                    
                    current_dir = Path(root)
                    relative_dir = current_dir.relative_to(self.repo_path)
                    
                    for file in files:
                        if file.startswith('.'):
                            continue
                        
                        file_path = current_dir / file
                        relative_path = file_path.relative_to(self.repo_path)
                        
                        # Check patterns
                        for category, patterns in KEY_FILE_PATTERNS.items():
                            for pattern in patterns:
                                if (pattern.endswith('/') and pattern[:-1] in str(relative_dir)) or \
                                   (pattern in file.lower()) or \
                                   (pattern.lower() == file.lower()) or \
                                   (file.lower().startswith(pattern.lower())):
                                    
                                    if category not in found_files:
                                        found_files[category] = []
                                    
                                    if len(found_files[category]) < 12:  # More files per category
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
                result += f"⚠️ Search limited due to error: {str(e)[:50]}\n\n"
            
            # Format detailed results
            if found_files:
                for category, files in found_files.items():
                    if files:
                        result += f"### {category}\n"
                        for file_info in files:
                            result += f"- **{file_info['path']}** "
                            result += f"({file_info['size']}, {file_info['type']})\n"
                        result += "\n"
            else:
                result += "📂 No obvious key files identified\n"
                # Improved fallback
                try:
                    first_files = list(self.repo_path.glob("*"))[:15]
                    if first_files:
                        result += "\n**First files found:**\n"
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
    
    def detailed_file_analysis(self, analysis_mode: str = "all") -> str:
        """Detailed analysis of files based on analysis mode"""
        def _operation():
            result = "## 🔬 Detailed File Analysis\n\n"
            
            # Identify files for detailed analysis
            analysis_targets = []
            
            # Important file patterns
            important_patterns = [
                'main.py', 'app.py', 'server.py', 'index.js', 'main.go',
                'README.md', 'setup.py', 'package.json', 'requirements.txt'
            ]
            
            try:
                # Search for files based on analysis mode
                for root, dirs, files in os.walk(self.repo_path):
                    dirs[:] = [d for d in dirs if not d.startswith('.') 
                              and d not in ['node_modules', '__pycache__', 'target', 'build', 'dist']]
                    
                    for file in files:
                        if file.startswith('.'):
                            continue
                        
                        file_path = Path(root) / file
                        relative_path = file_path.relative_to(self.repo_path)
                        
                        # Determine if file should be analyzed based on mode
                        should_analyze = False
                        priority = 0
                        
                        if analysis_mode == "all":
                            # Analyze ALL code files
                            if file_path.suffix.lower() in ['.py', '.js', '.ts', '.java', '.go', '.cpp', '.c', '.h', '.rs', '.php', '.rb', '.scala', '.kt']:
                                should_analyze = True
                                priority = 1
                            # High priority for specific files
                            if any(pattern in file.lower() for pattern in important_patterns):
                                priority = 10
                                
                        elif analysis_mode == "smart":
                            # Smart selection: important files + significant code files
                            if any(pattern in file.lower() for pattern in important_patterns):
                                should_analyze = True
                                priority = 10
                            elif file_path.suffix.lower() in ['.py', '.js', '.ts', '.java', '.go']:
                                try:
                                    if file_path.stat().st_size > 2000:  # > 2KB
                                        should_analyze = True
                                        priority = 5
                                except:
                                    pass
                                    
                        else:  # limited mode
                            # Limited to most important files only
                            if any(pattern in file.lower() for pattern in important_patterns):
                                should_analyze = True
                                priority = 10
                            elif file_path.suffix.lower() in ['.py', '.js', '.ts', '.java', '.go']:
                                try:
                                    if file_path.stat().st_size > 5000:  # > 5KB
                                        should_analyze = True
                                        priority = 3
                                except:
                                    pass
                        
                        if should_analyze:
                            # Apply limits based on mode
                            max_limit = float('inf') if analysis_mode == "all" else 50 if analysis_mode == "smart" else 20
                            if len(analysis_targets) < max_limit:
                                analysis_targets.append({
                                    'path': file_path,
                                    'relative_path': relative_path,
                                    'priority': priority
                                })
                
                # Sort by priority and size
                analysis_targets.sort(key=lambda x: (-x['priority'], -x['path'].stat().st_size if x['path'].exists() else 0))
                
                # Apply final limits for non-all modes
                if analysis_mode == "smart":
                    analysis_targets = analysis_targets[:50]
                elif analysis_mode == "limited":
                    analysis_targets = analysis_targets[:20]
                
            except Exception as e:
                result += f"⚠️ Error in file identification: {str(e)[:100]}\n\n"
                return result
            
            if not analysis_targets:
                result += "❌ No files identified for detailed analysis\n"
                return result
            
            mode_description = {
                "all": "ALL code files",
                "smart": "smart selection of important files", 
                "limited": "limited set of main files"
            }
            
            result += f"Analyzing {len(analysis_targets)} files ({mode_description.get(analysis_mode, 'selected files')}):\n\n"
            
            # Analyze each file
            for i, target in enumerate(analysis_targets, 1):
                try:
                    file_path = target['path']
                    relative_path = target['relative_path']
                    
                    if not file_path.exists():
                        continue
                    
                    result += f"### {i}. 📄 {relative_path}\n\n"
                    
                    # Basic information
                    size = file_path.stat().st_size
                    ext = file_path.suffix.lower()
                    language = self._get_language(ext)
                    
                    result += f"**Information:**\n"
                    result += f"- Size: {self._format_size(size)}\n"
                    result += f"- Language: {language}\n"
                    
                    # Read and analyze content
                    if size > 100 * 1024:  # 100KB
                        result += f"- Status: File too large for complete analysis\n\n"
                        continue
                    
                    try:
                        content = file_path.read_text(encoding='utf-8', errors='ignore')
                        lines = len([l for l in content.split('\n') if l.strip()])
                        
                        result += f"- Lines of code: {lines}\n"
                        
                        # Content-specific analysis
                        code_analysis = self._analyze_code_content(content, language)
                        result += f"- Analysis: {code_analysis}\n\n"
                        
                        # Show relevant snippet
                        if language != "Text" and lines > 5:
                            snippet = self._extract_relevant_snippet(content, language)
                            if snippet:
                                result += f"**Relevant snippet:**\n```{ext[1:] if ext else 'text'}\n{snippet}\n```\n\n"
                        
                    except (UnicodeDecodeError, PermissionError):
                        result += f"- Status: Error reading file\n\n"
                        continue
                    
                except Exception as e:
                    result += f"⚠️ Error in analysis of {target['relative_path']}: {str(e)[:50]}\n\n"
                    continue
            
            return result
        
        return self._safe_execute("detailed_file_analysis", _operation)
    
    def _analyze_code_content(self, content: str, language: str) -> str:
        """Language-specific content analysis"""
        if language == "Text":
            return "Text/documentation file"
        
        analysis = []
        
        try:
            lines = content.split('\n')
            code_lines = [l for l in lines if l.strip() and not l.strip().startswith('#') and not l.strip().startswith('//')]
            
            if language == "Python":
                # Python analysis
                imports = [l for l in lines if l.strip().startswith('import ') or l.strip().startswith('from ')]
                functions = len([l for l in lines if l.strip().startswith('def ')])
                classes = len([l for l in lines if l.strip().startswith('class ')])
                
                if imports:
                    main_imports = [imp.split()[1].split('.')[0] for imp in imports[:5] if len(imp.split()) > 1]
                    analysis.append(f"Main imports: {', '.join(main_imports[:3])}")
                
                if functions > 0:
                    analysis.append(f"{functions} functions")
                if classes > 0:
                    analysis.append(f"{classes} classes")
                    
                # Detect frameworks
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
                # JavaScript analysis
                functions = len(re.findall(r'function\s+\w+', content))
                arrow_functions = len(re.findall(r'\w+\s*=>\s*', content))
                const_vars = len([l for l in lines if l.strip().startswith('const ')])
                
                if functions > 0:
                    analysis.append(f"{functions} declared functions")
                if arrow_functions > 0:
                    analysis.append(f"{arrow_functions} arrow functions")
                if const_vars > 0:
                    analysis.append(f"{const_vars} constants")
                    
                # Detect frameworks/libraries
                if 'react' in content.lower():
                    analysis.append("React")
                if 'vue' in content.lower():
                    analysis.append("Vue.js")
                if 'angular' in content.lower():
                    analysis.append("Angular")
                if 'node' in content.lower():
                    analysis.append("Node.js")
                    
            elif language == "JSON":
                # JSON analysis
                try:
                    data = json.loads(content)
                    if isinstance(data, dict):
                        keys = list(data.keys())[:5]
                        analysis.append(f"Main keys: {', '.join(keys)}")
                except:
                    analysis.append("JSON with possible syntax error")
                    
            elif language in ["Java", "C++", "Go"]:
                # Analysis for compiled languages
                classes = len(re.findall(r'class\s+\w+', content))
                methods = len(re.findall(r'(public|private|protected).*?\w+\s*\(', content))
                
                if classes > 0:
                    analysis.append(f"{classes} classes")
                if methods > 0:
                    analysis.append(f"{methods} methods")
            
            # General analysis
            if len(code_lines) > 100:
                analysis.append("Extensive file")
            elif len(code_lines) < 20:
                analysis.append("Small file")
                
        except Exception:
            analysis.append("Limited analysis due to complex format")
        
        return "; ".join(analysis) if analysis else "Standard code"
    
    def _count_functions_classes(self, content: str, language: str) -> Tuple[int, int]:
        """Count functions and classes in code"""
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
        """Extract relevant code snippet"""
        lines = content.split('\n')
        
        # Look for interesting snippets
        if language == "Python":
            # Look for main, classes or important functions
            for i, line in enumerate(lines):
                if ('if __name__' in line or 
                    line.strip().startswith('class ') or 
                    line.strip().startswith('def main')):
                    return '\n'.join(lines[i:i+max_lines])
        
        elif language == "JavaScript":
            # Look for exports, main functions
            for i, line in enumerate(lines):
                if ('export' in line or 
                    'function main' in line or
                    'module.exports' in line):
                    return '\n'.join(lines[i:i+max_lines])
        
        # Fallback: first non-empty lines
        non_empty_lines = [l for l in lines if l.strip()]
        if non_empty_lines:
            return '\n'.join(non_empty_lines[:max_lines])
        
        return ""
    
    def _format_size(self, size: int) -> str:
        """Format file size"""
        if size < 1024:
            return f"{size}B"
        elif size < 1024*1024:
            return f"{size//1024}KB"
        else:
            return f"{size//(1024*1024)}MB"
    
    def _get_language(self, ext: str) -> str:
        """Identify language by extension"""
        return SUPPORTED_LANGUAGES.get(ext, 'Unknown')
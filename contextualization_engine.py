"""
Contextualization Engine for Coding Agent
EXPERIMENTAL: Creates rich context from web, user, and local code.

Toggle with: experimental.contextualization = True
"""
import os
import re
import json
import hashlib
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

try:
    import aiohttp
except ImportError:
    aiohttp = None


# Cache settings
CONTEXT_CACHE_PATH = Path(".cache/context")
CONTEXT_CACHE_TTL = timedelta(hours=24)


@dataclass
class CodeContext:
    """Context extracted from code."""
    file_path: str
    language: str = ""
    imports: List[str] = field(default_factory=list)
    classes: List[Dict[str, Any]] = field(default_factory=list)
    functions: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)
    summary: str = ""
    complexity_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WebContext:
    """Context gathered from the web."""
    query: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    documentation: List[Dict[str, Any]] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    related_issues: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UserContext:
    """Context from user interactions."""
    preferences: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    common_patterns: List[str] = field(default_factory=list)
    coding_style: Dict[str, Any] = field(default_factory=dict)
    known_issues: List[str] = field(default_factory=list)
    project_goals: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProjectContext:
    """Full project context."""
    project_name: str = ""
    project_type: str = ""
    languages: List[str] = field(default_factory=list)
    frameworks: List[str] = field(default_factory=list)
    structure: Dict[str, Any] = field(default_factory=dict)
    entry_points: List[str] = field(default_factory=list)
    test_framework: str = ""
    build_system: str = ""
    config_files: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CodeAnalyzer:
    """Analyzes code to extract context."""

    LANGUAGE_PATTERNS = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".java": "java",
        ".kt": "kotlin",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".cpp": "cpp",
        ".c": "c",
    }

    IMPORT_PATTERNS = {
        "python": [
            r'^import\s+([\w.]+)',
            r'^from\s+([\w.]+)\s+import',
        ],
        "javascript": [
            r'import\s+.*\s+from\s+[\'"]([^\'"]+)[\'"]',
            r'require\([\'"]([^\'"]+)[\'"]\)',
        ],
        "typescript": [
            r'import\s+.*\s+from\s+[\'"]([^\'"]+)[\'"]',
        ],
        "go": [
            r'import\s+[\'"]([^\'"]+)[\'"]',
            r'import\s+\(\s*[\'"]([^\'"]+)[\'"]',
        ],
        "java": [
            r'^import\s+([\w.]+);',
        ],
        "rust": [
            r'^use\s+([\w:]+)',
        ],
    }

    CLASS_PATTERNS = {
        "python": r'class\s+(\w+)(?:\([^)]*\))?:',
        "javascript": r'class\s+(\w+)(?:\s+extends\s+\w+)?\s*\{',
        "typescript": r'class\s+(\w+)(?:\s+extends\s+\w+)?(?:\s+implements\s+[\w,\s]+)?\s*\{',
        "java": r'(?:public\s+)?class\s+(\w+)',
        "go": r'type\s+(\w+)\s+struct\s*\{',
        "rust": r'(?:pub\s+)?struct\s+(\w+)',
    }

    FUNCTION_PATTERNS = {
        "python": r'def\s+(\w+)\s*\([^)]*\)',
        "javascript": r'(?:function\s+(\w+)|(\w+)\s*=\s*(?:async\s+)?(?:function|\([^)]*\)\s*=>))',
        "typescript": r'(?:function\s+(\w+)|(\w+)\s*=\s*(?:async\s+)?(?:function|\([^)]*\)\s*=>))',
        "java": r'(?:public|private|protected)?\s*(?:static\s+)?[\w<>]+\s+(\w+)\s*\([^)]*\)',
        "go": r'func\s+(?:\([^)]+\)\s+)?(\w+)\s*\(',
        "rust": r'(?:pub\s+)?fn\s+(\w+)',
    }

    @classmethod
    def detect_language(cls, file_path: Path) -> str:
        """Detect programming language from file extension."""
        return cls.LANGUAGE_PATTERNS.get(file_path.suffix.lower(), "unknown")

    @classmethod
    def extract_imports(cls, code: str, language: str) -> List[str]:
        """Extract import statements from code."""
        imports = []
        patterns = cls.IMPORT_PATTERNS.get(language, [])

        for pattern in patterns:
            matches = re.findall(pattern, code, re.MULTILINE)
            imports.extend(matches)

        return list(set(imports))

    @classmethod
    def extract_classes(cls, code: str, language: str) -> List[Dict[str, Any]]:
        """Extract class definitions from code."""
        classes = []
        pattern = cls.CLASS_PATTERNS.get(language)

        if pattern:
            for match in re.finditer(pattern, code):
                line_num = code[:match.start()].count('\n') + 1
                classes.append({
                    "name": match.group(1),
                    "line": line_num,
                })

        return classes

    @classmethod
    def extract_functions(cls, code: str, language: str) -> List[Dict[str, Any]]:
        """Extract function definitions from code."""
        functions = []
        pattern = cls.FUNCTION_PATTERNS.get(language)

        if pattern:
            for match in re.finditer(pattern, code):
                name = match.group(1) or (match.group(2) if match.lastindex >= 2 else None)
                if name:
                    line_num = code[:match.start()].count('\n') + 1
                    functions.append({
                        "name": name,
                        "line": line_num,
                    })

        return functions

    @classmethod
    def calculate_complexity(cls, code: str, language: str) -> float:
        """Calculate a simple complexity score for the code."""
        lines = len(code.splitlines())
        if lines == 0:
            return 0.0

        # Count complexity indicators
        complexity_patterns = [
            r'\bif\b', r'\belse\b', r'\bfor\b', r'\bwhile\b',
            r'\btry\b', r'\bcatch\b', r'\bexcept\b',
            r'\blambda\b', r'\bawait\b', r'\basync\b',
        ]

        complexity_count = 0
        for pattern in complexity_patterns:
            complexity_count += len(re.findall(pattern, code))

        # Normalize by lines of code
        score = min(complexity_count / (lines / 10), 10.0)
        return round(score, 2)

    @classmethod
    def analyze_file(cls, file_path: Path) -> Optional[CodeContext]:
        """Analyze a single code file."""
        if not file_path.exists():
            return None

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                code = f.read()
        except Exception:
            return None

        language = cls.detect_language(file_path)

        return CodeContext(
            file_path=str(file_path),
            language=language,
            imports=cls.extract_imports(code, language),
            classes=cls.extract_classes(code, language),
            functions=cls.extract_functions(code, language),
            complexity_score=cls.calculate_complexity(code, language),
        )


class ProjectAnalyzer:
    """Analyzes project structure and context."""

    CONFIG_FILES = {
        "package.json": ("nodejs", ["npm", "yarn"]),
        "requirements.txt": ("python", ["pip"]),
        "Pipfile": ("python", ["pipenv"]),
        "pyproject.toml": ("python", ["poetry", "pip"]),
        "setup.py": ("python", ["pip"]),
        "Cargo.toml": ("rust", ["cargo"]),
        "go.mod": ("go", ["go"]),
        "pom.xml": ("java", ["maven"]),
        "build.gradle": ("java", ["gradle"]),
        "Gemfile": ("ruby", ["bundler"]),
        "composer.json": ("php", ["composer"]),
    }

    FRAMEWORK_PATTERNS = {
        "react": [r'"react":', r"from 'react'", r'from "react"'],
        "vue": [r'"vue":', r"from 'vue'"],
        "angular": [r'"@angular/core":'],
        "django": [r"django", r"from django"],
        "flask": [r"from flask import", r"Flask\("],
        "fastapi": [r"from fastapi import", r"FastAPI\("],
        "express": [r'"express":', r"require('express')"],
        "spring": [r"org.springframework"],
        "rails": [r"class.*<.*ApplicationController"],
    }

    TEST_FRAMEWORKS = {
        "pytest": ["pytest.ini", "conftest.py"],
        "jest": ["jest.config.js", "jest.config.ts"],
        "mocha": [".mocharc.js", ".mocharc.json"],
        "junit": ["src/test/java"],
        "rspec": ["spec/"],
        "go_test": ["_test.go"],
    }

    @classmethod
    def analyze_project(cls, root_dir: Path = None) -> ProjectContext:
        """Analyze project structure."""
        root_dir = root_dir or Path.cwd()
        context = ProjectContext()

        # Detect project name
        context.project_name = root_dir.name

        # Find config files
        config_files = []
        for config_file, (lang, _) in cls.CONFIG_FILES.items():
            if (root_dir / config_file).exists():
                config_files.append(config_file)
                if lang not in context.languages:
                    context.languages.append(lang)

        context.config_files = config_files

        # Detect build system
        for config_file, (_, build_systems) in cls.CONFIG_FILES.items():
            if config_file in config_files:
                context.build_system = build_systems[0]
                break

        # Analyze file structure
        structure = {"directories": [], "file_counts": defaultdict(int)}

        for item in root_dir.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                structure["directories"].append(item.name)
            elif item.is_file():
                ext = item.suffix.lower()
                structure["file_counts"][ext] += 1

        context.structure = dict(structure)

        # Detect frameworks from code
        cls._detect_frameworks(root_dir, context)

        # Detect test framework
        for framework, indicators in cls.TEST_FRAMEWORKS.items():
            for indicator in indicators:
                if (root_dir / indicator).exists():
                    context.test_framework = framework
                    break

        # Find entry points
        entry_points = []
        for pattern in ["main.py", "index.js", "index.ts", "main.go", "Main.java", "main.rs"]:
            matches = list(root_dir.glob(f"**/{pattern}"))
            entry_points.extend([str(m.relative_to(root_dir)) for m in matches[:3]])

        context.entry_points = entry_points

        return context

    @classmethod
    def _detect_frameworks(cls, root_dir: Path, context: ProjectContext):
        """Detect frameworks used in the project."""
        # Check common config files for framework hints
        for config_file in context.config_files:
            config_path = root_dir / config_file
            if config_path.exists():
                try:
                    with open(config_path, "r") as f:
                        content = f.read()

                    for framework, patterns in cls.FRAMEWORK_PATTERNS.items():
                        for pattern in patterns:
                            if re.search(pattern, content):
                                if framework not in context.frameworks:
                                    context.frameworks.append(framework)
                                break
                except Exception:
                    pass


class WebContextGatherer:
    """Gathers context from the web."""

    DOCUMENTATION_SOURCES = {
        "python": "https://docs.python.org/3/",
        "javascript": "https://developer.mozilla.org/en-US/docs/Web/JavaScript",
        "typescript": "https://www.typescriptlang.org/docs/",
        "go": "https://golang.org/doc/",
        "rust": "https://doc.rust-lang.org/",
        "java": "https://docs.oracle.com/en/java/",
    }

    def __init__(self):
        self.console = Console()
        self.cache_path = CONTEXT_CACHE_PATH
        self.cache_path.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for a query."""
        return hashlib.sha256(query.encode()).hexdigest()[:16]

    def _get_cached(self, query: str) -> Optional[WebContext]:
        """Get cached web context."""
        cache_key = self._get_cache_key(query)
        cache_file = self.cache_path / f"{cache_key}.json"

        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)

                # Check TTL
                timestamp = datetime.fromisoformat(data.get("timestamp", ""))
                if datetime.now() - timestamp < CONTEXT_CACHE_TTL:
                    return WebContext(**data)
            except Exception:
                pass

        return None

    def _save_cache(self, query: str, context: WebContext):
        """Save web context to cache."""
        cache_key = self._get_cache_key(query)
        cache_file = self.cache_path / f"{cache_key}.json"

        context.timestamp = datetime.now().isoformat()

        try:
            with open(cache_file, "w") as f:
                json.dump(context.to_dict(), f, indent=2)
        except Exception:
            pass

    async def gather_context(self, query: str, language: str = "",
                            include_docs: bool = True,
                            include_issues: bool = True) -> WebContext:
        """Gather web context for a query."""
        # Check cache first
        cached = self._get_cached(query)
        if cached:
            return cached

        context = WebContext(query=query)

        if not aiohttp:
            self.console.print("[yellow]aiohttp not installed, web context limited[/yellow]")
            return context

        async with aiohttp.ClientSession() as session:
            tasks = []

            if include_docs and language in self.DOCUMENTATION_SOURCES:
                tasks.append(self._search_documentation(session, query, language))

            if include_issues:
                tasks.append(self._search_github_issues(session, query, language))

            tasks.append(self._search_stackoverflow(session, query, language))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, dict):
                    if "documentation" in result:
                        context.documentation.extend(result["documentation"])
                    if "issues" in result:
                        context.related_issues.extend(result["issues"])
                    if "examples" in result:
                        context.examples.extend(result["examples"])

        # Cache result
        self._save_cache(query, context)

        return context

    async def _search_documentation(self, session: aiohttp.ClientSession,
                                   query: str, language: str) -> Dict[str, Any]:
        """Search documentation for relevant information."""
        # Placeholder - in production would use actual API
        return {"documentation": []}

    async def _search_github_issues(self, session: aiohttp.ClientSession,
                                    query: str, language: str) -> Dict[str, Any]:
        """Search GitHub issues for related problems."""
        try:
            url = f"https://api.github.com/search/issues?q={query}+language:{language}&per_page=5"
            headers = {"Accept": "application/vnd.github.v3+json"}

            async with session.get(url, headers=headers, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    issues = []
                    for item in data.get("items", [])[:5]:
                        issues.append({
                            "title": item.get("title", ""),
                            "url": item.get("html_url", ""),
                            "state": item.get("state", ""),
                        })
                    return {"issues": issues}
        except Exception:
            pass

        return {"issues": []}

    async def _search_stackoverflow(self, session: aiohttp.ClientSession,
                                    query: str, language: str) -> Dict[str, Any]:
        """Search StackOverflow for examples and solutions."""
        try:
            url = (f"https://api.stackexchange.com/2.3/search/advanced?"
                   f"q={query}&tagged={language}&site=stackoverflow&pagesize=5")

            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    examples = []
                    for item in data.get("items", [])[:5]:
                        examples.append({
                            "title": item.get("title", ""),
                            "url": item.get("link", ""),
                            "score": item.get("score", 0),
                            "answered": item.get("is_answered", False),
                        })
                    return {"examples": examples}
        except Exception:
            pass

        return {"examples": []}


class UserContextManager:
    """Manages user context and preferences."""

    CONTEXT_FILE = Path(".cache/user_context.json")

    def __init__(self):
        self.console = Console()
        self.context = self._load_context()

    def _load_context(self) -> UserContext:
        """Load user context from file."""
        if self.CONTEXT_FILE.exists():
            try:
                with open(self.CONTEXT_FILE, "r") as f:
                    data = json.load(f)
                return UserContext(**data)
            except Exception:
                pass
        return UserContext()

    def save_context(self):
        """Save user context to file."""
        self.CONTEXT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(self.CONTEXT_FILE, "w") as f:
            json.dump(self.context.to_dict(), f, indent=2)

    def record_interaction(self, interaction_type: str, details: Dict[str, Any]):
        """Record a user interaction for learning."""
        self.context.history.append({
            "type": interaction_type,
            "timestamp": datetime.now().isoformat(),
            "details": details,
        })

        # Keep only recent history
        if len(self.context.history) > 100:
            self.context.history = self.context.history[-100:]

        self.save_context()

    def update_preference(self, key: str, value: Any):
        """Update a user preference."""
        self.context.preferences[key] = value
        self.save_context()

    def add_known_issue(self, issue: str):
        """Add a known issue to track."""
        if issue not in self.context.known_issues:
            self.context.known_issues.append(issue)
            self.save_context()

    def add_project_goal(self, goal: str):
        """Add a project goal."""
        if goal not in self.context.project_goals:
            self.context.project_goals.append(goal)
            self.save_context()

    def learn_coding_style(self, code_samples: List[str]):
        """Learn user's coding style from samples."""
        style = {
            "indent_style": "spaces",
            "indent_size": 4,
            "quote_style": "double",
            "trailing_comma": False,
            "max_line_length": 100,
        }

        # Analyze samples
        for code in code_samples:
            # Detect indent style
            if "\t" in code:
                style["indent_style"] = "tabs"
            else:
                # Count leading spaces
                for line in code.splitlines():
                    if line.startswith("    "):
                        style["indent_size"] = 4
                    elif line.startswith("  "):
                        style["indent_size"] = 2

            # Detect quote style
            single_quotes = code.count("'")
            double_quotes = code.count('"')
            style["quote_style"] = "single" if single_quotes > double_quotes else "double"

        self.context.coding_style = style
        self.save_context()


class ContextualizationEngine:
    """
    Main contextualization engine.
    Combines context from code, web, and user interactions.
    """

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.console = Console()
        self.code_analyzer = CodeAnalyzer()
        self.project_analyzer = ProjectAnalyzer()
        self.web_gatherer = WebContextGatherer() if enabled else None
        self.user_manager = UserContextManager() if enabled else None

    def analyze_current_directory(self) -> List[CodeContext]:
        """Analyze all code files in current directory."""
        if not self.enabled:
            return []

        contexts = []
        current = Path.cwd()

        # Analyze current directory
        for file_path in current.rglob("*"):
            if file_path.is_file() and file_path.suffix in CodeAnalyzer.LANGUAGE_PATTERNS:
                if any(ignore in str(file_path) for ignore in
                       [".git", "node_modules", "__pycache__", ".venv", "venv"]):
                    continue

                context = CodeAnalyzer.analyze_file(file_path)
                if context:
                    contexts.append(context)

        return contexts

    def analyze_parent_directory(self) -> List[CodeContext]:
        """Analyze code files in parent directory."""
        if not self.enabled:
            return []

        contexts = []
        parent = Path.cwd().parent

        for file_path in parent.rglob("*"):
            if file_path.is_file() and file_path.suffix in CodeAnalyzer.LANGUAGE_PATTERNS:
                if any(ignore in str(file_path) for ignore in
                       [".git", "node_modules", "__pycache__", ".venv", "venv"]):
                    continue

                context = CodeAnalyzer.analyze_file(file_path)
                if context:
                    contexts.append(context)

        return contexts

    def get_project_context(self) -> ProjectContext:
        """Get project-level context."""
        return ProjectAnalyzer.analyze_project()

    async def get_web_context(self, query: str, language: str = "") -> WebContext:
        """Get web context for a query."""
        if not self.enabled or not self.web_gatherer:
            return WebContext(query=query)

        return await self.web_gatherer.gather_context(query, language)

    def get_user_context(self) -> UserContext:
        """Get user context."""
        if not self.enabled or not self.user_manager:
            return UserContext()

        return self.user_manager.context

    async def build_full_context(self, query: str = "",
                                 include_code: bool = True,
                                 include_parent: bool = True,
                                 include_web: bool = True,
                                 include_user: bool = True) -> Dict[str, Any]:
        """Build comprehensive context from all sources."""
        if not self.enabled:
            return {"error": "Contextualization not enabled"}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            full_context = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
            }

            # Project context
            progress.add_task("Analyzing project...", total=None)
            full_context["project"] = self.get_project_context().to_dict()

            # Code context
            if include_code:
                progress.add_task("Analyzing current directory...", total=None)
                code_contexts = self.analyze_current_directory()
                full_context["current_code"] = [c.to_dict() for c in code_contexts[:50]]

            if include_parent:
                progress.add_task("Analyzing parent directory...", total=None)
                parent_contexts = self.analyze_parent_directory()
                full_context["parent_code"] = [c.to_dict() for c in parent_contexts[:20]]

            # Web context
            if include_web and query:
                progress.add_task("Gathering web context...", total=None)
                language = full_context["project"].get("languages", [""])[0]
                web_context = await self.get_web_context(query, language)
                full_context["web"] = web_context.to_dict()

            # User context
            if include_user:
                full_context["user"] = self.get_user_context().to_dict()

            # Generate summary
            full_context["summary"] = self._generate_summary(full_context)

            return full_context

    def _generate_summary(self, context: Dict[str, Any]) -> str:
        """Generate a summary of the context."""
        parts = []

        project = context.get("project", {})
        if project.get("project_name"):
            parts.append(f"Project: {project['project_name']}")
        if project.get("languages"):
            parts.append(f"Languages: {', '.join(project['languages'])}")
        if project.get("frameworks"):
            parts.append(f"Frameworks: {', '.join(project['frameworks'])}")

        code = context.get("current_code", [])
        if code:
            parts.append(f"Analyzed {len(code)} files in current directory")

        parent = context.get("parent_code", [])
        if parent:
            parts.append(f"Analyzed {len(parent)} files in parent directory")

        web = context.get("web", {})
        if web.get("examples"):
            parts.append(f"Found {len(web['examples'])} related examples")
        if web.get("related_issues"):
            parts.append(f"Found {len(web['related_issues'])} related issues")

        return "; ".join(parts)

    def display_context(self, context: Dict[str, Any]):
        """Display context in a formatted way."""
        self.console.print(Panel(
            context.get("summary", "No summary available"),
            title="Context Summary",
            border_style="cyan"
        ))

        project = context.get("project", {})
        if project:
            table = Table(title="Project Context")
            table.add_column("Property")
            table.add_column("Value")

            for key in ["project_name", "languages", "frameworks", "build_system", "test_framework"]:
                value = project.get(key, "")
                if isinstance(value, list):
                    value = ", ".join(value) if value else "-"
                table.add_row(key.replace("_", " ").title(), str(value) or "-")

            self.console.print(table)

        code = context.get("current_code", [])
        if code:
            self.console.print(f"\n[bold]Code Files Analyzed:[/bold] {len(code)}")

            # Group by language
            by_lang = defaultdict(int)
            for c in code:
                by_lang[c.get("language", "unknown")] += 1

            for lang, count in sorted(by_lang.items(), key=lambda x: -x[1]):
                self.console.print(f"  {lang}: {count} files")


# Global instance
_contextualization_engine: Optional[ContextualizationEngine] = None


def get_contextualization_engine() -> ContextualizationEngine:
    """Get or create the global contextualization engine."""
    global _contextualization_engine
    if _contextualization_engine is None:
        _contextualization_engine = ContextualizationEngine(enabled=False)
    return _contextualization_engine


def init_contextualization(enabled: bool = True) -> ContextualizationEngine:
    """Initialize contextualization engine."""
    global _contextualization_engine
    _contextualization_engine = ContextualizationEngine(enabled=enabled)
    return _contextualization_engine

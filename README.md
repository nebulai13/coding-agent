# Coding Agent

**Autonomous Code Optimization System**

An intelligent coding agent that iteratively analyzes, fixes, tests, optimizes, journals, and pushes code until it achieves flawless execution. Supports multiple AI providers, local AI on Apple Silicon, multi-agent orchestration, and web-based code search.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Configuration](#configuration)
6. [CLI Reference](#cli-reference)
7. [Modes of Operation](#modes-of-operation)
8. [AI Providers](#ai-providers)
9. [Local AI Setup](#local-ai-setup)
10. [Multi-Agent Orchestration](#multi-agent-orchestration)
11. [Code Search](#code-search)
12. [Local Code Search](#local-code-search)
13. [Experimental Features](#experimental-features)
14. [Self-Optimization](#self-optimization)
15. [Version Control](#version-control-system)
16. [Timed Sessions](#timed-sessions)
17. [User Interaction](#user-interaction)
18. [Journaling System](#journaling-system)
19. [Git Integration](#git-integration)
20. [API Reference](#api-reference)
21. [Architecture](#architecture)
22. [Troubleshooting](#troubleshooting)
23. [Contributing](#contributing)
24. [License](#license)

---

## Overview

Coding Agent automates the software development cycle through an intelligent loop:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   ANALYZE   │────▶│     FIX     │────▶│    TEST     │
└─────────────┘     └─────────────┘     └─────────────┘
       ▲                                       │
       │                                       ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    PUSH     │◀────│   JOURNAL   │◀────│  OPTIMIZE   │
└─────────────┘     └─────────────┘     └─────────────┘
       │
       └─────────────── REPEAT ──────────────▶
```

The agent continues iterating until:
- All tests pass
- Build succeeds
- No issues remain
- Or maximum iterations reached

---

## Features

### Core Features
- **Iterative Optimization**: Autonomous fix → test → optimize → journal → push loop
- **Multi-AI Support**: Claude, ChatGPT, Gemini, Perplexity, Junie (JetBrains)
- **Local AI**: Optimized for Apple Silicon M3/M4 using MLX, llama.cpp, or Ollama
- **Standalone Mode**: Works without AI using rule-based analysis and fixes
- **Code Search**: Find similar code, solutions, and examples across the web

### Intelligence Features
- **Multi-Agent Orchestration**: Run multiple AI agents in parallel
- **Consensus Voting**: Multiple agents vote on best solutions
- **Pipeline Processing**: Chain agents for specialized tasks
- **Swarm Intelligence**: Dynamic task allocation

### Developer Experience
- **User Interaction**: Asks for input, resources, and clarifications when needed
- **Rich Terminal UI**: Beautiful progress tracking and status display
- **Comprehensive Journaling**: Full audit trail of all activities
- **Git Integration**: Automatic commits and multi-account GitHub support

---

## Installation

### Prerequisites

- Python 3.10 or higher
- Git
- gh CLI (for GitHub operations): `brew install gh` (macOS) or see [cli.github.com](https://cli.github.com)

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/qwitch13/coding-agent.git
cd coding-agent

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Optional: Local AI Setup

```bash
# For Apple Silicon MLX support
pip install mlx mlx-lm

# For llama.cpp support
pip install llama-cpp-python

# For Ollama (easiest)
brew install ollama
ollama serve  # Start server
ollama pull codellama:7b-instruct
```

---

## Quick Start

### Interactive Mode

```bash
python main.py
```

This launches the interactive shell with commands like:
- `run <target>` - Run full optimization
- `fix <file>` - Fix issues
- `test` - Run tests
- `status` - Show status
- `help` - Show all commands

### Command Line

```bash
# Run agent on a directory
python main.py run src/

# Fix a specific file
python main.py fix main.py

# Optimize code
python main.py optimize utils.py

# Run tests
python main.py test

# Search for similar code
python main.py search "binary search tree implementation"
```

### With AI Providers

```bash
# Set API keys
export ANTHROPIC_API_KEY=your_claude_key
export OPENAI_API_KEY=your_chatgpt_key
export GOOGLE_API_KEY=your_gemini_key
export PERPLEXITY_API_KEY=your_perplexity_key

# Run with multi-agent mode
python main.py --mode multi_agent run src/
```

---

## Configuration

### Configuration File

Create `agent_config.json` in the project root:

```json
{
  "agent": {
    "mode": "single_ai",
    "max_iterations": 10,
    "test_command": "pytest",
    "build_command": "python -m py_compile",
    "lint_command": "ruff check",
    "max_parallel_agents": 4,
    "test_timeout": 300,
    "build_timeout": 120,
    "ai_timeout": 60,
    "journal_enabled": true,
    "verbose": false
  },
  "local_ai": {
    "enabled": true,
    "backend": "mlx",
    "model_name": "codellama-7b",
    "context_size": 4096,
    "gpu_layers": -1,
    "threads": 8
  },
  "git": {
    "auto_commit": false,
    "auto_push": false,
    "default_branch": "main",
    "commit_message_template": "[Agent] {action}: {description}",
    "github_accounts": ["nebulai13", "qwitch13"]
  }
}
```

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `ANTHROPIC_API_KEY` | Claude API key | No |
| `OPENAI_API_KEY` | ChatGPT API key | No |
| `GOOGLE_API_KEY` | Gemini API key | No |
| `PERPLEXITY_API_KEY` | Perplexity API key | No |
| `GITHUB_TOKEN` | GitHub personal access token | No |

### Configuration Priority

1. Command-line arguments (highest)
2. Environment variables
3. `agent_config.json`
4. Default values (lowest)

---

## CLI Reference

### Global Options

```bash
python main.py [OPTIONS] COMMAND [ARGS]

Options:
  -V, --version              Show version
  -v, --verbose              Verbose output
  -m, --mode MODE            Operation mode (standalone, single_ai, multi_agent)
```

### Commands

#### `run` - Full Optimization Loop

```bash
python main.py run <target> [OPTIONS]

Arguments:
  target    File or directory to process

Examples:
  python main.py run src/
  python main.py run main.py
  python main.py --mode multi_agent run .
```

#### `fix` - Fix Issues Only

```bash
python main.py fix <target>

Examples:
  python main.py fix utils.py
  python main.py fix src/modules/
```

#### `optimize` - Optimize Code

```bash
python main.py optimize <target>

Examples:
  python main.py optimize slow_function.py
```

#### `test` - Run Tests

```bash
python main.py test

# Uses test_command from config (default: pytest)
```

#### `search` - Search for Code

```bash
python main.py search "<query>" [OPTIONS]

Options:
  --language LANG    Filter by language (python, javascript, etc.)
  --source SOURCE    Search source (github, stackoverflow, all)
  --limit N          Maximum results (default: 20)

Examples:
  python main.py search "quick sort python"
  python main.py search "async http client" --language python
  python main.py search "React hooks" --source github
```

#### `status` - Show Status

```bash
python main.py status
```

#### `providers` - List AI Providers

```bash
python main.py providers
```

#### `push` - Commit and Push

```bash
python main.py push
```

#### `create-repos` - Create GitHub Repositories

```bash
python main.py create-repos [OPTIONS]

Options:
  -n, --name NAME         Repository name (default: coding-agent)
  -d, --description DESC  Repository description
  -p, --private           Create private repositories

Examples:
  python main.py create-repos --name my-project
  python main.py create-repos --name my-project --private
```

---

## Modes of Operation

### Standalone Mode

No AI required. Uses rule-based analysis and fixes.

```bash
python main.py --mode standalone run src/
```

**Capabilities:**
- Syntax error detection
- Linting with Ruff
- Import organization
- Basic code formatting
- Test execution

### Single AI Mode

Uses one AI provider for all operations.

```bash
python main.py --mode single_ai run src/
```

**Provider Selection Order:**
1. Claude (if available)
2. ChatGPT (if available)
3. Gemini (if available)
4. Local AI (if configured)
5. Perplexity (if available)

### Multi-Agent Mode

Uses multiple AI providers working together.

```bash
python main.py --mode multi_agent run src/
```

**Strategies:**
- `parallel` - Different agents work on different tasks
- `consensus` - Multiple agents vote on solutions
- `pipeline` - Chain of specialized agents
- `swarm` - Dynamic task allocation

---

## AI Providers

### Claude (Anthropic)

```bash
export ANTHROPIC_API_KEY=your_key
```

**Models:**
- `claude-sonnet-4-20250514` (default)
- `claude-opus-4-5-20251101`

**Best for:** Complex reasoning, code understanding, refactoring

### ChatGPT (OpenAI)

```bash
export OPENAI_API_KEY=your_key
```

**Models:**
- `gpt-4o` (default)
- `gpt-4-turbo`

**Best for:** General coding, explanations, diverse languages

### Gemini (Google)

```bash
export GOOGLE_API_KEY=your_key
```

**Models:**
- `gemini-2.0-flash` (default)
- `gemini-pro`

**Best for:** Fast responses, multi-modal understanding

### Perplexity

```bash
export PERPLEXITY_API_KEY=your_key
```

**Models:**
- `llama-3.1-sonar-large-128k-online` (default)

**Best for:** Research-backed solutions, up-to-date information

### Junie (JetBrains)

Requires JetBrains IDE with Junie plugin running.

**Best for:** IDE-integrated fixes, project-aware suggestions

---

## Local AI Setup

### MLX (Recommended for Apple Silicon)

```bash
pip install mlx mlx-lm

# Models are auto-downloaded from HuggingFace
```

**Configuration:**
```json
{
  "local_ai": {
    "enabled": true,
    "backend": "mlx",
    "model_name": "codellama-7b"
  }
}
```

### Ollama (Easiest Setup)

```bash
# Install
brew install ollama  # macOS
# or see https://ollama.ai for other platforms

# Start server
ollama serve

# Pull a model
ollama pull codellama:7b-instruct
```

**Configuration:**
```json
{
  "local_ai": {
    "enabled": true,
    "backend": "ollama",
    "model_name": "codellama:7b-instruct"
  }
}
```

### llama.cpp

```bash
pip install llama-cpp-python

# Download a GGUF model to models/ directory
```

**Configuration:**
```json
{
  "local_ai": {
    "enabled": true,
    "backend": "llama.cpp",
    "model_path": "models/codellama-7b-instruct.Q4_K_M.gguf"
  }
}
```

### Recommended Models

| Model | Size | Use Case | Performance |
|-------|------|----------|-------------|
| CodeLlama 7B | 4.1 GB | General code fixes | Fast |
| CodeLlama 13B | 7.9 GB | Complex reasoning | Medium |
| DeepSeek Coder 6.7B | 4.0 GB | Code generation | Fast |
| Qwen2.5 Coder 7B | 4.5 GB | Multi-language | Fast |
| StarCoder2 7B | 4.2 GB | Code completion | Fast |

---

## Multi-Agent Orchestration

### Parallel Strategy

Multiple agents work on different tasks simultaneously.

```python
from orchestrator import run_multi_agent

results = await run_multi_agent("src/", strategy="parallel")
```

**Use when:**
- Tasks are independent
- Speed is priority
- Multiple files need processing

### Consensus Strategy

Multiple agents solve the same problem and vote on the best solution.

```python
from orchestrator import AgentOrchestrator

orchestrator = AgentOrchestrator()
result = await orchestrator.run_consensus(code, issue, min_agents=3)
print(f"Confidence: {result.confidence}")
print(f"Winning solution: {result.winning_response}")
```

**Use when:**
- Critical fixes needed
- Want high confidence
- Ambiguous problems

### Pipeline Strategy

Chain agents for specialized processing.

```
Analyze → Fix → Review → Optimize
```

```python
result = await orchestrator.run_pipeline("main.py")
print(f"Approved: {result['approved']}")
```

**Use when:**
- Need code review
- Quality assurance required
- Complex refactoring

### Swarm Strategy

Dynamic task allocation with continuous improvement.

```python
results = await orchestrator.run_swarm("src/", max_iterations=10)
```

**Use when:**
- Large codebases
- Ongoing optimization
- Unknown scope of issues

---

## Code Search

The agent can search the web for similar code, solutions, and examples.

### Basic Usage

```bash
# Search for code examples
python main.py search "binary search tree python"

# Search specific source
python main.py search "React useState" --source stackoverflow

# Filter by language
python main.py search "sort algorithm" --language rust
```

### Programmatic Usage

```python
from code_search import CodeSearchManager

search = CodeSearchManager()

# Search GitHub
results = await search.search_github("async http client", language="python")

# Search StackOverflow
results = await search.search_stackoverflow("how to parse JSON")

# Search all sources
results = await search.search_all("binary tree implementation")

# Find similar code
similar = await search.find_similar_code(my_code_snippet)
```

### Search Sources

| Source | Description | Best For |
|--------|-------------|----------|
| GitHub | Code repositories | Libraries, implementations |
| StackOverflow | Q&A | How-to, debugging |
| Google | General search | Documentation, tutorials |
| Semantic Scholar | Academic papers | Algorithms, research |

### Search Result Format

```python
{
    "source": "github",
    "title": "Repository name or answer title",
    "url": "https://...",
    "code": "# Code snippet...",
    "language": "python",
    "relevance": 0.95,
    "metadata": {
        "stars": 1234,
        "forks": 56,
        "author": "username"
    }
}
```

---

## Local Code Search

Search your own local directories for code patterns and extract matches to your current project.

### Configure Search Directories

Add directories that the agent should search when looking for code:

```bash
# Add a directory
python main.py local-add ~/projects
python main.py local-add /path/to/libraries --name "My Libraries"

# List configured directories
python main.py local-list

# Remove a directory
python main.py local-remove ~/projects
```

### Search Local Directories

```bash
# Basic search
python main.py local-search "class MyClass"

# Search with regex
python main.py local-search "def \w+_handler" --regex

# Filter by language
python main.py local-search "async function" --language javascript

# Search and extract matches
python main.py local-search "authentication" --extract
```

### Programmatic Usage

```python
from local_code_search import LocalSearchManager, get_local_search_manager

manager = get_local_search_manager()

# Add search directories
manager.add_directory("~/projects")
manager.add_directory("/path/to/libs", name="Libraries")

# Search
results = manager.search("def calculate")

# Display results
results = manager.search_and_display("class Handler", show_context=True)

# Extract matches to current folder
summary = manager.extract_results(results)
print(f"Extracted {summary['success']} files to ./extracted/")
```

### Search Features

| Feature | Description |
|---------|-------------|
| **Pattern Matching** | Search by string or regex |
| **Language Filtering** | Filter by programming language |
| **Context Display** | Show lines around matches |
| **Relevance Scoring** | Results ranked by relevance |
| **File Extraction** | Copy matches to current project |
| **Directory Config** | Save directories for future sessions |

### Configuration File

Local search configuration is saved in `.local_search_config.json`:

```json
{
  "directories": [
    {
      "path": "/Users/me/projects",
      "name": "My Projects",
      "enabled": true,
      "languages": [],
      "added_at": "2024-01-15T10:30:00"
    }
  ],
  "default_languages": [],
  "max_file_size": 1048576,
  "max_results": 100
}
```

### Supported Languages

Python, JavaScript, TypeScript, Java, Kotlin, Go, Rust, C, C++, Ruby, PHP, Swift, Shell, SQL, HTML, CSS, YAML, JSON, Markdown

### Ignored Directories

The following are automatically ignored:
- `.git`, `.svn`, `.hg`
- `node_modules`, `__pycache__`
- `venv`, `.venv`, `.env`
- `build`, `dist`, `target`, `out`
- `.idea`, `.vscode`

---

## Experimental Features

Experimental features can be toggled on/off. These features may be unstable.

### Managing Features

```bash
# List all experimental features
python main.py experimental

# Enable a feature
python main.py experimental-enable self_optimization

# Enable dangerous features (requires --force)
python main.py experimental-enable self_update --force

# Disable a feature
python main.py experimental-disable self_optimization
```

### Available Features

| Feature | Description | Dangerous |
|---------|-------------|-----------|
| `self_optimization` | Learn from interactions, track trial/error | No |
| `self_update` | Agent can modify its own code | Yes |
| `contextualization` | Build context from web, user, code | No |
| `web_context` | Gather context from GitHub, StackOverflow | No |
| `user_learning` | Learn user preferences and style | No |
| `auto_strategy` | Auto-create fix strategies | No |

---

## Self-Optimization

When enabled, the agent learns from every fix attempt.

### Learning Database

The agent tracks:
- Error signatures and categories
- Fix strategies that worked
- Success rates and fix times
- Long-term stability of fixes

### Usage

```bash
# Enable self-optimization
python main.py experimental-enable self_optimization

# View learning statistics
python main.py learn-stats

# Run autonomous web learning (6 hours)
python main.py learn 6h python error-handling

# Run self-optimization task
python main.py optimize-self
```

### Web Learning

The agent can learn from web sources for extended periods:

```bash
# Learn for 6 hours about Python
python main.py learn 6h python

# Learn about specific topics
python main.py learn 2h async await error-handling

# Learn based on project context (auto-detects)
python main.py learn 1h
```

---

## Version Control System

Track code versions with ratings and smart rollback.

### Version Commands

```bash
# Show version history for a file
python main.py version src/main.py

# Show overall statistics
python main.py version

# Rollback to previous version
python main.py version-rollback src/main.py

# Rollback to best-rated version
python main.py version-rollback src/main.py --best

# Rate a version (1-5)
python main.py version-rate abc123 5 "Fixed the issue perfectly"
```

### Rating System

| Rating | Description |
|--------|-------------|
| 5 - Excellent | Quick fix, still working after weeks |
| 4 - Good | Worked well, no issues |
| 3 - Acceptable | Works but has minor issues |
| 2 - Poor | Caused problems, needed revision |
| 1 - Broken | Completely broken, reverted |

### Automatic Rating

- Versions stable for 7+ days get upgraded
- Versions rolled back from get downgraded
- Effectiveness score combines rating, stability, rollback history

---

## Timed Sessions

Run work sessions with deadlines and auto-save.

### Usage

```bash
# Work on src/ for 6 hours
python main.py work 6h src/

# Work with specific mode
python main.py work 2h main.py --mode optimize

# Work for 2 days
python main.py work 2d ./

# Resume a paused session
python main.py resume
```

### Session Features

- **Automatic checkpoints**: State saved every 5 minutes
- **Pause/Resume**: Press Ctrl+C to pause, resume later
- **Progress tracking**: Shows iterations, time remaining
- **Comprehensive journaling**: All actions recorded

### Session State

Saved to `.cache/session_state.json`:

```json
{
  "start_time": "2024-01-15T10:00:00",
  "deadline": "2024-01-15 16:00",
  "target": "src/",
  "mode": "fix",
  "iterations_completed": 15,
  "issues_fixed": 8,
  "status": "paused"
}
```

---

## User Interaction

The agent asks for user input when needed.

### API Key Requests

```
╔══════════════════════════════════════════════════════════════╗
║                     API Key Required                          ║
╠══════════════════════════════════════════════════════════════╣
║  Provider: Claude                                             ║
║  Environment Variable: ANTHROPIC_API_KEY                      ║
║                                                               ║
║  You can also set this in your shell:                        ║
║  export ANTHROPIC_API_KEY=your_key                           ║
╚══════════════════════════════════════════════════════════════╝

? How would you like to provide the API key?
  [1] Enter it now (will be used for this session only)
  [2] Skip this provider
  [3] Exit and set environment variable
```

### Clarification Requests

```
╔══════════════════════════════════════════════════════════════╗
║                   Clarification Needed                        ║
╠══════════════════════════════════════════════════════════════╣
║  Error in utils.py:                                          ║
║  TypeError: unsupported operand type(s)                      ║
║                                                               ║
║  Can you provide more context about this error?              ║
╚══════════════════════════════════════════════════════════════╝

? Select an option:
  [1] This is a known issue, proceed with standard fix
  [2] Skip this error for now
  [3] Let me explain the expected behavior
  [4] Other (specify)
```

### Code Review

```
╔══════════════════════════════════════════════════════════════╗
║                    Code Change Review                         ║
╠══════════════════════════════════════════════════════════════╣
║  File: utils.py                                               ║
║  Change: Fixed type error in calculate_total function        ║
╚══════════════════════════════════════════════════════════════╝

--- original
+++ modified
@@ -10,7 +10,7 @@
 def calculate_total(items):
-    return sum(items)
+    return sum(float(item) for item in items)

? Apply this change? [y/N]:
```

### Disabling Prompts

For CI/CD or automation:

```python
from user_interaction import set_auto_mode
set_auto_mode(True)  # Use defaults for all prompts
```

---

## Journaling System

All agent activities are logged for audit and debugging.

### Journal Location

```
journal/
├── agent_20240115_143022_a1b2c3.json  # Session journal
└── ...
```

### Journal Entry Types

| Type | Description |
|------|-------------|
| `iteration_start` | Beginning of optimization iteration |
| `iteration_end` | End of iteration with results |
| `analysis` | Code analysis results |
| `fix_attempt` | Fix attempt details |
| `test_run` | Test execution results |
| `build` | Build/compile results |
| `optimization` | Code optimization applied |
| `ai_interaction` | AI provider interaction |
| `agent_task` | Multi-agent task execution |
| `consensus` | Consensus voting result |
| `git_action` | Git operation |
| `error` | Error encountered |
| `session_end` | Session summary |

### Viewing Journal

```bash
python main.py journal
```

Or programmatically:

```python
from journal import get_journal

journal = get_journal()
summary = journal.get_session_summary()
print(f"Total iterations: {summary['total_iterations']}")
print(f"Fixes applied: {summary['successful_fixes']}")
```

### Exporting Reports

```python
journal = get_journal()

# Export as Markdown
report_path = journal.export_report("md")

# Export as JSON
report_path = journal.export_report("json")

# Export as plain text
report_path = journal.export_report("txt")
```

---

## Git Integration

### Automatic Commits

```python
from git_integration import GitManager

git = GitManager()

# Create agent-style commit
success, hash = git.create_agent_commit(
    action="fix",
    description="Resolved type error in utils.py"
)
```

### Multi-Account Push

```python
from git_integration import create_and_push_repos

# Create repos on all configured accounts and push
results = await create_and_push_repos(
    repo_name="my-project",
    description="Project description",
    private=False
)

for account, (success, url) in results["repo_creation"].items():
    print(f"{account}: {url}")
```

### GitHub Operations

```python
from git_integration import GitHubManager

gh = GitHubManager()

# Create PR
success, url = gh.create_pr(
    title="Fix: Resolved type errors",
    body="## Summary\n- Fixed type errors in utils.py",
    base="main"
)

# List repos
repos = gh.list_repos("username")
```

---

## API Reference

### AgentEngine

Main agent engine class.

```python
from agent_engine import AgentEngine

engine = AgentEngine(progress_callback=my_callback)

# Run optimization loop
iterations = await engine.run(
    target="src/",
    mode="fix",  # fix, optimize, full
    ai_provider="claude"
)

# Get summary
summary = engine.get_summary()
```

### AIProviderManager

Manages AI providers.

```python
from ai_providers import provider_manager

# Get available providers
providers = provider_manager.get_available_providers()

# Generate with fallback
response = await provider_manager.generate_with_fallback(
    prompt="Fix this code...",
    preferred_providers=["claude", "chatgpt"]
)

# Generate in parallel
responses = await provider_manager.generate_parallel(
    prompt="Analyze this code...",
    providers=["claude", "gemini"]
)
```

### LocalAIManager

Manages local AI models.

```python
from local_ai import local_ai_manager

# Load model
await local_ai_manager.load_model("codellama-7b")

# Generate
response = await local_ai_manager.generate(prompt)

# Generate code fix
fixed = await local_ai_manager.generate_code_fix(code, error)

# Get status
status = local_ai_manager.get_status()
```

### CodeSearchManager

Web code search functionality.

```python
from code_search import CodeSearchManager

search = CodeSearchManager()

# Search GitHub
results = await search.search_github(query, language="python")

# Search StackOverflow
results = await search.search_stackoverflow(query)

# Find similar code
similar = await search.find_similar_code(code_snippet)
```

### LocalSearchManager

Local directory code search.

```python
from local_code_search import get_local_search_manager

manager = get_local_search_manager()

# Configure directories
manager.add_directory("~/projects")
manager.list_directories()

# Search local files
results = manager.search("class Handler")

# Search with display
results = manager.search_and_display("async def", show_context=True)

# Extract to current folder
summary = manager.extract_results(results)
# Files extracted to ./extracted/
```

### UserInteractionManager

User interaction handling.

```python
from user_interaction import get_interaction_manager

interaction = get_interaction_manager()

# Ask confirmation
if interaction.ask_confirmation("Apply fix?"):
    apply_fix()

# Ask for input
value = interaction.ask_input("Enter test command:")

# Request resource
response = interaction.request_resource(
    resource_type="api_key",
    description="Claude API key needed"
)
```

---

## Architecture

### Project Structure

```
coding-agent/
├── main.py              # Entry point and CLI
├── config.py            # Configuration management
│   ├── Config           # Central config class
│   ├── AIProviderConfig # Provider settings
│   ├── LocalAIConfig    # Local AI settings
│   ├── GitConfig        # Git settings
│   └── AgentConfig      # Agent settings
│
├── ai_providers.py      # AI provider integrations
│   ├── BaseAIProvider   # Abstract base class
│   ├── ClaudeProvider   # Anthropic Claude
│   ├── ChatGPTProvider  # OpenAI GPT-4
│   ├── GeminiProvider   # Google Gemini
│   ├── PerplexityProvider # Perplexity AI
│   ├── JunieProvider    # JetBrains Junie
│   └── AIProviderManager # Provider coordination
│
├── local_ai.py          # Local AI support
│   ├── MLXBackend       # Apple MLX
│   ├── LlamaCppBackend  # llama.cpp
│   ├── OllamaBackend    # Ollama
│   └── LocalAIManager   # Backend management
│
├── agent_engine.py      # Core agent logic
│   ├── AgentEngine      # Main engine
│   ├── CodeAnalyzer     # Code analysis
│   ├── TestRunner       # Test execution
│   ├── BuildRunner      # Build execution
│   ├── CodeFixer        # Issue fixing
│   └── CodeOptimizer    # Code optimization
│
├── orchestrator.py      # Multi-agent coordination
│   ├── AgentOrchestrator # Main orchestrator
│   ├── TaskQueue        # Task management
│   ├── Agent            # Agent representation
│   └── ConsensusResult  # Voting results
│
├── code_search.py       # Web code search
│   ├── CodeSearchManager # Search coordination
│   ├── GitHubSearch     # GitHub search
│   ├── StackOverflowSearch # SO search
│   └── WebSearch        # General web search
│
├── local_code_search.py # Local directory search
│   ├── LocalSearchManager # Search management
│   ├── LocalCodeSearcher  # File searching
│   ├── LocalSearchConfig  # Directory config
│   └── SearchDirectory    # Directory settings
│
├── journal.py           # Activity logging
│   └── AgentJournal     # Journal management
│
├── git_integration.py   # Git operations
│   ├── GitManager       # Git operations
│   ├── GitHubManager    # GitHub API
│   └── GitHubAccountManager # Multi-account
│
├── terminal_ui.py       # Terminal interface
│   ├── TerminalUI       # Main UI
│   ├── ProgressTracker  # Progress display
│   └── ResultsDisplay   # Results display
│
├── user_interaction.py  # User prompts
│   ├── UserInteractionManager # Interaction handling
│   └── AgentPrompts     # Standard prompts
│
└── requirements.txt     # Dependencies
```

### Data Flow

```
User Input
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                     CLI (main.py)                        │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                  Agent Engine                            │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │
│  │ Analyze │─▶│   Fix   │─▶│  Test   │─▶│Optimize │    │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘    │
└─────────────────────────────────────────────────────────┘
    │                   │
    ▼                   ▼
┌─────────────┐    ┌─────────────┐
│ AI Providers│    │ Code Search │
│  - Claude   │    │  - GitHub   │
│  - ChatGPT  │    │  - StackOF  │
│  - Gemini   │    │  - Web      │
│  - Local AI │    └─────────────┘
└─────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                    Journal                               │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                  Git Integration                         │
│                 (Commit & Push)                          │
└─────────────────────────────────────────────────────────┘
```

---

## Troubleshooting

### Common Issues

#### "No AI providers available"

```bash
# Check if API keys are set
echo $ANTHROPIC_API_KEY
echo $OPENAI_API_KEY

# Or use standalone mode
python main.py --mode standalone run src/
```

#### "Test command not found"

```bash
# Install pytest
pip install pytest

# Or configure custom test command
python main.py config test_command "python -m unittest"
```

#### "Local AI model not loaded"

```bash
# For Ollama
ollama list  # Check installed models
ollama pull codellama:7b-instruct

# For MLX/llama.cpp - download model to models/
```

#### "GitHub authentication failed"

```bash
# Login with gh CLI
gh auth login

# Check status
gh auth status
```

### Debug Mode

```bash
python main.py --verbose run src/
```

### Logs

```bash
# View latest log
cat logs/agent_*.log | tail -100

# View journal
python main.py journal
```

---

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and test
4. Commit: `git commit -m "Add feature"`
5. Push: `git push origin feature-name`
6. Create Pull Request

### Code Style

- Python 3.10+ type hints
- Async/await for I/O operations
- Rich library for terminal output
- Dataclasses for data structures

---

## License

MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

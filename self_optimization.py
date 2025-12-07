"""
Self Optimization Module for Coding Agent
EXPERIMENTAL: Learn from interactions, track trial/error, self-update code.

Toggle with: experimental.self_optimization = True
"""
import os
import re
import json
import sqlite3
import hashlib
import ast
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import difflib

from rich.console import Console
from rich.table import Table
from rich.panel import Panel


# Database location
DB_PATH = Path(".cache/learning.db")
STRATEGIES_PATH = Path(".cache/strategies")
SELF_UPDATE_BACKUP = Path(".cache/backups")


class OutcomeRating(Enum):
    """Rating for fix outcomes."""
    EXCELLENT = 5  # Quick fix, still working after long time
    GOOD = 4       # Moderate time, still working
    ACCEPTABLE = 3 # Took time but worked
    POOR = 2       # Worked briefly, needed revision
    FAILED = 1     # Did not work or broke something


class FixCategory(Enum):
    """Categories of fixes for pattern matching."""
    SYNTAX_ERROR = "syntax_error"
    TYPE_ERROR = "type_error"
    IMPORT_ERROR = "import_error"
    RUNTIME_ERROR = "runtime_error"
    LOGIC_ERROR = "logic_error"
    PERFORMANCE = "performance"
    STYLE = "style"
    SECURITY = "security"
    REFACTOR = "refactor"
    UNKNOWN = "unknown"


@dataclass
class TrialRecord:
    """Record of a fix attempt."""
    id: str = ""
    timestamp: str = ""
    category: str = ""
    error_signature: str = ""
    error_message: str = ""
    file_path: str = ""
    original_code: str = ""
    fix_applied: str = ""
    fix_strategy: str = ""
    ai_provider: str = ""
    time_to_fix_seconds: float = 0.0
    tests_passed: bool = False
    build_passed: bool = False
    outcome_rating: int = 0
    still_working_after_days: int = 0
    reverted: bool = False
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Strategy:
    """A learned strategy for fixing issues."""
    id: str = ""
    name: str = ""
    category: str = ""
    error_pattern: str = ""
    fix_template: str = ""
    success_rate: float = 0.0
    avg_time_seconds: float = 0.0
    usage_count: int = 0
    last_used: str = ""
    created_at: str = ""
    source: str = ""  # "learned", "manual", "ai_suggested"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LearningDatabase:
    """SQLite database for storing learning data."""

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS trials (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                category TEXT,
                error_signature TEXT,
                error_message TEXT,
                file_path TEXT,
                original_code TEXT,
                fix_applied TEXT,
                fix_strategy TEXT,
                ai_provider TEXT,
                time_to_fix_seconds REAL,
                tests_passed INTEGER,
                build_passed INTEGER,
                outcome_rating INTEGER,
                still_working_after_days INTEGER,
                reverted INTEGER,
                notes TEXT
            );

            CREATE TABLE IF NOT EXISTS strategies (
                id TEXT PRIMARY KEY,
                name TEXT,
                category TEXT,
                error_pattern TEXT,
                fix_template TEXT,
                success_rate REAL,
                avg_time_seconds REAL,
                usage_count INTEGER,
                last_used TEXT,
                created_at TEXT,
                source TEXT
            );

            CREATE TABLE IF NOT EXISTS code_evolution (
                id TEXT PRIMARY KEY,
                file_path TEXT,
                version INTEGER,
                timestamp TEXT,
                code_hash TEXT,
                change_description TEXT,
                performance_score REAL,
                stability_score REAL
            );

            CREATE TABLE IF NOT EXISTS self_updates (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                target_file TEXT,
                change_type TEXT,
                description TEXT,
                original_code TEXT,
                new_code TEXT,
                approved INTEGER,
                applied INTEGER,
                rollback_available INTEGER
            );

            CREATE INDEX IF NOT EXISTS idx_trials_category ON trials(category);
            CREATE INDEX IF NOT EXISTS idx_trials_signature ON trials(error_signature);
            CREATE INDEX IF NOT EXISTS idx_strategies_category ON strategies(category);
            CREATE INDEX IF NOT EXISTS idx_strategies_pattern ON strategies(error_pattern);
        """)
        self.conn.commit()

    def record_trial(self, trial: TrialRecord) -> str:
        """Record a fix attempt trial."""
        if not trial.id:
            trial.id = hashlib.sha256(
                f"{trial.timestamp}{trial.file_path}{trial.error_message}".encode()
            ).hexdigest()[:16]

        if not trial.timestamp:
            trial.timestamp = datetime.now().isoformat()

        self.conn.execute("""
            INSERT OR REPLACE INTO trials VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, (
            trial.id, trial.timestamp, trial.category, trial.error_signature,
            trial.error_message, trial.file_path, trial.original_code,
            trial.fix_applied, trial.fix_strategy, trial.ai_provider,
            trial.time_to_fix_seconds, int(trial.tests_passed),
            int(trial.build_passed), trial.outcome_rating,
            trial.still_working_after_days, int(trial.reverted), trial.notes
        ))
        self.conn.commit()
        return trial.id

    def get_similar_trials(self, error_signature: str, limit: int = 10) -> List[TrialRecord]:
        """Find similar past trials."""
        cursor = self.conn.execute("""
            SELECT * FROM trials
            WHERE error_signature = ? OR error_signature LIKE ?
            ORDER BY outcome_rating DESC, time_to_fix_seconds ASC
            LIMIT ?
        """, (error_signature, f"%{error_signature[:20]}%", limit))

        return [TrialRecord(**dict(row)) for row in cursor.fetchall()]

    def get_best_strategy(self, category: str, error_pattern: str) -> Optional[Strategy]:
        """Get the best strategy for a given error type."""
        cursor = self.conn.execute("""
            SELECT * FROM strategies
            WHERE category = ? AND (error_pattern = ? OR ? LIKE '%' || error_pattern || '%')
            ORDER BY success_rate DESC, avg_time_seconds ASC
            LIMIT 1
        """, (category, error_pattern, error_pattern))

        row = cursor.fetchone()
        if row:
            return Strategy(**dict(row))
        return None

    def save_strategy(self, strategy: Strategy) -> str:
        """Save a new or updated strategy."""
        if not strategy.id:
            strategy.id = hashlib.sha256(
                f"{strategy.name}{strategy.category}{strategy.error_pattern}".encode()
            ).hexdigest()[:16]

        if not strategy.created_at:
            strategy.created_at = datetime.now().isoformat()

        self.conn.execute("""
            INSERT OR REPLACE INTO strategies VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, (
            strategy.id, strategy.name, strategy.category, strategy.error_pattern,
            strategy.fix_template, strategy.success_rate, strategy.avg_time_seconds,
            strategy.usage_count, strategy.last_used, strategy.created_at, strategy.source
        ))
        self.conn.commit()
        return strategy.id

    def update_strategy_stats(self, strategy_id: str, success: bool, time_seconds: float):
        """Update strategy statistics after use."""
        cursor = self.conn.execute(
            "SELECT success_rate, avg_time_seconds, usage_count FROM strategies WHERE id = ?",
            (strategy_id,)
        )
        row = cursor.fetchone()
        if row:
            old_rate = row["success_rate"]
            old_time = row["avg_time_seconds"]
            count = row["usage_count"]

            new_count = count + 1
            new_rate = ((old_rate * count) + (1.0 if success else 0.0)) / new_count
            new_time = ((old_time * count) + time_seconds) / new_count

            self.conn.execute("""
                UPDATE strategies
                SET success_rate = ?, avg_time_seconds = ?, usage_count = ?, last_used = ?
                WHERE id = ?
            """, (new_rate, new_time, new_count, datetime.now().isoformat(), strategy_id))
            self.conn.commit()

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get overall learning statistics."""
        stats = {}

        # Total trials
        cursor = self.conn.execute("SELECT COUNT(*) FROM trials")
        stats["total_trials"] = cursor.fetchone()[0]

        # Success rate
        cursor = self.conn.execute("""
            SELECT AVG(CASE WHEN tests_passed = 1 AND build_passed = 1 THEN 1.0 ELSE 0.0 END)
            FROM trials
        """)
        stats["overall_success_rate"] = cursor.fetchone()[0] or 0.0

        # Average fix time
        cursor = self.conn.execute("SELECT AVG(time_to_fix_seconds) FROM trials WHERE tests_passed = 1")
        stats["avg_fix_time_seconds"] = cursor.fetchone()[0] or 0.0

        # Best performing categories
        cursor = self.conn.execute("""
            SELECT category, AVG(outcome_rating) as avg_rating, COUNT(*) as count
            FROM trials
            GROUP BY category
            ORDER BY avg_rating DESC
            LIMIT 5
        """)
        stats["best_categories"] = [dict(row) for row in cursor.fetchall()]

        # Total strategies
        cursor = self.conn.execute("SELECT COUNT(*) FROM strategies")
        stats["total_strategies"] = cursor.fetchone()[0]

        # Top strategies
        cursor = self.conn.execute("""
            SELECT name, success_rate, usage_count
            FROM strategies
            ORDER BY success_rate DESC, usage_count DESC
            LIMIT 5
        """)
        stats["top_strategies"] = [dict(row) for row in cursor.fetchall()]

        return stats

    def record_self_update(self, target_file: str, change_type: str,
                          description: str, original_code: str, new_code: str) -> str:
        """Record a self-update attempt."""
        update_id = hashlib.sha256(
            f"{datetime.now().isoformat()}{target_file}".encode()
        ).hexdigest()[:16]

        self.conn.execute("""
            INSERT INTO self_updates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            update_id, datetime.now().isoformat(), target_file, change_type,
            description, original_code, new_code, 0, 0, 1
        ))
        self.conn.commit()
        return update_id

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


class ErrorAnalyzer:
    """Analyzes errors to create signatures and categorize them."""

    ERROR_PATTERNS = {
        FixCategory.SYNTAX_ERROR: [
            r"SyntaxError:",
            r"IndentationError:",
            r"TabError:",
            r"unexpected EOF",
            r"invalid syntax",
        ],
        FixCategory.TYPE_ERROR: [
            r"TypeError:",
            r"unsupported operand type",
            r"object is not callable",
            r"takes \d+ positional argument",
        ],
        FixCategory.IMPORT_ERROR: [
            r"ImportError:",
            r"ModuleNotFoundError:",
            r"No module named",
            r"cannot import name",
        ],
        FixCategory.RUNTIME_ERROR: [
            r"RuntimeError:",
            r"RecursionError:",
            r"MemoryError:",
            r"SystemError:",
        ],
        FixCategory.LOGIC_ERROR: [
            r"AssertionError:",
            r"ValueError:",
            r"KeyError:",
            r"IndexError:",
            r"AttributeError:",
        ],
        FixCategory.PERFORMANCE: [
            r"timeout",
            r"slow",
            r"performance",
            r"memory usage",
        ],
        FixCategory.SECURITY: [
            r"security",
            r"injection",
            r"XSS",
            r"CSRF",
            r"vulnerable",
        ],
    }

    @classmethod
    def categorize(cls, error_message: str) -> FixCategory:
        """Categorize an error message."""
        error_lower = error_message.lower()

        for category, patterns in cls.ERROR_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, error_message, re.IGNORECASE):
                    return category

        return FixCategory.UNKNOWN

    @classmethod
    def create_signature(cls, error_message: str, code_context: str = "") -> str:
        """Create a unique signature for an error."""
        # Extract key parts of the error
        error_type = ""
        if ":" in error_message:
            error_type = error_message.split(":")[0].strip()

        # Normalize the message
        normalized = re.sub(r'\d+', 'N', error_message)  # Replace numbers
        normalized = re.sub(r"'[^']*'", "'X'", normalized)  # Replace quoted strings
        normalized = re.sub(r'"[^"]*"', '"X"', normalized)  # Replace double-quoted strings
        normalized = re.sub(r'\s+', ' ', normalized).strip()[:200]  # Normalize whitespace

        # Create hash
        signature = hashlib.sha256(
            f"{error_type}:{normalized}".encode()
        ).hexdigest()[:32]

        return signature


class SelfOptimizer:
    """
    Main self-optimization engine.
    Learns from interactions, tracks success/failure, and can update its own code.
    """

    def __init__(self, enabled: bool = False, allow_self_update: bool = False):
        self.enabled = enabled
        self.allow_self_update = allow_self_update
        self.db = LearningDatabase() if enabled else None
        self.console = Console()
        self.current_trial: Optional[TrialRecord] = None
        self._start_time: Optional[datetime] = None

        # Ensure backup directory exists
        if allow_self_update:
            SELF_UPDATE_BACKUP.mkdir(parents=True, exist_ok=True)
            STRATEGIES_PATH.mkdir(parents=True, exist_ok=True)

    def start_trial(self, error_message: str, file_path: str,
                   original_code: str, ai_provider: str = "") -> str:
        """Start tracking a new fix trial."""
        if not self.enabled:
            return ""

        category = ErrorAnalyzer.categorize(error_message)
        signature = ErrorAnalyzer.create_signature(error_message, original_code)

        self.current_trial = TrialRecord(
            timestamp=datetime.now().isoformat(),
            category=category.value,
            error_signature=signature,
            error_message=error_message[:1000],
            file_path=file_path,
            original_code=original_code[:5000],
            ai_provider=ai_provider,
        )
        self._start_time = datetime.now()

        # Look for similar past trials
        similar = self.db.get_similar_trials(signature, limit=3)
        if similar:
            best = similar[0]
            self.console.print(f"[dim]Found {len(similar)} similar past trials. "
                             f"Best outcome: {best.outcome_rating}/5[/dim]")

            if best.fix_strategy:
                self.console.print(f"[dim]Recommended strategy: {best.fix_strategy}[/dim]")
                self.current_trial.fix_strategy = best.fix_strategy

        return signature

    def end_trial(self, fix_applied: str, tests_passed: bool,
                 build_passed: bool, notes: str = "") -> Optional[str]:
        """End the current trial and record results."""
        if not self.enabled or not self.current_trial:
            return None

        elapsed = (datetime.now() - self._start_time).total_seconds() if self._start_time else 0

        self.current_trial.fix_applied = fix_applied[:5000]
        self.current_trial.time_to_fix_seconds = elapsed
        self.current_trial.tests_passed = tests_passed
        self.current_trial.build_passed = build_passed
        self.current_trial.notes = notes

        # Calculate initial rating based on time and success
        if tests_passed and build_passed:
            if elapsed < 30:
                rating = OutcomeRating.EXCELLENT
            elif elapsed < 120:
                rating = OutcomeRating.GOOD
            else:
                rating = OutcomeRating.ACCEPTABLE
        elif tests_passed or build_passed:
            rating = OutcomeRating.POOR
        else:
            rating = OutcomeRating.FAILED

        self.current_trial.outcome_rating = rating.value

        trial_id = self.db.record_trial(self.current_trial)

        # Learn from this trial if successful
        if tests_passed and build_passed:
            self._learn_from_trial(self.current_trial)

        self.current_trial = None
        self._start_time = None

        return trial_id

    def _learn_from_trial(self, trial: TrialRecord):
        """Extract learnings from a successful trial."""
        # Check if we should create a new strategy
        existing = self.db.get_best_strategy(trial.category, trial.error_signature)

        if existing:
            # Update existing strategy stats
            self.db.update_strategy_stats(
                existing.id,
                trial.tests_passed and trial.build_passed,
                trial.time_to_fix_seconds
            )
        elif trial.outcome_rating >= OutcomeRating.GOOD.value:
            # Create new strategy from this trial
            strategy = Strategy(
                name=f"Auto-learned: {trial.category}",
                category=trial.category,
                error_pattern=trial.error_signature,
                fix_template=trial.fix_applied[:2000],
                success_rate=1.0,
                avg_time_seconds=trial.time_to_fix_seconds,
                usage_count=1,
                last_used=datetime.now().isoformat(),
                source="learned",
            )
            self.db.save_strategy(strategy)

    def get_recommended_fix(self, error_message: str, code: str) -> Optional[str]:
        """Get a recommended fix based on past learnings."""
        if not self.enabled:
            return None

        category = ErrorAnalyzer.categorize(error_message)
        signature = ErrorAnalyzer.create_signature(error_message, code)

        # Try to find a matching strategy
        strategy = self.db.get_best_strategy(category.value, signature)
        if strategy and strategy.success_rate > 0.7:
            return strategy.fix_template

        # Look at similar past trials
        similar = self.db.get_similar_trials(signature, limit=1)
        if similar and similar[0].outcome_rating >= OutcomeRating.GOOD.value:
            return similar[0].fix_applied

        return None

    def rate_past_fix(self, trial_id: str, still_working: bool, days_later: int):
        """Update rating for a past fix based on long-term outcome."""
        if not self.enabled:
            return

        cursor = self.db.conn.execute(
            "SELECT outcome_rating FROM trials WHERE id = ?", (trial_id,)
        )
        row = cursor.fetchone()
        if not row:
            return

        old_rating = row[0]

        if still_working and days_later > 7:
            # Upgrade rating if still working after a week
            new_rating = min(old_rating + 1, OutcomeRating.EXCELLENT.value)
        elif not still_working:
            # Downgrade if it broke
            new_rating = max(old_rating - 2, OutcomeRating.FAILED.value)
            self.db.conn.execute(
                "UPDATE trials SET reverted = 1 WHERE id = ?", (trial_id,)
            )
        else:
            new_rating = old_rating

        self.db.conn.execute("""
            UPDATE trials
            SET outcome_rating = ?, still_working_after_days = ?
            WHERE id = ?
        """, (new_rating, days_later, trial_id))
        self.db.conn.commit()

    def show_stats(self):
        """Display learning statistics."""
        if not self.enabled:
            self.console.print("[yellow]Self-optimization is not enabled[/yellow]")
            return

        stats = self.db.get_learning_stats()

        self.console.print(Panel(
            f"[bold]Learning Database Statistics[/bold]\n\n"
            f"Total trials: {stats['total_trials']}\n"
            f"Overall success rate: {stats['overall_success_rate']:.1%}\n"
            f"Average fix time: {stats['avg_fix_time_seconds']:.1f}s\n"
            f"Total strategies learned: {stats['total_strategies']}",
            title="Self-Optimization Stats",
            border_style="cyan"
        ))

        if stats["best_categories"]:
            table = Table(title="Best Performing Categories")
            table.add_column("Category")
            table.add_column("Avg Rating")
            table.add_column("Count")

            for cat in stats["best_categories"]:
                table.add_row(
                    cat["category"],
                    f"{cat['avg_rating']:.1f}/5",
                    str(cat["count"])
                )
            self.console.print(table)

        if stats["top_strategies"]:
            table = Table(title="Top Strategies")
            table.add_column("Name")
            table.add_column("Success Rate")
            table.add_column("Uses")

            for strat in stats["top_strategies"]:
                table.add_row(
                    strat["name"][:40],
                    f"{strat['success_rate']:.1%}",
                    str(strat["usage_count"])
                )
            self.console.print(table)

    # === Self-Update Capabilities ===

    def analyze_own_code(self, file_path: str) -> Dict[str, Any]:
        """Analyze agent's own code for potential improvements."""
        if not self.allow_self_update:
            return {"error": "Self-update not enabled"}

        try:
            with open(file_path, "r") as f:
                code = f.read()

            tree = ast.parse(code)

            analysis = {
                "file": file_path,
                "functions": [],
                "classes": [],
                "complexity_warnings": [],
                "improvement_suggestions": [],
            }

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_info = {
                        "name": node.name,
                        "lines": node.end_lineno - node.lineno + 1 if node.end_lineno else 0,
                        "args": len(node.args.args),
                    }
                    analysis["functions"].append(func_info)

                    # Check for overly complex functions
                    if func_info["lines"] > 50:
                        analysis["complexity_warnings"].append(
                            f"Function '{node.name}' is {func_info['lines']} lines - consider splitting"
                        )

                elif isinstance(node, ast.ClassDef):
                    methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    analysis["classes"].append({
                        "name": node.name,
                        "methods": len(methods),
                    })

            return analysis

        except Exception as e:
            return {"error": str(e)}

    def propose_self_update(self, target_file: str, change_type: str,
                           description: str, new_code: str) -> Optional[str]:
        """Propose an update to the agent's own code."""
        if not self.allow_self_update:
            self.console.print("[red]Self-update is not enabled[/red]")
            return None

        # Read original code
        try:
            with open(target_file, "r") as f:
                original_code = f.read()
        except FileNotFoundError:
            original_code = ""

        # Generate diff
        diff = difflib.unified_diff(
            original_code.splitlines(keepends=True),
            new_code.splitlines(keepends=True),
            fromfile=f"original/{target_file}",
            tofile=f"proposed/{target_file}",
        )
        diff_text = "".join(diff)

        if not diff_text:
            self.console.print("[yellow]No changes detected[/yellow]")
            return None

        # Record the proposal
        update_id = self.db.record_self_update(
            target_file, change_type, description, original_code, new_code
        )

        # Display proposal
        self.console.print(Panel(
            f"[bold]Self-Update Proposal[/bold]\n\n"
            f"ID: {update_id}\n"
            f"Target: {target_file}\n"
            f"Type: {change_type}\n"
            f"Description: {description}",
            title="Proposed Change",
            border_style="yellow"
        ))

        from rich.syntax import Syntax
        self.console.print(Syntax(diff_text, "diff", theme="monokai"))

        return update_id

    def apply_self_update(self, update_id: str, approved: bool = False) -> bool:
        """Apply a proposed self-update."""
        if not self.allow_self_update:
            return False

        cursor = self.db.conn.execute(
            "SELECT * FROM self_updates WHERE id = ?", (update_id,)
        )
        row = cursor.fetchone()
        if not row:
            self.console.print(f"[red]Update {update_id} not found[/red]")
            return False

        if not approved:
            self.console.print("[yellow]Update not approved[/yellow]")
            return False

        target_file = row["target_file"]
        new_code = row["new_code"]
        original_code = row["original_code"]

        # Create backup
        backup_path = SELF_UPDATE_BACKUP / f"{Path(target_file).name}.{update_id}.backup"
        try:
            shutil.copy2(target_file, backup_path)
        except FileNotFoundError:
            pass  # New file

        # Apply update
        try:
            with open(target_file, "w") as f:
                f.write(new_code)

            # Verify syntax
            ast.parse(new_code)

            # Mark as applied
            self.db.conn.execute(
                "UPDATE self_updates SET approved = 1, applied = 1 WHERE id = ?",
                (update_id,)
            )
            self.db.conn.commit()

            self.console.print(f"[green]Self-update {update_id} applied successfully[/green]")
            self.console.print(f"[dim]Backup saved to: {backup_path}[/dim]")
            return True

        except SyntaxError as e:
            self.console.print(f"[red]Syntax error in proposed code: {e}[/red]")
            # Restore backup
            if backup_path.exists():
                shutil.copy2(backup_path, target_file)
            return False

        except Exception as e:
            self.console.print(f"[red]Failed to apply update: {e}[/red]")
            if backup_path.exists():
                shutil.copy2(backup_path, target_file)
            return False

    def rollback_update(self, update_id: str) -> bool:
        """Rollback a previously applied self-update."""
        if not self.allow_self_update:
            return False

        cursor = self.db.conn.execute(
            "SELECT target_file, original_code, applied FROM self_updates WHERE id = ?",
            (update_id,)
        )
        row = cursor.fetchone()
        if not row:
            return False

        if not row["applied"]:
            self.console.print("[yellow]Update was not applied[/yellow]")
            return False

        try:
            with open(row["target_file"], "w") as f:
                f.write(row["original_code"])

            self.db.conn.execute(
                "UPDATE self_updates SET applied = 0 WHERE id = ?", (update_id,)
            )
            self.db.conn.commit()

            self.console.print(f"[green]Rolled back update {update_id}[/green]")
            return True

        except Exception as e:
            self.console.print(f"[red]Rollback failed: {e}[/red]")
            return False

    def optimize_self(self, target_files: List[str] = None) -> Dict[str, Any]:
        """
        Run self-optimization as a main task.
        Analyzes agent code and proposes improvements.
        """
        if not self.allow_self_update:
            return {"error": "Self-update not enabled"}

        if target_files is None:
            # Default to analyzing agent's own modules
            target_files = [
                "agent_engine.py",
                "ai_providers.py",
                "local_ai.py",
                "orchestrator.py",
                "self_optimization.py",
            ]

        results = {
            "analyzed": [],
            "proposals": [],
            "improvements_found": 0,
        }

        for file_path in target_files:
            if not Path(file_path).exists():
                continue

            analysis = self.analyze_own_code(file_path)
            results["analyzed"].append(analysis)

            if analysis.get("complexity_warnings"):
                results["improvements_found"] += len(analysis["complexity_warnings"])
                for warning in analysis["complexity_warnings"]:
                    results["proposals"].append({
                        "file": file_path,
                        "type": "complexity",
                        "suggestion": warning,
                    })

        # Display results
        self.console.print(Panel(
            f"[bold]Self-Optimization Analysis[/bold]\n\n"
            f"Files analyzed: {len(results['analyzed'])}\n"
            f"Improvements found: {results['improvements_found']}\n"
            f"Proposals generated: {len(results['proposals'])}",
            title="Self-Optimization Results",
            border_style="cyan"
        ))

        return results


# Global instance
_self_optimizer: Optional[SelfOptimizer] = None


def get_self_optimizer() -> SelfOptimizer:
    """Get or create the global self-optimizer instance."""
    global _self_optimizer
    if _self_optimizer is None:
        _self_optimizer = SelfOptimizer(enabled=False)
    return _self_optimizer


def init_self_optimizer(enabled: bool = True, allow_self_update: bool = False) -> SelfOptimizer:
    """Initialize self-optimizer with settings."""
    global _self_optimizer
    _self_optimizer = SelfOptimizer(enabled=enabled, allow_self_update=allow_self_update)
    return _self_optimizer

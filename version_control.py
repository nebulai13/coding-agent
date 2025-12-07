"""
Version Control System for Coding Agent
EXPERIMENTAL: Track code versions, rate effectiveness, rollback changes.

Provides:
- Version snapshots with metadata
- Rating system for version effectiveness
- Smart rollback to best-performing versions
- Version comparison and diff generation
- Integration with learning database
"""
import os
import re
import json
import sqlite3
import hashlib
import shutil
import difflib
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax


# Storage paths
VERSION_DB_PATH = Path(".cache/versions.db")
SNAPSHOTS_PATH = Path(".cache/snapshots")


class VersionRating(Enum):
    """Rating levels for version effectiveness."""
    EXCELLENT = 5    # Solved problem quickly, still working after weeks
    GOOD = 4         # Worked well, no issues
    ACCEPTABLE = 3   # Works but has minor issues
    POOR = 2         # Caused problems, needed revision
    BROKEN = 1       # Completely broken, immediately reverted
    UNRATED = 0      # Not yet rated


class VersionType(Enum):
    """Types of version changes."""
    FIX = "fix"                    # Bug fix
    FEATURE = "feature"            # New feature
    OPTIMIZATION = "optimization"  # Performance improvement
    REFACTOR = "refactor"          # Code refactoring
    ROLLBACK = "rollback"          # Rollback to previous version
    EXPERIMENT = "experiment"      # Experimental change
    STYLE = "style"                # Style/formatting change


@dataclass
class CodeVersion:
    """Represents a version of a code file."""
    id: str = ""
    file_path: str = ""
    version_number: int = 0
    timestamp: str = ""
    content_hash: str = ""
    change_type: str = VersionType.FIX.value
    description: str = ""
    rating: int = VersionRating.UNRATED.value
    rating_reason: str = ""
    rated_at: str = ""
    parent_version_id: str = ""
    is_current: bool = False

    # Effectiveness tracking
    tests_passed: bool = False
    build_passed: bool = False
    runtime_errors: int = 0
    days_stable: int = 0
    times_rolled_back_from: int = 0
    times_rolled_back_to: int = 0

    # Context
    error_fixed: str = ""
    ai_provider: str = ""
    strategy_used: str = ""

    # Metadata
    lines_added: int = 0
    lines_removed: int = 0
    complexity_delta: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def effectiveness_score(self) -> float:
        """Calculate an effectiveness score based on various factors."""
        score = 0.0

        # Base rating contribution (0-50)
        score += (self.rating / 5.0) * 50

        # Test/build success (0-20)
        if self.tests_passed:
            score += 10
        if self.build_passed:
            score += 10

        # Stability bonus (0-20)
        stability_bonus = min(self.days_stable * 2, 20)
        score += stability_bonus

        # Rollback penalty (-10 per rollback from this version)
        score -= self.times_rolled_back_from * 10

        # Rollback-to bonus (indicates this was a good version to return to)
        score += self.times_rolled_back_to * 5

        return max(0.0, min(100.0, score))


@dataclass
class VersionComparison:
    """Comparison between two versions."""
    version_a: str
    version_b: str
    diff_text: str = ""
    lines_changed: int = 0
    lines_added: int = 0
    lines_removed: int = 0
    better_version: str = ""
    comparison_reason: str = ""


class VersionDatabase:
    """SQLite database for version tracking."""

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or VERSION_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS versions (
                id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                version_number INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                change_type TEXT,
                description TEXT,
                rating INTEGER DEFAULT 0,
                rating_reason TEXT,
                rated_at TEXT,
                parent_version_id TEXT,
                is_current INTEGER DEFAULT 0,
                tests_passed INTEGER DEFAULT 0,
                build_passed INTEGER DEFAULT 0,
                runtime_errors INTEGER DEFAULT 0,
                days_stable INTEGER DEFAULT 0,
                times_rolled_back_from INTEGER DEFAULT 0,
                times_rolled_back_to INTEGER DEFAULT 0,
                error_fixed TEXT,
                ai_provider TEXT,
                strategy_used TEXT,
                lines_added INTEGER DEFAULT 0,
                lines_removed INTEGER DEFAULT 0,
                complexity_delta REAL DEFAULT 0.0,
                UNIQUE(file_path, version_number)
            );

            CREATE TABLE IF NOT EXISTS version_snapshots (
                version_id TEXT PRIMARY KEY,
                content TEXT,
                FOREIGN KEY (version_id) REFERENCES versions(id)
            );

            CREATE TABLE IF NOT EXISTS version_relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_version TEXT NOT NULL,
                to_version TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                notes TEXT,
                FOREIGN KEY (from_version) REFERENCES versions(id),
                FOREIGN KEY (to_version) REFERENCES versions(id)
            );

            CREATE TABLE IF NOT EXISTS rating_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version_id TEXT NOT NULL,
                old_rating INTEGER,
                new_rating INTEGER,
                reason TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (version_id) REFERENCES versions(id)
            );

            CREATE INDEX IF NOT EXISTS idx_versions_file ON versions(file_path);
            CREATE INDEX IF NOT EXISTS idx_versions_rating ON versions(rating);
            CREATE INDEX IF NOT EXISTS idx_versions_current ON versions(is_current);
        """)
        self.conn.commit()

    def save_version(self, version: CodeVersion, content: str) -> str:
        """Save a new version."""
        if not version.id:
            version.id = hashlib.sha256(
                f"{version.file_path}{version.timestamp}{version.content_hash}".encode()
            ).hexdigest()[:16]

        # Insert version record
        self.conn.execute("""
            INSERT OR REPLACE INTO versions VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, (
            version.id, version.file_path, version.version_number,
            version.timestamp, version.content_hash, version.change_type,
            version.description, version.rating, version.rating_reason,
            version.rated_at, version.parent_version_id, int(version.is_current),
            int(version.tests_passed), int(version.build_passed),
            version.runtime_errors, version.days_stable,
            version.times_rolled_back_from, version.times_rolled_back_to,
            version.error_fixed, version.ai_provider, version.strategy_used,
            version.lines_added, version.lines_removed, version.complexity_delta
        ))

        # Store content snapshot
        self.conn.execute("""
            INSERT OR REPLACE INTO version_snapshots VALUES (?, ?)
        """, (version.id, content))

        self.conn.commit()
        return version.id

    def get_version(self, version_id: str) -> Optional[CodeVersion]:
        """Get a version by ID."""
        cursor = self.conn.execute(
            "SELECT * FROM versions WHERE id = ?", (version_id,)
        )
        row = cursor.fetchone()
        if row:
            data = dict(row)
            data["is_current"] = bool(data["is_current"])
            data["tests_passed"] = bool(data["tests_passed"])
            data["build_passed"] = bool(data["build_passed"])
            return CodeVersion(**data)
        return None

    def get_version_content(self, version_id: str) -> Optional[str]:
        """Get the content of a version."""
        cursor = self.conn.execute(
            "SELECT content FROM version_snapshots WHERE version_id = ?",
            (version_id,)
        )
        row = cursor.fetchone()
        return row["content"] if row else None

    def get_file_versions(self, file_path: str, limit: int = 50) -> List[CodeVersion]:
        """Get all versions for a file."""
        cursor = self.conn.execute("""
            SELECT * FROM versions
            WHERE file_path = ?
            ORDER BY version_number DESC
            LIMIT ?
        """, (file_path, limit))

        versions = []
        for row in cursor.fetchall():
            data = dict(row)
            data["is_current"] = bool(data["is_current"])
            data["tests_passed"] = bool(data["tests_passed"])
            data["build_passed"] = bool(data["build_passed"])
            versions.append(CodeVersion(**data))

        return versions

    def get_current_version(self, file_path: str) -> Optional[CodeVersion]:
        """Get the current version of a file."""
        cursor = self.conn.execute("""
            SELECT * FROM versions
            WHERE file_path = ? AND is_current = 1
        """, (file_path,))
        row = cursor.fetchone()
        if row:
            data = dict(row)
            data["is_current"] = bool(data["is_current"])
            data["tests_passed"] = bool(data["tests_passed"])
            data["build_passed"] = bool(data["build_passed"])
            return CodeVersion(**data)
        return None

    def set_current_version(self, file_path: str, version_id: str):
        """Set the current version for a file."""
        self.conn.execute(
            "UPDATE versions SET is_current = 0 WHERE file_path = ?",
            (file_path,)
        )
        self.conn.execute(
            "UPDATE versions SET is_current = 1 WHERE id = ?",
            (version_id,)
        )
        self.conn.commit()

    def rate_version(self, version_id: str, rating: int, reason: str = ""):
        """Rate a version."""
        # Get old rating
        cursor = self.conn.execute(
            "SELECT rating FROM versions WHERE id = ?", (version_id,)
        )
        row = cursor.fetchone()
        old_rating = row["rating"] if row else 0

        # Update rating
        self.conn.execute("""
            UPDATE versions
            SET rating = ?, rating_reason = ?, rated_at = ?
            WHERE id = ?
        """, (rating, reason, datetime.now().isoformat(), version_id))

        # Record rating history
        self.conn.execute("""
            INSERT INTO rating_history (version_id, old_rating, new_rating, reason, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (version_id, old_rating, rating, reason, datetime.now().isoformat()))

        self.conn.commit()

    def record_rollback(self, from_version_id: str, to_version_id: str, notes: str = ""):
        """Record a rollback action."""
        # Increment rollback counters
        self.conn.execute(
            "UPDATE versions SET times_rolled_back_from = times_rolled_back_from + 1 WHERE id = ?",
            (from_version_id,)
        )
        self.conn.execute(
            "UPDATE versions SET times_rolled_back_to = times_rolled_back_to + 1 WHERE id = ?",
            (to_version_id,)
        )

        # Record relationship
        self.conn.execute("""
            INSERT INTO version_relationships (from_version, to_version, relationship_type, timestamp, notes)
            VALUES (?, ?, 'rollback', ?, ?)
        """, (from_version_id, to_version_id, datetime.now().isoformat(), notes))

        self.conn.commit()

    def get_best_versions(self, file_path: str, limit: int = 5) -> List[CodeVersion]:
        """Get the best-rated versions for a file."""
        cursor = self.conn.execute("""
            SELECT * FROM versions
            WHERE file_path = ? AND rating > 0
            ORDER BY rating DESC, days_stable DESC, times_rolled_back_to DESC
            LIMIT ?
        """, (file_path, limit))

        versions = []
        for row in cursor.fetchall():
            data = dict(row)
            data["is_current"] = bool(data["is_current"])
            data["tests_passed"] = bool(data["tests_passed"])
            data["build_passed"] = bool(data["build_passed"])
            versions.append(CodeVersion(**data))

        return versions

    def update_stability(self, version_id: str, days_stable: int):
        """Update the stability days for a version."""
        self.conn.execute(
            "UPDATE versions SET days_stable = ? WHERE id = ?",
            (days_stable, version_id)
        )
        self.conn.commit()

    def get_stats(self) -> Dict[str, Any]:
        """Get version control statistics."""
        stats = {}

        cursor = self.conn.execute("SELECT COUNT(*) FROM versions")
        stats["total_versions"] = cursor.fetchone()[0]

        cursor = self.conn.execute("SELECT COUNT(DISTINCT file_path) FROM versions")
        stats["tracked_files"] = cursor.fetchone()[0]

        cursor = self.conn.execute("SELECT AVG(rating) FROM versions WHERE rating > 0")
        stats["avg_rating"] = cursor.fetchone()[0] or 0.0

        cursor = self.conn.execute("""
            SELECT change_type, COUNT(*) as count
            FROM versions
            GROUP BY change_type
            ORDER BY count DESC
        """)
        stats["by_type"] = {row["change_type"]: row["count"] for row in cursor.fetchall()}

        cursor = self.conn.execute("""
            SELECT rating, COUNT(*) as count
            FROM versions WHERE rating > 0
            GROUP BY rating
            ORDER BY rating DESC
        """)
        stats["by_rating"] = {row["rating"]: row["count"] for row in cursor.fetchall()}

        return stats

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


class VersionControlManager:
    """High-level version control manager."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.db = VersionDatabase() if enabled else None
        self.console = Console()
        SNAPSHOTS_PATH.mkdir(parents=True, exist_ok=True)

    def _compute_hash(self, content: str) -> str:
        """Compute hash of content."""
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def _get_next_version_number(self, file_path: str) -> int:
        """Get the next version number for a file."""
        if not self.db:
            return 1

        versions = self.db.get_file_versions(file_path, limit=1)
        if versions:
            return versions[0].version_number + 1
        return 1

    def _compute_diff_stats(self, old_content: str, new_content: str) -> Tuple[int, int]:
        """Compute lines added and removed."""
        old_lines = old_content.splitlines()
        new_lines = new_content.splitlines()

        matcher = difflib.SequenceMatcher(None, old_lines, new_lines)

        added = 0
        removed = 0

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'insert':
                added += j2 - j1
            elif tag == 'delete':
                removed += i2 - i1
            elif tag == 'replace':
                removed += i2 - i1
                added += j2 - j1

        return added, removed

    def create_version(self, file_path: str, content: str,
                      change_type: VersionType, description: str,
                      error_fixed: str = "", ai_provider: str = "",
                      strategy_used: str = "") -> Optional[str]:
        """Create a new version of a file."""
        if not self.enabled:
            return None

        # Get current version for parent reference
        current = self.db.get_current_version(file_path)
        parent_id = current.id if current else ""

        # Get old content for diff stats
        old_content = ""
        if current:
            old_content = self.db.get_version_content(current.id) or ""

        lines_added, lines_removed = self._compute_diff_stats(old_content, content)

        version = CodeVersion(
            file_path=file_path,
            version_number=self._get_next_version_number(file_path),
            timestamp=datetime.now().isoformat(),
            content_hash=self._compute_hash(content),
            change_type=change_type.value,
            description=description,
            parent_version_id=parent_id,
            is_current=True,
            error_fixed=error_fixed,
            ai_provider=ai_provider,
            strategy_used=strategy_used,
            lines_added=lines_added,
            lines_removed=lines_removed,
        )

        version_id = self.db.save_version(version, content)
        self.db.set_current_version(file_path, version_id)

        self.console.print(f"[green]Created version {version.version_number} for {file_path}[/green]")

        return version_id

    def snapshot_current(self, file_path: str, description: str = "") -> Optional[str]:
        """Create a snapshot of the current file state."""
        if not self.enabled:
            return None

        path = Path(file_path)
        if not path.exists():
            self.console.print(f"[red]File not found: {file_path}[/red]")
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            self.console.print(f"[red]Error reading file: {e}[/red]")
            return None

        return self.create_version(
            file_path=str(path.absolute()),
            content=content,
            change_type=VersionType.FIX,
            description=description or f"Snapshot at {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )

    def rate_version(self, version_id: str, rating: VersionRating,
                    reason: str = "") -> bool:
        """Rate a version's effectiveness."""
        if not self.enabled:
            return False

        version = self.db.get_version(version_id)
        if not version:
            self.console.print(f"[red]Version not found: {version_id}[/red]")
            return False

        self.db.rate_version(version_id, rating.value, reason)

        rating_names = {
            5: "EXCELLENT",
            4: "GOOD",
            3: "ACCEPTABLE",
            2: "POOR",
            1: "BROKEN",
        }

        self.console.print(
            f"[green]Rated version {version.version_number} as "
            f"{rating_names.get(rating.value, 'UNKNOWN')}[/green]"
        )

        return True

    def rollback(self, file_path: str, target_version_id: str = None,
                to_best: bool = False) -> bool:
        """Rollback to a previous version."""
        if not self.enabled:
            return False

        current = self.db.get_current_version(file_path)
        if not current:
            self.console.print(f"[red]No versions found for: {file_path}[/red]")
            return False

        # Determine target version
        if to_best:
            best = self.db.get_best_versions(file_path, limit=1)
            if not best:
                self.console.print("[yellow]No rated versions to rollback to[/yellow]")
                return False
            target = best[0]
        elif target_version_id:
            target = self.db.get_version(target_version_id)
            if not target:
                self.console.print(f"[red]Version not found: {target_version_id}[/red]")
                return False
        else:
            # Rollback to previous version
            versions = self.db.get_file_versions(file_path, limit=2)
            if len(versions) < 2:
                self.console.print("[yellow]No previous version to rollback to[/yellow]")
                return False
            target = versions[1]

        # Get target content
        content = self.db.get_version_content(target.id)
        if not content:
            self.console.print("[red]Could not retrieve version content[/red]")
            return False

        # Apply rollback
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            self.console.print(f"[red]Error applying rollback: {e}[/red]")
            return False

        # Record rollback
        self.db.record_rollback(current.id, target.id,
                                f"Rolled back from v{current.version_number} to v{target.version_number}")

        # Create new version entry for the rollback
        self.create_version(
            file_path=file_path,
            content=content,
            change_type=VersionType.ROLLBACK,
            description=f"Rollback to version {target.version_number}"
        )

        self.console.print(
            f"[green]Rolled back to version {target.version_number} "
            f"(rating: {target.rating}/5)[/green]"
        )

        return True

    def compare_versions(self, version_a_id: str, version_b_id: str) -> Optional[VersionComparison]:
        """Compare two versions."""
        if not self.enabled:
            return None

        version_a = self.db.get_version(version_a_id)
        version_b = self.db.get_version(version_b_id)

        if not version_a or not version_b:
            return None

        content_a = self.db.get_version_content(version_a_id) or ""
        content_b = self.db.get_version_content(version_b_id) or ""

        diff = difflib.unified_diff(
            content_a.splitlines(keepends=True),
            content_b.splitlines(keepends=True),
            fromfile=f"v{version_a.version_number}",
            tofile=f"v{version_b.version_number}"
        )
        diff_text = "".join(diff)

        added, removed = self._compute_diff_stats(content_a, content_b)

        # Determine better version
        score_a = version_a.effectiveness_score
        score_b = version_b.effectiveness_score

        if score_a > score_b:
            better = version_a_id
            reason = f"v{version_a.version_number} has higher effectiveness ({score_a:.1f} vs {score_b:.1f})"
        elif score_b > score_a:
            better = version_b_id
            reason = f"v{version_b.version_number} has higher effectiveness ({score_b:.1f} vs {score_a:.1f})"
        else:
            better = ""
            reason = "Both versions have equal effectiveness scores"

        return VersionComparison(
            version_a=version_a_id,
            version_b=version_b_id,
            diff_text=diff_text,
            lines_changed=added + removed,
            lines_added=added,
            lines_removed=removed,
            better_version=better,
            comparison_reason=reason,
        )

    def show_history(self, file_path: str, limit: int = 10):
        """Display version history for a file."""
        if not self.enabled:
            self.console.print("[yellow]Version control is not enabled[/yellow]")
            return

        versions = self.db.get_file_versions(file_path, limit=limit)

        if not versions:
            self.console.print(f"[yellow]No versions found for: {file_path}[/yellow]")
            return

        table = Table(title=f"Version History: {Path(file_path).name}")
        table.add_column("V#", width=4)
        table.add_column("Date", width=16)
        table.add_column("Type", width=12)
        table.add_column("Rating", width=8)
        table.add_column("Score", width=7)
        table.add_column("Description", width=35)
        table.add_column("", width=3)

        rating_colors = {5: "green", 4: "green", 3: "yellow", 2: "red", 1: "red", 0: "dim"}

        for v in versions:
            rating_str = f"{v.rating}/5" if v.rating > 0 else "-"
            rating_color = rating_colors.get(v.rating, "white")
            current_marker = "[cyan]*[/cyan]" if v.is_current else ""

            table.add_row(
                str(v.version_number),
                v.timestamp[:16] if v.timestamp else "-",
                v.change_type,
                f"[{rating_color}]{rating_str}[/{rating_color}]",
                f"{v.effectiveness_score:.0f}",
                v.description[:32] + "..." if len(v.description) > 35 else v.description,
                current_marker,
            )

        self.console.print(table)
        self.console.print("[dim]* = current version[/dim]")

    def show_diff(self, version_a_id: str, version_b_id: str):
        """Display diff between two versions."""
        comparison = self.compare_versions(version_a_id, version_b_id)

        if not comparison:
            self.console.print("[red]Could not compare versions[/red]")
            return

        self.console.print(Panel(
            f"Lines added: +{comparison.lines_added}\n"
            f"Lines removed: -{comparison.lines_removed}\n"
            f"Better version: {comparison.better_version or 'equal'}\n"
            f"Reason: {comparison.comparison_reason}",
            title="Comparison",
            border_style="cyan"
        ))

        if comparison.diff_text:
            syntax = Syntax(comparison.diff_text, "diff", theme="monokai")
            self.console.print(syntax)

    def show_stats(self):
        """Display version control statistics."""
        if not self.enabled:
            self.console.print("[yellow]Version control is not enabled[/yellow]")
            return

        stats = self.db.get_stats()

        self.console.print(Panel(
            f"Total versions: {stats['total_versions']}\n"
            f"Tracked files: {stats['tracked_files']}\n"
            f"Average rating: {stats['avg_rating']:.1f}/5",
            title="Version Control Statistics",
            border_style="cyan"
        ))

        if stats["by_type"]:
            table = Table(title="Versions by Type")
            table.add_column("Type")
            table.add_column("Count")

            for type_name, count in stats["by_type"].items():
                table.add_row(type_name, str(count))

            self.console.print(table)

        if stats["by_rating"]:
            table = Table(title="Versions by Rating")
            table.add_column("Rating")
            table.add_column("Count")

            rating_names = {5: "Excellent", 4: "Good", 3: "Acceptable", 2: "Poor", 1: "Broken"}
            for rating, count in stats["by_rating"].items():
                table.add_row(rating_names.get(rating, str(rating)), str(count))

            self.console.print(table)

    def get_best_version_for_error(self, file_path: str, error_type: str) -> Optional[CodeVersion]:
        """Find the best version that fixed a similar error."""
        if not self.enabled:
            return None

        # Search for versions that fixed similar errors
        cursor = self.db.conn.execute("""
            SELECT * FROM versions
            WHERE file_path = ? AND error_fixed LIKE ? AND rating >= 4
            ORDER BY rating DESC, days_stable DESC
            LIMIT 1
        """, (file_path, f"%{error_type}%"))

        row = cursor.fetchone()
        if row:
            data = dict(row)
            data["is_current"] = bool(data["is_current"])
            data["tests_passed"] = bool(data["tests_passed"])
            data["build_passed"] = bool(data["build_passed"])
            return CodeVersion(**data)

        return None

    def auto_rate_old_versions(self):
        """Automatically upgrade ratings for stable old versions."""
        if not self.enabled:
            return

        # Find versions that have been stable for a while
        threshold_date = (datetime.now() - timedelta(days=7)).isoformat()

        cursor = self.db.conn.execute("""
            SELECT id, rating, timestamp FROM versions
            WHERE rating BETWEEN 3 AND 4
            AND timestamp < ?
            AND times_rolled_back_from = 0
        """, (threshold_date,))

        upgraded = 0
        for row in cursor.fetchall():
            # Calculate days stable
            created = datetime.fromisoformat(row["timestamp"])
            days_stable = (datetime.now() - created).days

            self.db.update_stability(row["id"], days_stable)

            # Upgrade rating if very stable
            if days_stable >= 7 and row["rating"] < 5:
                new_rating = min(row["rating"] + 1, 5)
                self.db.rate_version(
                    row["id"], new_rating,
                    f"Auto-upgraded after {days_stable} days of stability"
                )
                upgraded += 1

        if upgraded:
            self.console.print(f"[dim]Auto-upgraded {upgraded} stable versions[/dim]")


# Global instance
_version_manager: Optional[VersionControlManager] = None


def get_version_manager() -> VersionControlManager:
    """Get or create the global version manager."""
    global _version_manager
    if _version_manager is None:
        _version_manager = VersionControlManager(enabled=True)
    return _version_manager


def init_version_control(enabled: bool = True) -> VersionControlManager:
    """Initialize version control manager."""
    global _version_manager
    _version_manager = VersionControlManager(enabled=enabled)
    return _version_manager

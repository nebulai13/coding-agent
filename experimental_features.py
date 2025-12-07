"""
Experimental Features Toggle System for Coding Agent
Manages experimental features that can be enabled/disabled.

WARNING: Experimental features may be unstable and change without notice.
"""
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

from rich.console import Console
from rich.table import Table
from rich.panel import Panel


# Configuration file
EXPERIMENTAL_CONFIG = Path(".cache/experimental.json")


class FeatureStatus(Enum):
    """Status of an experimental feature."""
    DISABLED = "disabled"
    ENABLED = "enabled"
    TESTING = "testing"  # Enabled but logging extra debug info


@dataclass
class ExperimentalFeature:
    """Definition of an experimental feature."""
    name: str
    description: str
    status: str = FeatureStatus.DISABLED.value
    enabled_at: Optional[str] = None
    warning: str = ""
    requires: List[str] = field(default_factory=list)  # Other features this depends on
    conflicts: List[str] = field(default_factory=list)  # Features this conflicts with
    dangerous: bool = False  # If True, requires extra confirmation
    version_added: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Feature definitions
FEATURE_DEFINITIONS = {
    "self_optimization": ExperimentalFeature(
        name="self_optimization",
        description="Learn from interactions, track trial/error success rates, "
                   "and use past learnings to improve future fixes.",
        warning="Stores interaction history in a local database.",
        requires=[],
        conflicts=[],
        dangerous=False,
        version_added="1.1.0",
    ),
    "self_update": ExperimentalFeature(
        name="self_update",
        description="Allow the agent to modify its own code to implement "
                   "new strategies and improve performance.",
        warning="DANGEROUS: Allows code self-modification. Always creates backups. "
               "Review proposed changes carefully before approving.",
        requires=["self_optimization"],
        conflicts=[],
        dangerous=True,
        version_added="1.1.0",
    ),
    "contextualization": ExperimentalFeature(
        name="contextualization",
        description="Build rich context from web sources, user preferences, "
                   "and code in ./ and ../ directories.",
        warning="Makes web requests to gather documentation and examples. "
               "Caches results locally.",
        requires=[],
        conflicts=[],
        dangerous=False,
        version_added="1.1.0",
    ),
    "web_context": ExperimentalFeature(
        name="web_context",
        description="Gather context from GitHub, StackOverflow, and documentation sites.",
        warning="Makes external API calls.",
        requires=["contextualization"],
        conflicts=[],
        dangerous=False,
        version_added="1.1.0",
    ),
    "user_learning": ExperimentalFeature(
        name="user_learning",
        description="Learn user preferences, coding style, and common patterns "
                   "to provide personalized suggestions.",
        warning="Stores user interaction history locally.",
        requires=["contextualization"],
        conflicts=[],
        dangerous=False,
        version_added="1.1.0",
    ),
    "auto_strategy": ExperimentalFeature(
        name="auto_strategy",
        description="Automatically create and test new fix strategies based on "
                   "accumulated learnings.",
        warning="May propose unconventional solutions based on past successes.",
        requires=["self_optimization"],
        conflicts=[],
        dangerous=False,
        version_added="1.1.0",
    ),
    "parallel_learning": ExperimentalFeature(
        name="parallel_learning",
        description="Share learning data across multiple agent instances "
                   "for faster knowledge accumulation.",
        warning="Requires network access if sharing across machines.",
        requires=["self_optimization"],
        conflicts=[],
        dangerous=False,
        version_added="1.1.0",
    ),
}


class ExperimentalFeaturesManager:
    """Manages experimental feature toggles."""

    def __init__(self, config_path: Path = None):
        self.config_path = config_path or EXPERIMENTAL_CONFIG
        self.console = Console()
        self.features: Dict[str, ExperimentalFeature] = {}
        self._callbacks: Dict[str, List[Callable]] = {}
        self._load_config()

    def _load_config(self):
        """Load feature configuration from file."""
        # Start with defaults
        self.features = {k: ExperimentalFeature(**v.to_dict()) for k, v in FEATURE_DEFINITIONS.items()}

        # Override with saved config
        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    data = json.load(f)

                for name, saved in data.get("features", {}).items():
                    if name in self.features:
                        self.features[name].status = saved.get("status", FeatureStatus.DISABLED.value)
                        self.features[name].enabled_at = saved.get("enabled_at")

            except Exception as e:
                self.console.print(f"[yellow]Warning: Could not load experimental config: {e}[/yellow]")

    def _save_config(self):
        """Save feature configuration to file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": "1.0",
            "last_updated": datetime.now().isoformat(),
            "features": {
                name: {"status": f.status, "enabled_at": f.enabled_at}
                for name, f in self.features.items()
            }
        }

        with open(self.config_path, "w") as f:
            json.dump(data, f, indent=2)

    def is_enabled(self, feature_name: str) -> bool:
        """Check if a feature is enabled."""
        feature = self.features.get(feature_name)
        if not feature:
            return False
        return feature.status in [FeatureStatus.ENABLED.value, FeatureStatus.TESTING.value]

    def enable(self, feature_name: str, testing: bool = False,
               force: bool = False) -> tuple[bool, str]:
        """Enable an experimental feature."""
        feature = self.features.get(feature_name)
        if not feature:
            return False, f"Unknown feature: {feature_name}"

        # Check for dangerous features
        if feature.dangerous and not force:
            return False, (f"Feature '{feature_name}' is marked as DANGEROUS. "
                          f"Use --force to enable. Warning: {feature.warning}")

        # Check dependencies
        for req in feature.requires:
            if not self.is_enabled(req):
                return False, f"Feature '{feature_name}' requires '{req}' to be enabled first"

        # Check conflicts
        for conflict in feature.conflicts:
            if self.is_enabled(conflict):
                return False, f"Feature '{feature_name}' conflicts with enabled feature '{conflict}'"

        # Enable the feature
        feature.status = FeatureStatus.TESTING.value if testing else FeatureStatus.ENABLED.value
        feature.enabled_at = datetime.now().isoformat()

        self._save_config()
        self._trigger_callbacks(feature_name, "enabled")

        return True, f"Feature '{feature_name}' enabled"

    def disable(self, feature_name: str) -> tuple[bool, str]:
        """Disable an experimental feature."""
        feature = self.features.get(feature_name)
        if not feature:
            return False, f"Unknown feature: {feature_name}"

        # Check if other features depend on this
        for name, f in self.features.items():
            if feature_name in f.requires and self.is_enabled(name):
                return False, f"Cannot disable '{feature_name}': feature '{name}' depends on it"

        feature.status = FeatureStatus.DISABLED.value
        feature.enabled_at = None

        self._save_config()
        self._trigger_callbacks(feature_name, "disabled")

        return True, f"Feature '{feature_name}' disabled"

    def toggle(self, feature_name: str) -> tuple[bool, str]:
        """Toggle an experimental feature."""
        if self.is_enabled(feature_name):
            return self.disable(feature_name)
        else:
            return self.enable(feature_name)

    def list_features(self, show_all: bool = False) -> List[Dict[str, Any]]:
        """List all experimental features."""
        features = []
        for name, feature in self.features.items():
            if show_all or feature.status != FeatureStatus.DISABLED.value:
                features.append({
                    "name": name,
                    "description": feature.description,
                    "status": feature.status,
                    "dangerous": feature.dangerous,
                    "warning": feature.warning,
                    "requires": feature.requires,
                })
        return features

    def display_features(self, show_all: bool = True):
        """Display features in a formatted table."""
        table = Table(title="Experimental Features", show_header=True)
        table.add_column("Feature", style="cyan")
        table.add_column("Status")
        table.add_column("Description")
        table.add_column("Danger", width=6)

        for name, feature in self.features.items():
            if not show_all and feature.status == FeatureStatus.DISABLED.value:
                continue

            status_color = {
                FeatureStatus.ENABLED.value: "green",
                FeatureStatus.TESTING.value: "yellow",
                FeatureStatus.DISABLED.value: "dim",
            }.get(feature.status, "white")

            danger = "[red]YES[/red]" if feature.dangerous else ""

            table.add_row(
                name,
                f"[{status_color}]{feature.status}[/{status_color}]",
                feature.description[:50] + "..." if len(feature.description) > 50 else feature.description,
                danger,
            )

        self.console.print(table)

        # Show warnings for enabled features
        for name, feature in self.features.items():
            if self.is_enabled(name) and feature.warning:
                self.console.print(f"[yellow]Warning ({name}):[/yellow] {feature.warning}")

    def show_feature_details(self, feature_name: str):
        """Show detailed information about a feature."""
        feature = self.features.get(feature_name)
        if not feature:
            self.console.print(f"[red]Unknown feature: {feature_name}[/red]")
            return

        status_color = {
            FeatureStatus.ENABLED.value: "green",
            FeatureStatus.TESTING.value: "yellow",
            FeatureStatus.DISABLED.value: "red",
        }.get(feature.status, "white")

        details = f"""[bold]Feature:[/bold] {feature.name}
[bold]Status:[/bold] [{status_color}]{feature.status}[/{status_color}]
[bold]Description:[/bold] {feature.description}
[bold]Version Added:[/bold] {feature.version_added}
[bold]Dangerous:[/bold] {"Yes" if feature.dangerous else "No"}"""

        if feature.requires:
            details += f"\n[bold]Requires:[/bold] {', '.join(feature.requires)}"

        if feature.conflicts:
            details += f"\n[bold]Conflicts:[/bold] {', '.join(feature.conflicts)}"

        if feature.enabled_at:
            details += f"\n[bold]Enabled At:[/bold] {feature.enabled_at}"

        self.console.print(Panel(details, title=f"Feature: {feature_name}", border_style="cyan"))

        if feature.warning:
            self.console.print(Panel(
                feature.warning,
                title="Warning",
                border_style="yellow"
            ))

    def register_callback(self, feature_name: str, callback: Callable):
        """Register a callback for when a feature is toggled."""
        if feature_name not in self._callbacks:
            self._callbacks[feature_name] = []
        self._callbacks[feature_name].append(callback)

    def _trigger_callbacks(self, feature_name: str, action: str):
        """Trigger callbacks for a feature."""
        for callback in self._callbacks.get(feature_name, []):
            try:
                callback(feature_name, action)
            except Exception as e:
                self.console.print(f"[yellow]Callback error: {e}[/yellow]")

    def get_enabled_features(self) -> List[str]:
        """Get list of enabled feature names."""
        return [name for name, f in self.features.items()
                if f.status in [FeatureStatus.ENABLED.value, FeatureStatus.TESTING.value]]

    def enable_all(self, force: bool = False) -> Dict[str, str]:
        """Enable all features (respecting dependencies)."""
        results = {}

        # Enable in dependency order
        enabled_count = 0
        while enabled_count < len(self.features):
            made_progress = False

            for name, feature in self.features.items():
                if self.is_enabled(name):
                    continue

                # Check if all dependencies are met
                deps_met = all(self.is_enabled(req) for req in feature.requires)
                if deps_met:
                    if feature.dangerous and not force:
                        results[name] = "skipped (dangerous, use --force)"
                    else:
                        success, msg = self.enable(name, force=force)
                        results[name] = msg
                        if success:
                            made_progress = True
                            enabled_count += 1

            if not made_progress:
                break

        return results

    def disable_all(self) -> Dict[str, str]:
        """Disable all features."""
        results = {}

        # Disable in reverse dependency order
        for name in reversed(list(self.features.keys())):
            if self.is_enabled(name):
                success, msg = self.disable(name)
                results[name] = msg

        return results


# Global instance
_experimental_manager: Optional[ExperimentalFeaturesManager] = None


def get_experimental_manager() -> ExperimentalFeaturesManager:
    """Get or create the global experimental features manager."""
    global _experimental_manager
    if _experimental_manager is None:
        _experimental_manager = ExperimentalFeaturesManager()
    return _experimental_manager


def is_feature_enabled(feature_name: str) -> bool:
    """Convenience function to check if a feature is enabled."""
    return get_experimental_manager().is_enabled(feature_name)


def require_feature(feature_name: str):
    """Decorator to require an experimental feature."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not is_feature_enabled(feature_name):
                console = Console()
                console.print(f"[yellow]Feature '{feature_name}' is not enabled.[/yellow]")
                console.print(f"[dim]Enable with: experimental enable {feature_name}[/dim]")
                return None
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Initialize experimental features based on current config
def initialize_experimental_features():
    """Initialize all enabled experimental features."""
    from rich.console import Console
    console = Console()

    manager = get_experimental_manager()
    enabled = manager.get_enabled_features()

    if enabled:
        console.print(f"[dim]Experimental features enabled: {', '.join(enabled)}[/dim]")

    # Initialize self-optimization if enabled
    if manager.is_enabled("self_optimization"):
        from self_optimization import init_self_optimizer
        allow_update = manager.is_enabled("self_update")
        init_self_optimizer(enabled=True, allow_self_update=allow_update)

    # Initialize contextualization if enabled
    if manager.is_enabled("contextualization"):
        from contextualization_engine import init_contextualization
        init_contextualization(enabled=True)

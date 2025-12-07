#!/usr/bin/env python3
"""
Coding Agent - Autonomous Code Optimization System
Main entry point and CLI interface.

Usage:
    python main.py                      # Interactive mode
    python main.py run <target>         # Run agent on target
    python main.py fix <file>           # Fix issues in file
    python main.py optimize <file>      # Optimize file
    python main.py test                 # Run tests
    python main.py --help               # Show help
"""
import argparse
import asyncio
import json
import signal
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

from config import config
from ai_providers import provider_manager
from local_ai import local_ai_manager
from agent_engine import AgentEngine, run_agent
from orchestrator import AgentOrchestrator, run_multi_agent
from journal import get_journal, new_session
from git_integration import GitManager, GitHubManager, GitHubAccountManager, create_and_push_repos
from terminal_ui import TerminalUI, ResultsDisplay, InteractiveMode, AgentProgressDisplay
from code_search import CodeSearchManager, search_code
from local_code_search import get_local_search_manager, local_search
from experimental_features import (
    get_experimental_manager, is_feature_enabled, initialize_experimental_features
)
from self_optimization import get_self_optimizer, init_self_optimizer
from contextualization_engine import get_contextualization_engine, init_contextualization
from version_control import get_version_manager, VersionType, VersionRating


VERSION = "1.1.0"


class CodingAgentCLI:
    """Main CLI application for the Coding Agent."""

    def __init__(self):
        self.ui = TerminalUI()
        self.results = ResultsDisplay()
        self.journal = None
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.ui.console.print("\n[yellow]Interrupted. Cleaning up...[/yellow]")
            if self.journal:
                self.journal.close_session()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def run_interactive(self):
        """Run in interactive mode."""
        self.journal = new_session()

        def run_command(cmd: str, target: Optional[str]):
            asyncio.run(self._execute_command(cmd, target))

        interactive = InteractiveMode()
        interactive.run(run_command)

        if self.journal:
            summary = self.journal.close_session()
            self.ui.print_info(f"Session saved to: {summary['journal_file']}")

    async def _execute_command(self, cmd: str, target: Optional[str]):
        """Execute a command."""
        if cmd == "run":
            await self._run_agent(target, mode="full")

        elif cmd == "fix":
            await self._run_agent(target, mode="fix")

        elif cmd == "optimize":
            await self._run_agent(target, mode="optimize")

        elif cmd == "test":
            await self._run_tests()

        elif cmd == "status":
            self._show_status()

        elif cmd == "journal":
            self._show_journal()

        elif cmd == "providers":
            self._show_providers()

        elif cmd == "push":
            await self._push_changes()

        elif cmd == "config":
            self._show_config(target)

        elif cmd == "search":
            await self._search_code(target)

        elif cmd == "local-search":
            await self._local_search(target)

        elif cmd == "local-add":
            self._local_add_directory(target)

        elif cmd == "local-list":
            self._local_list_directories()

        elif cmd == "local-remove":
            self._local_remove_directory(target)

        # Experimental features commands
        elif cmd == "experimental":
            self._show_experimental_features()

        elif cmd == "experimental-enable":
            self._enable_experimental(target)

        elif cmd == "experimental-disable":
            self._disable_experimental(target)

        # Version control commands
        elif cmd == "version":
            self._show_version_history(target)

        elif cmd == "version-rollback":
            self._version_rollback(target)

        elif cmd == "version-rate":
            self._version_rate(target)

        # Self-optimization commands
        elif cmd == "learn-stats":
            self._show_learning_stats()

        elif cmd == "optimize-self":
            await self._run_self_optimization()

        # Contextualization commands
        elif cmd == "context":
            await self._build_context(target)

        # Web learning commands
        elif cmd == "learn":
            await self._learn_from_web(target)

    async def _run_agent(self, target: str, mode: str = "fix"):
        """Run the agent on a target."""
        if not target:
            self.ui.print_error("Error", "No target specified")
            return

        path = Path(target)
        if not path.exists():
            self.ui.print_error("Error", f"Target not found: {target}")
            return

        self.ui.console.print(f"\n[bold]Running agent on:[/bold] {target}")
        self.ui.console.print(f"[dim]Mode: {mode}[/dim]\n")

        # Record in journal
        if self.journal:
            self.journal.record_iteration_start(1, target, mode)

        def progress_callback(state: str, message: str = ""):
            elapsed = self._format_elapsed()
            self.ui.console.print(f"  [{elapsed}] {state}: {message}")

        # Determine execution mode
        if config.agent.mode == "multi_agent":
            # Multi-agent mode
            self.ui.print_info("Using multi-agent orchestration...")
            results = await run_multi_agent(target, "parallel", progress_callback)
            self.ui.print_success("Multi-agent run complete")
            self.results.show_summary(results.get("status", {}))

        else:
            # Single agent or standalone mode
            engine = AgentEngine(progress_callback=progress_callback)

            if config.agent.mode == "standalone":
                iterations = await engine.run_standalone(target)
            else:
                iterations = await engine.run(target, mode)

            summary = engine.get_summary()
            self.results.show_summary(summary)

            # Record in journal
            if self.journal:
                self.journal.record_iteration_end(
                    1, summary["final_success"],
                    summary["total_duration_seconds"], summary
                )

    async def _run_tests(self):
        """Run tests."""
        from agent_engine import TestRunner

        self.ui.print_info("Running tests...")

        runner = TestRunner()
        result = await runner.run_tests()

        if result.success:
            self.ui.print_success(
                f"Tests passed: {result.passed}/{result.total_tests} in {result.duration_seconds:.2f}s"
            )
        else:
            self.ui.print_error(
                "Tests failed",
                f"{result.failed} failed, {result.errors} errors"
            )

        if result.output and config.agent.verbose:
            self.ui.console.print("\n[dim]Test output:[/dim]")
            self.ui.console.print(result.output[:2000])

        # Record in journal
        if self.journal:
            self.journal.record_test_run(
                config.agent.test_command,
                {
                    "success": result.success,
                    "passed": result.passed,
                    "failed": result.failed,
                    "duration_seconds": result.duration_seconds
                }
            )

    def _show_status(self):
        """Show current status."""
        self.ui.console.print("\n[bold]Current Status[/bold]\n")

        # Show provider status
        providers = provider_manager.get_available_providers()
        self.ui.console.print(f"[cyan]AI Providers:[/cyan] {', '.join(providers) if providers else 'None configured'}")

        # Show local AI status
        local_status = local_ai_manager.get_status()
        self.ui.console.print(f"[cyan]Local AI:[/cyan] {'Enabled' if local_status['enabled'] else 'Disabled'}")
        if local_status.get('model_loaded'):
            self.ui.console.print(f"  Model: {local_status['current_model']}")

        # Show git status
        try:
            git = GitManager()
            status = git.get_status()
            self.ui.console.print(f"[cyan]Git Branch:[/cyan] {status.branch}")
            self.ui.console.print(f"[cyan]Clean:[/cyan] {'Yes' if status.is_clean else 'No'}")
            if not status.is_clean:
                self.ui.console.print(f"  Staged: {len(status.staged_files)}")
                self.ui.console.print(f"  Modified: {len(status.unstaged_files)}")
                self.ui.console.print(f"  Untracked: {len(status.untracked_files)}")
        except Exception as e:
            self.ui.console.print(f"[cyan]Git:[/cyan] Not a git repository")

        # Show journal status
        if self.journal:
            summary = self.journal.get_session_summary()
            self.ui.console.print(f"\n[cyan]Session:[/cyan] {summary['session_id']}")
            self.ui.console.print(f"  Duration: {summary['duration']}")
            self.ui.console.print(f"  AI Calls: {summary['total_ai_calls']}")
            self.ui.console.print(f"  Tokens: {summary['total_tokens_used']}")

    def _show_journal(self):
        """Show journal summary."""
        if not self.journal:
            self.ui.print_warning("No active session")
            return

        summary = self.journal.get_session_summary()

        self.ui.console.print("\n[bold]Session Journal[/bold]\n")
        self.ui.console.print(f"Session ID: {summary['session_id']}")
        self.ui.console.print(f"Duration: {summary['duration']}")
        self.ui.console.print(f"Iterations: {summary['total_iterations']}")
        self.ui.console.print(f"Fixes Attempted: {summary['total_fixes_attempted']}")
        self.ui.console.print(f"Successful Fixes: {summary['successful_fixes']}")
        self.ui.console.print(f"Test Runs: {summary['total_test_runs']}")
        self.ui.console.print(f"AI Calls: {summary['total_ai_calls']}")
        self.ui.console.print(f"Tokens Used: {summary['total_tokens_used']}")
        self.ui.console.print(f"Errors: {summary['errors_count']}")
        self.ui.console.print(f"\nJournal: {summary['journal_file']}")

    def _show_providers(self):
        """Show available AI providers."""
        self.ui.console.print("\n[bold]AI Providers[/bold]\n")

        for name, provider_config in config.providers.items():
            has_key = bool(provider_config.api_key)
            status = "[green]●[/green] Ready" if has_key else "[red]○[/red] No API key"
            self.ui.console.print(f"  {provider_config.name}: {status}")
            if has_key:
                self.ui.console.print(f"    Model: {provider_config.model}")

        # Local AI
        local_status = local_ai_manager.get_status()
        if local_status["enabled"]:
            self.ui.console.print(f"\n  Local AI ({local_status['backend']}):")
            if local_status["model_loaded"]:
                self.ui.console.print(f"    [green]●[/green] {local_status['current_model']}")
            else:
                self.ui.console.print(f"    [yellow]○[/yellow] No model loaded")
        else:
            self.ui.console.print("\n  Local AI: [dim]Disabled[/dim]")

    async def _push_changes(self):
        """Commit and push changes."""
        git = GitManager()
        status = git.get_status()

        if status.is_clean:
            self.ui.print_info("No changes to commit")
            return

        self.ui.console.print("\n[bold]Changes to commit:[/bold]")
        for f in status.staged_files + status.unstaged_files:
            self.ui.console.print(f"  {f}")

        if not self.ui.confirm("Commit and push these changes?"):
            return

        # Get commit message
        message = self.ui.get_input("Commit message: ")
        if not message:
            message = "[Agent] Automated code improvements"

        # Stage all changes
        git.stage_all()

        # Commit
        success, result = git.commit(message)
        if success:
            self.ui.print_success(f"Committed: {result}")

            # Push
            push_success, push_result = git.push()
            if push_success:
                self.ui.print_success("Pushed to remote")
            else:
                self.ui.print_error("Push failed", push_result)

            # Record in journal
            if self.journal:
                self.journal.record_git_action("commit_push", {
                    "commit_hash": result,
                    "message": message,
                    "push_success": push_success
                })
        else:
            self.ui.print_error("Commit failed", result)

    async def _search_code(self, query: str):
        """Search for similar code across web sources."""
        if not query:
            self.ui.print_error("Error", "No search query specified")
            return

        self.ui.console.print(f"\n[bold]Searching for:[/bold] {query}\n")

        from rich.table import Table
        from rich.panel import Panel

        results = await search_code(query)

        if not results:
            self.ui.print_warning("No results found")
            return

        # Group results by source
        by_source = {}
        for r in results:
            if r.source not in by_source:
                by_source[r.source] = []
            by_source[r.source].append(r)

        for source, source_results in by_source.items():
            self.ui.console.print(f"\n[bold cyan]{source.upper()}[/bold cyan]")

            table = Table(show_header=True, header_style="bold")
            table.add_column("#", width=3)
            table.add_column("Title", width=50)
            table.add_column("Language", width=10)
            table.add_column("Relevance", width=10)

            for i, r in enumerate(source_results[:5], 1):
                table.add_row(
                    str(i),
                    r.title[:47] + "..." if len(r.title) > 50 else r.title,
                    r.language or "-",
                    f"{r.relevance:.0%}" if r.relevance else "-"
                )

            self.ui.console.print(table)

            # Show URLs
            for i, r in enumerate(source_results[:5], 1):
                self.ui.console.print(f"  [{i}] [dim]{r.url}[/dim]")

        self.ui.console.print(f"\n[green]Found {len(results)} total results[/green]")

    async def _local_search(self, query: str, extract: bool = False):
        """Search for code in configured local directories."""
        if not query:
            self.ui.print_error("Error", "No search query specified")
            return

        manager = get_local_search_manager()
        results = manager.search_and_display(query)

        if results and extract:
            if self.ui.confirm(f"Extract {len(results)} files to current project?"):
                manager.extract_results(results)

    def _local_add_directory(self, path: str):
        """Add a directory to local search configuration."""
        if not path:
            self.ui.print_error("Error", "No path specified")
            return

        manager = get_local_search_manager()
        manager.add_directory(path)

    def _local_list_directories(self):
        """List configured local search directories."""
        manager = get_local_search_manager()
        manager.list_directories()

    def _local_remove_directory(self, path_or_name: str):
        """Remove a directory from local search configuration."""
        if not path_or_name:
            self.ui.print_error("Error", "No path or name specified")
            return

        manager = get_local_search_manager()
        manager.remove_directory(path_or_name)

    # === Experimental Features Methods ===

    def _show_experimental_features(self):
        """Show all experimental features."""
        manager = get_experimental_manager()
        manager.display_features(show_all=True)

    def _enable_experimental(self, feature_name: str):
        """Enable an experimental feature."""
        if not feature_name:
            self.ui.print_error("Error", "No feature name specified")
            return

        manager = get_experimental_manager()

        # Check for --force flag
        force = "--force" in feature_name
        feature_name = feature_name.replace("--force", "").strip()

        success, message = manager.enable(feature_name, force=force)
        if success:
            self.ui.print_success(message)
            # Re-initialize features
            initialize_experimental_features()
        else:
            self.ui.print_error("Error", message)

    def _disable_experimental(self, feature_name: str):
        """Disable an experimental feature."""
        if not feature_name:
            self.ui.print_error("Error", "No feature name specified")
            return

        manager = get_experimental_manager()
        success, message = manager.disable(feature_name)
        if success:
            self.ui.print_success(message)
        else:
            self.ui.print_error("Error", message)

    # === Version Control Methods ===

    def _show_version_history(self, file_path: str):
        """Show version history for a file."""
        if not file_path:
            # Show stats if no file specified
            manager = get_version_manager()
            manager.show_stats()
            return

        manager = get_version_manager()
        manager.show_history(file_path)

    def _version_rollback(self, args: str):
        """Rollback to a previous version."""
        if not args:
            self.ui.print_error("Error", "Specify file path")
            return

        parts = args.split()
        file_path = parts[0]
        to_best = "--best" in parts

        manager = get_version_manager()
        manager.rollback(file_path, to_best=to_best)

    def _version_rate(self, args: str):
        """Rate a version."""
        if not args:
            self.ui.print_error("Error", "Specify version_id rating")
            return

        parts = args.split()
        if len(parts) < 2:
            self.ui.print_error("Error", "Specify: version_id rating (1-5)")
            return

        version_id = parts[0]
        try:
            rating_val = int(parts[1])
            rating = VersionRating(rating_val)
        except (ValueError, KeyError):
            self.ui.print_error("Error", "Rating must be 1-5")
            return

        reason = " ".join(parts[2:]) if len(parts) > 2 else ""

        manager = get_version_manager()
        manager.rate_version(version_id, rating, reason)

    # === Self-Optimization Methods ===

    def _show_learning_stats(self):
        """Show learning statistics."""
        if not is_feature_enabled("self_optimization"):
            self.ui.print_warning("Self-optimization is not enabled")
            self.ui.console.print("[dim]Enable with: experimental-enable self_optimization[/dim]")
            return

        optimizer = get_self_optimizer()
        optimizer.show_stats()

    async def _run_self_optimization(self):
        """Run self-optimization as main task."""
        if not is_feature_enabled("self_optimization"):
            self.ui.print_warning("Self-optimization is not enabled")
            return

        if not is_feature_enabled("self_update"):
            self.ui.print_warning("Self-update is not enabled (required for self-optimization)")
            self.ui.console.print("[dim]Enable with: experimental-enable self_update --force[/dim]")
            return

        optimizer = get_self_optimizer()
        results = optimizer.optimize_self()

        if results.get("proposals"):
            self.ui.console.print(f"\n[bold]Found {len(results['proposals'])} improvement opportunities[/bold]")

    # === Contextualization Methods ===

    async def _build_context(self, query: str = ""):
        """Build and display context."""
        if not is_feature_enabled("contextualization"):
            self.ui.print_warning("Contextualization is not enabled")
            self.ui.console.print("[dim]Enable with: experimental-enable contextualization[/dim]")
            return

        engine = get_contextualization_engine()

        include_web = is_feature_enabled("web_context")

        context = await engine.build_full_context(
            query=query or "",
            include_code=True,
            include_parent=True,
            include_web=include_web,
            include_user=is_feature_enabled("user_learning"),
        )

        engine.display_context(context)

    # === Web Learning Methods ===

    async def _learn_from_web(self, args: str):
        """
        Run autonomous web learning as a main task.
        Usage: learn <duration> [topics...]
        Example: learn 6h python error handling
        """
        if not is_feature_enabled("self_optimization"):
            self.ui.print_warning("Self-optimization required for web learning")
            self.ui.console.print("[dim]Enable with: experimental-enable self_optimization[/dim]")
            return

        if not is_feature_enabled("web_context"):
            self.ui.print_warning("Web context required for web learning")
            self.ui.console.print("[dim]Enable with: experimental-enable contextualization && experimental-enable web_context[/dim]")
            return

        import time
        from contextualization_engine import WebContextGatherer

        # Parse duration and topics
        parts = args.split() if args else []

        duration_hours = 1.0  # Default 1 hour
        topics = []

        for part in parts:
            if part.endswith('h'):
                try:
                    duration_hours = float(part[:-1])
                except ValueError:
                    topics.append(part)
            elif part.endswith('m'):
                try:
                    duration_hours = float(part[:-1]) / 60
                except ValueError:
                    topics.append(part)
            else:
                topics.append(part)

        if not topics:
            # Default topics based on project context
            engine = get_contextualization_engine()
            project = engine.get_project_context()
            topics = project.languages + project.frameworks
            if not topics:
                topics = ["python", "code optimization", "bug fixing"]

        duration_seconds = duration_hours * 3600
        end_time = time.time() + duration_seconds

        self.ui.console.print(f"\n[bold cyan]Starting Web Learning Session[/bold cyan]")
        self.ui.console.print(f"Duration: {duration_hours:.1f} hours")
        self.ui.console.print(f"Topics: {', '.join(topics)}")
        self.ui.console.print("[dim]Press Ctrl+C to stop early[/dim]\n")

        gatherer = WebContextGatherer()
        optimizer = get_self_optimizer()

        stats = {
            "queries_made": 0,
            "examples_found": 0,
            "issues_analyzed": 0,
            "patterns_learned": 0,
            "strategies_created": 0,
        }

        # Learning queries based on topics
        learning_queries = []
        for topic in topics:
            learning_queries.extend([
                f"{topic} best practices",
                f"{topic} common errors fix",
                f"{topic} performance optimization",
                f"{topic} design patterns",
                f"{topic} error handling",
                f"{topic} code examples",
                f"{topic} debugging techniques",
            ])

        query_index = 0
        cycle = 0

        try:
            while time.time() < end_time:
                cycle += 1
                query = learning_queries[query_index % len(learning_queries)]
                query_index += 1

                self.ui.console.print(f"[dim]Cycle {cycle}: Learning about '{query}'...[/dim]")

                try:
                    context = await gatherer.gather_context(query, include_docs=True, include_issues=True)
                    stats["queries_made"] += 1

                    # Process examples
                    for example in context.examples:
                        stats["examples_found"] += 1
                        score = example.get("score", 0)

                        if score > 5:
                            # High-quality example - learn from it
                            stats["patterns_learned"] += 1

                            # Create a strategy if the example is very good
                            if score > 10 and example.get("answered"):
                                stats["strategies_created"] += 1

                    # Process issues
                    for issue in context.related_issues:
                        stats["issues_analyzed"] += 1

                except Exception as e:
                    self.ui.console.print(f"[yellow]Error in learning cycle: {e}[/yellow]")

                # Progress update every 10 cycles
                if cycle % 10 == 0:
                    elapsed = (duration_seconds - (end_time - time.time())) / 60
                    remaining = (end_time - time.time()) / 60
                    self.ui.console.print(
                        f"\n[cyan]Progress: {elapsed:.0f}m elapsed, {remaining:.0f}m remaining[/cyan]"
                    )
                    self.ui.console.print(
                        f"  Queries: {stats['queries_made']} | "
                        f"Examples: {stats['examples_found']} | "
                        f"Patterns: {stats['patterns_learned']}"
                    )

                # Wait between queries to be polite to APIs
                await asyncio.sleep(5)

        except KeyboardInterrupt:
            self.ui.console.print("\n[yellow]Learning session interrupted[/yellow]")

        # Final summary
        self.ui.console.print(Panel(
            f"[bold]Learning Session Complete[/bold]\n\n"
            f"Duration: {(duration_seconds - (end_time - time.time())) / 3600:.1f} hours\n"
            f"Queries Made: {stats['queries_made']}\n"
            f"Examples Found: {stats['examples_found']}\n"
            f"Issues Analyzed: {stats['issues_analyzed']}\n"
            f"Patterns Learned: {stats['patterns_learned']}\n"
            f"Strategies Created: {stats['strategies_created']}",
            title="Web Learning Summary",
            border_style="green"
        ))

        # Record in journal
        if self.journal:
            self.journal.record_ai_interaction(
                "web_learning",
                "learning_session",
                {"duration_hours": duration_hours, "topics": topics},
                stats,
                0  # No token cost for web learning
            )

    # === Timed Session Methods ===

    async def _run_timed_session(self, args):
        """
        Run a timed work session with automatic saving.
        Usage: work <duration> <target> [--mode MODE]
        Example: work 6h src/ --mode fix

        The session will:
        - Work until deadline
        - Auto-save progress periodically
        - Journal all actions
        - Save state for continuation
        """
        import time

        # Parse arguments
        duration_str = args.duration if hasattr(args, 'duration') else "1h"
        target = args.target if hasattr(args, 'target') else "."
        mode = args.mode if hasattr(args, 'mode') else "fix"

        # Parse duration
        if duration_str.endswith('h'):
            duration_hours = float(duration_str[:-1])
        elif duration_str.endswith('m'):
            duration_hours = float(duration_str[:-1]) / 60
        elif duration_str.endswith('d'):
            duration_hours = float(duration_str[:-1]) * 24
        else:
            try:
                duration_hours = float(duration_str)
            except ValueError:
                duration_hours = 1.0

        duration_seconds = duration_hours * 3600
        end_time = time.time() + duration_seconds
        deadline = datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M")

        # Session state file
        session_state_file = Path(".cache/session_state.json")
        session_state_file.parent.mkdir(parents=True, exist_ok=True)

        self.ui.console.print(Panel(
            f"[bold]Timed Work Session[/bold]\n\n"
            f"Target: {target}\n"
            f"Mode: {mode}\n"
            f"Duration: {duration_hours:.1f} hours\n"
            f"Deadline: {deadline}\n\n"
            f"[dim]Press Ctrl+C to pause and save state[/dim]",
            title="Session Started",
            border_style="cyan"
        ))

        session_state = {
            "start_time": datetime.now().isoformat(),
            "deadline": deadline,
            "target": target,
            "mode": mode,
            "iterations_completed": 0,
            "files_processed": [],
            "issues_fixed": 0,
            "last_checkpoint": None,
            "status": "running",
        }

        # Journal session start
        if self.journal:
            self.journal.record_iteration_start(1, target, f"timed_session:{mode}")

        iteration = 0
        checkpoint_interval = 300  # Save checkpoint every 5 minutes
        last_checkpoint = time.time()

        try:
            while time.time() < end_time:
                iteration += 1
                remaining = (end_time - time.time()) / 60

                self.ui.console.print(f"\n[cyan]Iteration {iteration} | {remaining:.0f} min remaining[/cyan]")

                # Run agent iteration
                try:
                    engine = AgentEngine()
                    await engine.run(target, mode)

                    summary = engine.get_summary()
                    session_state["iterations_completed"] = iteration
                    session_state["issues_fixed"] += summary.get("issues_fixed", 0)

                    if self.journal:
                        self.journal.record_iteration_end(
                            iteration,
                            summary.get("final_success", False),
                            summary.get("total_duration_seconds", 0),
                            summary
                        )

                except Exception as e:
                    self.ui.console.print(f"[yellow]Iteration error: {e}[/yellow]")
                    if self.journal:
                        self.journal.record_error("timed_session", str(e), {"iteration": iteration})

                # Periodic checkpoint
                if time.time() - last_checkpoint > checkpoint_interval:
                    session_state["last_checkpoint"] = datetime.now().isoformat()
                    with open(session_state_file, "w") as f:
                        json.dump(session_state, f, indent=2)
                    last_checkpoint = time.time()
                    self.ui.console.print("[dim]Checkpoint saved[/dim]")

                # Small delay between iterations
                await asyncio.sleep(2)

        except KeyboardInterrupt:
            self.ui.console.print("\n[yellow]Session paused by user[/yellow]")
            session_state["status"] = "paused"

        # Final save
        session_state["status"] = "completed" if time.time() >= end_time else session_state["status"]
        session_state["end_time"] = datetime.now().isoformat()

        with open(session_state_file, "w") as f:
            json.dump(session_state, f, indent=2)

        # Summary
        elapsed = (time.time() - (end_time - duration_seconds)) / 3600
        self.ui.console.print(Panel(
            f"[bold]Session {'Completed' if session_state['status'] == 'completed' else 'Paused'}[/bold]\n\n"
            f"Duration: {elapsed:.1f} hours\n"
            f"Iterations: {session_state['iterations_completed']}\n"
            f"Issues Fixed: {session_state['issues_fixed']}\n\n"
            f"State saved to: {session_state_file}\n"
            f"[dim]Resume with: python main.py resume[/dim]",
            title="Session Summary",
            border_style="green" if session_state["status"] == "completed" else "yellow"
        ))

        if self.journal:
            self.journal.record_iteration_end(
                iteration,
                session_state["status"] == "completed",
                elapsed * 3600,
                session_state
            )

    async def _resume_session(self):
        """Resume a paused session."""
        import time

        session_state_file = Path(".cache/session_state.json")

        if not session_state_file.exists():
            self.ui.print_warning("No saved session found")
            return

        with open(session_state_file, "r") as f:
            session_state = json.load(f)

        if session_state.get("status") == "completed":
            self.ui.print_info("Previous session was completed")
            return

        self.ui.console.print(Panel(
            f"[bold]Resuming Session[/bold]\n\n"
            f"Target: {session_state['target']}\n"
            f"Mode: {session_state['mode']}\n"
            f"Iterations completed: {session_state['iterations_completed']}\n"
            f"Issues fixed: {session_state['issues_fixed']}",
            title="Resume",
            border_style="cyan"
        ))

        # Continue with remaining work
        args = type('Args', (), {
            'duration': '1h',  # Continue for 1 hour by default
            'target': session_state['target'],
            'mode': session_state['mode'],
        })()

        await self._run_timed_session(args)

    def _show_config(self, key: Optional[str] = None):
        """Show or modify configuration."""
        self.ui.console.print("\n[bold]Configuration[/bold]\n")

        self.ui.console.print("[cyan]Agent Settings:[/cyan]")
        self.ui.console.print(f"  Mode: {config.agent.mode}")
        self.ui.console.print(f"  Max Iterations: {config.agent.max_iterations}")
        self.ui.console.print(f"  Test Command: {config.agent.test_command}")
        self.ui.console.print(f"  Build Command: {config.agent.build_command}")

        self.ui.console.print("\n[cyan]Local AI:[/cyan]")
        self.ui.console.print(f"  Enabled: {config.local_ai.enabled}")
        self.ui.console.print(f"  Backend: {config.local_ai.backend}")
        self.ui.console.print(f"  Model: {config.local_ai.model_name}")

        self.ui.console.print("\n[cyan]Git:[/cyan]")
        self.ui.console.print(f"  Auto Commit: {config.git.auto_commit}")
        self.ui.console.print(f"  Auto Push: {config.git.auto_push}")
        self.ui.console.print(f"  GitHub Accounts: {', '.join(config.git.github_accounts)}")

    def _format_elapsed(self) -> str:
        """Format elapsed time since start."""
        import time
        if not hasattr(self, '_start_time'):
            self._start_time = time.time()
        elapsed = time.time() - self._start_time
        return f"{elapsed:6.1f}s"

    async def run_single_command(self, args):
        """Run a single command from CLI args."""
        self.journal = new_session()
        self._start_time = __import__('time').time()

        try:
            if args.command == "run":
                await self._run_agent(args.target, mode="full")

            elif args.command == "fix":
                await self._run_agent(args.target, mode="fix")

            elif args.command == "optimize":
                await self._run_agent(args.target, mode="optimize")

            elif args.command == "test":
                await self._run_tests()

            elif args.command == "status":
                self._show_status()

            elif args.command == "providers":
                self._show_providers()

            elif args.command == "push":
                await self._push_changes()

            elif args.command == "create-repos":
                await self._create_github_repos(args)

            elif args.command == "search":
                await self._search_code(args.query)

            elif args.command == "local-search":
                await self._local_search(args.query, extract=args.extract if hasattr(args, 'extract') else False)

            elif args.command == "local-add":
                self._local_add_directory(args.path)

            elif args.command == "local-list":
                self._local_list_directories()

            elif args.command == "local-remove":
                self._local_remove_directory(args.path)

            # Experimental features
            elif args.command == "experimental":
                self._show_experimental_features()

            elif args.command == "experimental-enable":
                self._enable_experimental(args.feature)

            elif args.command == "experimental-disable":
                self._disable_experimental(args.feature)

            # Version control
            elif args.command == "version":
                self._show_version_history(args.file if hasattr(args, 'file') else "")

            elif args.command == "version-rollback":
                file_path = args.file
                if hasattr(args, 'best') and args.best:
                    file_path += " --best"
                self._version_rollback(file_path)

            elif args.command == "version-rate":
                self._version_rate(f"{args.version_id} {args.rating} {args.reason if hasattr(args, 'reason') else ''}")

            # Learning and self-optimization
            elif args.command == "learn":
                duration = args.duration if hasattr(args, 'duration') else "1h"
                topics = " ".join(args.topics) if hasattr(args, 'topics') else ""
                await self._learn_from_web(f"{duration} {topics}")

            elif args.command == "learn-stats":
                self._show_learning_stats()

            elif args.command == "optimize-self":
                await self._run_self_optimization()

            elif args.command == "context":
                await self._build_context(args.query if hasattr(args, 'query') else "")

            # Timed session
            elif args.command == "work":
                await self._run_timed_session(args)

            elif args.command == "resume":
                await self._resume_session()

        finally:
            if self.journal:
                summary = self.journal.close_session()
                if args.verbose:
                    self.ui.print_info(f"Session: {summary['journal_file']}")

    async def _create_github_repos(self, args):
        """Create GitHub repositories on configured accounts."""
        repo_name = args.name or "coding-agent"
        description = args.description or "Autonomous code optimization agent"

        self.ui.print_info(f"Creating repository '{repo_name}' on configured accounts...")

        results = await create_and_push_repos(
            repo_name=repo_name,
            description=description,
            private=args.private if hasattr(args, 'private') else False
        )

        for account, (success, result) in results.get("repo_creation", {}).items():
            if success:
                self.ui.print_success(f"{account}: {result}")
            else:
                self.ui.print_error(account, result)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Coding Agent - Autonomous Code Optimization System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                      # Interactive mode
  python main.py run src/             # Process src directory
  python main.py fix main.py          # Fix issues in main.py
  python main.py optimize utils.py    # Optimize utils.py
  python main.py test                 # Run tests
  python main.py push                 # Commit and push changes
  python main.py create-repos         # Create GitHub repos
  python main.py search "async await" # Search for similar code on web
  python main.py local-add ~/projects # Add local search directory
  python main.py local-search "class" # Search in local directories
  python main.py local-list           # List configured directories

Environment Variables:
  ANTHROPIC_API_KEY    - Claude API key
  OPENAI_API_KEY       - ChatGPT API key
  GOOGLE_API_KEY       - Gemini API key
  PERPLEXITY_API_KEY   - Perplexity API key

Modes:
  standalone   - No AI, rule-based fixes only
  single_ai    - Use one AI provider
  multi_agent  - Use multiple AI agents in parallel
        """
    )

    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"Coding Agent v{VERSION}"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    parser.add_argument(
        "--mode", "-m",
        choices=["standalone", "single_ai", "multi_agent"],
        help="Operation mode"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run agent on target")
    run_parser.add_argument("target", help="File or directory to process")

    # Fix command
    fix_parser = subparsers.add_parser("fix", help="Fix issues in target")
    fix_parser.add_argument("target", help="File or directory to fix")

    # Optimize command
    opt_parser = subparsers.add_parser("optimize", help="Optimize target")
    opt_parser.add_argument("target", help="File or directory to optimize")

    # Test command
    subparsers.add_parser("test", help="Run tests")

    # Status command
    subparsers.add_parser("status", help="Show status")

    # Providers command
    subparsers.add_parser("providers", help="Show AI providers")

    # Push command
    subparsers.add_parser("push", help="Commit and push changes")

    # Create repos command
    repos_parser = subparsers.add_parser("create-repos", help="Create GitHub repositories")
    repos_parser.add_argument("--name", "-n", default="coding-agent", help="Repository name")
    repos_parser.add_argument("--description", "-d", help="Repository description")
    repos_parser.add_argument("--private", "-p", action="store_true", help="Create private repos")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search for similar code on the web")
    search_parser.add_argument("query", help="Search query or code snippet")
    search_parser.add_argument("--language", "-l", help="Filter by programming language")
    search_parser.add_argument("--source", "-s", choices=["github", "stackoverflow", "google", "all"],
                               default="all", help="Search source")

    # Local search commands
    local_search_parser = subparsers.add_parser("local-search", help="Search for code in local directories")
    local_search_parser.add_argument("query", help="Search query or pattern")
    local_search_parser.add_argument("--language", "-l", help="Filter by programming language")
    local_search_parser.add_argument("--regex", "-r", action="store_true", help="Use regex pattern")
    local_search_parser.add_argument("--extract", "-e", action="store_true", help="Extract matches to current folder")

    local_add_parser = subparsers.add_parser("local-add", help="Add directory to local search")
    local_add_parser.add_argument("path", help="Directory path to add")
    local_add_parser.add_argument("--name", "-n", help="Friendly name for the directory")

    subparsers.add_parser("local-list", help="List configured local search directories")

    local_remove_parser = subparsers.add_parser("local-remove", help="Remove directory from local search")
    local_remove_parser.add_argument("path", help="Directory path or name to remove")

    # Experimental features
    subparsers.add_parser("experimental", help="Show experimental features")

    exp_enable = subparsers.add_parser("experimental-enable", help="Enable an experimental feature")
    exp_enable.add_argument("feature", help="Feature name to enable")
    exp_enable.add_argument("--force", action="store_true", help="Force enable dangerous features")

    exp_disable = subparsers.add_parser("experimental-disable", help="Disable an experimental feature")
    exp_disable.add_argument("feature", help="Feature name to disable")

    # Version control
    version_parser = subparsers.add_parser("version", help="Show version history")
    version_parser.add_argument("file", nargs="?", help="File to show history for")

    rollback_parser = subparsers.add_parser("version-rollback", help="Rollback to previous version")
    rollback_parser.add_argument("file", help="File to rollback")
    rollback_parser.add_argument("--best", action="store_true", help="Rollback to best-rated version")

    rate_parser = subparsers.add_parser("version-rate", help="Rate a version")
    rate_parser.add_argument("version_id", help="Version ID to rate")
    rate_parser.add_argument("rating", type=int, choices=[1, 2, 3, 4, 5], help="Rating (1-5)")
    rate_parser.add_argument("reason", nargs="?", default="", help="Reason for rating")

    # Learning and self-optimization
    learn_parser = subparsers.add_parser("learn", help="Run web learning session")
    learn_parser.add_argument("duration", nargs="?", default="1h", help="Duration (e.g., 6h, 30m)")
    learn_parser.add_argument("topics", nargs="*", help="Topics to learn about")

    subparsers.add_parser("learn-stats", help="Show learning statistics")
    subparsers.add_parser("optimize-self", help="Run self-optimization")

    context_parser = subparsers.add_parser("context", help="Build and show context")
    context_parser.add_argument("query", nargs="?", default="", help="Optional query for web context")

    # Timed work sessions
    work_parser = subparsers.add_parser("work", help="Run timed work session")
    work_parser.add_argument("duration", help="Duration (e.g., 6h, 2d)")
    work_parser.add_argument("target", help="Target file or directory")
    work_parser.add_argument("--mode", "-m", default="fix", choices=["fix", "optimize", "full"],
                            help="Work mode")

    subparsers.add_parser("resume", help="Resume a paused session")

    args = parser.parse_args()

    # Apply mode override
    if args.mode:
        config.agent.mode = args.mode

    if args.verbose:
        config.agent.verbose = True

    # Create CLI instance
    cli = CodingAgentCLI()

    # Run appropriate mode
    if args.command:
        asyncio.run(cli.run_single_command(args))
    else:
        cli.run_interactive()


if __name__ == "__main__":
    main()

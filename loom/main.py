"""Loom CLI entry point.

Help is wired so that `loom`, `loom -h`, `loom --help`, and `loom -?` all
print usage. Same on every subcommand.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from .core import orchestrator
from .utils import cache as response_cache
from .utils import storage

HELP_OPTIONS = ["-h", "--help", "-?"]

app = typer.Typer(
    name="loom",
    help="Loom — weave batch LLM jobs across OpenAI, Anthropic, and Google.",
    context_settings={"help_option_names": HELP_OPTIONS},
    no_args_is_help=True,
    add_completion=False,
)

console = Console()


def _print_prompt_errors(errors: dict[str, str]) -> None:
    if not errors:
        return
    console.print(f"[red]{len(errors)} error(s):[/red]")
    for cid, msg in sorted(errors.items()):
        console.print(f"  [dim]{cid}:[/dim] {msg}")


class Provider(str, Enum):
    openai = "openai"
    anthropic = "anthropic"
    google = "google"
    openrouter = "openrouter"


@app.command(
    "run",
    help=(
        "Run prompts from a JSON or CSV file. Default: submit as a provider batch job "
        "(cheap, async — fetch later). Use --sync to call the provider synchronously "
        "and write results immediately (no fetch step). OpenRouter only supports --sync."
    ),
    context_settings={"help_option_names": HELP_OPTIONS},
)
def run_cmd(
    file: Path = typer.Option(..., "--file", "-f", exists=True, help="Input .json or .csv file."),
    provider: Provider = typer.Option(..., "--provider", "-p", help="LLM provider."),
    model: str = typer.Option(..., "--model", "-m", help="Model identifier (provider-specific)."),
    col: Optional[str] = typer.Option(None, "--col", "-c", help="Prompt column name (required for CSV)."),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Override env API key."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path."),
    sync: bool = typer.Option(
        False,
        "--sync/--batch",
        help="Run prompts synchronously and write output directly, skipping the fetch step (default: --batch).",
    ),
    workers: int = typer.Option(
        8, "--workers", "-w", min=1, help="Concurrent workers for --sync mode."
    ),
    no_cache: bool = typer.Option(
        False, "--no-cache", help="Disable the on-disk response cache (only meaningful with --sync)."
    ),
    force: bool = typer.Option(
        False, "--force", help="Overwrite an existing output file without prompting (only meaningful with --sync)."
    ),
    with_meta: bool = typer.Option(
        False,
        "--with-meta",
        help="Add 'llm_provider' and 'llm_model' columns/fields to the output, alongside 'llm_response'.",
    ),
) -> None:
    if provider == Provider.openrouter and not sync:
        console.print(
            "[red]Error:[/red] OpenRouter has no batch API. Re-run with [bold]--sync[/bold]."
        )
        raise typer.Exit(code=1)

    if sync:
        _run_sync(
            file=file,
            provider=provider,
            model=model,
            col=col,
            api_key=api_key,
            output=output,
            workers=workers,
            use_cache=not no_cache,
            force=force,
            with_meta=with_meta,
        )
        return

    console.print("[bold cyan]Spinning the threads...[/bold cyan] submitting batch.")
    try:
        meta = orchestrator.run_batch(
            file_path=file,
            provider_name=provider.value,
            model=model,
            column=col,
            api_key=api_key,
            output_path=output,
            with_meta=with_meta,
        )
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)

    console.print(f"[green]Batch submitted.[/green] id=[bold]{meta.batch_id}[/bold] provider={meta.provider}")
    console.print(f"Metadata saved to ~/.loom/batches/. Run [bold]loom fetch --id {meta.batch_id}[/bold] later.")


def _run_sync(
    file: Path,
    provider: Provider,
    model: str,
    col: Optional[str],
    api_key: Optional[str],
    output: Optional[Path],
    workers: int,
    use_cache: bool,
    force: bool,
    with_meta: bool,
) -> None:
    console.print(
        f"[bold cyan]Weaving live...[/bold cyan] provider={provider.value} model={model} "
        f"workers={workers} cache={'on' if use_cache else 'off'}"
    )
    try:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn("[cyan]cache={task.fields[hits]}[/cyan]"),
            TextColumn("[red]errors={task.fields[errors]}[/red]"),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:
            task_id = progress.add_task("Generating", total=1, hits=0, errors=0)

            def _on_progress(done: int, total: int, hits: int, errors: int) -> None:
                progress.update(task_id, total=total, completed=done, hits=hits, errors=errors)

            out_path, total, hits, errors, error_messages = orchestrator.generate_sync(
                file_path=file,
                provider_name=provider.value,
                model=model,
                column=col,
                api_key=api_key,
                output_path=output,
                workers=workers,
                use_cache=use_cache,
                force=force,
                with_meta=with_meta,
                on_progress=_on_progress,
            )
    except orchestrator.SyncOutputExistsError as exc:
        console.print(
            f"[yellow]Warning:[/yellow] output file [bold]{exc.out_path}[/bold] already exists."
        )
        if not typer.confirm("Overwrite?", default=False):
            console.print("[dim]Aborted. Re-run with --force to overwrite.[/dim]")
            raise typer.Exit(code=0)
        out_path, total, hits, errors, error_messages = orchestrator.generate_sync(
            file_path=file,
            provider_name=provider.value,
            model=model,
            column=col,
            api_key=api_key,
            output_path=output,
            workers=workers,
            use_cache=use_cache,
            force=True,
            with_meta=with_meta,
        )
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)

    status_color = "green" if errors == 0 else "yellow"
    console.print(
        f"[{status_color}]Done.[/{status_color}] "
        f"wrote [bold]{out_path}[/bold] "
        f"({total} prompts, {hits} cache hits, {errors} errors)"
    )
    _print_prompt_errors(error_messages)


@app.command(
    "fetch",
    help="Check status / download results for a batch.",
    context_settings={"help_option_names": HELP_OPTIONS},
)
def fetch_cmd(
    batch_id: Optional[str] = typer.Option(None, "--id", "-i", help="Batch id to fetch."),
    all_: bool = typer.Option(True, "--all/--no-all", "-a", help="Fetch all pending batches (default)."),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Override env API key."),
    keep: bool = typer.Option(
        False,
        "--keep",
        "-k",
        help="Keep the metadata file in ~/.loom/batches/ after a successful fetch (default: delete it).",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite existing output files without prompting.",
    ),
) -> None:
    if batch_id:
        all_ = False

    def _confirm_overwrite(meta, out_path) -> bool:
        console.print(
            f"[yellow]Warning:[/yellow] output file [bold]{out_path}[/bold] "
            f"already exists (batch {meta.batch_id})."
        )
        return typer.confirm("Overwrite?", default=False)

    targets = []
    if all_:
        results = orchestrator.fetch_all(
            api_key=api_key, keep=keep, force=force, on_conflict=_confirm_overwrite
        )
        targets.extend(results)
    else:
        try:
            targets.append(
                orchestrator.fetch_batch(batch_id, api_key=api_key, keep=keep, force=force)
            )
        except orchestrator.OutputExistsError as exc:
            if not _confirm_overwrite(exc.meta, exc.out_path):
                console.print("[dim]Skipped. Re-run with --force to overwrite, or move the existing file.[/dim]")
                raise typer.Exit(code=0)
            try:
                targets.append(
                    orchestrator.fetch_batch(batch_id, api_key=api_key, keep=keep, force=True)
                )
            except Exception as e:  # noqa: BLE001
                console.print(f"[red]Error:[/red] {e}")
                raise typer.Exit(code=1)
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(code=1)

    status_help = {
        "validating": "Provider has accepted the batch and is queueing/preparing it; no work has started yet.",
        "in_progress": "Provider is actively running the prompts; check back later.",
        "completed": "All prompts finished and results were downloaded.",
        "failed": "Provider reported the batch as failed; results are not available.",
        "expired": "Batch exceeded the provider's time limit before completing.",
        "cancelled": "Batch was cancelled before completing.",
        "unknown": "Last fetch attempt raised an error (e.g. invalid id, auth, or network); re-run to retry.",
    }

    for meta, done, prompt_errors in targets:
        if done:
            suffix = "" if keep else " [dim](metadata removed)[/dim]"
            console.print(
                f"[green]Fabric complete.[/green] id={meta.batch_id} -> [bold]{meta.output_path}[/bold]{suffix}"
            )
            _print_prompt_errors(prompt_errors)
        else:
            console.print(
                f"[yellow]Checking the loom...[/yellow] id={meta.batch_id} status={meta.status}"
            )
            explanation = status_help.get(meta.status)
            if explanation:
                console.print(f"  [dim]{explanation}[/dim]")
            _print_prompt_errors(prompt_errors)


@app.command(
    "list",
    help="List all known batches with their last-known status.",
    context_settings={"help_option_names": HELP_OPTIONS},
)
def list_cmd() -> None:
    batches = storage.list_batches()
    if not batches:
        console.print("[dim]No batches stored under ~/.loom/batches/.[/dim]")
        return
    t = Table(title="Loom batches")
    t.add_column("batch_id")
    t.add_column("provider")
    t.add_column("model")
    t.add_column("status")
    t.add_column("created_at")
    t.add_column("file")
    for m in batches:
        t.add_row(
            m.batch_id,
            m.provider,
            m.model,
            m.status,
            m.created_at.isoformat(timespec="seconds"),
            Path(m.original_file_path).name,
        )
    console.print(t)


@app.command(
    "tokens",
    help=(
        "Count input tokens for every prompt in a file using the provider's "
        "token-counting API. Available for Anthropic and Google. OpenAI and "
        "OpenRouter do not expose a remote token-counting API."
    ),
    context_settings={"help_option_names": HELP_OPTIONS},
)
def tokens_cmd(
    file: Path = typer.Option(..., "--file", "-f", exists=True, help="Input .json or .csv file."),
    provider: Provider = typer.Option(..., "--provider", "-p", help="LLM provider."),
    model: str = typer.Option(..., "--model", "-m", help="Model identifier (provider-specific)."),
    col: Optional[str] = typer.Option(None, "--col", "-c", help="Prompt column name (required for CSV)."),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Override env API key."),
    workers: int = typer.Option(8, "--workers", "-w", min=1, help="Concurrent workers."),
) -> None:
    try:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn("[cyan]est_total≈{task.fields[est]}[/cyan]"),
            TextColumn("[red]errors={task.fields[errors]}[/red]"),
            TimeElapsedColumn(),
            TextColumn("[yellow]eta[/yellow]"),
            TimeRemainingColumn(compact=True),
            console=console,
            transient=False,
        ) as progress:
            task_id = progress.add_task(
                "Counting tokens", total=1, errors=0, est="—"
            )

            def _on_progress(done: int, total: int, errors: int, tokens_so_far: int) -> None:
                successful = max(0, done - errors)
                if successful > 0:
                    avg = tokens_so_far / successful
                    est = int(round(avg * total))
                    est_str = f"{est:,}"
                else:
                    est_str = "—"
                progress.update(
                    task_id,
                    total=total,
                    completed=done,
                    errors=errors,
                    est=est_str,
                )

            total_tokens, total, errors = orchestrator.count_tokens(
                file_path=file,
                provider_name=provider.value,
                model=model,
                column=col,
                api_key=api_key,
                workers=workers,
                on_progress=_on_progress,
            )
    except orchestrator.TokenCountingNotSupported as e:
        console.print(f"[yellow]Token counting not available:[/yellow] {e}")
        raise typer.Exit(code=2)
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)

    status_color = "green" if errors == 0 else "yellow"
    console.print(
        f"[{status_color}]Total input tokens: [bold]{total_tokens:,}[/bold][/{status_color}] "
        f"across {total} prompts (provider={provider.value}, model={model}, errors={errors})"
    )


cache_app = typer.Typer(
    name="cache",
    help="Manage the on-disk response cache used by 'loom run --sync'.",
    context_settings={"help_option_names": HELP_OPTIONS},
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(cache_app, name="cache")


@cache_app.command(
    "clear",
    help="Delete every cached response under ~/.loom/cache/.",
    context_settings={"help_option_names": HELP_OPTIONS},
)
def cache_clear_cmd(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
) -> None:
    cache_dir = response_cache.CACHE_DIR
    if not cache_dir.exists():
        console.print("[dim]No cache directory exists yet — nothing to clear.[/dim]")
        return
    count = sum(1 for _ in cache_dir.glob("*.json"))
    if count == 0:
        console.print("[dim]Cache is already empty.[/dim]")
        return
    if not yes and not typer.confirm(
        f"Delete {count} cached response{'s' if count != 1 else ''} from {cache_dir}?",
        default=False,
    ):
        console.print("[dim]Aborted.[/dim]")
        raise typer.Exit(code=0)
    n = response_cache.clear()
    console.print(f"[green]Cache cleared.[/green] Removed {n} file{'s' if n != 1 else ''}.")


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()

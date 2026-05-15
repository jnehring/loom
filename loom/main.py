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
from rich.table import Table

from .core import orchestrator
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


class Provider(str, Enum):
    openai = "openai"
    anthropic = "anthropic"
    google = "google"


@app.command(
    "run",
    help="Submit a new batch job from a JSON or CSV file.",
    context_settings={"help_option_names": HELP_OPTIONS},
)
def run_cmd(
    file: Path = typer.Option(..., "--file", "-f", exists=True, help="Input .json or .csv file."),
    provider: Provider = typer.Option(..., "--provider", "-p", help="LLM provider."),
    model: str = typer.Option(..., "--model", "-m", help="Model identifier (provider-specific)."),
    col: Optional[str] = typer.Option(None, "--col", "-c", help="Prompt column name (required for CSV)."),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Override env API key."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path."),
) -> None:
    console.print("[bold cyan]Spinning the threads...[/bold cyan] submitting batch.")
    try:
        meta = orchestrator.run_batch(
            file_path=file,
            provider_name=provider.value,
            model=model,
            column=col,
            api_key=api_key,
            output_path=output,
        )
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)

    console.print(f"[green]Batch submitted.[/green] id=[bold]{meta.batch_id}[/bold] provider={meta.provider}")
    console.print(f"Metadata saved to ~/.loom/batches/. Run [bold]loom fetch --id {meta.batch_id}[/bold] later.")


@app.command(
    "fetch",
    help="Check status / download results for a batch.",
    context_settings={"help_option_names": HELP_OPTIONS},
)
def fetch_cmd(
    batch_id: Optional[str] = typer.Option(None, "--id", "-i", help="Batch id to fetch."),
    all_: bool = typer.Option(False, "--all", "-a", help="Fetch all pending batches."),
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
    if not batch_id and not all_:
        console.print("[red]Provide --id or --all.[/red]")
        raise typer.Exit(code=2)

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

    for meta, done in targets:
        if done:
            suffix = "" if keep else " [dim](metadata removed)[/dim]"
            console.print(
                f"[green]Fabric complete.[/green] id={meta.batch_id} -> [bold]{meta.output_path}[/bold]{suffix}"
            )
        else:
            console.print(
                f"[yellow]Checking the loom...[/yellow] id={meta.batch_id} status={meta.status}"
            )


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


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()

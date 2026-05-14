from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from .core import orchestrator
from .utils import storage


app = typer.Typer(help="Loom — weave LLM batch jobs across providers.", no_args_is_help=True)
console = Console()


def _apply_api_key(provider: str, api_key: Optional[str]) -> Optional[str]:
    """If the user passed --api-key, expose it via the matching env var for the provider."""
    if not api_key:
        return None
    import os
    env_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
    }
    var = env_map.get(provider.lower())
    if var:
        os.environ[var] = api_key
    return api_key


@app.callback()
def _root() -> None:
    # Load .env from CWD if present. Existing env vars take precedence.
    load_dotenv(override=False)


@app.command()
def run(
    file: Path = typer.Option(..., "--file", "-f", exists=True, readable=True, help="Input .json or .csv"),
    provider: str = typer.Option(..., "--provider", "-p", help="openai | anthropic | google"),
    model: str = typer.Option(..., "--model", "-m", help="Model name (e.g. gpt-4o-mini, claude-3-5-sonnet-latest, gemini-2.5-flash)"),
    col: Optional[str] = typer.Option(None, "--col", help="CSV column containing the prompt (required for .csv)"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Override API key from env"),
) -> None:
    """Submit a batch."""
    _apply_api_key(provider, api_key)
    console.print(f"[dim]Spinning the threads...[/dim] submitting to [bold]{provider}[/bold] ({model})")
    meta = orchestrator.submit_batch(
        file_path=file,
        provider_name=provider,
        model=model,
        column=col,
        api_key=api_key,
    )
    console.print(f"[green]✓[/green] Batch submitted: [bold]{meta.batch_id}[/bold]")
    console.print(f"  State saved to ~/.loom/batches/{meta.provider}_{meta.batch_id}.json")


@app.command()
def fetch(
    id: Optional[str] = typer.Option(None, "--id", help="Specific batch id to fetch"),
    all_: bool = typer.Option(False, "--all", help="Fetch every locally-tracked batch"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Override API key from env"),
) -> None:
    """Fetch results for one batch (--id) or every pending batch (--all)."""
    if not id and not all_:
        raise typer.BadParameter("Provide either --id or --all.")

    targets: list[str]
    if all_:
        targets = [m.batch_id for m in storage.list_all()]
        if not targets:
            console.print("[dim]No batches tracked locally.[/dim]")
            return
    else:
        assert id is not None
        targets = [id]

    for batch_id in targets:
        meta = storage.load(batch_id)
        _apply_api_key(meta.provider, api_key)
        console.print(f"[dim]Checking the loom...[/dim] {meta.provider}/{batch_id}")
        try:
            updated, out = orchestrator.fetch_batch(batch_id, api_key=api_key)
        except Exception as e:
            console.print(f"  [red]✗[/red] {e}")
            continue
        if out is None:
            # Re-read the live status for display.
            from .providers import get_provider
            st = get_provider(meta.provider, api_key=api_key).status(batch_id)
            progress = ""
            if st.total:
                pct = int(100 * (st.completed or 0) / st.total)
                progress = f" — {st.completed}/{st.total} ({pct}%)"
            console.print(f"  [yellow]…[/yellow] still weaving: {st.raw}{progress}")
        else:
            console.print(f"  [green]✓[/green] Fabric complete. Results → [bold]{out}[/bold]")


@app.command("list")
def list_cmd() -> None:
    """Show every batch this machine is tracking."""
    rows = storage.list_all()
    if not rows:
        console.print("[dim]No batches tracked locally.[/dim]")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("Batch ID")
    table.add_column("Provider")
    table.add_column("Model")
    table.add_column("Input")
    table.add_column("Type")
    table.add_column("Created (UTC)")
    for m in rows:
        table.add_row(
            m.batch_id, m.provider, m.model,
            Path(m.original_file_path).name, m.file_type,
            m.created_at.strftime("%Y-%m-%d %H:%M"),
        )
    console.print(table)


@app.command()
def forget(batch_id: str = typer.Argument(..., help="Batch id to drop from local state")) -> None:
    """Remove a batch from local tracking (does not cancel it remotely)."""
    storage.delete(batch_id)
    console.print(f"[green]✓[/green] Forgot {batch_id}.")


if __name__ == "__main__":
    app()

"""CLI for acatome-extract."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(
    name="acatome-extract", help="Extract scientific papers into .acatome bundles."
)


@app.command()
def extract(
    path: Path = typer.Argument(..., help="PDF file or directory to extract"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output directory"),
    no_verify: bool = typer.Option(
        False, "--no-verify", help="Skip metadata verification"
    ),
):
    """Extract PDF(s) into .acatome bundle(s)."""
    from acatome_extract.pipeline import extract as do_extract
    from acatome_extract.pipeline import extract_dir

    verify = not no_verify

    if path.is_file():
        bundle = do_extract(path, output_dir=output, verify=verify)
        typer.echo(f"✓ {bundle}")
    elif path.is_dir():
        result = extract_dir(path, output_dir=output, verify=verify)
        typer.echo(
            f"✓ {len(result['succeeded'])} extracted, {len(result['failed'])} failed"
        )
    else:
        typer.echo(f"Error: {path} not found", err=True)
        raise typer.Exit(1)


@app.command()
def enrich(
    path: Path = typer.Argument(..., help=".acatome bundle or directory to enrich"),
    profile: str = typer.Option("default", "--profile", "-p", help="Embedding profile"),
    summarize: bool = typer.Option(
        True, "--summarize/--no-summarize", help="Generate summaries"
    ),
    summarizer: str = typer.Option(
        "", "--summarizer", help="litellm model spec (e.g. ollama/qwen3.5:9b)"
    ),
):
    """Add embeddings and summaries to a bundle (Phase 2)."""
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("acatome_extract.enrich").setLevel(logging.INFO)

    from acatome_extract.enrich import enrich as do_enrich
    from acatome_meta.config import load_config

    cfg = load_config()
    sm = summarizer or cfg.extract.enrich.summarizer

    bundles = [path] if path.is_file() else sorted(path.glob("*.acatome"))
    for b in bundles:
        try:
            do_enrich(b, profiles=[profile], summarize=summarize, summarizer=sm)
            typer.echo(f"✓ {b.name}")
        except Exception as e:
            typer.echo(f"✗ {b.name}: {e}", err=True)


@app.command()
def update_meta(
    path: Path = typer.Argument(..., help=".acatome bundle or directory"),
    no_verify: bool = typer.Option(False, "--no-verify", help="Skip verification"),
):
    """Re-run metadata lookup on existing bundle(s)."""
    from acatome_extract.bundle import read_bundle, update_bundle
    from acatome_meta.lookup import lookup
    from acatome_meta.verify import verify_metadata

    bundles = [path] if path.is_file() else sorted(path.glob("*.acatome"))
    for b in bundles:
        try:
            data = read_bundle(b)
            header = data["header"]
            bundle_path_str = str(b.resolve())
            # Re-lookup from bundle's stored PDF path or DOI
            new_header = lookup(doi=header.get("doi"), title=header.get("title"))
            # Merge — keep existing fields, update from new lookup
            for key in [
                "title",
                "authors",
                "year",
                "doi",
                "arxiv_id",
                "journal",
                "abstract",
                "entry_type",
                "s2_id",
                "source",
            ]:
                if new_header.get(key):
                    header[key] = new_header[key]
            # Re-verify
            if not no_verify and header.get("first_pages_text"):
                verified, warnings = verify_metadata(header, header["first_pages_text"])
                header["verified"] = verified
                header["verify_warnings"] = warnings
            data["header"] = header
            update_bundle(data, b)
            typer.echo(f"✓ {b.name}")
        except Exception as e:
            typer.echo(f"✗ {b.name}: {e}", err=True)


@app.command()
def strip(
    path: Path = typer.Argument(..., help=".acatome bundle or directory"),
    profile: str = typer.Option("default", "--profile", "-p", help="Profile to strip"),
):
    """Remove an embedding profile from a bundle."""
    from acatome_extract.bundle import read_bundle, update_bundle

    bundles = [path] if path.is_file() else sorted(path.glob("*.acatome"))
    for b in bundles:
        try:
            data = read_bundle(b)
            stripped = 0
            for block in data.get("blocks", []):
                embs = block.get("embeddings", {})
                if profile in embs:
                    del embs[profile]
                    stripped += 1
            update_bundle(data, b)
            typer.echo(f"✓ {b.name}: stripped {stripped} embeddings for '{profile}'")
        except Exception as e:
            typer.echo(f"✗ {b.name}: {e}", err=True)


@app.command()
def attach(
    parent: str = typer.Argument(..., help="Parent paper slug, DOI, or ref_id"),
    pdf: Path = typer.Argument(..., help="Supplement PDF file"),
    name: str = typer.Option(
        "", "--name", "-n", help="Supplement name (default: auto-detect from filename)"
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Bundle output dir"
    ),
    no_enrich: bool = typer.Option(False, "--no-enrich", help="Skip enrichment"),
    summarizer: str = typer.Option("", "--summarizer", help="litellm model spec"),
):
    """Extract a supplement PDF and attach it to an existing paper.

    Auto-detects supplement name from filename: paper_S1.pdf → "s1".
    Everything after the last '_' before .pdf becomes the supplement name.
    """
    import re

    from acatome_extract.pipeline import extract as do_extract

    if not pdf.is_file():
        typer.echo(f"Error: {pdf} not found", err=True)
        raise typer.Exit(1)

    # Auto-detect supplement name from filename
    if not name:
        stem = pdf.stem
        if "_" in stem:
            name = stem.rsplit("_", 1)[-1].lower()
        else:
            typer.echo("Cannot auto-detect supplement name. Use --name.", err=True)
            raise typer.Exit(1)

    typer.echo(f"Extracting {pdf.name} as supplement '{name}' for {parent}...")
    bundle = do_extract(pdf, output_dir=output)

    if not no_enrich:
        from acatome_extract.enrich import enrich as do_enrich
        from acatome_meta.config import load_config

        cfg = load_config()
        sm = summarizer or cfg.extract.enrich.summarizer
        do_enrich(bundle, profiles=["default"], summarize=True, summarizer=sm)

    try:
        from acatome_store.store import Store
    except ImportError:
        typer.echo(f"✓ Bundle: {bundle}")
        typer.echo(
            "acatome-store not installed — skipping ingest. Ingest manually.", err=True
        )
        raise typer.Exit(0)

    store = Store()
    ref_id = store.ingest_supplement(parent, bundle, name)
    store.close()
    typer.echo(f"✓ Supplement '{name}' attached to ref_id={ref_id}")


@app.command()
def watch(
    path: Path = typer.Argument(..., help="Directory to watch for new PDFs"),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Bundle output dir"
    ),
    no_recursive: bool = typer.Option(
        False, "--no-recursive", help="Don't watch subdirectories"
    ),
    no_backfill: bool = typer.Option(
        False, "--no-backfill", help="Don't process existing PDFs on startup"
    ),
    no_enrich: bool = typer.Option(False, "--no-enrich", help="Skip enrichment"),
    no_ingest: bool = typer.Option(False, "--no-ingest", help="Skip store ingestion"),
    summarizer: str = typer.Option(
        "", "--summarizer", help="litellm model spec (e.g. ollama/qwen3.5:9b)"
    ),
    keep: bool = typer.Option(False, "--keep", help="Don't move PDFs after processing"),
    user: str = typer.Option(
        "", "--user", "-u", help="User name for ingest log (default: OS user)"
    ),
    debounce: float = typer.Option(
        3.0, "--debounce", help="File stability wait (secs)"
    ),
    poll: bool = typer.Option(
        False, "--poll", help="Use polling observer (for network mounts)"
    ),
):
    """Watch a directory and auto-ingest new PDFs.

    Monitors for new PDF files and runs: extract → enrich → ingest.
    Processed PDFs move to completed/, failures to errors/.
    Recursive by default.
    """
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
    # Ensure extraction and watch loggers are visible
    for name in ("acatome.watch", "acatome_extract.marker", "acatome_extract.enrich"):
        logging.getLogger(name).setLevel(logging.INFO)

    from acatome_extract.watch import watch as do_watch

    do_watch(
        path,
        output_dir=output,
        recursive=not no_recursive,
        backfill=not no_backfill,
        enrich=not no_enrich,
        summarizer=summarizer,
        ingest=not no_ingest,
        keep=keep,
        user=user,
        debounce=debounce,
        use_polling=poll,
    )


if __name__ == "__main__":
    app()

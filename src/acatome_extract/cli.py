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
    doc_type: str = typer.Option(
        "article",
        "--type",
        "-t",
        help="Document type: article, datasheet, manual, techreport, notes, other",
    ),
    skip_existing: bool = typer.Option(
        False, "--skip-existing", help="Skip PDFs that already have a .acatome bundle"
    ),
):
    """Extract PDF(s) into .acatome bundle(s)."""
    from acatome_extract.pipeline import extract as do_extract

    verify = not no_verify

    if path.is_file():
        bundle = do_extract(path, output_dir=output, verify=verify, doc_type=doc_type)
        typer.echo(f"✓ {bundle}")
    elif path.is_dir():
        pdfs = sorted(path.resolve().glob("*.pdf"))
        if not pdfs:
            typer.echo(f"No PDFs found in {path}")
            raise typer.Exit(1)
        typer.echo(f"Found {len(pdfs)} PDFs in {path}")
        succeeded, failed, skipped = 0, 0, 0
        for i, pdf in enumerate(pdfs, 1):
            # Skip if .acatome bundle already exists next to the PDF
            if skip_existing and pdf.with_suffix(".acatome").exists():
                skipped += 1
                typer.echo(f"  [{i}/{len(pdfs)}] ⏭ {pdf.name} (bundle exists)")
                continue
            try:
                bundle = do_extract(pdf, output_dir=output, verify=verify, doc_type=doc_type)
                succeeded += 1
                typer.echo(f"  [{i}/{len(pdfs)}] ✓ {pdf.name} → {bundle.name}")
            except Exception as e:
                failed += 1
                typer.echo(f"  [{i}/{len(pdfs)}] ✗ {pdf.name}: {e}", err=True)
        typer.echo(
            f"\nDone: {succeeded} extracted, {skipped} skipped, {failed} failed"
        )
    else:
        typer.echo(f"Error: {path} not found", err=True)
        raise typer.Exit(1)


@app.command()
def enrich(
    path: Path = typer.Argument(..., help=".acatome bundle or directory to enrich"),
    profile: str = typer.Option("default", "--profile", "-p", help="Embedding profile"),
    summarize: bool = typer.Option(
        False, "--summarize/--no-summarize", help="Generate LLM summaries (default: off)"
    ),
    summarizer: str = typer.Option(
        "", "--summarizer", help="litellm model spec (e.g. ollama/qwen3.5:9b)"
    ),
    skip_existing: bool = typer.Option(
        False, "--skip-existing", help="Skip bundles that already have LLM summaries"
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

    from acatome_extract.bundle import read_bundle
    from acatome_extract.enrich import enrich as do_enrich
    from acatome_meta.config import load_config

    cfg = load_config()
    sm = summarizer or cfg.extract.enrich.summarizer

    bundles = [path] if path.is_file() else sorted(path.rglob("*.acatome"))
    skipped = 0
    for b in bundles:
        try:
            if skip_existing and summarize:
                data = read_bundle(b)
                if _has_llm_summaries(data):
                    skipped += 1
                    continue
            do_enrich(b, profiles=[profile], summarize=summarize, summarizer=sm)
            typer.echo(f"✓ {b.name}")
        except Exception as e:
            typer.echo(f"✗ {b.name}: {e}", err=True)
    if skipped:
        typer.echo(f"⊘ {skipped} bundles skipped (already have LLM summaries)")


@app.command()
def update_meta(
    path: Path = typer.Argument(..., help=".acatome bundle or directory"),
    no_verify: bool = typer.Option(False, "--no-verify", help="Skip verification"),
):
    """Re-run metadata lookup on existing bundle(s)."""
    from acatome_extract.bundle import read_bundle, update_bundle
    from acatome_meta.lookup import lookup
    from acatome_meta.verify import verify_metadata

    bundles = [path] if path.is_file() else sorted(path.rglob("*.acatome"))
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

    bundles = [path] if path.is_file() else sorted(path.rglob("*.acatome"))
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
    no_summarize: bool = typer.Option(True, "--no-summarize/--summarize", help="Skip LLM summaries (RAKE still runs at extract)"),
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
        summarize=not no_summarize,
        summarizer=summarizer,
        ingest=not no_ingest,
        keep=keep,
        user=user,
        debounce=debounce,
        use_polling=poll,
    )


def _has_llm_summaries(data: dict) -> bool:
    """Check if any block in the bundle already has an LLM summary."""
    for b in data.get("blocks", []):
        for key in b.get("summaries", {}):
            if key.startswith("llm:"):
                return True
    return False


@app.command()
def migrate(
    path: Path = typer.Argument(..., help=".acatome bundle or directory to migrate"),
    rake: bool = typer.Option(
        True, "--rake/--no-rake", help="Add RAKE summaries to blocks that lack them"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report what would change without writing"
    ),
):
    """Migrate bundles from old summary format to new summaries dict.

    Converts block["summary"] → block["summaries"]["llm:unknown"] (or {} if null).
    Optionally adds RAKE summaries (default: yes, instant).
    """
    from acatome_extract.bundle import read_bundle, update_bundle
    from acatome_extract.pipeline import _rake_summarize_blocks

    bundles = [path] if path.is_file() else sorted(path.rglob("*.acatome"))
    if not bundles:
        typer.echo("No .acatome files found.", err=True)
        raise typer.Exit(1)

    migrated = 0
    already_ok = 0
    raked = 0

    for b in bundles:
        try:
            data = read_bundle(b)
            changed = False

            # Migrate blocks
            for block in data.get("blocks", []):
                if "summary" in block and "summaries" not in block:
                    old = block.pop("summary")
                    block["summaries"] = {}
                    if old:
                        block["summaries"]["llm:unknown"] = old
                    changed = True
                elif "summaries" not in block:
                    block["summaries"] = {}
                    if "summary" in block:
                        block.pop("summary")
                    changed = True

            # Migrate enrichment_meta paper_summary → paper_summaries
            em = data.get("enrichment_meta") or {}
            if em.get("paper_summary") and not em.get("paper_summaries"):
                em["paper_summaries"] = {"llm:unknown": em["paper_summary"]}
                data["enrichment_meta"] = em
                changed = True

            # Add RAKE summaries
            if rake:
                blocks = data.get("blocks", [])
                before_keys = sum(
                    1 for bl in blocks if bl.get("summaries", {}).get("rake")
                )
                blocks = _rake_summarize_blocks(blocks)
                after_keys = sum(
                    1 for bl in blocks if bl.get("summaries", {}).get("rake")
                )
                if after_keys > before_keys:
                    data["blocks"] = blocks
                    changed = True
                    raked += 1

            if changed:
                if not dry_run:
                    update_bundle(data, b)
                migrated += 1
                typer.echo(f"✓ {b.name}")
            else:
                already_ok += 1
        except Exception as e:
            typer.echo(f"✗ {b.name}: {e}", err=True)

    prefix = "[dry-run] " if dry_run else ""
    typer.echo(
        f"{prefix}{migrated} migrated, {already_ok} already current"
        + (f", {raked} got RAKE summaries" if rake else "")
    )


if __name__ == "__main__":
    app()
